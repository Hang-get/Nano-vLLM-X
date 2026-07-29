import pickle
import torch
import torch.distributed as dist
import numpy as np
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model, load_model_arch_from_config
from nanovllm.v1.sample.rejection_sampler import RejectionSampler
from nanovllm.v1.spec_decode.ngram_proposer import NgramProposer


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        model_cls = load_model_arch_from_config(hf_config)
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.speculative_config = config.speculative_config
        if self.speculative_config is not None:
            self.drafter = NgramProposer(
                prompt_lookup_min=self.speculative_config.prompt_lookup_min,
                prompt_lookup_max=self.speculative_config.prompt_lookup_max,
                num_speculative_tokens=self.speculative_config.num_speculative_tokens,
                max_model_len=config.max_model_len,
                max_num_seqs=config.max_num_seqs,
            )
            self.rejection_sampler = RejectionSampler(self.sampler)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = max(1, hf_config.num_key_value_heads // self.world_size)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def propose_draft_token_ids(self, seqs: list[Sequence]) -> list[list[int]]:
        if self.speculative_config is None:
            raise RuntimeError("Speculative decoding is not configured")
        max_model_len = self.config.max_model_len
        num_tokens_no_spec = np.empty(len(seqs), dtype=np.int32)
        token_ids_cpu = np.zeros((len(seqs), max_model_len), dtype=np.int32)
        for index, seq in enumerate(seqs):
            seq_len = min(len(seq), max_model_len)
            num_tokens_no_spec[index] = seq_len
            token_ids_cpu[index, :seq_len] = np.asarray(
                seq.token_ids[:seq_len], dtype=np.int32
            )
        return self.drafter.propose(num_tokens_no_spec, token_ids_cpu)

    def prepare_spec_decode(
        self,
        seqs: list[Sequence],
        draft_token_ids: list[list[int]],
        reservations: list[dict[str, list[int] | int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids: list[int] = []
        positions: list[int] = []
        draft_row_indices: list[int] = []
        bonus_row_indices: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables_list: list[list[int]] = []
        offset = 0
        for seq, seq_draft_token_ids, reservation in zip(
            seqs, draft_token_ids, reservations
        ):
            seq_start_pos = seq.num_computed_tokens
            seq_total_len = len(seq) + len(seq_draft_token_ids)
            seq_input_ids = seq.token_ids[seq_start_pos:] + seq_draft_token_ids
            q_len = len(seq_input_ids)
            if q_len == 0:
                raise RuntimeError("Speculative decode requires an uncomputed token")
            block_table = seq.block_table + reservation["new_block_ids"]
            input_ids.extend(seq_input_ids)
            positions.extend(range(seq_start_pos, seq_total_len))
            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seq_total_len)
            max_seqlen_q = max(max_seqlen_q, q_len)
            max_seqlen_k = max(max_seqlen_k, seq_total_len)
            block_tables_list.append(block_table)
            for position in range(seq_start_pos, seq_total_len):
                block_id = block_table[position // self.block_size]
                slot_mapping.append(block_id * self.block_size + position % self.block_size)

            num_drafts = len(seq_draft_token_ids)
            draft_start = q_len - num_drafts - 1
            if num_drafts:
                draft_row_indices.extend(
                    range(offset + draft_start, offset + draft_start + num_drafts)
                )
            bonus_row_indices.append(offset + q_len - 1)
            offset += q_len

        input_ids_tensor = torch.tensor(
            input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions_tensor = torch.tensor(
            positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        verify_row_indices = torch.tensor(
            draft_row_indices + bonus_row_indices,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping_tensor,
            None,
            self.prepare_block_tables_from_lists(block_tables_list),
        )
        return input_ids_tensor, positions_tensor, verify_row_indices

    def prepare_block_tables_from_lists(self, block_tables_list: list[list[int]]):
        max_len = max(len(block_table) for block_table in block_tables_list)
        block_tables = [
            block_table + [-1] * (max_len - len(block_table))
            for block_table in block_tables_list
        ]
        return torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

    def run_spec_decode(
        self,
        seqs: list[Sequence],
        draft_token_ids: list[list[int]],
        reservations: list[dict[str, list[int] | int]],
    ) -> list[list[int]] | None:
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        input_ids, positions, verify_row_indices = self.prepare_spec_decode(
            seqs, draft_token_ids, reservations
        )
        verify_logits = self.run_model(
            input_ids,
            positions,
            is_prefill=True,
            spec_row_indices=verify_row_indices,
        )
        token_ids = (
            self.rejection_sampler(draft_token_ids, verify_logits, temperatures)
            if self.rank == 0
            else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        spec_row_indices: torch.Tensor | None = None,
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)
            if spec_row_indices is not None:
                return self.model.compute_logits(
                    hidden_states, return_all_logits=True
                )[spec_row_indices]
            return self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
