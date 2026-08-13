import atexit
import importlib.metadata
import os
from types import SimpleNamespace

import pytest
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.models.model_output import TargetModelOutput
from nanovllm.utils.context import reset_context, set_context


pytestmark = [pytest.mark.cuda, pytest.mark.model_weights]


def _checkpoint_path(name):
    path = os.environ.get(name)
    if not path:
        pytest.skip(f"{name} is not set")
    if not os.path.isdir(path):
        pytest.skip(f"{name} does not point to a local directory: {path}")
    return path


@pytest.fixture(scope="module")
def eagle_llm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    target_path = _checkpoint_path("NANOVLLM_TARGET_MODEL")
    draft_path = _checkpoint_path("NANOVLLM_EAGLE3_MODEL")
    llm = LLM(
        target_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        speculative_config={
            "method": "eagle3",
            "draft_model": draft_path,
            "num_speculative_tokens": 5,
        },
    )
    try:
        yield llm
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


def _set_single_prefill_context(num_tokens, device):
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
        slot_mapping=torch.full(
            (num_tokens,),
            -1,
            dtype=torch.int32,
            device=device,
        ),
    )


def _target_decode_one(runner, seq, token_id, position, device):
    block_tables = torch.tensor(
        [seq.block_table],
        dtype=torch.int32,
        device=device,
    )
    set_context(
        False,
        slot_mapping=torch.tensor(
            [seq.block_table[position // runner.block_size] * runner.block_size
             + position % runner.block_size],
            dtype=torch.int32,
            device=device,
        ),
        context_lens=torch.tensor(
            [position + 1],
            dtype=torch.int32,
            device=device,
        ),
        block_tables=block_tables,
    )
    try:
        hidden = runner.model(
            torch.tensor([token_id], dtype=torch.long, device=device),
            torch.tensor([position], dtype=torch.long, device=device),
        )
        return runner.model.lm_head(hidden, return_all_logits=True)
    finally:
        reset_context()


def test_strict_checkpoint_loading_has_no_unresolved_weights(eagle_llm):
    report = eagle_llm.model_runner.draft_load_report

    assert report is not None
    assert report.missing == ()
    assert report.unexpected == ()
    assert report.injected == ("model.embed_tokens.weight",)


def test_target_logits_are_unchanged_by_auxiliary_capture(eagle_llm):
    runner = eagle_llm.model_runner
    seq = Sequence([1, 2, 3, 4])
    eagle_llm.scheduler.block_manager.allocate(seq)
    try:
        input_ids, positions = runner.prepare_prefill([seq])
        plain_hidden = runner.model(input_ids, positions)
        plain_logits = runner.model.lm_head(
            plain_hidden,
            return_all_logits=True,
        )
        reset_context()

        input_ids, positions = runner.prepare_prefill([seq])
        captured = runner.model(
            input_ids,
            positions,
            runner.speculative_config.auxiliary_layer_ids,
        )
        assert isinstance(captured, TargetModelOutput)
        captured_logits = runner.model.lm_head(
            captured.hidden_states,
            return_all_logits=True,
        )
        torch.testing.assert_close(
            captured_logits,
            plain_logits,
            rtol=2e-2,
            atol=2e-2,
        )
    finally:
        reset_context()
        eagle_llm.scheduler.block_manager.deallocate(seq)


@pytest.mark.parametrize("num_tokens", [4, 8])
def test_target_verification_logits_match_token_by_token_decode(
    eagle_llm,
    num_tokens,
):
    runner = eagle_llm.model_runner
    manager = eagle_llm.scheduler.block_manager
    token_ids = list(range(1, num_tokens + 1))
    step_seq = Sequence(token_ids)
    verify_seq = Sequence(token_ids)
    manager.allocate(step_seq)
    manager.allocate(verify_seq)
    device = runner.kv_cache.device
    prefix_length = 2
    try:
        step_logits = []
        for position, token_id in enumerate(token_ids):
            step_logits.append(
                _target_decode_one(
                    runner,
                    step_seq,
                    token_id,
                    position,
                    device,
                )
            )
        step_logits = torch.cat(step_logits, dim=0)

        for position in range(prefix_length):
            _target_decode_one(
                runner,
                verify_seq,
                token_ids[position],
                position,
                device,
            )
        suffix = token_ids[prefix_length:]
        suffix_positions = list(range(prefix_length, num_tokens))
        slot_mapping = [
            verify_seq.block_table[position // runner.block_size]
            * runner.block_size
            + position % runner.block_size
            for position in suffix_positions
        ]
        set_context(
            True,
            cu_seqlens_q=torch.tensor(
                [0, len(suffix)],
                dtype=torch.int32,
                device=device,
            ),
            cu_seqlens_k=torch.tensor(
                [0, num_tokens],
                dtype=torch.int32,
                device=device,
            ),
            max_seqlen_q=len(suffix),
            max_seqlen_k=num_tokens,
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                device=device,
            ),
            block_tables=torch.tensor(
                [verify_seq.block_table],
                dtype=torch.int32,
                device=device,
            ),
        )
        try:
            verification_hidden = runner.model(
                torch.tensor(suffix, dtype=torch.long, device=device),
                torch.tensor(
                    suffix_positions,
                    dtype=torch.long,
                    device=device,
                ),
            )
            verification_logits = runner.model.lm_head(
                verification_hidden,
                return_all_logits=True,
            )
        finally:
            reset_context()

        torch.testing.assert_close(
            verification_logits,
            step_logits[prefix_length:],
            rtol=2e-2,
            atol=2e-2,
        )
    finally:
        reset_context()
        manager.deallocate(step_seq)
        manager.deallocate(verify_seq)


@pytest.mark.parametrize("num_tokens", [4, 8])
def test_draft_hidden_and_logits_match_vllm_adapter_semantics(
    eagle_llm,
    num_tokens,
    capsys,
):
    try:
        import vllm
        from vllm.model_executor.models.qwen3_eagle3 import (
            Eagle3Qwen3ForCausalLM,
            Qwen3Eagle3Model,
        )
    except Exception as exc:
        pytest.skip(f"vLLM Qwen3 EAGLE3 adapter unavailable: {exc}")

    version = importlib.metadata.version("vllm")
    commit = getattr(vllm, "__commit__", "unknown")
    print(f"vLLM reference version: {version}; commit: {commit}")
    draft = eagle_llm.model_runner.draft_model
    device = draft.model.fc.weight.device
    generator = torch.Generator(device=device).manual_seed(1234 + num_tokens)
    auxiliary = torch.randn(
        num_tokens,
        3 * draft.config.hidden_size,
        generator=generator,
        device=device,
        dtype=draft.model.fc.weight.dtype,
    )
    input_ids = torch.arange(1, num_tokens + 1, device=device)
    positions = torch.arange(num_tokens, device=device)

    reference_clm = SimpleNamespace(
        model=SimpleNamespace(
            use_aux_hidden_state=True,
            norm_before_fc=False,
            fc_norm=None,
            fc=draft.model.fc,
        )
    )
    reference_fused = Eagle3Qwen3ForCausalLM.combine_hidden_states(
        reference_clm,
        auxiliary,
    )
    actual_fused = draft.combine_hidden_states(auxiliary)
    torch.testing.assert_close(actual_fused, reference_fused)

    _set_single_prefill_context(num_tokens, device)
    try:
        actual_hidden, actual_auxiliary = draft(
            input_ids,
            positions,
            actual_fused,
        )
    finally:
        reset_context()

    reference_model = SimpleNamespace(
        embed_input_ids=lambda ids: draft.model.embed_tokens(ids),
        layers=draft.model.layers,
        norm=draft.model.norm,
        norm_output=False,
    )
    _set_single_prefill_context(num_tokens, device)
    try:
        reference_hidden, reference_auxiliary = Qwen3Eagle3Model.forward(
            reference_model,
            input_ids,
            positions,
            actual_fused,
        )
    finally:
        reset_context()
    torch.testing.assert_close(actual_hidden, reference_hidden, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        actual_auxiliary,
        reference_auxiliary,
        rtol=2e-2,
        atol=2e-2,
    )

    reference_logits_model = SimpleNamespace(
        logits_processor=lambda lm_head, hidden: draft.lm_head(
            hidden,
            return_all_logits=True,
        ),
        lm_head=None,
        draft_id_to_target_id=draft.draft_id_to_target_id,
        config=draft.config,
    )
    actual_logits = draft.compute_logits(actual_hidden)
    reference_logits = Eagle3Qwen3ForCausalLM.compute_logits(
        reference_logits_model,
        reference_hidden,
    )
    torch.testing.assert_close(actual_logits, reference_logits)
    assert version in capsys.readouterr().out


def test_eagle3_generates_ragged_repeated_and_block_boundary_requests(eagle_llm):
    prompts = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        list(range(1, 256)),
    ]
    params = [
        SamplingParams(temperature=0.8, max_tokens=1),
        SamplingParams(temperature=0.8, max_tokens=7),
        SamplingParams(temperature=0.8, max_tokens=4),
    ]

    outputs = eagle_llm.generate(prompts, params, use_tqdm=False)

    assert len(outputs) == 3
    assert len(outputs[0]["token_ids"]) <= 1
    assert len(outputs[1]["token_ids"]) <= 7
    assert len(outputs[2]["token_ids"]) <= 4


def test_eos_prefill_releases_eagle_state(eagle_llm, monkeypatch):
    runner = eagle_llm.model_runner
    eos_token_id = eagle_llm.scheduler.eos

    class EosSampler:
        def __call__(self, logits, temperatures):
            return torch.full(
                (logits.size(0),),
                eos_token_id,
                dtype=torch.long,
                device=logits.device,
            )

    monkeypatch.setattr(runner, "sampler", EosSampler())
    outputs = eagle_llm.generate(
        [[1, 2, 3, 4]],
        SamplingParams(temperature=0.8, max_tokens=4),
        use_tqdm=False,
    )

    assert outputs[0]["token_ids"] == [eos_token_id]
    assert runner.eagle3_proposer.states == {}


def test_repeated_generation_does_not_monotonically_leak_allocated_memory(
    eagle_llm,
):
    prompt = [[1, 2, 3, 4]]
    params = SamplingParams(temperature=0.8, max_tokens=8)
    allocated = []
    for _ in range(3):
        eagle_llm.generate(prompt, params, use_tqdm=False)
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())

    assert allocated[-1] <= allocated[0] + 16 * 1024 * 1024
