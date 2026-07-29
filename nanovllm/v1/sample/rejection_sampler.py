import torch
import torch.nn as nn
import triton
import triton.language as tl
from nanovllm.layers.sampler import Sampler
from nanovllm.v1.spec_decode.types import SpecDecodeResult

PLACEHOLDER_TOKEN_ID = -1


@triton.jit
def _sample_recovered_tokens_kernel(
    output_token_ids_ptr,       # [num_tokens]
    cu_num_draft_tokens_ptr,    # [batch_size]
    draft_token_ids_ptr,        # [num_tokens]
    draft_probs_ptr,            # [num_tokens, vocab_size]
    target_probs_ptr,           # [num_tokens, vocab_size]
    q_ptr,
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int32)
    start_idx = tl.zeros((), dtype=tl.int32)
    if req_idx > 0:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx).to(tl.int32)
    num_draft_tokens = end_idx - start_idx

    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    row_idx = start_idx + pos
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + row_idx)
        prob = tl.load(
            target_probs_ptr + row_idx * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0.0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + row_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0.0,
        )
        target_prob = tl.load(
            target_probs_ptr + row_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0.0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0.0)

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + row_idx, recovered_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def _rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    accepted_counts_ptr,  # [batch_size]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size]
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int32)
    start_idx = tl.zeros((), dtype=tl.int32)
    if req_idx > 0:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx).to(tl.int32)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    accepted_count = tl.zeros((), dtype=tl.int32)
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            draft_token_id_i32 = draft_token_id.to(tl.int32)

            if NO_DRAFT_PROBS:
                draft_prob = 1.0
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id_i32
                )

            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id_i32
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
                accepted_count += 1
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)

            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
    tl.store(accepted_counts_ptr + req_idx, accepted_count)


@triton.jit
def _uniform_probs_kernel(
    output_ptr,
    num_tokens,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    uniform = tl.rand(seed, offsets)
    uniform = tl.maximum(uniform, 1e-7)
    tl.store(output_ptr + offsets, uniform, mask=mask)


class RejectionSampler(nn.Module):
    """
    A lightweight rejection sampler for speculative decoding.

    This implementation is intentionally simple and is sufficient for
    ngram-based drafting (where draft_probs is None).
    """

    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler

    def forward(
        self,
        draft_token_ids: list[list[int]],
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
        acceptance_uniforms: torch.Tensor | None = None,
        recovery_noise: torch.Tensor | None = None,
    ) -> SpecDecodeResult:
        """
        Args:
            draft_token_ids:
                Draft tokens proposed by drafter. Shape: [bs, <=k]
            logits:
                Flattened target logits in the same layout as vLLM:
                [sum(num_draft_tokens) + bs, vocab_size].
                First part is per-draft-position logits, second part is
                one bonus-token logit row for each request.
            temperatures:
                Sampling temperatures. Shape: [bs]
            draft_probs:
                Optional draft model probabilities for each draft position.
                Shape: [sum(num_draft_tokens), vocab_size]. For ngram, keep None.
            acceptance_uniforms:
                Optional injected acceptance uniforms in [0, 1].
                Shape: [sum(num_draft_tokens)].
            recovery_noise:
                Optional injected exponential recovery noise.
                Shape: [batch_size, vocab_size].
        Returns:
            Output token IDs and the number of actually accepted draft tokens
            for each request.
        """
        batch_size = len(draft_token_ids)
        if batch_size == 0:
            return SpecDecodeResult(
                output_token_ids=[],
                accepted_draft_counts=[],
            )
        if temperatures.ndim != 1 or temperatures.size(0) != batch_size:
            raise ValueError("temperatures must have shape [batch_size]")

        num_draft_tokens = [len(x) for x in draft_token_ids]
        total_num_draft_tokens = sum(num_draft_tokens)
        expected_rows = total_num_draft_tokens + batch_size

        if logits.ndim != 2 or logits.size(0) != expected_rows:
            raise ValueError(
                f"logits must have shape [{expected_rows}, vocab_size], "
                f"got {tuple(logits.shape)}"
            )
        if draft_probs is not None:
            expected_shape = (total_num_draft_tokens, logits.size(-1))
            if tuple(draft_probs.shape) != expected_shape:
                raise ValueError(
                    "draft_probs shape mismatch: "
                    f"expected {expected_shape}, got {tuple(draft_probs.shape)}"
                )
            if draft_probs.device != logits.device:
                raise ValueError("draft_probs must be on the logits device")
            if draft_probs.dtype != torch.float32:
                raise ValueError("draft_probs must have dtype torch.float32")

        if recovery_noise is not None:
            expected_noise_shape = (batch_size, logits.size(-1))
            if tuple(recovery_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"recovery_noise must have shape {expected_noise_shape}"
                )
            if recovery_noise.device != logits.device:
                raise ValueError("recovery_noise must be on the logits device")
            if not bool(torch.isfinite(recovery_noise).all()) or bool(
                (recovery_noise <= 0).any()
            ):
                raise ValueError(
                    "recovery_noise values must be finite and positive"
                )

        if acceptance_uniforms is not None:
            expected_uniform_shape = (total_num_draft_tokens,)
            if tuple(acceptance_uniforms.shape) != expected_uniform_shape:
                raise ValueError(
                    f"acceptance_uniforms must have shape {expected_uniform_shape}"
                )
            if acceptance_uniforms.device != logits.device:
                raise ValueError(
                    "acceptance_uniforms must be on the logits device"
                )
            if not bool(torch.isfinite(acceptance_uniforms).all()) or bool(
                ((acceptance_uniforms < 0) | (acceptance_uniforms > 1)).any()
            ):
                raise ValueError(
                    "acceptance_uniforms values must be finite and in [0, 1]"
                )

        bonus_logits = logits[total_num_draft_tokens:]
        if recovery_noise is None:
            bonus_token_ids = self.sampler(bonus_logits, temperatures).to(torch.int64)
        else:
            bonus_probs = torch.softmax(
                bonus_logits.to(torch.float32)
                / temperatures.to(torch.float32).unsqueeze(-1),
                dim=-1,
            )
            bonus_token_ids = bonus_probs.div(recovery_noise).argmax(dim=-1)

        if total_num_draft_tokens == 0:
            return SpecDecodeResult(
                output_token_ids=[
                    [int(bonus_token_ids[i].item())] for i in range(batch_size)
                ],
                accepted_draft_counts=[0] * batch_size,
            )

        target_logits = logits[:total_num_draft_tokens].to(torch.float32)
        token_temperatures = _expand_batch_to_tokens(
            temperatures.to(torch.float32),
            num_draft_tokens,
            total_num_draft_tokens,
            target_logits.device,
        )
        target_logits = target_logits.div(token_temperatures.unsqueeze(-1))
        target_probs = torch.softmax(target_logits, dim=-1, dtype=torch.float32)

        max_spec_len = max(num_draft_tokens)
        cu_num_draft_tokens = torch.tensor(
            num_draft_tokens,
            dtype=torch.int32,
            device=logits.device,
        ).cumsum(dim=0)
        flat_draft_token_ids = torch.tensor(
            [token_id for row in draft_token_ids for token_id in row],
            dtype=torch.int64,
            device=logits.device,
        )
        if acceptance_uniforms is None:
            uniform_probs = generate_uniform_probs(
                total_num_draft_tokens,
                logits.device,
            )
        else:
            uniform_probs = acceptance_uniforms.to(torch.float32).contiguous()
        recovered_token_ids = sample_recovered_tokens(
            num_draft_tokens,
            cu_num_draft_tokens,
            flat_draft_token_ids,
            draft_probs,
            target_probs,
            logits.device,
            recovery_noise,
        )

        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int64,
            device=logits.device,
        )
        accepted_counts = torch.empty(
            (batch_size,), dtype=torch.int32, device=logits.device
        )
        no_draft_probs = draft_probs is None
        draft_probs_for_kernel = (
            torch.empty_like(target_probs) if no_draft_probs else draft_probs
        )

        _rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            accepted_counts,
            cu_num_draft_tokens,
            flat_draft_token_ids,
            draft_probs_for_kernel,
            target_probs,
            bonus_token_ids.to(torch.int64),
            recovered_token_ids,
            uniform_probs,
            max_spec_len,
            target_probs.size(-1),
            NO_DRAFT_PROBS=no_draft_probs,
        )

        output_token_ids_cpu = output_token_ids.cpu().tolist()
        output_rows = [
            [token_id for token_id in row if token_id != PLACEHOLDER_TOKEN_ID]
            for row in output_token_ids_cpu
        ]
        return SpecDecodeResult(
            output_token_ids=output_rows,
            accepted_draft_counts=accepted_counts.cpu().tolist(),
        )


def generate_uniform_probs(
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if num_tokens == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if device.type != "cuda":
        raise RuntimeError("Triton uniform sampling requires CUDA device.")

    uniform_probs = torch.empty(
        (num_tokens,),
        dtype=torch.float32,
        device=device,
    )
    seed = int(
        torch.randint(
            0,
            2**31 - 1,
            (1,),
            device="cpu",
            dtype=torch.int64,
        ).item()
    )
    block_size = 1024
    grid = (triton.cdiv(num_tokens, block_size),)
    _uniform_probs_kernel[grid](
        uniform_probs,
        num_tokens,
        seed,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return uniform_probs


def sample_recovered_tokens(
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    device: torch.device,
    recovery_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if device.type != "cuda":
        raise RuntimeError("Triton recovered token sampling requires CUDA device.")

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.numel()
    vocab_size = target_probs.size(-1)
    if recovery_noise is None:
        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        q.exponential_()
    else:
        q = recovery_noise.to(torch.float32).contiguous()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    draft_probs_for_kernel = target_probs if draft_probs is None else draft_probs
    max_spec_len = max(num_draft_tokens)
    _sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs_for_kernel,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


def _expand_batch_to_tokens(
    values: torch.Tensor,
    num_tokens_per_request: list[int],
    total_num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    expanded = torch.empty(total_num_tokens, dtype=values.dtype, device=device)
    start = 0
    for idx, count in enumerate(num_tokens_per_request):
        if count > 0:
            expanded[start : start + count] = values[idx]
            start += count
    return expanded


def reference_rejection_sample(
    draft_token_ids: list[list[int]],
    target_logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_probs: torch.Tensor | None,
    acceptance_uniforms: torch.Tensor,
    recovery_uniforms: torch.Tensor,
) -> SpecDecodeResult:
    batch_size = len(draft_token_ids)
    if temperatures.ndim != 1 or temperatures.numel() != batch_size:
        raise ValueError("temperatures must have shape [batch_size]")
    lengths = [len(row) for row in draft_token_ids]
    total_drafts = sum(lengths)
    expected_rows = total_drafts + batch_size
    if target_logits.ndim != 2 or target_logits.size(0) != expected_rows:
        raise ValueError(
            f"target_logits must have shape [{expected_rows}, vocab_size]"
        )
    vocab_size = target_logits.size(1)
    if draft_probs is not None and tuple(draft_probs.shape) != (
        total_drafts,
        vocab_size,
    ):
        raise ValueError("draft_probs shape mismatch")
    if draft_probs is not None and draft_probs.device != target_logits.device:
        raise ValueError("draft_probs must be on the target_logits device")
    if tuple(acceptance_uniforms.shape) != (total_drafts,):
        raise ValueError("acceptance_uniforms shape mismatch")
    if acceptance_uniforms.device != target_logits.device:
        raise ValueError("acceptance_uniforms must be on the target_logits device")
    if not bool(torch.isfinite(acceptance_uniforms).all()) or bool(
        ((acceptance_uniforms < 0) | (acceptance_uniforms > 1)).any()
    ):
        raise ValueError(
            "acceptance_uniforms values must be finite and in [0, 1]"
        )
    if tuple(recovery_uniforms.shape) != (batch_size, vocab_size):
        raise ValueError("recovery_uniforms shape mismatch")
    if recovery_uniforms.device != target_logits.device:
        raise ValueError("recovery_uniforms must be on the target_logits device")
    if not bool(torch.isfinite(recovery_uniforms).all()) or bool(
        (recovery_uniforms <= 0).any()
    ):
        raise ValueError(
            "recovery_uniforms values must be finite and positive"
        )

    token_temperatures = _expand_batch_to_tokens(
        temperatures.to(torch.float32),
        lengths,
        total_drafts,
        target_logits.device,
    )
    if total_drafts:
        target_probs = torch.softmax(
            target_logits[:total_drafts].to(torch.float32)
            / token_temperatures.unsqueeze(-1),
            dim=-1,
        )
    else:
        target_probs = target_logits.new_empty((0, vocab_size), dtype=torch.float32)
    bonus_probs = torch.softmax(
        target_logits[total_drafts:].to(torch.float32)
        / temperatures.to(torch.float32).unsqueeze(-1),
        dim=-1,
    )

    output_rows = []
    accepted_counts = []
    offset = 0
    for request_idx, request_tokens in enumerate(draft_token_ids):
        output = []
        accepted_count = 0
        rejected = False
        noise = recovery_uniforms[request_idx].to(torch.float32)
        for position, token_id in enumerate(request_tokens):
            row_idx = offset + position
            p = target_probs[row_idx]
            if draft_probs is None:
                draft_token_probability = 1.0
            else:
                draft_token_probability = float(draft_probs[row_idx, token_id])
            target_token_probability = float(p[token_id])
            ratio = (
                target_token_probability / draft_token_probability
                if draft_token_probability > 0
                else 0.0
            )
            if float(acceptance_uniforms[row_idx]) <= min(1.0, ratio):
                output.append(token_id)
                accepted_count += 1
                continue

            if draft_probs is None:
                recovered = p.clone()
                recovered[token_id] = 0
            else:
                recovered = torch.clamp(p - draft_probs[row_idx], min=0)
            recovered_mass = recovered.sum()
            if float(recovered_mass) <= 0:
                raise ValueError("recovered distribution has zero mass")
            recovered = recovered / recovered_mass
            output.append(int((recovered / noise).argmax().item()))
            rejected = True
            break
        if not rejected:
            output.append(int((bonus_probs[request_idx] / noise).argmax().item()))
        output_rows.append(output)
        accepted_counts.append(accepted_count)
        offset += len(request_tokens)

    return SpecDecodeResult(
        output_token_ids=output_rows,
        accepted_draft_counts=accepted_counts,
    )
