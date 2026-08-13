# EAGLE3 Inference Mode Design

> **Implementation status (2026-08-13): Implemented.**
>
> `ModelRunner._warmup_eagle3_draft_step`, `run_eagle3_prefill`,
> `run_eagle3_spec_decode`, `run_eagle3_tree_propose`, and
> `run_eagle3_tree_verify` execute under `@torch.inference_mode()`. Tree
> verification is split into propose/verify entry points, so the original
> three-method scope does not describe every current EAGLE3 boundary.
>
> Current source of truth: `nanovllm/engine/model_runner.py` and
> `tests/engine/test_eagle3_flow.py`.

## Goal

Prevent EAGLE3 execution from retaining autograd graphs during inference. This
reduces the transient activation peak for target and draft forwards without
changing sampling, proposal, verification, or KV-cache semantics.

## Scope

Add `@torch.inference_mode()` to these `ModelRunner` methods:

- `run_eagle3_prefill`
- `run_eagle3_spec_decode`
- `_warmup_eagle3_draft_step`

The decorators cover nested `Eagle3Proposer` target and draft forwards during
the production EAGLE3 path. Warmup batch size and sequence length are not part
of this change.

## Behavior

The public return values and call order remain unchanged. Target auxiliary
states, draft probabilities, rejection sampling, state commits, and context
cleanup all continue to use the same tensors and control flow. The only
intentional change is that tensors created inside the decorated methods do not
track gradients.

## Tests

Extend EAGLE3 flow tests with recording fakes that assert inference mode is
enabled while executing target prefill, draft proposal, target verification,
and the direct draft warmup step. Existing flow tests remain responsible for
token, anchor, and scheduler behavior.

## Non-Goals

- Change the warmup batch size or prompt length.
- Change EAGLE3 model weights, proposal length, rejection sampling, or KV
  allocation.
- Refactor the generic `Eagle3Proposer` API.
