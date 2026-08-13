# EAGLE3 Inference Mode Implementation Plan

> **Completion status (2026-08-13): Completed.**
>
> The unchecked boxes below are retained for traceability. The documented
> inference-mode boundaries are present in `nanovllm/engine/model_runner.py`.
> The final runtime also has tree-specific `run_eagle3_tree_propose` and
> `run_eagle3_tree_verify` boundaries, both decorated with
> `@torch.inference_mode()`. Current test coverage is in
> `tests/engine/test_eagle3_flow.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all EAGLE3 target and draft inference forwards without autograd graph retention.

**Architecture:** Add `torch.inference_mode()` at the three `ModelRunner` execution boundaries that own EAGLE3 target or draft forwards. The boundaries cover nested proposer calls while preserving existing tensor flow, rejection sampling, and KV-cache state behavior.

**Tech Stack:** Python 3.12, PyTorch, pytest, CUDA/Triton inference runtime.

## Global Constraints

- Do not change warmup batch size or sequence length.
- Do not change EAGLE3 proposal, verification, rejection-sampling, or KV-cache semantics.
- Do not modify the existing user change in `nanovllm/v1/sample/rejection_sampler.py`.

---

### Task 1: Guard EAGLE3 Forwards With Inference Mode

**Files:**
- Modify: `nanovllm/engine/model_runner.py:158-211,526-630`
- Modify: `tests/engine/test_eagle3_flow.py:15-170`

**Interfaces:**
- Consumes: `torch.inference_mode()` and the existing `ModelRunner` methods.
- Produces: `run_eagle3_prefill`, `run_eagle3_spec_decode`, and `_warmup_eagle3_draft_step` execute their complete bodies with `torch.is_inference_mode_enabled() is True`.

- [ ] **Step 1: Write failing inference-mode tests**

Update `RecordingTargetModel.__call__` and `RecordingProposer.prefill` / `propose` to record `torch.is_inference_mode_enabled()`. Assert that both `test_runner_eagle3_prefill_orders_target_draft_and_root_sampling` and `test_runner_eagle3_verify_commits_anchor_by_explicit_accepted_count` observe `True` for every target and draft forward.

Add this focused warmup test:

```python
def test_eagle3_draft_warmup_uses_inference_mode(monkeypatch):
    class RecordingDraft:
        def combine_hidden_states(self, hidden):
            assert torch.is_inference_mode_enabled()
            return hidden[:, :2]

        def __call__(self, input_ids, positions, hidden):
            assert torch.is_inference_mode_enabled()
            return hidden, hidden

        def compute_logits(self, hidden):
            assert torch.is_inference_mode_enabled()
            return torch.zeros((hidden.size(0), 7))

    runner = ModelRunner.__new__(ModelRunner)
    runner.draft_model = RecordingDraft()
    runner.eagle3_proposer = SimpleNamespace(
        states={1: SimpleNamespace(anchor_hidden_states=torch.zeros(6), draft_num_computed_tokens=2)}
    )
    monkeypatch.setattr("nanovllm.engine.model_runner.reset_context", lambda: None)
    runner._warmup_eagle3_draft_step([FakeSequence(1, [1, 2, 3], 2)])
```

Import `SimpleNamespace` from `types` in the test module.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
py -3.12 -m pytest tests/engine/test_eagle3_flow.py -q
```

Expected: the new inference-mode assertions fail because the EAGLE3 methods currently execute with inference mode disabled.

- [ ] **Step 3: Add the execution-boundary decorators**

In `nanovllm/engine/model_runner.py`, add the existing PyTorch decorator immediately above each method:

```python
@torch.inference_mode()
def _warmup_eagle3_draft_step(self, seqs: list[Sequence]) -> None:
    ...

@torch.inference_mode()
def run_eagle3_prefill(self, seqs: list[Sequence]) -> list[int] | None:
    ...

@torch.inference_mode()
def run_eagle3_spec_decode(
    self,
    seqs: list[Sequence],
    reservations: list[SpecReservation],
) -> tuple[DraftProposal, SpecDecodeResult] | None:
    ...
```

Do not alter method bodies or add decorators to `Eagle3Proposer`; the three outer boundaries cover all production EAGLE3 forwards and maintain the proposer as a reusable component.

- [ ] **Step 4: Run focused flow tests**

Run:

```powershell
py -3.12 -m pytest tests/engine/test_eagle3_flow.py -q
```

Expected: PASS. The assertions prove inference mode is active, while the existing tests prove token ordering, anchor selection, and scheduler call ordering are unchanged.

- [ ] **Step 5: Run the CPU EAGLE3 test subset**

Run:

```powershell
py -3.12 -m pytest tests/engine/test_eagle3_flow.py tests/v1/spec_decode/test_eagle3_proposer.py tests/models/test_qwen3_eagle3.py -q -m "not cuda and not model_weights"
```

Expected: PASS. CUDA/model-weight integration tests remain outside this CPU verification step.

- [ ] **Step 6: Commit the implementation**

Run:

```powershell
git add nanovllm/engine/model_runner.py tests/engine/test_eagle3_flow.py
git commit -m "fix: run EAGLE3 forwards in inference mode"
```

Confirm `nanovllm/v1/sample/rejection_sampler.py` is not staged before committing.
