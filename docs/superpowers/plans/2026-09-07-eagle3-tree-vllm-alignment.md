# EAGLE3 Tree vLLM Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align Nano-vLLM-X tree EAGLE3 verification with the vLLM reference data flow, restore exact greedy token equivalence with target-only decoding, and then reduce avoidable tree verification overhead.

**Architecture:** Keep Nano-vLLM-X's existing Scheduler/ModelRunner/BlockManager interfaces. Use vLLM's established semantics as the reference for root handling, tree row layout, target sampling, accepted-path selection, anchor state updates, and KV commit. Add instrumentation and focused regression tests before changing runtime behavior; optimize only after correctness passes.

**Tech Stack:** Python 3.12, PyTorch, Triton, FlashAttention, pytest, Hugging Face Transformers.

## Global Constraints

- EAGLE3 validation uses `Qwen/Qwen3-14B` with `thoughtworks/Qwen3-14B-Eagle3`.
- EAGLE3 remains single GPU, `tensor_parallel_size=1`, and `enforce_eager=True`.
- Greedy correctness tests use `temperature=0.0`; target-only token IDs are authoritative.
- No performance claim is valid until token-level correctness passes.
- The local machine may not have CUDA/PyTorch; use static checks locally and run model tests on A800.

### Task 1: Capture a deterministic correctness trace

**Files:**
- Modify: `check_eagle3_tree_correctness.py`
- Modify: `tests/test_eagle3_correctness.py`

- [ ] Add an optional `--dump-json` path that records target/tree token IDs and configuration for a failing repeat.
- [ ] Add tests for deterministic dump contents and first-mismatch reporting.
- [ ] Run the pure helper tests with `pytest --noconftest` and compile the checker.

### Task 2: Verify tree row and path semantics against the reference

**Files:**
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `nanovllm/v1/sample/rejection_sampler.py`
- Modify: `nanovllm/v1/spec_decode/types.py`
- Create/modify: `tests/v1/sample/test_tree_verifier.py`

- [ ] Add unit tests for a one-request depth-2 topology: root row, child row, target greedy token, accepted path, and bonus token.
- [ ] Confirm that every tree row samples from target logits at the row representing the current node, and that the root row is used only for the pending root decision.
- [ ] Confirm accepted paths are parent-consistent and output tokens are exactly path tokens followed by one target bonus/recovery token.
- [ ] Make the smallest implementation change needed for any failing row/path assertion.

### Task 3: Verify position IDs, mask, and KV commit

**Files:**
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `nanovllm/layers/attention.py`
- Modify: `nanovllm/v1/spec_decode/tree_kv.py`
- Modify: `nanovllm/v1/spec_decode/eagle3_proposer.py`
- Create/modify: `tests/engine/test_eagle3_tree.py`

- [ ] Add tests asserting root position, draft-node positions, ancestor visibility, and prompt visibility for a depth-2 and depth-3 topology.
- [ ] Add tests asserting accepted-path Target KV staging is copied into persistent slots at consecutive target positions.
- [ ] Add tests asserting the selected verification auxiliary row becomes the next draft anchor.
- [ ] Compare the first failing A800 trace at `max_tokens=1`, `2`, and `3`; fix only the first failing boundary.

### Task 4: Add tree cost instrumentation

**Files:**
- Modify: `nanovllm/v1/spec_decode/types.py`
- Modify: `nanovllm/engine/scheduler.py`
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `bench_eagle3_tree.py`

- [ ] Record per-run proposed tree nodes, accepted path length, effective depth, draft time, verify time, and sampling time.
- [ ] Include these fields in JSON summaries without changing existing field names.
- [ ] Add unit tests for metric accumulation and per-repeat serialization.

### Task 5: Apply low-risk performance optimizations

**Files:**
- Modify: `nanovllm/v1/spec_decode/eagle3_proposer.py`
- Modify: `nanovllm/layers/attention.py`
- Modify: `nanovllm/v1/spec_decode/tree_kv.py`
- Modify: `tests/v1/spec_decode/test_eagle3_proposer.py`

- [ ] Cache the Attention module list in `TreeDraftKVManager` instead of traversing `model.modules()` for every fork/commit.
- [ ] Reuse tree prompt-KV temporary buffers where shape permits, preserving request isolation.
- [ ] Remove avoidable Python-side repeated conversions and topology lookups on the hot path.
- [ ] Re-run correctness tests after each optimization.

### Task 6: Validate on A800 and update documentation

**Files:**
- Modify: `docs/Speculative-Decoding.md`
- Modify: `README.md`

- [ ] Run correctness at `tree_top_k=2, tree_max_depth=2/3` with `max_tokens=1/2/3/32`.
- [ ] Run the default tree configuration only after all token sequences match target-only.
- [ ] Run repeated benchmark measurements and report actual, not projected, speedups.
- [ ] Document any remaining limitation and the exact A800 command/output.
