# EAGLE3 TreeAttention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for inline, task-by-task implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TreeAttention speculative decoding for EAGLE3 when `tree_top_k >= 2`, while retaining the current bit-for-bit linear path for `tree_top_k == 1`.

**Architecture:** Draft generation builds a BFS token tree and maintains branch-local Draft KV through copy-on-write. Target verification runs root plus all draft nodes with an ancestor-only SDPA mask, persists root K/V, stages Draft-node Target K/V, and scatters only the accepted path after rank verification. The linear configuration bypasses all tree code.

**Tech Stack:** Python, PyTorch SDPA, Triton KV store kernel, pytest, existing Qwen3/EAGLE3 model runner.

## Global Constraints

- `tree_top_k == 1` uses the existing linear `RejectionSampler`, FlashAttention, and reservation behavior.
- Tree mode requires `tree_top_k >= 2` and `tree_max_depth >= 1`.
- Target verification performs one model forward per speculative round; accepted Target K/V comes from a per-round staging buffer, not a replay forward.
- Root Target K/V is stored in the primary cache; non-root tree K/V is staged until rank acceptance.
- Tree mode uses flattened model inputs and reshapes only inside `Attention._tree_attention`; SDPA uses GQA or an explicit K/V-head repeat fallback.

---

### Task 1: Tree Configuration and Topology

**Files:**
- Modify: `nanovllm/config.py`
- Modify: `nanovllm/v1/spec_decode/types.py`
- Test: `tests/test_config.py`
- Test: `tests/v1/spec_decode/test_types.py`

**Interfaces:**
- Produces `TreeTopology`, `DraftProposal.tree_topologies`, and `SpecDecodeResult.accepted_paths`.
- Produces validated `SpeculativeConfig.tree_top_k`, `tree_max_depth`, and `tree_prune_ratio`.

- [ ] Write failing tests for invalid tree config, BFS topology, ancestor mask, and RoPE positions.
- [ ] Run: `pytest tests/test_config.py tests/v1/spec_decode/test_types.py -q`
- [ ] Add config validation and `TreeTopology.build_tree_internal_mask()`.
- [ ] Extend proposal/result dataclasses while preserving existing constructors in linear tests.
- [ ] Re-run the focused tests.

### Task 2: Tree Rank Verifier

**Files:**
- Modify: `nanovllm/v1/sample/rejection_sampler.py`
- Test: `tests/v1/sample/test_rejection_sampler_reference.py`

**Interfaces:**
- Produces `RankVerifier.forward(proposal, logits, temperatures) -> SpecDecodeResult`.
- Consumes per-request root/node logit row maps and `TreeTopology`.

- [ ] Write CPU reference tests covering root rejection, accepted branch traversal, and leaf-node bonus sampling.
- [ ] Run: `pytest tests/v1/sample/test_rejection_sampler_reference.py -q`
- [ ] Implement `RankVerifier` without draft probability inputs or a global bonus row.
- [ ] Re-run the focused tests.

### Task 3: Tree Draft KV and Proposal Generation

**Files:**
- Modify: `nanovllm/v1/spec_decode/types.py`
- Modify: `nanovllm/v1/spec_decode/eagle3_proposer.py`
- Test: `tests/v1/spec_decode/test_eagle3_proposer.py`

**Interfaces:**
- Produces `TreeDraftKVManager` and tree-mode `Eagle3Proposer.propose()`.
- Consumes split Draft/Target reservations and produces BFS token IDs plus topology.

- [ ] Write failing tests for COW fork/write/release and pruned BFS proposal shapes.
- [ ] Run: `pytest tests/v1/spec_decode/test_eagle3_proposer.py -q`
- [ ] Implement Draft COW pool allocation and layer-batched top-k expansion.
- [ ] Preserve the existing proposer implementation for `tree_top_k == 1`.
- [ ] Re-run the focused tests.

### Task 4: Tree Attention and Target KV Staging

**Files:**
- Modify: `nanovllm/utils/context.py`
- Modify: `nanovllm/layers/attention.py`
- Create: `nanovllm/v1/spec_decode/tree_kv.py`
- Test: `tests/layers/test_tree_attention.py`

**Interfaces:**
- Produces `TreeTargetKVStager.stage()` and `commit_path()`.
- Adds tree context fields for masks, flattened root row indices, batch shape, and staging.
- Produces `Attention._tree_attention()` with root cache writes and staged Draft K/V.

- [ ] Write failing CPU/CUDA tests for root/ancestor masks, root-only primary-cache writes, GQA layout, and accepted-path scatter.
- [ ] Run: `pytest tests/layers/test_tree_attention.py -q`
- [ ] Implement tree context, staging buffers, prompt gather, SDPA reshape/transpose, and GQA fallback.
- [ ] Re-run the focused tests.

### Task 5: Tree Model Runner and Scheduler Integration

**Files:**
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `nanovllm/engine/scheduler.py`
- Modify: `nanovllm/engine/llm_engine.py`
- Test: `tests/engine/test_eagle3_flow.py`
- Test: `tests/engine/test_scheduler_eagle3.py`

**Interfaces:**
- Produces tree `prepare_spec_decode()` with flattened padded input and row maps.
- Produces split `SpecReservation` pools and effective tree depth.
- Commits Target staged K/V before scheduler cache commit.

- [ ] Write failing flow tests for root/node logits, no global bonus row, staging commit, and depth fallback.
- [ ] Run: `pytest tests/engine/test_eagle3_flow.py tests/engine/test_scheduler_eagle3.py -q`
- [ ] Implement tree runner path, split reservations, commit/release ordering, and strict legacy branch selection.
- [ ] Re-run the focused tests.

### Task 6: Regression and Integration Verification

**Files:**
- Modify: `tests/integration/test_eagle3_qwen3_4b.py`
- Modify: `tests/engine/test_eagle3_flow.py`

- [ ] Add a fixed-seed `tree_top_k == 1` regression against the current linear output and acceptance counts.
- [ ] Add tree acceptance and max-depth integration coverage guarded by checkpoint availability.
- [ ] Run: `pytest tests/v1/spec_decode tests/v1/sample tests/engine -q`
- [ ] Run the full available suite: `pytest -q`.
- [ ] Commit implementation and tests in reviewable task-sized commits.
