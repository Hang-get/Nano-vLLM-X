# Qwen3 EAGLE3 Linear Speculative Decoding Implementation Plan

> **Completion status (2026-08-13): Historical plan completed; later tree work
> extends its linear runtime.**
>
> The unchecked boxes below are retained as the original implementation record,
> not as remaining work. The implemented linear contract is selected by
> `speculative_config={"method": "eagle3", "tree_top_k": 1}`. The plan's
> former no-tree non-goal is no longer current: tree mode is implemented by the
> 2026-08-04 tree-attention work. EAGLE3 remains one-GPU, eager-only, and
> prefix-cache-disabled. The current supported pair is
> `Qwen/Qwen3-14B` plus `thoughtworks/Qwen3-14B-Eagle3`; its 40960-token
> target limit should initially be benchmarked at 4096 tokens on A800 80G.
> The historical 4B details below are retained for traceability. See
> `docs/Speculative-Decoding.md` for current usage.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 nano-vLLM-MS2 中为 `Qwen/Qwen3-4B-Instruct-2507` target 和 `andyjjrt/Qwen3-4B-Instruct-2507-Eagle3` draft 实现单卡 eager、固定长度、线性 EAGLE3 speculative decoding。

**Architecture:** 保留现有 target verification 的“pending root + drafts”布局，在 target prefill/verification 中捕获 layers `(2, 18, 33)` 的辅助 hidden states。独立 `Eagle3Proposer` 管理 draft 模型、`d2t` 映射、draft KV 和 per-request anchor；target/draft 使用相同逻辑 block ID 和独立物理 KV tensors，rejection sampler 显式返回接受长度。

**Tech Stack:** Python 3.12、PyTorch >= 2.5、Transformers >= 5.2、FlashAttention >= 2.7.4、Triton >= 3.1、safetensors、pytest。

## Global Constraints

- Target checkpoint 固定为 `Qwen/Qwen3-4B-Instruct-2507`，本地目录加载。
- Draft checkpoint 固定为 `andyjjrt/Qwen3-4B-Instruct-2507-Eagle3` revision `408d111ec6cde42f2784f50cd14189d626a6eae4`，本地目录加载。
- 首版只允许 `tensor_parallel_size == 1`、`enforce_eager is True`、prefix cache disabled。
- Target auxiliary layer IDs 固定为 `(2, 18, 33)`。
- 最大上下文长度不得超过 draft checkpoint 的 2048。
- 稳定边界必须满足 `draft_num_computed_tokens == max(target_num_computed_tokens - 1, 0)`。
- Pending root 已由 target 分布采样，不计入 proposed/accepted draft 指标。
- N-gram speculative decoding 的行为和配置必须保持兼容。
- 不实现动态候选树、tree attention、tensor parallel 或 CUDA graph。
- 设计依据：`docs/superpowers/specs/2026-07-26-eagle3-linear-speculative-decoding-design.md`。

---

## File Map

- `nanovllm/config.py`：解析并校验 N-gram/EAGLE3 配置，加载 draft HF config。
- `nanovllm/v1/spec_decode/eagle3_config.py`：所选 checkpoint pair 的不可变兼容契约。
- `nanovllm/v1/spec_decode/types.py`：proposal、reservation、request state 和验收结果。
- `nanovllm/models/model_output.py`：target forward 的结构化 hidden-state 输出。
- `nanovllm/models/qwen3.py`：按需捕获 target layers `(2, 18, 33)`。
- `nanovllm/models/qwen3_eagle3.py`：一层 Qwen3 EAGLE3 draft architecture。
- `nanovllm/utils/eagle3_loader.py`：严格的 draft 权重映射、注入和加载报告。
- `nanovllm/v1/spec_decode/eagle3_proposer.py`：draft prefill、线性 proposal、状态提交与释放。
- `nanovllm/v1/sample/rejection_sampler.py`：CPU reference、draft probability 验收、显式 accepted counts。
- `nanovllm/engine/block_manager.py`：prefix-cache 开关和共享逻辑 block 生命周期。
- `nanovllm/engine/scheduler.py`：先预留 draft budget、显式 accepted counts、抢占事件。
- `nanovllm/engine/model_runner.py`：target/draft 联合加载、KV 分配和端到端编排。
- `nanovllm/engine/llm_engine.py`：方法分派和 Eagle state 生命周期通知。
- `tests/`：CPU 单元测试、CUDA kernel 对齐测试、真实 checkpoint 集成测试。
- `example_sd.py`、`docs/Speculative-Decoding.md`：EAGLE3 使用方式和能力边界。

---

### Task 1: Test Harness and Checkpoint Contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `nanovllm/config.py`
- Create: `nanovllm/v1/spec_decode/eagle3_config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `validate_eagle3_checkpoint_pair(target_config, draft_config, *, target_path, draft_path) -> tuple[int, int, int]`
- Produces: `SpeculativeConfig.draft_model`, `.draft_hf_config`, `.auxiliary_layer_ids`
- Consumes: local target/draft directories and `AutoConfig.from_pretrained`

- [ ] **Step 1: Add pytest as a test dependency and define markers**

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
    "pytest>=8.3,<9",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "cuda: requires a CUDA device and installed GPU kernels",
    "model_weights: requires local target and draft checkpoint directories",
]
```

- [ ] **Step 2: Write failing checkpoint/config tests**

Create `tests/conftest.py` with a non-autouse `single_rank_dist` fixture that monkeypatches `torch.distributed.get_rank` to `0` and `get_world_size` to `1`.

Create `tests/test_config.py` with fixtures returning `SimpleNamespace` configs containing the exact target and draft values from the design. Cover:

```python
def test_eagle3_config_resolves_contract(tmp_path, monkeypatch):
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    configs = {
        str(target_dir): make_target_config(num_hidden_layers=36),
        str(draft_dir): make_draft_config(),
    }
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda path: configs[str(path)],
    )

    config = Config(
        str(target_dir),
        enforce_eager=True,
        speculative_config={
            "method": "eagle3",
            "draft_model": str(draft_dir),
            "num_speculative_tokens": 5,
        },
    )

    assert config.max_model_len == 2048
    assert config.enable_prefix_cache is False
    assert config.speculative_config.auxiliary_layer_ids == (2, 18, 33)
    assert config.speculative_config.draft_hf_config is configs[str(draft_dir)]
```

Also test: missing draft directory, wrong target architecture, wrong draft architecture, every incompatible hidden/intermediate/head/KV-head/head-dim/vocabulary/dtype field, target layer count other than 36, draft position limit other than 2048, `enforce_eager=False`, TP > 1, `num_speculative_tokens < 1`, and unchanged N-gram construction. Error assertions must include the checkpoint role/path, field name, expected value and actual value.

- [ ] **Step 3: Run tests and verify the expected failure**

Run: `py -3.12 -m pytest tests/test_config.py -q`

Expected: collection fails because `eagle3_config.py` and EAGLE3 fields do not exist.

- [ ] **Step 4: Implement the exact checkpoint contract**

Create `nanovllm/v1/spec_decode/eagle3_config.py`:

```python
TARGET_ARCH = "Qwen3ForCausalLM"
DRAFT_ARCH = "LlamaForCausalLMEagle3"
AUXILIARY_LAYER_IDS = (2, 18, 33)


def _architecture(config, role: str, path: str) -> str:
    architectures = getattr(config, "architectures", None)
    if not architectures:
        raise ValueError(
            f"{role} checkpoint {path}: architectures expected one value, got "
            f"{architectures!r}"
        )
    return architectures[0]


def _dtype_name(config) -> str:
    value = getattr(config, "dtype", None)
    if value is None:
        value = getattr(config, "torch_dtype", None)
    return str(value).removeprefix("torch.")


def _require(role: str, path: str, name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(
            f"{role} checkpoint {path}: {name} expected {expected!r}, "
            f"got {actual!r}"
        )


def validate_eagle3_checkpoint_pair(
    target_config,
    draft_config,
    *,
    target_path: str,
    draft_path: str,
) -> tuple[int, int, int]:
    target_values = {
        "architecture": _architecture(target_config, "target", target_path),
        "hidden_size": target_config.hidden_size,
        "intermediate_size": target_config.intermediate_size,
        "num_hidden_layers": target_config.num_hidden_layers,
        "num_attention_heads": target_config.num_attention_heads,
        "num_key_value_heads": target_config.num_key_value_heads,
        "head_dim": target_config.head_dim,
        "vocab_size": target_config.vocab_size,
        "max_position_embeddings": target_config.max_position_embeddings,
        "bos_token_id": target_config.bos_token_id,
        "eos_token_id": target_config.eos_token_id,
        "attention_bias": target_config.attention_bias,
        "hidden_act": target_config.hidden_act,
        "rms_norm_eps": target_config.rms_norm_eps,
        "rope_theta": target_config.rope_theta,
        "tie_word_embeddings": target_config.tie_word_embeddings,
        "dtype": _dtype_name(target_config),
    }
    target_expected = {
        "architecture": TARGET_ARCH,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "attention_bias": False,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
    }
    draft_values = {
        "architecture": _architecture(draft_config, "draft", draft_path),
        "hidden_size": draft_config.hidden_size,
        "intermediate_size": draft_config.intermediate_size,
        "num_hidden_layers": draft_config.num_hidden_layers,
        "num_attention_heads": draft_config.num_attention_heads,
        "num_key_value_heads": draft_config.num_key_value_heads,
        "head_dim": draft_config.head_dim,
        "vocab_size": draft_config.vocab_size,
        "draft_vocab_size": draft_config.draft_vocab_size,
        "max_position_embeddings": draft_config.max_position_embeddings,
        "bos_token_id": draft_config.bos_token_id,
        "eos_token_id": draft_config.eos_token_id,
        "attention_bias": draft_config.attention_bias,
        "hidden_act": draft_config.hidden_act,
        "rms_norm_eps": draft_config.rms_norm_eps,
        "rope_theta": draft_config.rope_theta,
        "tie_word_embeddings": draft_config.tie_word_embeddings,
        "dtype": _dtype_name(draft_config),
    }
    draft_expected = {
        "architecture": DRAFT_ARCH,
        "hidden_size": 2560,
        "intermediate_size": 12288,
        "num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "draft_vocab_size": 32000,
        "max_position_embeddings": 2048,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "attention_bias": False,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
    }
    for name, expected in target_expected.items():
        _require("target", target_path, name, target_values[name], expected)
    for name, expected in draft_expected.items():
        _require("draft", draft_path, name, draft_values[name], expected)
    return AUXILIARY_LAYER_IDS
```

Extend `SpeculativeConfig` with `draft_model`, non-init `draft_hf_config`, and non-init `auxiliary_layer_ids`. Add `Config.enable_prefix_cache: bool = True`. In `Config.__post_init__`, construct `SpeculativeConfig` after loading target config; for `method == "eagle3"`, require both directories and their `config.json` files, load draft config, call the contract with both paths, force prefix cache off, enforce eager/single-rank, and cap `max_model_len` by both configs. The draft uses the target tokenizer; do not attempt to load a tokenizer from the draft directory.

- [ ] **Step 5: Run config tests**

Run: `py -3.12 -m pytest tests/test_config.py -q`

Expected: all config tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml nanovllm/config.py nanovllm/v1/spec_decode/eagle3_config.py tests/conftest.py tests/test_config.py
git commit -m "feat: validate EAGLE3 checkpoint configuration"
```

---

### Task 2: Structured Speculative-Decoding Types

**Files:**
- Create: `nanovllm/v1/spec_decode/types.py`
- Create: `tests/v1/spec_decode/test_types.py`

**Interfaces:**
- Produces: `DraftProposal`, `SpecReservation`, `SpecDecodeResult`, `Eagle3RequestState`
- Produces: `DraftProposal.truncate(new_lengths) -> DraftProposal`

- [ ] **Step 1: Write failing shape and truncation tests**

Create `tests/v1/spec_decode/test_types.py`:

```python
def test_draft_proposal_truncate_preserves_request_major_rows():
    proposal = DraftProposal(
        token_ids=[[10, 11, 12], [20, 21]],
        probabilities=torch.arange(5 * 7, dtype=torch.float32).reshape(5, 7),
        lengths=[3, 2],
    )

    truncated = proposal.truncate([2, 1])

    assert truncated.token_ids == [[10, 11], [20]]
    assert truncated.lengths == [2, 1]
    torch.testing.assert_close(
        truncated.probabilities,
        torch.cat([proposal.probabilities[0:2], proposal.probabilities[3:4]]),
    )
```

Add tests rejecting mismatched token lengths, probability rows, accepted-count batch size, and invalid reservation lengths.

- [ ] **Step 2: Run tests and verify import failure**

Run: `py -3.12 -m pytest tests/v1/spec_decode/test_types.py -q`

Expected: FAIL because `nanovllm.v1.spec_decode.types` does not exist.

- [ ] **Step 3: Implement the data contracts**

Create `nanovllm/v1/spec_decode/types.py` with these exact public fields:

```python
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SpecReservation:
    draft_len: int
    new_block_ids: list[int]

    def __post_init__(self):
        if self.draft_len < 0:
            raise ValueError("draft_len must be non-negative")


@dataclass
class DraftProposal:
    token_ids: list[list[int]]
    probabilities: torch.Tensor
    lengths: list[int]

    def __post_init__(self):
        if self.lengths != [len(row) for row in self.token_ids]:
            raise ValueError("lengths must match token_ids")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must be rank 2")
        if self.probabilities.size(0) != sum(self.lengths):
            raise ValueError("probability rows must equal sum(lengths)")

    def truncate(self, new_lengths: list[int]) -> "DraftProposal":
        if len(new_lengths) != len(self.lengths):
            raise ValueError("new_lengths batch size mismatch")
        rows = []
        token_ids = []
        offset = 0
        for old_len, new_len, request_tokens in zip(
            self.lengths, new_lengths, self.token_ids
        ):
            if not 0 <= new_len <= old_len:
                raise ValueError("new length exceeds proposed length")
            rows.append(self.probabilities[offset : offset + new_len])
            token_ids.append(request_tokens[:new_len])
            offset += old_len
        probabilities = torch.cat(rows, dim=0) if rows else self.probabilities[:0]
        return DraftProposal(token_ids, probabilities, new_lengths)


@dataclass
class SpecDecodeResult:
    output_token_ids: list[list[int]]
    accepted_draft_counts: list[int]

    def __post_init__(self):
        if len(self.output_token_ids) != len(self.accepted_draft_counts):
            raise ValueError("accepted-count batch size mismatch")
        if any(count < 0 for count in self.accepted_draft_counts):
            raise ValueError("accepted counts must be non-negative")


@dataclass
class Eagle3RequestState:
    anchor_hidden_states: torch.Tensor
    draft_num_computed_tokens: int
    valid: bool = True
```

- [ ] **Step 4: Run tests**

Run: `py -3.12 -m pytest tests/v1/spec_decode/test_types.py -q`

Expected: all type tests pass.

- [ ] **Step 5: Commit**

```bash
git add nanovllm/v1/spec_decode/types.py tests/v1/spec_decode/test_types.py
git commit -m "feat: add speculative decoding data contracts"
```

---

### Task 3: Qwen3 Auxiliary Hidden-State Capture

**Files:**
- Create: `nanovllm/models/model_output.py`
- Modify: `nanovllm/models/qwen3.py`
- Create: `tests/models/test_qwen3_aux.py`

**Interfaces:**
- Produces: `TargetModelOutput(hidden_states, auxiliary_hidden_states)`
- Produces: `Qwen3ForCausalLM.forward(input_ids, positions, auxiliary_layer_ids=()) -> torch.Tensor | TargetModelOutput`

- [ ] **Step 1: Write failing capture tests with fake decoder layers**

Build a `Qwen3Model` instance via `__new__`, initialize it as an `nn.Module`, and install deterministic fake embedding/layers/norm. Assert:

```python
normal = model(input_ids, positions)
captured = model(input_ids, positions, auxiliary_layer_ids=(0, 2))

torch.testing.assert_close(captured.hidden_states, normal)
assert captured.auxiliary_hidden_states.shape == (input_ids.numel(), 2 * hidden_size)
torch.testing.assert_close(captured.auxiliary_hidden_states, expected_aux)
```

Also assert duplicate, unsorted, negative, and out-of-range layer IDs raise `ValueError`.

- [ ] **Step 2: Run tests and verify signature failure**

Run: `py -3.12 -m pytest tests/models/test_qwen3_aux.py -q`

Expected: FAIL because the Qwen3 forward path does not accept auxiliary layer IDs.

- [ ] **Step 3: Add structured target output and capture logic**

Create `nanovllm/models/model_output.py`:

```python
from dataclasses import dataclass
import torch


@dataclass
class TargetModelOutput:
    hidden_states: torch.Tensor
    auxiliary_hidden_states: torch.Tensor
```

Change `Qwen3Model.forward` to validate IDs, collect `hidden_states + residual` immediately after selected decoder layers, preserve the normal tensor return when no IDs are supplied, and concatenate selected values on the last dimension:

```python
def forward(self, input_ids, positions, auxiliary_layer_ids=()):
    if tuple(sorted(set(auxiliary_layer_ids))) != tuple(auxiliary_layer_ids):
        raise ValueError("auxiliary_layer_ids must be sorted and unique")
    if auxiliary_layer_ids and (
        auxiliary_layer_ids[0] < 0 or auxiliary_layer_ids[-1] >= len(self.layers)
    ):
        raise ValueError("auxiliary layer index out of range")

    hidden_states = self.embed_tokens(input_ids)
    residual = None
    auxiliary_hidden_states = []
    for layer_idx, layer in enumerate(self.layers):
        hidden_states, residual = layer(positions, hidden_states, residual)
        if layer_idx in auxiliary_layer_ids:
            auxiliary_hidden_states.append(hidden_states + residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    if not auxiliary_layer_ids:
        return hidden_states
    return TargetModelOutput(
        hidden_states=hidden_states,
        auxiliary_hidden_states=torch.cat(auxiliary_hidden_states, dim=-1),
    )
```

Thread the same optional argument through `Qwen3ForCausalLM.forward`.

- [ ] **Step 4: Run capture and existing import tests**

Run: `py -3.12 -m pytest tests/models/test_qwen3_aux.py tests/test_config.py -q`

Expected: all tests pass and the normal Qwen3 return remains a tensor.

- [ ] **Step 5: Commit**

```bash
git add nanovllm/models/model_output.py nanovllm/models/qwen3.py tests/models/test_qwen3_aux.py
git commit -m "feat: capture Qwen3 EAGLE3 auxiliary states"
```

---

### Task 4: Qwen3 EAGLE3 Draft Model and Strict Loader

**Files:**
- Create: `nanovllm/models/qwen3_eagle3.py`
- Create: `nanovllm/utils/eagle3_loader.py`
- Create: `tests/models/test_qwen3_eagle3.py`
- Create: `tests/utils/test_eagle3_loader.py`

**Interfaces:**
- Produces: `Qwen3Eagle3ForCausalLM(config, target_embedding)` without allocating a second full embedding
- Produces: `Qwen3Eagle3ForCausalLM.forward(input_ids, positions, fused_hidden_states) -> tuple[torch.Tensor, torch.Tensor]`（post-norm logits hidden，pre-norm recurrent auxiliary hidden）
- Produces: `combine_hidden_states([N, 7680]) -> [N, 2560]`
- Produces: `compute_logits([N, 2560]) -> [N, 151936]`
- Produces: `load_eagle3_weights(model, path, target_embedding) -> WeightLoadReport`

- [ ] **Step 1: Write failing model-shape and d2t tests**

Using a tiny synthetic config with `hidden_size=8`, `vocab_size=13`, `draft_vocab_size=4`, one layer and fake single-rank distributed functions, test:

```python
model.draft_id_to_target_id.copy_(torch.tensor([0, 2, 4, 6]))
model.lm_head.weight.copy_(known_weight)
logits = model.compute_logits(hidden_states)

assert logits.shape == (2, 13)
targets = torch.tensor([0, 3, 6, 9])
torch.testing.assert_close(logits[:, targets], expected_small_logits)
assert torch.isneginf(logits[:, [1, 2, 4, 5, 7, 8, 10, 11, 12]]).all()
```

Assert the first layer QKV input width is `2 * hidden_size`, `combine_hidden_states` accepts exactly `3 * hidden_size`, and the constructor installs the exact target embedding module/parameter by identity without first allocating a second 151936-row embedding. With dropout disabled, compare a short sequence processed in one prefill against token-by-token cached forwards at every position and require matching hidden states/logits.

- [ ] **Step 2: Write failing synthetic safetensors loader tests**

Create a tiny checkpoint with keys `midlayer.*`, `fc.weight`, `norm.weight`, `lm_head.weight`, `d2t`, and `t2d`. Assert the load report contains `t2d` in `skipped`, `embed_tokens.weight` in `injected`, and no unresolved missing/unexpected keys. Add negative tests for missing `d2t`, unknown tensors, wrong tensor shapes, duplicate/out-of-range d2t targets.

- [ ] **Step 3: Run model/loader tests and verify import failures**

Run: `py -3.12 -m pytest tests/models/test_qwen3_eagle3.py tests/utils/test_eagle3_loader.py -q`

Expected: FAIL because the model and loader modules do not exist.

- [ ] **Step 4: Implement the draft model**

Implement `Qwen3Eagle3DecoderLayer` using the existing Qwen3 attention/MLP primitives. The first layer must replace `qkv_proj` with `QKVParallelLinear(2 * hidden_size, head_dim, num_attention_heads, num_key_value_heads, bias=attention_bias)`, normalize embeddings and fused hidden states separately, concatenate them for attention, and preserve the un-concatenated fused hidden state as the residual.

Implement `Qwen3Eagle3ForCausalLM` with these public methods:

```python
def combine_hidden_states(self, auxiliary_hidden_states):
    if auxiliary_hidden_states.size(-1) != 3 * self.config.hidden_size:
        raise ValueError("expected three concatenated target hidden states")
    return self.model.fc(auxiliary_hidden_states)


def compute_logits(self, hidden_states):
    draft_logits = self.lm_head(hidden_states, return_all_logits=True)
    draft_ids = torch.arange(
        self.config.draft_vocab_size,
        device=draft_logits.device,
        dtype=torch.long,
    )
    target_ids = draft_ids + self.draft_id_to_target_id
    logits = draft_logits.new_full(
        (draft_logits.size(0), self.config.vocab_size),
        float("-inf"),
    )
    logits[:, target_ids] = draft_logits
    return logits
```

`Qwen3Eagle3ForCausalLM.__init__(config, target_embedding)` validates target embedding vocabulary width, hidden size, dtype and device, then installs that module directly; it must not construct a temporary full-vocabulary draft embedding. `forward(input_ids, positions, fused_hidden_states)` embeds `input_ids`, separately normalizes the embedding and fused hidden state, concatenates them for the first attention projection, and returns `(post_norm_hidden_states, pre_norm_auxiliary_hidden_states)`。前者交给 `compute_logits`，后者作为下一 recurrent draft step 的 hidden input，与 vLLM Qwen3 EAGLE3 adapter 的 `norm_output=False` 语义一致。The draft model owns `fc`, one `midlayer`, final norm, 32000-row LM head and non-trainable `draft_id_to_target_id`. It declares packed QKV and gate/up mappings.

- [ ] **Step 5: Implement strict weight loading**

Create `WeightLoadReport(consumed, skipped, injected, missing, unexpected)` and a loader that performs these exact mappings:

```python
def map_eagle3_weight_name(name: str):
    if "t2d" in name:
        return None, None
    if "d2t" in name:
        return "draft_id_to_target_id", None
    name = name.replace("midlayer.", "model.layers.0.")
    for source, target, shard in (
        ("q_proj", "qkv_proj", "q"),
        ("k_proj", "qkv_proj", "k"),
        ("v_proj", "qkv_proj", "v"),
        ("gate_proj", "gate_up_proj", 0),
        ("up_proj", "gate_up_proj", 1),
    ):
        if source in name:
            return name.replace(source, target), shard
    if not name.startswith("lm_head.") and not name.startswith("model."):
        name = "model." + name
    return name, None
```

Record `model.model.embed_tokens.weight` as injected after verifying that it is identical to the already loaded target embedding; validate shape/dtype/device, validate `target_ids = arange(32000) + d2t` is unique and within `[0, 151936)`, then reject every unresolved missing/unexpected tensor.

- [ ] **Step 6: Run model and loader tests**

Run: `py -3.12 -m pytest tests/models/test_qwen3_eagle3.py tests/utils/test_eagle3_loader.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add nanovllm/models/qwen3_eagle3.py nanovllm/utils/eagle3_loader.py tests/models/test_qwen3_eagle3.py tests/utils/test_eagle3_loader.py
git commit -m "feat: load Qwen3 EAGLE3 draft model"
```

---

### Task 5: Draft Alignment, Prefill, Proposal, and State

**Files:**
- Create: `nanovllm/v1/spec_decode/eagle3_proposer.py`
- Create: `tests/v1/spec_decode/test_eagle3_proposer.py`

**Interfaces:**
- Consumes: `Qwen3Eagle3ForCausalLM`, `DraftProposal`, `SpecReservation`, `Eagle3RequestState`
- Produces: `Eagle3Proposer.prefill(seqs, target_auxiliary_hidden_states, target_num_computed_tokens) -> None`
- Produces: `Eagle3Proposer.propose(seqs, reservations, temperatures) -> DraftProposal`
- Produces: `Eagle3Proposer.commit(seqs, verification_auxiliary_hidden_states, accepted_draft_counts, new_target_num_computed_tokens) -> None`
- Produces: `Eagle3Proposer.release(seq_ids) -> None`

- [ ] **Step 1: Write failing pure alignment tests**

Test the two central helpers without CUDA:

```python
tokens, features, positions = build_draft_prefill_inputs(
    token_ids=[10, 11, 12, 13],
    target_auxiliary_hidden_states=torch.arange(4 * 6).reshape(4, 6),
)
assert tokens == [11, 12, 13]
torch.testing.assert_close(features, target_auxiliary_hidden_states[:-1])
assert positions == [0, 1, 2]

assert draft_position_for_target_position(1) == 0
assert draft_position_for_target_position(7) == 6
assert committed_draft_length(0) == 0
assert committed_draft_length(8) == 7
```

Test `select_next_anchor`: for accepted counts `[0, 2]`, select verification rows for the old root and second accepted draft respectively. Test release removes request state and commit enforces the stable-length invariant.

- [ ] **Step 2: Run tests and verify import failure**

Run: `py -3.12 -m pytest tests/v1/spec_decode/test_eagle3_proposer.py -q`

Expected: FAIL because `eagle3_proposer.py` does not exist.

- [ ] **Step 3: Implement alignment and state helpers**

Add these exact invariants:

```python
def draft_position_for_target_position(target_position: int) -> int:
    if target_position < 1:
        raise ValueError("target position must have a predecessor")
    return target_position - 1


def committed_draft_length(target_num_computed_tokens: int) -> int:
    return max(target_num_computed_tokens - 1, 0)
```

`build_draft_prefill_inputs` returns `token_ids[1:]`, auxiliary rows `[:-1]`, and positions `range(len(token_ids) - 1)`. Empty and one-token prompts return empty tensors without calling the draft model.

- [ ] **Step 4: Implement the proposer lifecycle**

`Eagle3Proposer` stores `states: dict[int, Eagle3RequestState]`. `prefill` builds shifted request-major inputs, runs `combine_hidden_states`, fills draft KV, and stores each request's last target auxiliary row as its anchor. It receives explicit target computed lengths because scheduler postprocess has not yet mutated `Sequence` during runner prefill.

Every public lifecycle method rejects mismatched batch sizes before mutating state: `prefill` requires one auxiliary tensor and computed length per sequence, `propose` requires one reservation and temperature per sequence, and `commit` requires one verification tensor, accepted count and new computed length per sequence.

`propose` performs exactly `max(reservation.draft_len + 1)` batched iterations. For request length `K`, iterations `0..K-1` apply per-request temperature, sample one token, save its full target-vocab probability row, and feed the draft hidden output into the next iteration. Iteration `K` processes the final sampled token only to fill its draft KV slot; its logits are discarded. For `K == 0`, the single iteration processes the pending root and writes its KV without proposing a token.

The public lifecycle must follow these exact signatures and state transitions:

```python
class Eagle3Proposer:
    def prefill(
        self,
        seqs: list[Sequence],
        target_auxiliary_hidden_states: list[torch.Tensor],
        target_num_computed_tokens: list[int],
    ) -> None:
        for seq, auxiliary, target_length in zip(
            seqs, target_auxiliary_hidden_states, target_num_computed_tokens
        ):
            self.states[seq.seq_id] = Eagle3RequestState(
                anchor_hidden_states=auxiliary[-1].detach(),
                draft_num_computed_tokens=committed_draft_length(target_length),
            )
        self._run_shifted_prefill(seqs, target_auxiliary_hidden_states)

    def propose(
        self,
        seqs: list[Sequence],
        reservations: list[SpecReservation],
        temperatures: torch.Tensor,
    ) -> DraftProposal:
        request_tokens = [[] for _ in seqs]
        request_probabilities = [[] for _ in seqs]
        current_tokens = [seq.last_token for seq in seqs]
        current_hidden = [self.states[seq.seq_id].anchor_hidden_states for seq in seqs]
        max_steps = max((item.draft_len + 1 for item in reservations), default=0)
        for step in range(max_steps):
            active = [
                idx for idx, reservation in enumerate(reservations)
                if step <= reservation.draft_len
            ]
            logits, next_hidden = self._run_step(
                seqs, reservations, active, step, current_tokens, current_hidden
            )
            sampling_rows = [
                row for row, request_idx in enumerate(active)
                if step < reservations[request_idx].draft_len
            ]
            if not sampling_rows:
                continue
            sampling_requests = [active[row] for row in sampling_rows]
            active_temperatures = temperatures[sampling_requests].to(torch.float32)
            probabilities = torch.softmax(
                logits[sampling_rows].to(torch.float32)
                / active_temperatures.unsqueeze(-1),
                dim=-1,
            )
            sampled = probabilities.div(
                torch.empty_like(probabilities).exponential_().clamp_min_(1e-10)
            ).argmax(dim=-1)
            for row, request_idx in enumerate(sampling_requests):
                token_id = int(sampled[row].item())
                request_tokens[request_idx].append(token_id)
                request_probabilities[request_idx].append(probabilities[row])
                current_tokens[request_idx] = token_id
                current_hidden[request_idx] = next_hidden[sampling_rows[row]]
        probability_rows = [row for request in request_probabilities for row in request]
        probabilities = (
            torch.stack(probability_rows)
            if probability_rows
            else temperatures.new_empty((0, self.target_vocab_size))
        )
        return DraftProposal(
            token_ids=request_tokens,
            probabilities=probabilities,
            lengths=[len(row) for row in request_tokens],
        )

    def release(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.states.pop(seq_id, None)
```

`_run_shifted_prefill` creates the request-major `input_ids[1:]`/auxiliary `[:-1]` tensors and draft prefill attention context. `_run_step` creates a decode context for active requests with draft position `state.draft_num_computed_tokens + step`, uses `reservation.new_block_ids` together with the shared committed block table, runs `combine_hidden_states` only for `step == 0`, runs the draft model, expands draft logits through `d2t`, and returns full target-vocab logits plus the draft hidden output.

Return probability rows in request-major order, not step-major order. At every stable boundary (after target/scheduler computed lengths have been supplied) assert:

```python
state.draft_num_computed_tokens == max(seq.num_computed_tokens - 1, 0)
```

`commit` receives target verification auxiliary rows, accepted counts and explicit new target computed lengths. It selects row `accepted_count` per request as the next anchor and sets `draft_num_computed_tokens = committed_draft_length(new_target_num_computed_tokens[i])`. It also validates that the new draft length advances by exactly `accepted_count + 1`, corresponding to the pending root plus accepted draft prefix. `release(seq_ids)` deletes states idempotently.

- [ ] **Step 5: Test the proposer with a deterministic fake draft model**

The fake model must record tokens, features, positions and slot mappings and return fixed logits. Assert two-request ragged proposals have lengths `[3, 1]`, request-major probabilities, shifted positions, and forward counts `[4, 2]`; compare that batch result with running each request separately. Add a zero-length reservation test proving that one cache-fill forward occurs while the proposal contains zero token/probability rows. Assert no request accesses past `draft_len + 1` draft slots.

Run: `py -3.12 -m pytest tests/v1/spec_decode/test_eagle3_proposer.py -q`

Expected: all proposer tests pass.

- [ ] **Step 6: Commit**

```bash
git add nanovllm/v1/spec_decode/eagle3_proposer.py tests/v1/spec_decode/test_eagle3_proposer.py
git commit -m "feat: add linear EAGLE3 proposer state machine"
```

---

### Task 6: Exact Rejection Sampling with Explicit Accepted Counts

**Files:**
- Modify: `nanovllm/v1/sample/rejection_sampler.py`
- Create: `tests/v1/sample/test_rejection_sampler_reference.py`
- Create: `tests/v1/sample/test_rejection_sampler_cuda.py`

**Interfaces:**
- Consumes: request-major `DraftProposal`, target logits and temperatures
- Produces: `SpecDecodeResult`
- Produces: `reference_rejection_sample(draft_token_ids, target_logits, temperatures, draft_probs, acceptance_uniforms, recovery_uniforms) -> SpecDecodeResult`

- [ ] **Step 1: Write CPU reference tests**

Use a four-token vocabulary and injected acceptance/recovery uniforms. Cover all-accept, first rejection, middle rejection, zero drafts, ragged batch and temperature. Include a 50,000-sample statistical test with a fixed generator asserting output frequencies are within `0.015` absolute error of the target distribution.

- [ ] **Step 2: Write a CUDA parity test**

Mark with `@pytest.mark.cuda`; inject identical acceptance uniforms and exponential recovery noise into CPU and Triton paths and assert identical tokens and accepted counts.

- [ ] **Step 3: Run CPU tests and verify contract failure**

Run: `py -3.12 -m pytest tests/v1/sample/test_rejection_sampler_reference.py -q`

Expected: FAIL because the reference function and `SpecDecodeResult` return do not exist.

- [ ] **Step 4: Implement the CPU reference and update the sampler return**

The CPU reference must calculate `p = softmax(target_logits / temperature)` and use the already temperature-adjusted `q` from `DraftProposal.probabilities`. At position `i`, accept when `u <= min(1, p[token] / q[token])`; on rejection sample from normalized `clamp(p - q, min=0)`, stop the request, and return the accepted prefix plus recovered token. If all drafts are accepted, append the bonus sampled from the final target row.

Update the Triton wrapper to return:

```python
return SpecDecodeResult(
    output_token_ids=output_rows,
    accepted_draft_counts=accepted_counts,
)
```

Extend `_rejection_random_sample_kernel` with an `accepted_counts_ptr`. Initialize the request count to zero, increment it only on an actual acceptance branch, stop incrementing after rejection, and store the count before returning. Copy this explicit tensor to CPU together with the output rows and construct `SpecDecodeResult`; never infer acceptance by comparing token IDs because a recovered token may equal the rejected draft token. Validate draft probability shape and device before launching kernels. Preserve `draft_probs=None` for N-gram.

- [ ] **Step 5: Run CPU and CUDA tests**

Run CPU: `py -3.12 -m pytest tests/v1/sample/test_rejection_sampler_reference.py -q`

Run CUDA where available: `py -3.12 -m pytest tests/v1/sample/test_rejection_sampler_cuda.py -q -m cuda`

Expected: CPU tests pass; CUDA test passes on a CUDA host or is skipped by its explicit availability guard.

- [ ] **Step 6: Commit**

```bash
git add nanovllm/v1/sample/rejection_sampler.py tests/v1/sample/test_rejection_sampler_reference.py tests/v1/sample/test_rejection_sampler_cuda.py
git commit -m "feat: return explicit speculative acceptance results"
```

---

### Task 7: KV Budget, Prefix-Cache Disablement, and Scheduler Lifecycle

**Files:**
- Modify: `nanovllm/engine/block_manager.py`
- Modify: `nanovllm/engine/scheduler.py`
- Modify: `nanovllm/engine/sequence.py`
- Create: `tests/engine/test_block_manager_eagle3.py`
- Create: `tests/engine/test_scheduler_eagle3.py`

**Interfaces:**
- Produces: `Scheduler.reserve_spec_budget(seqs, requested_lengths) -> list[SpecReservation]`
- Produces: `Scheduler.get_eagle3_requested_lengths(seqs) -> list[int]`
- Produces: `Scheduler.pop_preempted_seq_ids() -> list[int]`
- Produces: `Scheduler.speculative_method -> str | None`
- Consumes: `SpecDecodeResult.accepted_draft_counts`

- [ ] **Step 1: Write failing block-manager tests**

Assert `enable_prefix_cache=False` never reuses a hash-matched block, does not increment `num_cached_tokens`, and still supports speculative reserve/commit across a block boundary.

- [ ] **Step 2: Write failing scheduler tests**

Cover requested lengths `[5, 3, 0]`, limited capacity truncation, explicit accepted counts `[0, 2, 0]`, a recovered-token collision with the rejected draft ID, EOS during accepted output, and preemption IDs. For an unfinished request, assert computed lengths use:

```python
expected_num_computed = old_sequence_length + accepted_draft_count
```

For a request stopped by EOS or `max_tokens`, use the number of accepted draft tokens actually appended before the stop. Do not infer acceptance by token equality.

- [ ] **Step 3: Run tests and verify failures**

Run: `py -3.12 -m pytest tests/engine/test_block_manager_eagle3.py tests/engine/test_scheduler_eagle3.py -q`

Expected: FAIL because prefix-cache configuration and budget reservation do not exist.

- [ ] **Step 4: Add the prefix-cache switch**

Add `enable_prefix_cache` to `BlockManager.__init__`. When false, `allocate` always allocates new blocks and `_finalize_computed_blocks` returns without hashing. Leave current N-gram/default behavior unchanged when true.

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache=True):
        self.enable_prefix_cache = enable_prefix_cache
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    def _finalize_computed_blocks(self, seq, num_computed_tokens):
        if not self.enable_prefix_cache:
            return
        num_full_blocks = num_computed_tokens // self.block_size
        for block_idx in range(num_full_blocks):
            self._finalize_one_block(seq, block_idx)

    def _finalize_one_block(self, seq, block_idx):
        block_id = seq.block_table[block_idx]
        block = self.blocks[block_id]
        if block.hash != -1:
            return
        token_ids = seq.block(block_idx)
        prefix = self.blocks[seq.block_table[block_idx - 1]].hash \
            if block_idx > 0 else -1
        block_hash = self.compute_hash(token_ids, prefix)
        block.update(block_hash, token_ids)
        self.hash_to_block_id[block_hash] = block_id
```

`allocate` skips all hash lookups and allocates from `free_block_ids` when prefix cache is disabled. In `may_append`, handle `len(seq) % block_size == 1` by allocating the new block without requiring the preceding full block to have a hash; handle `== 0` as a no-op rather than finalizing a hash. Preserve the existing hash assertions and updates only inside the enabled branch.

- [ ] **Step 5: Reserve budget before EAGLE3 proposal**

Implement `get_eagle3_requested_lengths` with the exact per-request formula below, then implement `reserve_spec_budget` by clamping each requested length to `get_num_appendable_tokens`, reserving blocks, and returning `SpecReservation`. Keep the old N-gram method as a wrapper that calls the budget method after N-gram token proposal and slices token IDs to reservation lengths.

```python
remaining_output_budget = seq.max_tokens - seq.num_completion_tokens
remaining_context = self.max_model_len - len(seq)
requested = max(
    0,
    min(
        self.num_speculative_tokens,
        remaining_output_budget - 1,
        remaining_context,
    ),
)
```

The reservation method must allocate sequentially so each request sees capacity remaining after earlier reservations:

```python
def reserve_spec_budget(self, seqs, requested_lengths):
    reservations = []
    for seq, requested in zip(seqs, requested_lengths):
        draft_len = min(requested, self.block_manager.get_num_appendable_tokens(seq))
        reservations.append(
            SpecReservation(
                draft_len=draft_len,
                new_block_ids=self.block_manager.reserve_spec_append(seq, draft_len),
            )
        )
    return reservations
```

Initialize `speculative_method` from `config.speculative_config`, and pass `config.enable_prefix_cache` into `BlockManager`. Change `postprocess_spec_decode` to accept `SpecDecodeResult`; reject counts outside `0..proposal_length`, use each count as the upper bound, count how many accepted draft outputs are actually appended before EOS/`max_tokens`, and use that committed count for metrics and `num_computed_tokens`. Track preempted IDs in a list populated by `preempt` and cleared by `pop_preempted_seq_ids`. The N-gram reservation wrapper and N-gram engine path also consume `SpecDecodeResult`, so this type change cannot bypass existing speculative postprocess.

- [ ] **Step 6: Run scheduler tests**

Run: `py -3.12 -m pytest tests/engine/test_block_manager_eagle3.py tests/engine/test_scheduler_eagle3.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add nanovllm/engine/block_manager.py nanovllm/engine/scheduler.py nanovllm/engine/sequence.py tests/engine/test_block_manager_eagle3.py tests/engine/test_scheduler_eagle3.py
git commit -m "feat: manage EAGLE3 speculative KV lifecycle"
```

---

### Task 8: ModelRunner and LLMEngine End-to-End Orchestration

**Files:**
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `nanovllm/engine/llm_engine.py`
- Create: `tests/engine/test_eagle3_flow.py`

**Interfaces:**
- Consumes: all interfaces from Tasks 1-7
- Produces: `ModelRunner.run_eagle3_prefill(seqs) -> list[int] | None`
- Produces: `ModelRunner.run_eagle3_spec_decode(seqs, reservations) -> tuple[DraftProposal, SpecDecodeResult] | None`
- Produces: `ModelRunner.release_eagle_states(seq_ids) -> None`

- [ ] **Step 1: Write a failing fake-model flow test**

Construct `LLMEngine` and `ModelRunner` via `__new__`, then inject fake target, draft proposer, sampler and scheduler inputs. Verify this exact event order:

```text
prefill: target_with_aux -> draft_prefill_and_store_anchor -> sample_root
decode: reserve_budget -> draft_propose -> target_verify_with_aux
        -> rejection_sample -> select_next_anchor -> scheduler_commit
```

For accepted counts `0`, `2`, and all accepted, assert the next anchor row index equals the accepted count and finished/preempted IDs release state.

- [ ] **Step 2: Run the flow test and verify missing methods**

Run: `py -3.12 -m pytest tests/engine/test_eagle3_flow.py -q`

Expected: FAIL because EAGLE3 runner methods do not exist.

- [ ] **Step 3: Load and warm both models**

In `ModelRunner.__init__`, branch on `method == "eagle3"`: instantiate `Qwen3Eagle3ForCausalLM(draft_config, self.model.model.embed_tokens)` so no duplicate full embedding is allocated, load strict draft weights, construct `Eagle3Proposer`, and warm target feature capture, one draft step and target verification before KV allocation.

Compute joint block bytes:

```python
target_block_bytes = (
    2 * target_layers * block_size * target_kv_heads * target_head_dim * dtype.itemsize
)
draft_block_bytes = (
    2 * draft_layers * block_size * draft_kv_heads * draft_head_dim * dtype.itemsize
)
config.num_kvcache_blocks = available_bytes // (
    target_block_bytes + draft_block_bytes
)
```

Allocate target/draft KV tensors with the same block count and attach them to their respective attention modules.

- [ ] **Step 4: Implement EAGLE3 prefill**

Run target prefill with `(2, 18, 33)`, select normal LM logits from `TargetModelOutput.hidden_states`, split auxiliary rows per request, call shifted draft prefill with explicit target computed lengths `[len(seq) for seq in seqs]`, save each request's last auxiliary row, and sample the first completion token using the existing target sampler. Reset global attention context after target and draft forwards.

- [ ] **Step 5: Implement EAGLE3 decode/verify**

`LLMEngine.step` asks `Scheduler.get_eagle3_requested_lengths` for the per-request upper bounds and calls `reserve_spec_budget` before invoking the runner. Pass those reservations into `ModelRunner.run_eagle3_spec_decode`, then into `Eagle3Proposer.propose`. Target verification input remains `seq.token_ids[seq.num_computed_tokens:] + draft_tokens`, which is pending root plus drafts. Return both selected verification logits and request-major query auxiliary rows. Invoke rejection sampling with `proposal.probabilities`; before scheduler postprocess changes sequence lengths, call proposer commit with `new_target_num_computed_tokens = [len(seq) + accepted for seq, accepted in zip(seqs, result.accepted_draft_counts)]`. Requests later stopped by EOS/`max_tokens` are immediately released, so their temporary predicted commit cannot survive a stable boundary. Return `(proposal, result)` to `LLMEngine.step`, which passes both objects to scheduler postprocess.

For reservation length zero, verification still processes the pending root, samples the bonus row, stores the root auxiliary row as the next anchor, and increments fallback metrics.

- [ ] **Step 6: Wire lifecycle notifications in `LLMEngine.step`**

Use the EAGLE3-specific prefill and decode methods only for `method == "eagle3"`. After scheduling, release `scheduler.pop_preempted_seq_ids()`. After postprocess, release states for finished sequences. Keep the existing N-gram path unchanged.

The `LLMEngine.step` branch must have this ownership order:

```python
seqs, is_prefill = self.scheduler.schedule()
preempted_ids = self.scheduler.pop_preempted_seq_ids()
if preempted_ids:
    self.model_runner.call("release_eagle_states", preempted_ids)

if is_prefill and self.scheduler.speculative_method == "eagle3":
    token_ids = self.model_runner.call("run_eagle3_prefill", seqs)
    num_decode_tokens = self.scheduler.postprocess(seqs, token_ids)
elif not is_prefill and self.scheduler.speculative_method == "eagle3":
    requested = self.scheduler.get_eagle3_requested_lengths(seqs)
    reservations = self.scheduler.reserve_spec_budget(seqs, requested)
    proposal, result = self.model_runner.call(
        "run_eagle3_spec_decode", seqs, reservations
    )
    num_decode_tokens = self.scheduler.postprocess_spec_decode(
        seqs, result, proposal, reservations
    )
else:
    if not is_prefill and self.scheduler.speculative_method == "ngram":
        draft_token_ids = self.model_runner.propose_draft_token_ids(seqs)
        draft_token_ids, reservations = self.scheduler.reserve_spec_decode(
            seqs, draft_token_ids
        )
        result = self.model_runner.call(
            "run_spec_decode", seqs, draft_token_ids, reservations
        )
        num_decode_tokens = self.scheduler.postprocess_spec_decode(
            seqs, result, draft_token_ids, reservations
        )
    else:
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        num_decode_tokens = self.scheduler.postprocess(seqs, token_ids)

finished_ids = [seq.seq_id for seq in seqs if seq.is_finished]
if finished_ids and self.scheduler.speculative_method == "eagle3":
    self.model_runner.call("release_eagle_states", finished_ids)
```

- [ ] **Step 7: Run flow and full CPU tests**

Run: `py -3.12 -m pytest tests/engine/test_eagle3_flow.py tests -q -m "not cuda and not model_weights"`

Expected: all CPU tests pass.

- [ ] **Step 8: Commit**

```bash
git add nanovllm/engine/model_runner.py nanovllm/engine/llm_engine.py tests/engine/test_eagle3_flow.py
git commit -m "feat: orchestrate linear EAGLE3 decoding"
```

---

### Task 9: Real-Checkpoint Verification, Metrics, Example, and Documentation

**Files:**
- Modify: `nanovllm/engine/scheduler.py`
- Modify: `nanovllm/engine/llm_engine.py`
- Modify: `nanovllm/v1/spec_decode/types.py`
- Modify: `example_sd.py`
- Modify: `docs/Speculative-Decoding.md`
- Create: `tests/integration/test_eagle3_qwen3_4b.py`
- Create: `bench_eagle3.py`

**Interfaces:**
- Produces public EAGLE3 metrics and reproducible integration/benchmark commands

- [ ] **Step 1: Write the real-checkpoint integration test**

Read target/draft directories from `NANOVLLM_TARGET_MODEL` and `NANOVLLM_EAGLE3_MODEL`; skip with an explicit reason when absent. Import the installed vLLM reference adapter for alignment cases and skip those cases separately with its version/import error when unavailable. Mark `cuda` and `model_weights`. Test:

- strict checkpoint loading with zero unresolved weights;
- target logits unchanged by feature capture within BF16 tolerance;
- for fixed 4-token and 8-token ID sequences, fused inputs, per-position draft hidden states and expanded logits match the installed vLLM Qwen3 EAGLE3 reference adapter; record the vLLM version/commit in test output;
- multi-round EAGLE3 target verification logits match ordinary token-by-token target forwards within BF16 tolerance;
- batch sizes 1 and 2 with ragged remaining `max_tokens`;
- EOS, 256-token block boundary and repeated requests;
- no monotonic increase in `torch.cuda.memory_allocated()` after warmup and cleanup.

- [ ] **Step 2: Add metrics before running the integration test**

Add a frozen `SpecDecodeMetrics` dataclass in `types.py` with `draft_tokens_proposed`, `draft_tokens_accepted`, `mean_effective_draft_length`, `fallback_decode_count`, `draft_time_ms`, `verify_time_ms`, and `sampling_time_ms`. Track counters and `perf_counter` durations in the runner/scheduler, expose `LLMEngine.spec_decode_metrics -> SpecDecodeMetrics`, and retain the existing `acceptance_rate` property.

- [ ] **Step 3: Run real-checkpoint tests**

PowerShell:

```powershell
$env:NANOVLLM_TARGET_MODEL='D:\models\Qwen3-4B-Instruct-2507'
$env:NANOVLLM_EAGLE3_MODEL='D:\models\Qwen3-4B-Instruct-2507-Eagle3'
py -3.12 -m pytest tests/integration/test_eagle3_qwen3_4b.py -q -m "cuda and model_weights"
```

Expected: all integration cases pass on a CUDA host with both local checkpoints and a compatible vLLM reference install. If the environment lacks any prerequisite, report the explicit skip output and do not claim the corresponding runtime/reference validation.

- [ ] **Step 4: Add the example and benchmark**

Update `example_sd.py` to select N-gram or EAGLE3 without hard-coded private paths. Add `bench_eagle3.py` accepting target/draft paths and running identical prompts/sampling parameters for target-only and EAGLE3 modes. Print output tokens, elapsed time, throughput, acceptance rate, effective draft length, fallback count and phase timings.

- [ ] **Step 5: Document exact usage and limitations**

Update `docs/Speculative-Decoding.md` with:

```python
llm = LLM(
    target_model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=2048,
    speculative_config={
        "method": "eagle3",
        "draft_model": draft_model_path,
        "num_speculative_tokens": 5,
    },
)
```

State the exact model pair, eager/single-GPU/2048 limits, disabled prefix cache, and that repository performance numbers are only valid after executing `bench_eagle3.py` on named hardware.

- [ ] **Step 6: Run the final verification matrix**

Run:

```powershell
py -3.12 -m pytest tests -q -m "not cuda and not model_weights"
py -3.12 -m pytest tests/v1/sample/test_rejection_sampler_cuda.py -q -m cuda
git diff --check
```

Then run the real-checkpoint test and `bench_eagle3.py` when model paths and CUDA are available. Record pass/skip counts and benchmark hardware in the handoff; do not convert skipped GPU checks into success claims.

- [ ] **Step 7: Commit**

```bash
git add nanovllm/engine/scheduler.py nanovllm/engine/llm_engine.py nanovllm/v1/spec_decode/types.py example_sd.py docs/Speculative-Decoding.md tests/integration/test_eagle3_qwen3_4b.py bench_eagle3.py
git commit -m "test: verify Qwen3 EAGLE3 decoding end to end"
```

---

## Execution Checkpoints

- After Task 1: configuration rejects every unsupported runtime/checkpoint combination before CUDA allocation.
- After Task 4: the real draft checkpoint can be loaded structurally, but generation is not yet enabled.
- After Task 6: sampling correctness is independently testable from model execution.
- After Task 8: fake-model end-to-end EAGLE3 flow is complete.
- After Task 9: real-checkpoint correctness and measured performance are available when CUDA/model prerequisites exist.
