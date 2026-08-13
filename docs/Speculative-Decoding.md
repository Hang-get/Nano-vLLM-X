# Speculative Decoding

Nano-vLLM-X supports N-gram prompt lookup and EAGLE3 decoding. Both methods
keep target-model sampling authoritative. EAGLE3 has two modes:

- **Linear mode** (`tree_top_k=1`, the default) uses draft probabilities and
  rejection sampling. Accepted drafts plus the recovery or bonus token follow
  the target distribution.
- **Tree mode** (`tree_top_k>=2`) expands a top-k draft tree, verifies its
  root and draft nodes in one target-model forward pass, and samples a path
  only from target-model distributions.

## EAGLE3 Model Pair

The supported Qwen3-14B checkpoint pair is:

- Target: `Qwen/Qwen3-14B`, revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Draft: `thoughtworks/Qwen3-14B-Eagle3`, revision
  `af02e393da5821ad04d17333622678c57a56f860`
- Target auxiliary layers: `(2, 20, 37)`

The Thoughtworks draft stores `d2t` as `int32`, has architecture
`LlamaForCausalLM`, and intentionally omits several base-model config fields.
The runtime inherits the target's 40960-token context limit and validates the
fields that are part of this checkpoint contract. Neither model uses YaRN, so
do not enable YaRN for this pair.

Download both checkpoints to local directories. Runtime loading does not fetch
weights or a draft tokenizer. The target tokenizer is used for both models.

## EAGLE3 Usage

```python
from nanovllm import LLM, SamplingParams

target_model_path = "/models/Qwen3-14B"
draft_model_path = "/models/Qwen3-14B-Eagle3"

llm = LLM(
    target_model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.80,
    speculative_config={
        "method": "eagle3",
        "draft_model": draft_model_path,
        "num_speculative_tokens": 5,
    },
)

outputs = llm.generate(
    ["Explain speculative decoding."],
    SamplingParams(temperature=0.8, max_tokens=128),
)
print(outputs[0]["text"])
print(llm.spec_decode_metrics)
```

EAGLE3 validates the checkpoint architecture, dimensions, vocabulary, dtype,
and applicable token IDs before allocating CUDA memory. Incompatible checkpoints
fail with a field-specific error. It does not verify a Hugging Face revision
from a local directory; download the listed revisions explicitly.

## EAGLE3 Tree Mode

Tree mode is enabled only when `tree_top_k >= 2`. The tree depth includes the
pending root token, so a depth of `D` can accept at most `D - 1` draft tokens
and emits one target-sampled token. `num_speculative_tokens` controls only the
linear mode.

```python
llm = LLM(
    target_model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    speculative_config={
        "method": "eagle3",
        "draft_model": draft_model_path,
        "tree_top_k": 2,
        "tree_max_depth": 4,
        "tree_prune_ratio": 0.0,
    },
)
```

- `tree_top_k` must be at least `1`. A value of `1` selects the linear path.
- `tree_max_depth` must be at least `1` in tree mode.
- `tree_prune_ratio` must be in `[0, 1]`. At `0`, each expanded node keeps up
  to `tree_top_k` candidates. A positive value discards candidates whose
  probability is below `tree_prune_ratio * best_candidate_probability`; `1`
  keeps only the best candidate per expanded node.

Tree mode reserves separate pools for persistent target-KV append blocks and
transient draft copy-on-write blocks. Capacity pressure can reduce the
effective tree depth for an individual request. Pruned draft blocks are
released before target verification, and only the accepted target/draft path
is committed.

## Current EAGLE3 Limits

- One GPU and `tensor_parallel_size == 1`.
- Eager execution only: `enforce_eager=True`.
- The effective context length is the minimum of `max_model_len` and the target
  checkpoint limit. The supported Qwen3-14B target is limited to 40960 tokens;
  use 4096 as the initial A800 80G benchmark setting.
- Prefix caching is disabled.
- Tensor parallelism and CUDA Graph are unavailable in EAGLE3 mode.
- Tree configuration is global to an `LLM` instance; a batch cannot mix linear
  and tree EAGLE3 requests.

The target and draft use the same logical block IDs but separate physical KV
tensors. KV capacity is computed from the combined bytes per target and draft
block. Tree mode additionally uses transient draft copy-on-write and target
staging storage for its candidates.

## N-gram Usage

```python
llm = LLM(
    target_model_path,
    enforce_eager=True,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 2,
    },
)
```

N-gram proposals remain deterministic and do not provide draft probabilities.
The rejection sampler preserves this existing behavior while returning explicit
accepted counts.

## Metrics

`llm.spec_decode_metrics` returns an immutable snapshot with:

- proposed and accepted draft-token counts;
- mean effective draft length per speculative request;
- zero-draft fallback count;
- cumulative draft, target verification, and sampling time in milliseconds.

`llm.acceptance_rate` remains available. The pending root token is sampled by
the target and is excluded from proposed and accepted draft counts.

## Verification And Benchmarking

CPU checks:

```powershell
py -3.12 -m pytest tests -q -m "not cuda and not model_weights"
```

CUDA and local-checkpoint checks:

```powershell
$env:NANOVLLM_TARGET_MODEL='D:\models\Qwen3-14B'
$env:NANOVLLM_EAGLE3_MODEL='D:\models\Qwen3-14B-Eagle3'
py -3.12 -m pytest tests/integration/test_eagle3_qwen3_14b.py -q -m "cuda and model_weights"
```

Benchmark target-only and EAGLE3 with identical prompts and sampling settings:

```powershell
py -3.12 bench_eagle3.py `
  --target-model $env:NANOVLLM_TARGET_MODEL `
  --draft-model $env:NANOVLLM_EAGLE3_MODEL `
  --max-model-len 4096 `
  --temperature 0.6 `
  --gpu-memory-utilization 0.80
```

Benchmark target-only and N-gram decoding with repeated token prompts that make
prompt lookup measurable:

```powershell
py -3.12 bench.py --model D:\models\Qwen3-0.6B
```

Both benchmark scripts report the same JSON schema: end-to-end output
throughput, TTFT, completion latency, TPOT, acceptance rate, proposed and
accepted draft-token counts, effective draft length, fallback count, and the
draft/verification/sampling time breakdown. `throughput_speedup` compares the
speculative run directly with the target-only baseline from the same script.

Performance numbers are valid only when reported with the command, checkpoint
revision, software versions, and named GPU hardware. This repository does not
claim EAGLE3 speedups without running `bench_eagle3.py` in that environment.
