<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-X

Nano-vLLM-X is a lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

N-gram and EAGLE3 speculative decoding are configured through
`speculative_config`. N-gram uses Numba for prompt lookup and Triton for GPU
rejection sampling. EAGLE3 supports linear and optional tree-shaped proposals;
see [Speculative Decoding](docs/Speculative-Decoding.md) for the supported
checkpoint pair and runtime constraints.

```bash
pip install git+https://github.com/Hang-get/Nano-vLLM-X.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` or `example_sd.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM(
    "/YOUR/MODEL/PATH",
    enforce_eager=True,
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 2,
    },
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM-X."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## EAGLE3 Tree Decoding

EAGLE3 requires the compatible local target and draft checkpoints, one GPU,
and eager execution. `tree_top_k=1` keeps the linear EAGLE3 path. Set
`tree_top_k >= 2` and `tree_max_depth >= 1` to generate a pruned top-k token
tree that the target model verifies in one forward pass per speculative round.

```python
llm = LLM(
    "/models/Qwen3-4B-Instruct-2507",
    enforce_eager=True,
    tensor_parallel_size=1,
    speculative_config={
        "method": "eagle3",
        "draft_model": "/models/Qwen3-4B-Instruct-2507-Eagle3",
        "num_speculative_tokens": 5,  # Used by the linear path.
        "tree_top_k": 2,
        "tree_max_depth": 4,           # Includes the pending root token.
        "tree_prune_ratio": 0.0,
    },
)
```

## Benchmark

Use `bench.py` to compare target-only decoding with N-gram speculative decoding:

```powershell
py -3.12 bench.py --model D:\models\Qwen3-0.6B
```

Use `bench_eagle3.py` to compare target-only decoding with EAGLE3:

```powershell
py -3.12 bench_eagle3.py `
  --target-model D:\models\Qwen3-4B-Instruct-2507 `
  --draft-model D:\models\Qwen3-4B-Instruct-2507-Eagle3
```

Both scripts emit JSON with end-to-end throughput, throughput speedup, TTFT,
completion latency, TPOT, acceptance rate, draft proposal and acceptance
counts, effective draft length, fallback count, and draft/verification/sampling
time attribution.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM-X    | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hang-get/Nano-vLLM-X&type=Date)](https://www.star-history.com/#Hang-get/Nano-vLLM-X&Date)
