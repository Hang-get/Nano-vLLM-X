import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

import torch

from bench_utils import run_mode, speedup, summarize_benchmark_runs
from nanovllm import SamplingParams


PROMPTS = [
    "Explain why speculative decoding preserves the target distribution.",
    "Write a short checklist for reviewing a CUDA inference kernel.",
    "Summarize the tradeoffs between latency and throughput in LLM serving.",
    "Give three examples of request-level KV cache lifecycle bugs.",
    "Compare eager execution and CUDA Graph execution for inference latency.",
    "Explain why fixed output lengths matter in throughput benchmarks.",
    "Describe how KV cache capacity affects speculative decoding trees.",
    "List the validation checks needed before loading a draft checkpoint.",
    "Explain the difference between greedy decoding and temperature sampling.",
    "Give a concise incident report for a GPU out-of-memory failure.",
    "Summarize the role of the draft-to-target vocabulary mapping.",
    "Write a checklist for reproducing an inference performance regression.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare target-only, linear EAGLE3, and TreeAttention EAGLE3."
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--tree-top-k", type=int, default=3)
    parser.add_argument("--tree-max-depth", type=int, default=4)
    parser.add_argument("--tree-prune-ratio", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.tree_top_k < 2:
        parser.error("--tree-top-k must be at least 2 for the tree benchmark")
    if args.tree_max_depth < 1:
        parser.error("--tree-max-depth must be at least 1")
    if not 0.0 <= args.tree_prune_ratio <= 1.0:
        parser.error("--tree-prune-ratio must be in [0, 1]")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be non-negative")
    return args


def eagle3_config(args) -> dict:
    return {
        "method": "eagle3",
        "draft_model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def nvidia_driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (IndexError, OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict:
    device = torch.cuda.get_device_properties(0)
    prompt_text = "\n".join(PROMPTS).encode("utf-8")
    return {
        "git_revision": git_revision(),
        "nanovllm_version": package_version("nano-vllm-x"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": nvidia_driver_version(),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": device.name,
        "gpu_memory_bytes": device.total_memory,
        "packages": {
            name: package_version(name)
            for name in (
                "transformers",
                "flash-attn",
                "triton",
                "safetensors",
            )
        },
        "prompt_count": len(PROMPTS),
        "prompt_sha256": hashlib.sha256(prompt_text).hexdigest(),
    }


def latest_output_token_counts(runs: dict[str, list[dict]]) -> dict[str, int]:
    return {mode: values[-1]["output_tokens"] for mode, values in runs.items()}


def main():
    args = parse_args()
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    base_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    modes = (
        ("target_only", base_kwargs),
        (
            "linear_eagle3",
            {**base_kwargs, "speculative_config": eagle3_config(args)},
        ),
        (
            "tree_eagle3",
            {
                **base_kwargs,
                "speculative_config": {
                    **eagle3_config(args),
                    "tree_top_k": args.tree_top_k,
                    "tree_max_depth": args.tree_max_depth,
                    "tree_prune_ratio": args.tree_prune_ratio,
                },
            },
        ),
    )
    runs = {mode: [] for mode, _ in modes}
    execution_order = []
    for repeat in range(args.repeats):
        ordered_modes = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
        execution_order.append([mode for mode, _ in ordered_modes])
        for mode, llm_kwargs in ordered_modes:
            result = run_mode(
                mode=mode,
                model=args.target_model,
                prompts=PROMPTS,
                sampling=sampling,
                seed=args.seed + repeat,
                llm_kwargs=llm_kwargs,
                warmup_prompt=PROMPTS[repeat % len(PROMPTS)],
                warmup_tokens=args.warmup_tokens,
            )
            result["repeat"] = repeat
            runs[mode].append(result)

        output_counts = latest_output_token_counts(runs)
        if len(set(output_counts.values())) != 1:
            raise RuntimeError(
                "benchmark modes generated different output token counts; "
                "results are not comparable"
            )

    summaries = {mode: summarize_benchmark_runs(values) for mode, values in runs.items()}
    baseline = summaries["target_only"]
    linear_eagle3 = summaries["linear_eagle3"]
    tree_eagle3 = summaries["tree_eagle3"]
    result = {
        "environment": environment_record(),
        "configuration": vars(args),
        "measurement": {
            "ignore_eos": True,
            "output_length_policy": "force_max_tokens",
            "acceptance_rate_scope": (
                "Measured with ignore_eos=True, so EOS does not end requests; "
                "acceptance rates can differ from an EOS-respecting workload."
            ),
            "request_warmup": args.warmup_tokens > 0,
            "mode_order_by_repeat": execution_order,
        },
        "linear_eagle3_mean_throughput_speedup": speedup(
            baseline, linear_eagle3
        ),
        "tree_eagle3_mean_throughput_speedup": speedup(baseline, tree_eagle3),
        "summaries": [baseline, linear_eagle3, tree_eagle3],
        "results": [run for mode, _ in modes for run in runs[mode]],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
