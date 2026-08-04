import argparse
import json

import torch

from bench_utils import run_mode, speedup
from nanovllm import SamplingParams


PROMPTS = [
    "Explain why speculative decoding preserves the target distribution.",
    "Write a short checklist for reviewing a CUDA inference kernel.",
    "Summarize the tradeoffs between latency and throughput in LLM serving.",
    "Give three examples of request-level KV cache lifecycle bugs.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare target-only, linear EAGLE3, and TreeAttention EAGLE3."
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--tree-top-k", type=int, default=3)
    parser.add_argument("--tree-max-depth", type=int, default=4)
    parser.add_argument("--tree-prune-ratio", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.tree_top_k < 2:
        parser.error("--tree-top-k must be at least 2 for the tree benchmark")
    if args.tree_max_depth < 1:
        parser.error("--tree-max-depth must be at least 1")
    if not 0.0 <= args.tree_prune_ratio <= 1.0:
        parser.error("--tree-prune-ratio must be in [0, 1]")
    return args


def eagle3_config(args) -> dict:
    return {
        "method": "eagle3",
        "draft_model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
    }


def main():
    args = parse_args()
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    base_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    baseline = run_mode(
        mode="target_only",
        model=args.target_model,
        prompts=PROMPTS,
        sampling=sampling,
        seed=args.seed,
        llm_kwargs=base_kwargs,
    )
    linear_eagle3 = run_mode(
        mode="linear_eagle3",
        model=args.target_model,
        prompts=PROMPTS,
        sampling=sampling,
        seed=args.seed,
        llm_kwargs={
            **base_kwargs,
            "speculative_config": eagle3_config(args),
        },
    )
    tree_eagle3 = run_mode(
        mode="tree_eagle3",
        model=args.target_model,
        prompts=PROMPTS,
        sampling=sampling,
        seed=args.seed,
        llm_kwargs={
            **base_kwargs,
            "speculative_config": {
                **eagle3_config(args),
                "tree_top_k": args.tree_top_k,
                "tree_max_depth": args.tree_max_depth,
                "tree_prune_ratio": args.tree_prune_ratio,
            },
        },
    )
    result = {
        "hardware": torch.cuda.get_device_name(0),
        "configuration": vars(args),
        "linear_eagle3_speedup": speedup(baseline, linear_eagle3),
        "tree_eagle3_speedup": speedup(baseline, tree_eagle3),
        "results": [baseline, linear_eagle3, tree_eagle3],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
