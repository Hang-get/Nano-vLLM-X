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
        description="Compare target-only and linear EAGLE3 decoding."
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


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
    eagle3 = run_mode(
        mode="eagle3",
        model=args.target_model,
        prompts=PROMPTS,
        sampling=sampling,
        seed=args.seed,
        llm_kwargs={
            **base_kwargs,
            "speculative_config": {
                "method": "eagle3",
                "draft_model": args.draft_model,
                "num_speculative_tokens": args.num_speculative_tokens,
            },
        },
    )
    result = {
        "hardware": torch.cuda.get_device_name(0),
        "configuration": vars(args),
        "throughput_speedup": speedup(baseline, eagle3),
        "results": [baseline, eagle3],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
