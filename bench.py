import argparse
import json
from random import Random

import torch

from bench_utils import run_mode, speedup
from nanovllm import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare target-only and N-gram speculative decoding."
    )
    parser.add_argument("--model", required=True, help="Local Qwen3 checkpoint")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--prompt-lookup-min", type=int, default=1)
    parser.add_argument("--prompt-lookup-max", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def build_repeated_prompts(args) -> list[list[int]]:
    rng = Random(args.seed)
    pattern = [rng.randrange(10000) for _ in range(min(32, args.input_length))]
    return [
        (pattern * ((args.input_length + len(pattern) - 1) // len(pattern)))[
            : args.input_length
        ]
        for _ in range(args.num_requests)
    ]


def main():
    args = parse_args()
    prompts = build_repeated_prompts(args)
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
    baseline = run_mode(
        mode="target_only",
        model=args.model,
        prompts=prompts,
        sampling=sampling,
        seed=args.seed,
        llm_kwargs=base_kwargs,
    )
    ngram = run_mode(
        mode="ngram",
        model=args.model,
        prompts=prompts,
        sampling=sampling,
        seed=args.seed,
        llm_kwargs={
            **base_kwargs,
            "speculative_config": {
                "method": "ngram",
                "num_speculative_tokens": args.num_speculative_tokens,
                "prompt_lookup_min": args.prompt_lookup_min,
                "prompt_lookup_max": args.prompt_lookup_max,
            },
        },
    )
    result = {
        "hardware": torch.cuda.get_device_name(0),
        "configuration": vars(args),
        "throughput_speedup": speedup(baseline, ngram),
        "results": [baseline, ngram],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
