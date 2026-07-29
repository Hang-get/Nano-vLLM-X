import argparse
import atexit
from dataclasses import asdict
import json
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


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


def run_mode(args, mode):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    speculative_config = None
    if mode == "eagle3":
        speculative_config = {
            "method": "eagle3",
            "draft_model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    llm = LLM(
        args.target_model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
    )
    try:
        llm.reset_spec_decode_metrics()
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        started = perf_counter()
        outputs = llm.generate(PROMPTS, sampling, use_tqdm=False)
        elapsed = perf_counter() - started
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        metrics = asdict(llm.spec_decode_metrics)
        return {
            "mode": mode,
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "throughput_tokens_per_second": output_tokens / elapsed,
            "acceptance_rate": llm.acceptance_rate,
            **metrics,
        }
    finally:
        atexit.unregister(llm.exit)
        llm.exit()
        del llm
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    results = [run_mode(args, "target_only"), run_mode(args, "eagle3")]
    print(f"Hardware: {hardware}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
