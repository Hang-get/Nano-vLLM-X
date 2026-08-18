import argparse
import atexit
import gc

import torch

from eagle3_correctness import compare_token_sequences
from nanovllm import LLM, SamplingParams


PROMPTS = [
    "Explain why speculative decoding preserves the target distribution.",
    "Describe how KV cache capacity affects speculative decoding trees.",
    "Explain the difference between greedy decoding and temperature sampling.",
    "Write a short checklist for reviewing a CUDA inference kernel.",
    "Summarize the tradeoffs between latency and throughput in LLM serving.",
    "List the validation checks needed before loading a draft checkpoint.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare target-only and tree EAGLE3 token sequences "
            "under greedy decoding."
        )
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--tree-top-k", type=int, default=2)
    parser.add_argument("--tree-max-depth", type=int, default=3)
    parser.add_argument("--tree-prune-ratio", type=float, default=0.0)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force max_tokens when true; use --no-ignore-eos for natural EOS.",
    )
    args = parser.parse_args()
    if args.tree_top_k < 2:
        parser.error("--tree-top-k must be at least 2")
    if args.tree_max_depth < 1:
        parser.error("--tree-max-depth must be at least 1")
    if not 0.0 <= args.tree_prune_ratio <= 1.0:
        parser.error("--tree-prune-ratio must be in [0, 1]")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def run_generation(args, *, tree: bool, seed: int) -> list[list[int]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    llm_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if tree:
        llm_kwargs["speculative_config"] = {
            "method": "eagle3",
            "draft_model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "tree_top_k": args.tree_top_k,
            "tree_max_depth": args.tree_max_depth,
            "tree_prune_ratio": args.tree_prune_ratio,
        }

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    llm = LLM(args.target_model, **llm_kwargs)
    try:
        outputs = llm.generate(PROMPTS, sampling, use_tqdm=False)
        return [list(output["token_ids"]) for output in outputs]
    finally:
        atexit.unregister(llm.exit)
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    print(
        "Checking target-only vs tree EAGLE3: "
        f"prompts={len(PROMPTS)}, max_tokens={args.max_tokens}, "
        f"ignore_eos={args.ignore_eos}, repeats={args.repeats}"
    )

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        target_outputs = run_generation(args, tree=False, seed=seed)
        tree_outputs = run_generation(args, tree=True, seed=seed)
        mismatches = compare_token_sequences(target_outputs, tree_outputs)
        if mismatches:
            print(f"[FAIL] repeat={repeat}, mismatches={len(mismatches)}")
            for mismatch in mismatches:
                print(
                    f"  request={mismatch['request']} "
                    f"position={mismatch['position']} "
                    f"target_len={mismatch['expected_length']} "
                    f"tree_len={mismatch['actual_length']}"
                )
                print(f"    target tokens: {mismatch['expected_tokens']}")
                print(f"    tree tokens:   {mismatch['actual_tokens']}")
            raise SystemExit(1)
        print(f"[PASS] repeat={repeat}: all token sequences identical")

    print("CORRECTNESS: PASS")


if __name__ == "__main__":
    main()
