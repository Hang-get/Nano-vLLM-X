import argparse

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run speculative decoding.")
    parser.add_argument("--model", required=True, help="Local target checkpoint")
    parser.add_argument("--method", choices=("ngram", "eagle3"), default="ngram")
    parser.add_argument("--draft-model", help="Local EAGLE3 draft checkpoint")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.method == "eagle3" and not args.draft_model:
        raise SystemExit("--draft-model is required for --method eagle3")

    speculative_config = {
        "method": args.method,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    if args.method == "eagle3":
        speculative_config["draft_model"] = args.draft_model
    else:
        speculative_config.update(
            prompt_lookup_min=1,
            prompt_lookup_max=2,
        )

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048 if args.method == "eagle3" else 4096,
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
    )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "Introduce yourself.",
        "List all prime numbers below 100.",
    ]
    prompts = [
        llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print(f"Acceptance rate: {llm.acceptance_rate:.4f}")
    print(f"Speculative metrics: {llm.spec_decode_metrics}")


if __name__ == "__main__":
    main()
