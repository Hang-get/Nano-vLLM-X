import atexit
from dataclasses import asdict
from statistics import median
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": sum(values) / len(values) if values else 0.0,
        "p50_ms": median(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95),
    }


def run_mode(
    *,
    mode: str,
    model: str,
    prompts: list[str] | list[list[int]],
    sampling: SamplingParams,
    seed: int,
    llm_kwargs: dict,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    llm = LLM(model, **llm_kwargs)
    try:
        llm.reset_spec_decode_metrics()
        request_ids = []
        for prompt in prompts:
            llm.add_request(prompt, sampling)
            request_ids.append(llm.scheduler.waiting[-1].seq_id)

        started = perf_counter()
        first_token_at: dict[int, float] = {}
        completed_at: dict[int, float] = {}
        output_lengths: dict[int, int] = {}
        while not llm.is_finished():
            outputs, _ = llm.step()
            now = perf_counter()
            for sequence in llm.scheduler.running:
                if sequence.num_completion_tokens > 0:
                    first_token_at.setdefault(sequence.seq_id, now)
            for sequence_id, token_ids in outputs:
                first_token_at.setdefault(sequence_id, now)
                completed_at[sequence_id] = now
                output_lengths[sequence_id] = len(token_ids)

        finished = perf_counter()
        ttft_ms = [
            (first_token_at[sequence_id] - started) * 1000
            for sequence_id in request_ids
            if sequence_id in first_token_at
        ]
        completion_latency_ms = [
            (completed_at[sequence_id] - started) * 1000
            for sequence_id in request_ids
            if sequence_id in completed_at
        ]
        tpot_ms = []
        for sequence_id in request_ids:
            token_count = output_lengths.get(sequence_id, 0)
            if token_count > 1 and sequence_id in first_token_at:
                tpot_ms.append(
                    (completed_at[sequence_id] - first_token_at[sequence_id])
                    * 1000
                    / (token_count - 1)
                )

        elapsed_seconds = finished - started
        output_tokens = sum(output_lengths.values())
        speculative_metrics = asdict(llm.spec_decode_metrics)
        speculative_time_ms = sum(
            speculative_metrics[name]
            for name in ("draft_time_ms", "verify_time_ms", "sampling_time_ms")
        )
        return {
            "mode": mode,
            "requests": len(request_ids),
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed_seconds,
            "throughput_tokens_per_second": (
                output_tokens / elapsed_seconds if elapsed_seconds else 0.0
            ),
            "ttft": summarize(ttft_ms),
            "completion_latency": summarize(completion_latency_ms),
            "tpot": summarize(tpot_ms),
            "acceptance_rate": llm.acceptance_rate,
            "speculative_time_ms": speculative_time_ms,
            "speculative_time_fraction": (
                speculative_time_ms / (elapsed_seconds * 1000)
                if elapsed_seconds
                else 0.0
            ),
            **speculative_metrics,
        }
    finally:
        atexit.unregister(llm.exit)
        llm.exit()
        del llm
        torch.cuda.empty_cache()


def speedup(baseline: dict, speculative: dict) -> float:
    baseline_throughput = baseline["throughput_tokens_per_second"]
    if baseline_throughput == 0:
        return 0.0
    return speculative["throughput_tokens_per_second"] / baseline_throughput
