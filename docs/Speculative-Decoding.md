# Speculative Decoding

N-gram speculative decoding accelerates target-model generation by proposing a
short continuation from repeated token patterns in the existing context. The
target model verifies the proposal in a single forward pass.

## N-gram Flow

1. `NgramProposer` searches each sequence for its longest repeated suffix and
   proposes up to `num_speculative_tokens` continuation tokens.
2. The model runner evaluates all proposal positions plus one bonus position
   for every request.
3. `RejectionSampler` accepts the longest valid draft prefix. On the first
   rejection it samples a replacement token; when all draft tokens are accepted
   it appends the bonus token.
4. The scheduler commits KV-cache blocks only for accepted draft tokens and
   releases unused reservations.

## Configuration

```python
from nanovllm import LLM

llm = LLM(
    "/YOUR/MODEL/PATH",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 2,
    },
)
```

`prompt_lookup_min` and `prompt_lookup_max` define the repeated n-gram range.
The engine falls back to a normal target-model token whenever no continuation
can be proposed.

## Metrics

`LLM.acceptance_rate` reports accepted draft tokens divided by proposed draft
tokens. Call `LLM.reset_spec_decode_metrics()` before a measured run.
