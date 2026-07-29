import os

import numpy as np
from numba import get_num_threads, jit, njit, prange, set_num_threads


class NgramProposer:
    def __init__(
        self,
        prompt_lookup_min: int = 1,
        prompt_lookup_max: int = 3,
        num_speculative_tokens: int = 2,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
    ):
        assert max_model_len is not None, "max_model_len must be specified"
        assert max_num_seqs is not None, "max_num_seqs must be specified"
        self.min_n = prompt_lookup_min
        self.max_n = prompt_lookup_max
        self.k = num_speculative_tokens
        self.max_model_len = max_model_len
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros(max_num_seqs, dtype=np.int32)
        self.num_tokens_threshold = 8192
        cpu_count = os.cpu_count()
        self.num_numba_thread_available = min(8, cpu_count // 2) if cpu_count else 1

        self.propose(
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, max_model_len), dtype=np.int32),
        )

    def propose(
        self,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        num_requests = int(num_tokens_no_spec.shape[0])
        if token_ids_cpu.shape[0] != num_requests:
            raise ValueError(
                "token_ids_cpu batch size must match num_tokens_no_spec, "
                f"got {token_ids_cpu.shape[0]} and {num_requests}."
            )
        valid_ngram_requests = [
            i
            for i, num_tokens in enumerate(num_tokens_no_spec)
            if 0 < num_tokens < self.max_model_len
        ]
        return self.batch_propose(
            num_requests,
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list[int],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        valid_ngram_request_set = set(valid_ngram_requests)
        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            if np.sum(num_tokens_no_spec) >= self.num_tokens_threshold:
                set_num_threads(
                    max(1, min(self.num_numba_thread_available, num_ngram_requests))
                )
            else:
                set_num_threads(1)
            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )
            set_num_threads(original_num_numba_threads)

        return [
            self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
            if i in valid_ngram_request_set and self.valid_ngram_num_drafts[i] > 0
            else []
            for i in range(num_requests)
        ]


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list[int],
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        request_id = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[request_id]
        proposed_tokens = _find_longest_matched_ngram_and_propose_tokens(
            token_ids_cpu[request_id][:num_tokens], min_n, max_n, max_model_len, k
        )
        valid_ngram_draft[request_id][: len(proposed_tokens)] = proposed_tokens
        valid_ngram_num_drafts[request_id] = len(proposed_tokens)


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty(0, dtype=origin_tokens.dtype)
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty(0, dtype=origin_tokens.dtype)

    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty(0, dtype=origin_tokens.dtype)
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
