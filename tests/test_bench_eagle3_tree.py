import pytest

from bench_eagle3_tree import validate_repeat_output_token_counts
from bench_utils import speedup


def test_repeat_output_token_validation_reads_each_mode_at_the_requested_repeat():
    runs = {
        "target_only": [{"output_tokens": 12}, {"output_tokens": 24}],
        "linear_eagle3": [{"output_tokens": 12}, {"output_tokens": 24}],
        "tree_eagle3": [{"output_tokens": 12}, {"output_tokens": 24}],
    }

    assert validate_repeat_output_token_counts(runs, 1) == {
        "target_only": 24,
        "linear_eagle3": 24,
        "tree_eagle3": 24,
    }


def test_repeat_output_token_validation_rejects_inconsistent_outputs():
    runs = {
        "target_only": [{"output_tokens": 12}],
        "linear_eagle3": [{"output_tokens": 11}],
        "tree_eagle3": [{"output_tokens": 12}],
    }

    with pytest.raises(ValueError, match="repeat 0"):
        validate_repeat_output_token_counts(runs, 0)


def test_speedup_uses_summary_mean_throughput_field():
    baseline_summary = {"throughput_tokens_per_second": 100.0}
    tree_summary = {"throughput_tokens_per_second": 125.0}

    assert speedup(baseline_summary, tree_summary) == 1.25
