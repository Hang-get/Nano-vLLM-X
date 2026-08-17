from bench_eagle3_tree import latest_output_token_counts


def test_latest_output_token_counts_reads_the_latest_run_per_mode():
    runs = {
        "target_only": [{"output_tokens": 12}, {"output_tokens": 24}],
        "linear_eagle3": [{"output_tokens": 12}, {"output_tokens": 24}],
        "tree_eagle3": [{"output_tokens": 12}, {"output_tokens": 24}],
    }

    assert latest_output_token_counts(runs) == {
        "target_only": 24,
        "linear_eagle3": 24,
        "tree_eagle3": 24,
    }
