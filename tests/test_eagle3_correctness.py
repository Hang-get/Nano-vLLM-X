import pytest

from eagle3_correctness import compare_token_sequences


def test_compare_token_sequences_accepts_identical_outputs():
    outputs = [[1, 2, 3], [8, 13]]

    assert compare_token_sequences(outputs, outputs) == []


def test_compare_token_sequences_reports_first_token_mismatch():
    mismatches = compare_token_sequences(
        [[10, 20, 30, 40]],
        [[10, 20, 99]],
    )

    assert mismatches == [
        {
            "request": 0,
            "position": 2,
            "expected_length": 4,
            "actual_length": 3,
            "expected_tokens": [30, 40],
            "actual_tokens": [99],
        }
    ]


def test_compare_token_sequences_rejects_different_request_counts():
    with pytest.raises(ValueError, match="request count"):
        compare_token_sequences([[1]], [[1], [2]])
