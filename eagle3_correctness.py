from typing import Any, Dict, List, Sequence, Union


def compare_token_sequences(
    expected_outputs: Sequence[Sequence[int]],
    actual_outputs: Sequence[Sequence[int]],
) -> List[Dict[str, Union[int, List[int]]]]:
    """Return first token mismatch details for each request."""
    if len(expected_outputs) != len(actual_outputs):
        raise ValueError(
            "request count differs: "
            f"expected {len(expected_outputs)}, actual {len(actual_outputs)}"
        )

    mismatches = []
    for request, (expected, actual) in enumerate(
        zip(expected_outputs, actual_outputs)
    ):
        position = next(
            (
                index
                for index, (expected_token, actual_token) in enumerate(
                    zip(expected, actual)
                )
                if expected_token != actual_token
            ),
            min(len(expected), len(actual)),
        )
        if position == len(expected) and position == len(actual):
            continue
        mismatches.append(
            {
                "request": request,
                "position": position,
                "expected_length": len(expected),
                "actual_length": len(actual),
                "expected_tokens": list(expected[position : position + 5]),
                "actual_tokens": list(actual[position : position + 5]),
            }
        )
    return mismatches


def build_correctness_report(
    *,
    configuration: Dict[str, Any],
    target_outputs: Sequence[Sequence[int]],
    tree_outputs: Sequence[Sequence[int]],
    mismatches: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a JSON-serializable report for a correctness run."""
    return {
        "configuration": dict(configuration),
        "target_outputs": [list(output) for output in target_outputs],
        "tree_outputs": [list(output) for output in tree_outputs],
        "mismatches": [dict(mismatch) for mismatch in mismatches],
        "passed": not mismatches,
    }
