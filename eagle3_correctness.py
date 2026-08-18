from typing import Dict, List, Sequence, Union


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
