"""
Deprecated legacy evaluator entrypoint.

Use `project/scripts/evaluate_score_based_v3.py` for evaluation.
"""

import sys


def main() -> int:
    print(
        "Deprecated script: `project/scripts/evaluate_score_based.py` is disabled.\n"
        "Use: python project/scripts/evaluate_score_based_v3.py --help"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
