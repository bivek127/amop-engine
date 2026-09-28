"""Seed a real incident into an ALREADY-RUNNING app.py process, without
restarting it -- edits the tiny control file app.py re-reads every loop
iteration. This is the live-demo trigger for spec 6.1/8.7's log-based
anomaly detection: start app.py healthy (rate 0), let `amop watch` poll
a few clean cycles, then run this to push the real, live error rate over
whatever threshold AMOP_ERROR_RATE_THRESHOLD_PCT is configured to, and
watch the NEXT poll cycle discover it on its own.

Usage:
    python3 seed_error.py --log-path <path> --rate 0.5   # 50% empty carts
    python3 seed_error.py --log-path <path> --rate 0     # back to healthy
"""

import argparse
import json
import os
from pathlib import Path

from app import _control_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default=os.environ.get("AMOP_LOG_SOURCE_PATH"))
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Fraction (0.0-1.0) of new orders that are empty carts (the seeded bug).",
    )
    args = parser.parse_args()
    if not args.log_path:
        parser.error("--log-path (or $AMOP_LOG_SOURCE_PATH) is required")

    control_path = _control_path(Path(args.log_path))
    control_path.write_text(json.dumps({"empty_cart_rate": max(0.0, min(1.0, args.rate))}))
    print(f"empty_cart_rate set to {args.rate} in {control_path}")


if __name__ == "__main__":
    main()
