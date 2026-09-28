"""A minimal, long-running order-processing service -- the real "sample
app" for demonstrating spec 6.1/8.7's log/metric-based anomaly
detection. Unlike every other fixture in this directory (buggy_calculator,
task_tracker, ...), this one is not driven by pytest inside the ephemeral
sandbox -- it is meant to be started as a real background process on the
host, so `amop watch`'s log-threshold check (amop/orchestrator/
monitoring.py) has a real, currently-running source to poll against.
No prior milestone demo has used a running app like this one; it did not
exist before this task.

The bug is real, not simulated: process_order() computes an order's
average item price as sum(prices) / len(items), which is a genuine
ZeroDivisionError for an empty cart. Most simulated orders are healthy
(non-empty); the fraction that are empty carts is controlled by a tiny
on-disk control file, checked once per loop iteration, so an operator
can "seed" a real incident into the ALREADY-RUNNING process (edit the
control file) without restarting it -- closer to how a real production
error rate actually spikes than a restart-with-new-flags would be. See
seed_error.py.

Log line format (must match amop/tools/monitoring.py's parser exactly):
  "<ISO-8601 UTC timestamp> <INFO|WARNING|ERROR> <message>"
"""

import argparse
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path


def _control_path(log_path: Path) -> Path:
    return log_path.with_suffix(log_path.suffix + ".control.json")


def read_empty_cart_rate(control_path: Path) -> float:
    try:
        data = json.loads(control_path.read_text())
        return max(0.0, min(1.0, float(data.get("empty_cart_rate", 0.0))))
    except (FileNotFoundError, ValueError, TypeError):
        return 0.0


def make_order(order_id: int, *, empty: bool) -> dict:
    if empty:
        return {"id": order_id, "items": []}
    n = random.randint(1, 5)
    return {"id": order_id, "items": [{"price": round(random.uniform(3, 40), 2)} for _ in range(n)]}


def process_order(order: dict) -> float:
    """The real, seeded bug: an empty cart divides by zero. No guard --
    this is exactly the kind of unhandled-edge-case bug every other
    fixture in this project seeds deliberately (e.g. buggy_calculator's
    own division bug), just running continuously instead of under pytest.
    """
    items = order["items"]
    total = sum(i["price"] for i in items)
    return total / len(items)


def log_line(log_path: Path, level: str, message: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    with log_path.open("a") as fh:
        fh.write(f"{ts} {level} {message}\n")


def run(log_path: Path, interval: float) -> None:
    control_path = _control_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not control_path.exists():
        control_path.write_text(json.dumps({"empty_cart_rate": 0.0}))

    log_line(log_path, "INFO", f"monitored_service started, pid={os.getpid()}")
    order_id = 0
    while True:
        order_id += 1
        rate = read_empty_cart_rate(control_path)
        order = make_order(order_id, empty=random.random() < rate)
        try:
            avg = process_order(order)
            log_line(log_path, "INFO", f"order {order_id} processed avg_item_price={avg:.2f}")
        except ZeroDivisionError:
            log_line(
                log_path,
                "ERROR",
                f"order {order_id} failed: division by zero -- empty cart (0 items)",
            )
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        default=os.environ.get("AMOP_LOG_SOURCE_PATH"),
        help="Where to write log lines. Defaults to $AMOP_LOG_SOURCE_PATH.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between simulated orders (default 1.0).",
    )
    args = parser.parse_args()
    if not args.log_path:
        parser.error("--log-path (or $AMOP_LOG_SOURCE_PATH) is required")

    try:
        run(Path(args.log_path), args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
