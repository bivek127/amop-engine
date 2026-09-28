# monitored_service

A real, long-running order-processing service used to demonstrate spec
6.1/8.7's log/metric-based anomaly detection -- `amop watch` discovering
a real production-style incident from log data, with no GitHub issue and
no human-created task involved.

Unlike every other fixture in `tests/fixtures/`, this is not run by
pytest inside the sandbox. It's a plain, standalone Python process you
start yourself; it writes real log lines to a real file, which
`amop/tools/monitoring.py`'s tools (and, through them, `amop watch`)
read from the host, the same way a real operator's log-tailing setup
would.

## The seeded bug

`process_order()` divides an order's total by its item count. An empty
cart (0 items) is a genuine, unguarded `ZeroDivisionError` -- not
simulated, not mocked. Most generated orders are healthy; the fraction
that are empty carts is controlled by a small on-disk control file that
the running process re-reads every loop iteration, so an incident can be
"seeded" into an already-running process without restarting it.

## Live demo: watch AMOP discover a real incident on its own

Terminal 1 -- start the monitored app, healthy (0% error rate):

```bash
cd /Users/bivekmohanbhattarai/bmb/amop-engine/tests/fixtures/monitored_service
source /Users/bivekmohanbhattarai/bmb/amop-engine/.venv/bin/activate && python3 app.py --log-path /tmp/monitored_service.log --interval 1.0
```

Terminal 2 -- start `amop watch`, pointed at whatever repo you already
use for `--repo`/`--local-path` (the same repo Milestone 9's GitHub demo
used works fine -- the log source is independent of what's in that
repo's issue tracker). A short window/threshold makes the demo fast
instead of needing to wait a real 5 minutes:

```bash
source /Users/bivekmohanbhattarai/bmb/amop-engine/.venv/bin/activate && AMOP_LOG_SOURCE_PATH=/tmp/monitored_service.log AMOP_LOG_SOURCE_REPO=<your-repo-slug> AMOP_ERROR_RATE_THRESHOLD_PCT=5 AMOP_ERROR_RATE_WINDOW_MINUTES=1 python3 -m amop.cli.main watch --repo <your-repo-slug> --local-path <path-to-local-clone> --interval 15
```

Watch a few clean cycles go by (`error_rate_pct=0.00 -- OK`).

Terminal 3 -- seed a real incident into the still-running app.py, no
restart:

```bash
cd /Users/bivekmohanbhattarai/bmb/amop-engine/tests/fixtures/monitored_service
source /Users/bivekmohanbhattarai/bmb/amop-engine/.venv/bin/activate && python3 seed_error.py --log-path /tmp/monitored_service.log --rate 0.5
```

Within the next 1-2 poll cycles, Terminal 2 should show the threshold
rule firing on its own (`AnomalyAlert (logs): ...`), a real task created,
and TRIAGING -> INVESTIGATING -- with nobody having run `amop fix` or
created a task by hand.

To return to healthy: `python3 seed_error.py --log-path /tmp/monitored_service.log --rate 0`.
