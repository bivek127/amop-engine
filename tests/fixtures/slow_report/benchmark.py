"""Benchmark entry point for the Optimizer fixture.

A __main__ block that does a fixed, deterministic amount of work, so a
before/after comparison measures the code rather than the input.
"""

from app.aggregate import summarize

# Fixed input: same work every run, so two benchmarks are comparable.
VISITOR_IDS = [f"visitor-{i % 900}" for i in range(9000)]

if __name__ == "__main__":
    summarize(VISITOR_IDS)
