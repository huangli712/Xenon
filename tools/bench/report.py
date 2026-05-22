#!/usr/bin/env python3
"""Parse benchmark stdout and annotate performance change bands.

Usage:
    python tools/bench/report.py --input <stdout.txt> --baseline <baseline.json>

Regression bands (27-benchmark §6.4):
    delta > 20%  → failure  (report as regression)
    delta > 5%   → warning  (flag for investigation)
    delta <= 5%  → noise    (within measurement noise)
"""

import argparse
import json
import re
import sys

# Matches three benchmark output formats produced by W28 benches:
#   1. Plain:               "elem_add_f64/65536: 12345 ns"
#   2. 2D shape:            "sum_2d_axis0/256x256: 12345 ns"
#   3. SIMD/parallel path:  "simd_add_compare_f32/65536/simd: 12345 ns"
#                           "par_sum_compare_i64/16777216/parallel: 12345 ns"
BENCH_LINE_RE = re.compile(
    r"^(?P<name>[\w]+)/(?P<size>\d+(?:x\d+)?)(?:/(?P<path>scalar|simd|serial|parallel))?: (?P<ns>\d+) ns"
)


def parse_bench_output(path: str) -> dict[str, int]:
    """Parse stdout lines into {key: median_ns}.

    Key format: "<name>/<size>" or "<name>/<size>/<path>" for comparison
    benches. The path suffix lets SIMD-on vs SIMD-off invocations live in
    the same baseline file without collisions.
    """
    results: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            m = BENCH_LINE_RE.search(line)
            if not m:
                continue
            name = m.group("name")
            size = m.group("size")
            path_suffix = m.group("path")
            ns = int(m.group("ns"))
            key = f"{name}/{size}/{path_suffix}" if path_suffix else f"{name}/{size}"
            results[key] = ns
    return results


def classify_delta(delta: float) -> str:
    """Classify performance change against 27-benchmark §6.4 thresholds."""
    if delta > 0.20:
        return "failure"
    if delta > 0.05:
        return "warning"
    return "noise"


def load_baseline(path: str) -> dict[str, int]:
    """Load baseline from JSON file."""
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark regression reporter")
    parser.add_argument("--input", required=True, help="Benchmark stdout text file")
    parser.add_argument("--baseline", required=True, help="Baseline JSON file")
    parser.add_argument("--output", default=None, help="Write JSON report (default: stdout)")
    args = parser.parse_args()

    current = parse_bench_output(args.input)
    baseline = load_baseline(args.baseline)

    report = {}
    all_ok = True
    for key, cur_ns in current.items():
        base_ns = baseline.get(key)
        if base_ns is None:
            print(f"WARNING: no baseline for {key}", file=sys.stderr)
            continue
        delta = (cur_ns - base_ns) / base_ns
        band = classify_delta(delta)
        report[key] = {"current_ns": cur_ns, "baseline_ns": base_ns, "delta": delta, "band": band}
        if band == "failure":
            print(f"REGRESSION: {key} ({delta:+.1%})", file=sys.stderr)
            all_ok = False
        elif band == "warning":
            print(f"WARNING: {key} ({delta:+.1%})", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()