#!/usr/bin/env python3
"""
CLI entry point for the held-out batch evaluation. Uses the exact same
server.batch.run_held_out_batch() function as the /run_batch API
endpoint, so the numbers you get here are identical to what the
dashboard shows - one code path, no hand-edits.

Usage:
    python scripts/run_batch.py --n 100 --seed-start 1000
    python scripts/run_batch.py --golden-trap
    python scripts/run_batch.py --ablation --n 40
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.db import init_db
from server.batch import run_held_out_batch, run_ablation


def main():
    parser = argparse.ArgumentParser(description="Run FraudCourt's held-out batch evaluation.")
    parser.add_argument("--n", type=int, default=40, help="Number of cases (ignored with --golden-trap)")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--golden-trap", action="store_true", help="Run the 15-case hand-authored trap suite instead")
    parser.add_argument("--ablation", action="store_true", help="Run naive-vs-full ablation instead")
    parser.add_argument("--out", default=None, help="Output JSON path (default: reports/<name>.json)")
    args = parser.parse_args()

    init_db()
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if args.ablation:
        result = run_ablation(n=args.n, seed_start=args.seed_start)
        out_path = args.out or os.path.join(reports_dir, "ablation_summary.json")
    else:
        result = run_held_out_batch(n=args.n, seed_start=args.seed_start, use_golden_trap=args.golden_trap)
        default_name = "golden_trap_summary.json" if args.golden_trap else "component_shift_summary.json"
        out_path = args.out or os.path.join(reports_dir, default_name)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
