from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.data import DEFAULT_OUTPUT_DIR, write_monitoring_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference and live-like monitoring feature batches.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_monitoring_inputs(output_dir=args.output_dir, sample_count=args.sample_count)
    print(f"Wrote reference features to {summary['reference_path']}")
    print(f"Wrote current features to {summary['current_path']}")
    print(f"Wrote input summary for {summary['row_count']} rows per batch")


if __name__ == "__main__":
    main()
