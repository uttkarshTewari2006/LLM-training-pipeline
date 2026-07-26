from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.data import DEFAULT_OUTPUT_DIR
from src.monitoring.report import generate_drift_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Evidently drift report for monitoring batches.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_drift_report(output_dir=args.output_dir)
    print(f"Wrote Evidently HTML report to {summary['html_report_path']}")
    print(f"Wrote Evidently JSON report to {summary['json_report_path']}")
    print("Largest mean shifts:")
    for item in summary["largest_mean_shifts"]:
        print(f"- {item['feature']}: {item['mean_delta']:.6f}")


if __name__ == "__main__":
    main()
