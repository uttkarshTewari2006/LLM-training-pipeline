from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.monitoring.data import DEFAULT_OUTPUT_DIR, load_monitoring_inputs


DRIFT_REPORT_HTML = "drift_report.html"
DRIFT_REPORT_JSON = "drift_report.json"
DRIFT_SUMMARY_JSON = "drift_summary.json"


def build_drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    mean_delta = (current.mean(numeric_only=True) - reference.mean(numeric_only=True)).sort_values(
        key=lambda series: series.abs(),
        ascending=False,
    )
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "feature_count": int(len(reference.columns)),
        "largest_mean_shifts": [
            {"feature": feature, "mean_delta": float(delta)}
            for feature, delta in mean_delta.head(5).items()
        ],
        "interpretation": (
            "This batch intentionally shifts color and contrast statistics to exercise drift detection. "
            "A drift report is an investigation trigger, not proof that model accuracy changed."
        ),
    }


def generate_drift_report(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, current = load_monitoring_inputs(output_dir=output_dir)

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference, current)

    html_path = output_dir / DRIFT_REPORT_HTML
    json_path = output_dir / DRIFT_REPORT_JSON
    summary_path = output_dir / DRIFT_SUMMARY_JSON

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    summary = build_drift_summary(reference, current)
    summary["html_report_path"] = str(html_path)
    summary["json_report_path"] = str(json_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
