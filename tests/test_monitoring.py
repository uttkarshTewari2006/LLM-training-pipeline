from pathlib import Path

import pandas as pd

from src.monitoring.data import CURRENT_FILE, REFERENCE_FILE, build_monitoring_frames, write_monitoring_inputs
from src.monitoring.report import DRIFT_REPORT_HTML, DRIFT_SUMMARY_JSON, build_drift_summary, generate_drift_report


def test_build_monitoring_frames_has_matching_feature_schema() -> None:
    reference, current = build_monitoring_frames(sample_count=16)

    assert len(reference) == 16
    assert len(current) == 16
    assert list(reference.columns) == list(current.columns)
    assert "brightness_mean" in reference.columns


def test_write_monitoring_inputs_creates_reference_and_current_files(tmp_path: Path) -> None:
    summary = write_monitoring_inputs(output_dir=tmp_path, sample_count=8)

    assert (tmp_path / REFERENCE_FILE).exists()
    assert (tmp_path / CURRENT_FILE).exists()
    assert summary["row_count"] == 8
    assert len(pd.read_csv(tmp_path / REFERENCE_FILE)) == 8
    assert len(pd.read_csv(tmp_path / CURRENT_FILE)) == 8


def test_build_drift_summary_reports_largest_mean_shifts() -> None:
    reference, current = build_monitoring_frames(sample_count=32)
    summary = build_drift_summary(reference, current)

    assert summary["reference_rows"] == 32
    assert summary["current_rows"] == 32
    assert len(summary["largest_mean_shifts"]) == 5


def test_generate_drift_report_writes_artifacts(tmp_path: Path) -> None:
    write_monitoring_inputs(output_dir=tmp_path, sample_count=32)

    summary = generate_drift_report(output_dir=tmp_path)

    assert (tmp_path / DRIFT_REPORT_HTML).exists()
    assert (tmp_path / DRIFT_SUMMARY_JSON).exists()
    assert summary["html_report_path"] == str(tmp_path / DRIFT_REPORT_HTML)
