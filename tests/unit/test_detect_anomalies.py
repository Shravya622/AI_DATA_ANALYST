"""
Unit tests for app.anomaly_detector.detect_anomalies.

Covers all four acceptance criteria from task 10.1:
  AC1 – A DataFrame with a known outlier is flagged correctly.
  AC2 – Each AnomalyRecord.explanation names the column and states the reason.
  AC3 – A DataFrame with no numeric columns returns the expected message.
  AC4 – AnomalyReport.count == len(AnomalyReport.flagged_rows).
"""

import pandas as pd
import pytest

from app.anomaly_detector import detect_anomalies, _NO_NUMERIC_MESSAGE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zscore_df() -> pd.DataFrame:
    """Return a DataFrame where the last row is a clear Z-score outlier.

    19 tightly-clustered values of 1.0 followed by a single extreme value of
    100.0 → |z| ≈ 4.25, which comfortably exceeds the default threshold of 3.0.
    """
    values = [1.0] * 19 + [100.0]
    return pd.DataFrame({"score": values})


def _make_iqr_df() -> pd.DataFrame:
    """Return a DataFrame where row 9 is a clear IQR outlier (value 200)."""
    values = [10.0, 11.0, 12.0, 11.5, 10.5, 12.0, 11.0, 10.0, 11.0, 200.0]
    return pd.DataFrame({"price": values})


# ---------------------------------------------------------------------------
# AC1 – Known outlier is flagged
# ---------------------------------------------------------------------------


class TestZscoreMethod:
    def test_known_outlier_is_flagged(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        flagged_indices = [r.row_index for r in report.flagged_rows]
        assert 19 in flagged_indices, "Row 19 (value=100) should be flagged"

    def test_non_outlier_rows_not_flagged(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        flagged_indices = [r.row_index for r in report.flagged_rows]
        for i in range(19):
            assert i not in flagged_indices, f"Row {i} should not be flagged"

    def test_flagged_record_has_correct_column_and_value(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        outlier = next(r for r in report.flagged_rows if r.row_index == 19)
        assert outlier.column == "score"
        assert outlier.value == pytest.approx(100.0)

    def test_z_score_is_populated(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        outlier = next(r for r in report.flagged_rows if r.row_index == 19)
        assert outlier.z_score is not None
        assert abs(outlier.z_score) > 3.0

    def test_custom_threshold(self):
        """A lower threshold should flag more rows."""
        df = pd.DataFrame({"x": [1.0, 2.0, 1.0, 2.0, 10.0]})
        tight = detect_anomalies(df, method="zscore", threshold=0.5)
        loose = detect_anomalies(df, method="zscore", threshold=3.0)
        assert tight.count >= loose.count


class TestIqrMethod:
    def test_known_outlier_is_flagged(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        flagged_indices = [r.row_index for r in report.flagged_rows]
        assert 9 in flagged_indices, "Row 9 (value=200) should be flagged"

    def test_non_outlier_rows_not_flagged(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        flagged_indices = [r.row_index for r in report.flagged_rows]
        for i in range(9):
            assert i not in flagged_indices, f"Row {i} should not be flagged"

    def test_z_score_is_none_for_iqr(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        for record in report.flagged_rows:
            assert record.z_score is None, "IQR records should have z_score=None"


# ---------------------------------------------------------------------------
# AC2 – Explanation names the column and states the statistical reason
# ---------------------------------------------------------------------------


class TestExplanations:
    def test_zscore_explanation_names_column(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        outlier = next(r for r in report.flagged_rows if r.row_index == 19)
        assert "score" in outlier.explanation

    def test_zscore_explanation_contains_std_deviation_phrase(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        outlier = next(r for r in report.flagged_rows if r.row_index == 19)
        assert "standard deviations" in outlier.explanation

    def test_zscore_explanation_contains_value(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        outlier = next(r for r in report.flagged_rows if r.row_index == 19)
        assert "100" in outlier.explanation

    def test_iqr_explanation_names_column(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        outlier = next(r for r in report.flagged_rows if r.row_index == 9)
        assert "price" in outlier.explanation

    def test_iqr_explanation_contains_range(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        outlier = next(r for r in report.flagged_rows if r.row_index == 9)
        assert "expected range" in outlier.explanation

    def test_iqr_explanation_contains_value(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        outlier = next(r for r in report.flagged_rows if r.row_index == 9)
        assert "200" in outlier.explanation


# ---------------------------------------------------------------------------
# AC3 – No numeric columns returns message
# ---------------------------------------------------------------------------


class TestNoNumericColumns:
    def test_message_is_set(self):
        df = pd.DataFrame({"name": ["alice", "bob"], "city": ["NY", "LA"]})
        report = detect_anomalies(df)

        assert report.message == _NO_NUMERIC_MESSAGE

    def test_flagged_rows_is_empty(self):
        df = pd.DataFrame({"name": ["alice", "bob"]})
        report = detect_anomalies(df)

        assert report.flagged_rows == []

    def test_count_is_zero(self):
        df = pd.DataFrame({"name": ["alice", "bob"]})
        report = detect_anomalies(df)

        assert report.count == 0


# ---------------------------------------------------------------------------
# AC4 – count == len(flagged_rows)
# ---------------------------------------------------------------------------


class TestCountConsistency:
    def test_count_equals_flagged_rows_length_zscore(self):
        df = _make_zscore_df()
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        assert report.count == len(report.flagged_rows)

    def test_count_equals_flagged_rows_length_iqr(self):
        df = _make_iqr_df()
        report = detect_anomalies(df, method="iqr")

        assert report.count == len(report.flagged_rows)

    def test_count_equals_flagged_rows_no_anomalies(self):
        """Uniform data should produce no anomalies; count still matches."""
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 1.0]})
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        assert report.count == len(report.flagged_rows)

    def test_count_equals_flagged_rows_no_numeric(self):
        df = pd.DataFrame({"tag": ["a", "b", "c"]})
        report = detect_anomalies(df)

        assert report.count == len(report.flagged_rows)

    def test_count_equals_flagged_rows_multiple_columns(self):
        """Multiple numeric columns — each may contribute flagged rows."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 100.0],
                "b": [5.0, 6.0, 5.5, 6.0, 5.0, 5.5, 6.0, 5.0, 5.5, -200.0],
            }
        )
        report = detect_anomalies(df, method="zscore", threshold=2.0)

        assert report.count == len(report.flagged_rows)


# ---------------------------------------------------------------------------
# Edge-cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_identical_values_no_flags(self):
        """Zero std deviation — nothing should be flagged."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]})
        report = detect_anomalies(df, method="zscore", threshold=3.0)

        assert report.count == 0

    def test_nan_values_ignored(self):
        """NaN entries should be skipped without raising errors."""
        # 8 values near 1.0, one NaN, one extreme outlier (100.0).
        # With threshold=2.0 the outlier's |z| exceeds the threshold.
        values = [1.0, 1.0, 1.0, 1.0, float("nan"), 1.0, 1.0, 1.0, 1.0, 100.0]
        df = pd.DataFrame({"x": values})
        report = detect_anomalies(df, method="zscore", threshold=2.0)

        # Row 9 (value 100) should still be flagged despite the NaN at row 4
        flagged_indices = [r.row_index for r in report.flagged_rows]
        assert 9 in flagged_indices

    def test_default_method_is_zscore(self):
        """Calling without method= should behave like method='zscore'."""
        df = _make_zscore_df()
        default_report = detect_anomalies(df)
        zscore_report = detect_anomalies(df, method="zscore")

        assert default_report.count == zscore_report.count

    def test_empty_dataframe_returns_empty_report(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        report = detect_anomalies(df)

        assert report.count == 0
        assert report.flagged_rows == []
