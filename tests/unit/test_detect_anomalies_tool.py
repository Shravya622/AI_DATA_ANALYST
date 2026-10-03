"""
Unit tests for the detect_anomalies_tool and its registry registration.

Requirements covered: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.detect_anomalies import detect_anomalies_tool
from app.tools.registry import TOOLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """DataFrame with no outliers (all values within 1 std of mean)."""
    return pd.DataFrame({"value": [10.0, 11.0, 10.5, 9.8, 10.2]})


@pytest.fixture()
def outlier_df() -> pd.DataFrame:
    """DataFrame with a tight cluster of ~10s and one obvious outlier (100).

    With 20 values near 10 the outlier of 100 achieves z ≈ 4.36, comfortably
    above the default threshold of 3.0, making it reliably detected.
    """
    data = [10.0] * 20 + [100.0]
    return pd.DataFrame({"value": data})


@pytest.fixture()
def no_numeric_df() -> pd.DataFrame:
    """DataFrame with only string columns — no anomalies can be computed."""
    return pd.DataFrame({"category": ["a", "b", "c"]})


# ---------------------------------------------------------------------------
# Return structure tests
# ---------------------------------------------------------------------------


class TestDetectAnomaliesToolReturnStructure:
    def test_returns_dict(self, clean_df):
        result = detect_anomalies_tool(clean_df)
        assert isinstance(result, dict)

    def test_has_count_key(self, clean_df):
        result = detect_anomalies_tool(clean_df)
        assert "count" in result

    def test_has_flagged_rows_key(self, clean_df):
        result = detect_anomalies_tool(clean_df)
        assert "flagged_rows" in result

    def test_count_equals_len_flagged_rows(self, outlier_df):
        result = detect_anomalies_tool(outlier_df)
        assert result["count"] == len(result["flagged_rows"])


# ---------------------------------------------------------------------------
# Anomaly detection correctness
# ---------------------------------------------------------------------------


class TestDetectAnomaliesToolBehavior:
    def test_detects_outlier_zscore(self, outlier_df):
        """The value 100 should be flagged as a Z-score outlier (z ≈ 4.36)."""
        result = detect_anomalies_tool(outlier_df, method="zscore", threshold=3.0)
        assert result["count"] >= 1
        flagged_values = [r["value"] for r in result["flagged_rows"]]
        assert 100.0 in flagged_values

    def test_detects_outlier_iqr(self, outlier_df):
        """The value 100 should be flagged using the IQR method."""
        result = detect_anomalies_tool(outlier_df, method="iqr")
        assert result["count"] >= 1
        flagged_values = [r["value"] for r in result["flagged_rows"]]
        assert 100.0 in flagged_values

    def test_no_outliers_in_clean_data(self, clean_df):
        """A uniform dataset should produce no flagged rows."""
        result = detect_anomalies_tool(clean_df, method="zscore", threshold=3.0)
        assert result["count"] == 0
        assert result["flagged_rows"] == []

    def test_no_numeric_columns_returns_message(self, no_numeric_df):
        """A DataFrame with no numeric columns should return count=0 and a message."""
        result = detect_anomalies_tool(no_numeric_df)
        assert result["count"] == 0
        assert result["flagged_rows"] == []
        assert result.get("message") is not None
        assert len(result["message"]) > 0

    def test_flagged_row_has_required_keys(self, outlier_df):
        """Each flagged row record must have the expected AnomalyRecord fields."""
        result = detect_anomalies_tool(outlier_df)
        assert result["count"] >= 1
        row = result["flagged_rows"][0]
        assert "row_index" in row
        assert "column" in row
        assert "value" in row
        assert "explanation" in row

    def test_custom_threshold_affects_results(self, outlier_df):
        """A very high threshold should flag nothing; a low one should flag more."""
        high_threshold = detect_anomalies_tool(outlier_df, method="zscore", threshold=100.0)
        low_threshold = detect_anomalies_tool(outlier_df, method="zscore", threshold=0.5)
        assert high_threshold["count"] < low_threshold["count"]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestDetectAnomaliesToolRegistry:
    def test_registered_in_tools(self):
        """detect_anomalies must be a key in the TOOLS registry."""
        assert "detect_anomalies" in TOOLS

    def test_registry_entry_has_json_schema(self):
        """The registry entry must carry a valid OpenAI-format JSON schema."""
        entry = TOOLS["detect_anomalies"]
        schema = entry.json_schema
        assert schema.get("type") == "function"
        assert "function" in schema
        fn = schema["function"]
        assert fn.get("name") == "detect_anomalies"
        assert "parameters" in fn

    def test_schema_method_enum(self):
        """The 'method' parameter schema must list 'zscore' and 'iqr' as enum values."""
        schema = TOOLS["detect_anomalies"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "method" in props
        assert set(props["method"]["enum"]) == {"zscore", "iqr"}

    def test_schema_threshold_is_number(self):
        """The 'threshold' parameter must be typed as 'number' in the schema."""
        schema = TOOLS["detect_anomalies"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "threshold" in props
        assert props["threshold"]["type"] == "number"

    def test_registry_callable_is_detect_anomalies_tool(self):
        """The callable stored in the registry must be detect_anomalies_tool."""
        assert TOOLS["detect_anomalies"].callable is detect_anomalies_tool
