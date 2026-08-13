import pandas as pd
import pytest

from data_designer_retrieval_sdg.convert import _compute_split, SUPPORTED_SPLIT_STRATEGIES


def test_compute_split_rejects_unsupported_strategy():
    df = pd.DataFrame({"file_name": []})
    with pytest.raises(ValueError, match="Unsupported split_strategy"):
        _compute_split(df, 0.8, 0.1, 42, "not_a_strategy", None)


def test_supported_strategies_contains_expected_values():
    assert SUPPORTED_SPLIT_STRATEGIES == {"random", "dedupped", "cluster"}
