"""Unit tests for feature engineering."""

import pandas as pd
import pytest
from src.features.engineering import (
    extract_time_features,
    hash_high_cardinality,
    add_interaction_features,
    build_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "hour": [14102100, 14102101, 14102114],
        "banner_pos": [0, 1, 0],
        "site_id": ["abc123", "def456", "abc123"],
        "site_domain": ["domain1", "domain2", "domain1"],
        "site_category": ["arts", "sports", "arts"],
        "app_id": ["app1", "app2", "app1"],
        "app_domain": ["appdomain1", "appdomain2", "appdomain1"],
        "app_category": ["games", "news", "games"],
        "device_id": ["dev1", "dev2", "dev1"],
        "device_ip": ["1.2.3.4", "5.6.7.8", "1.2.3.4"],
        "device_model": ["modelA", "modelB", "modelA"],
        "device_type": [1, 0, 1],
        "device_make": ["makeA", "makeB", "makeA"],
        "C1": [1005, 1005, 1005],
        "C14": [21689, 21689, 21689],
        "C15": [320, 320, 320],
        "C16": [50, 50, 50],
        "C17": [112, 112, 112],
        "C18": [0, 0, 0],
        "C19": [35, 35, 35],
        "C20": [-1, -1, -1],
        "C21": [79, 79, 79],
        "click": [0, 1, 0],
    })


def test_extract_time_features(sample_df):
    result = extract_time_features(sample_df)
    assert "hour_of_day" in result.columns
    assert "day_of_week" in result.columns
    assert result["hour_of_day"].iloc[0] == 0   # last 2 digits of 14102100
    assert result["hour_of_day"].iloc[1] == 1   # last 2 digits of 14102101
    assert result["hour_of_day"].iloc[2] == 14  # last 2 digits of 14102114


def test_hash_high_cardinality_is_deterministic(sample_df):
    result1 = hash_high_cardinality(sample_df)
    result2 = hash_high_cardinality(sample_df)
    assert result1["site_id"].tolist() == result2["site_id"].tolist()


def test_hash_high_cardinality_bounded(sample_df):
    result = hash_high_cardinality(sample_df)
    assert result["site_id"].between(0, 2**18 - 1).all()


def test_add_interaction_features(sample_df):
    df = hash_high_cardinality(sample_df)
    result = add_interaction_features(df)
    assert "site_x_device" in result.columns
    assert "app_x_banner" in result.columns


def test_build_features_returns_encoders(sample_df):
    df, encoders = build_features(sample_df)
    assert isinstance(encoders, dict)
    assert len(encoders) > 0


def test_build_features_encoder_reuse(sample_df):
    df1, encoders = build_features(sample_df)
    df2, encoders2 = build_features(sample_df, encoders=encoders)
    assert encoders2 is encoders  # same object passed through
