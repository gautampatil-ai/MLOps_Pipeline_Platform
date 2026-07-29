"""
Unit Tests for Data Preprocessing & Cleaning Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from app.preprocessing.cleaner import DataCleaner


def test_cleaner_missing_value_imputation(sample_classification_df):
    """Tests numeric and categorical missing value imputation."""
    cleaner = DataCleaner()
    cleaned_df = cleaner.fit_transform(sample_classification_df, target_column="target")

    # Assert no null values remain in feature columns
    assert cleaned_df["feature_num1"].isnull().sum() == 0
    assert cleaned_df["feature_cat"].isnull().sum() == 0


def test_cleaner_categorical_encoding(sample_classification_df):
    """Tests categorical feature encoding into numeric format."""
    cleaner = DataCleaner()
    cleaned_df = cleaner.fit_transform(sample_classification_df, target_column="target")

    # Assert categorical column was converted to numeric/dummified dtype
    assert not pd.api.types.is_object_dtype(cleaned_df["feature_cat"])
    assert pd.api.types.is_numeric_dtype(cleaned_df["feature_cat"])


def test_cleaner_outlier_clipping(sample_classification_df):
    """Tests IQR outlier detection and capping/clipping mechanism."""
    # Inject extreme outlier
    df = sample_classification_df.copy()
    df.loc[0, "feature_num2"] = 10000.0

    cleaner = DataCleaner(handle_outliers=True)
    cleaned_df = cleaner.fit_transform(df, target_column="target")

    # Extreme value should be clipped down to upper bound
    assert cleaned_df.loc[0, "feature_num2"] < 10000.0


def test_cleaner_transform_unseen_data(sample_classification_df):
    """Tests fitted cleaner transformation on new inference batch data."""
    cleaner = DataCleaner()
    train_cleaned = cleaner.fit_transform(sample_classification_df, target_column="target")

    # Create new unseen sample batch
    new_data = pd.DataFrame({
        "feature_num1": [1.5, np.nan],
        "feature_num2": [50.0, 75.0],
        "feature_cat": ["A", "B"],
    })

    transformed_df = cleaner.transform(new_data)
    assert transformed_df["feature_num1"].isnull().sum() == 0
    assert len(transformed_df) == 2
