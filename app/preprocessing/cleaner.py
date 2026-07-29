"""
Automatic Data Cleaning and Feature Preprocessing Engine.
Provides modular transformers for missing value imputation, duplicate removal,
outlier detection/clipping, categorical encoding, feature scaling, and feature engineering.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from app.core.logger import logger


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Production-grade scikit-learn compatible transformer for dataset cleaning,
    imputation, outlier management, encoding, and scaling.
    """

    def __init__(
        self,
        impute_numeric: str = "median",
        impute_categorical: str = "most_frequent",
        drop_duplicates: bool = True,
        outlier_method: Optional[str] = "iqr",
        outlier_threshold: float = 1.5,
        encoding_method: str = "onehot",
        scaling_method: str = "standard",
    ):
        self.impute_numeric = impute_numeric
        self.impute_categorical = impute_categorical
        self.drop_duplicates = drop_duplicates
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method

        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.imputation_values_: Dict[str, Union[float, str]] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.category_mappings_: Dict[str, Dict[str, int]] = {}
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.ohe_encoder_: Optional[OneHotEncoder] = None

    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identifies numeric and categorical columns."""
        numeric_cols = df.select_dtypes(include=["number", "float64", "int64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return numeric_cols, categorical_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataCleaner":
        """
        Learns dataset statistics (imputation values, outlier boundaries, encoding dictionaries, scaling parameters).
        """
        df = X.copy()
        if self.drop_duplicates:
            df = df.drop_duplicates()

        self.numeric_cols_, self.categorical_cols_ = self._detect_column_types(df)

        # 1. Numeric Imputation Values
        for col in self.numeric_cols_:
            if self.impute_numeric == "mean":
                self.imputation_values_[col] = df[col].mean()
            elif self.impute_numeric == "median":
                self.imputation_values_[col] = df[col].median()
            elif self.impute_numeric == "zero":
                self.imputation_values_[col] = 0.0

        # 2. Categorical Imputation Values
        for col in self.categorical_cols_:
            if self.impute_categorical == "most_frequent":
                mode_val = df[col].mode(dropna=True)
                self.imputation_values_[col] = mode_val.iloc[0] if not mode_val.empty else "Missing"
            else:
                self.imputation_values_[col] = "Missing"

        # 3. Outlier Bound Calculations (IQR / Z-Score)
        if self.outlier_method == "iqr":
            for col in self.numeric_cols_:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - (self.outlier_threshold * iqr)
                upper = q3 + (self.outlier_threshold * iqr)
                self.outlier_bounds_[col] = (lower, upper)
        elif self.outlier_method == "zscore":
            for col in self.numeric_cols_:
                mean = df[col].mean()
                std = df[col].std(ddof=0)
                std = std if std > 1e-8 else 1.0
                lower = mean - (self.outlier_threshold * std)
                upper = mean + (self.outlier_threshold * std)
                self.outlier_bounds_[col] = (lower, upper)

        # 4. Categorical Encoding Fits
        if self.categorical_cols_:
            if self.encoding_method == "onehot":
                # Fill missing before fitting encoder
                filled_cat = df[self.categorical_cols_].fillna(
                    {c: self.imputation_values_.get(c, "Missing") for c in self.categorical_cols_}
                )
                self.ohe_encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                self.ohe_encoder_.fit(filled_cat.astype(str))
            elif self.encoding_method == "label":
                for col in self.categorical_cols_:
                    unique_vals = df[col].dropna().unique().tolist()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    mapping["<UNKNOWN>"] = -1
                    self.category_mappings_[col] = mapping

        # 5. Scaler Fit
        # Simulate preprocessing output shape for scaling fit
        df_imputed = df.copy()
        for c in self.numeric_cols_:
            df_imputed[c] = df_imputed[c].fillna(self.imputation_values_.get(c, 0.0))

        if self.scaling_method == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.scaling_method == "robust":
            self.scaler_ = RobustScaler()

        if self.scaler_ and self.numeric_cols_:
            self.scaler_.fit(df_imputed[self.numeric_cols_])

        logger.info(f"DataCleaner fitted on {len(df)} records ({len(self.numeric_cols_)} num, {len(self.categorical_cols_)} cat).")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned cleaning, imputation, outlier handling, encoding, and scaling to the input dataframe.
        """
        df = X.copy()

        if self.drop_duplicates:
            df = df.drop_duplicates()

        # 1. Missing Value Imputation
        for col, imp_val in self.imputation_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(imp_val)

        # 2. Outlier Management (Clipping)
        if self.outlier_method and self.outlier_bounds_:
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in df.columns:
                    df[col] = np.clip(df[col], lower, upper)

        # 3. Categorical Encoding Execution
        encoded_dfs: List[pd.DataFrame] = []
        if self.categorical_cols_:
            existing_cats = [c for c in self.categorical_cols_ if c in df.columns]

            if self.encoding_method == "onehot" and self.ohe_encoder_ and existing_cats:
                cat_data = df[existing_cats].astype(str)
                encoded_arr = self.ohe_encoder_.transform(cat_data)
                feature_names = self.ohe_encoder_.get_feature_names_out(existing_cats)
                ohe_df = pd.DataFrame(encoded_arr, columns=feature_names, index=df.index)
                df = df.drop(columns=existing_cats)
                encoded_dfs.append(ohe_df)

            elif self.encoding_method == "label" and existing_cats:
                for col in existing_cats:
                    mapping = self.category_mappings_.get(col, {})
                    df[col] = df[col].map(lambda x: mapping.get(x, mapping.get("<UNKNOWN>", -1)))

        # 4. Feature Scaling Execution
        existing_nums = [c for c in self.numeric_cols_ if c in df.columns]
        if self.scaler_ and existing_nums:
            scaled_arr = self.scaler_.transform(df[existing_nums])
            df[existing_nums] = pd.DataFrame(scaled_arr, columns=existing_nums, index=df.index)

        # Merge OHE encoded features if generated
        if encoded_dfs:
            df = pd.concat([df] + encoded_dfs, axis=1)

        logger.info(f"DataCleaner transformed dataframe to shape {df.shape}.")
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits parameters and transforms input data in a single step."""
        return self.fit(X, y).transform(X)
