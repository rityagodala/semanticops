"""
Semantic-aware feature extractor.

Uses the SchemaInterpretation to build features that understand
the physical meaning of each column — not just its raw values.

Key techniques:
  - Lag features for time-series dependencies
  - Rolling statistics (mean, std, min, max)
  - Cyclic encoding (sin/cos) for cyclic columns (hour, day-of-week)
  - Interaction features for semantically related columns
  - Missing value strategies informed by column role
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from semanticops.schema_agent import SchemaInterpretation, ColumnRole


@dataclass
class FeatureSet:
    """Result of feature extraction."""

    X: pd.DataFrame          # feature matrix
    y: pd.Series             # target series
    feature_names: list[str]
    target_name: str
    n_samples: int
    n_features: int
    extraction_log: list[str]


class FeatureExtractor:
    """
    Builds a feature matrix from a raw DataFrame using schema interpretation.

    Example:
        schema = agent.interpret(df.columns.tolist())
        extractor = FeatureExtractor(schema, lag_windows=[1, 3, 6, 24])
        feature_set = extractor.extract(df)
        print(feature_set.X.shape)
    """

    def __init__(
        self,
        schema: SchemaInterpretation,
        lag_windows: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        self.schema = schema
        self.lag_windows = lag_windows or [1, 3, 6]
        self.rolling_windows = rolling_windows or [6, 24]
        self._log: list[str] = []

    def extract(self, df: pd.DataFrame) -> FeatureSet:
        """
        Extract features from a raw DataFrame.

        Args:
            df: Raw dataframe with columns matching schema
        Returns:
            FeatureSet ready for model training
        """
        self._log = []
        df = df.copy()

        # Parse timestamp if available
        if self.schema.timestamp_column and self.schema.timestamp_column in df.columns:
            df[self.schema.timestamp_column] = pd.to_datetime(
                df[self.schema.timestamp_column], errors="coerce"
            )
            df = df.set_index(self.schema.timestamp_column).sort_index()
            self._log.append(f"Set {self.schema.timestamp_column} as DatetimeIndex")

        # Drop identifier columns
        drop_cols = [c for c in self.schema.drop_columns if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            self._log.append(f"Dropped identifier columns: {drop_cols}")

        # Separate target
        target_col = self.schema.target_column
        if target_col is None or target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame. "
                f"Available: {list(df.columns)}"
            )
        y_raw = df[target_col].copy()
        feature_df = df.drop(columns=[target_col])

        # Build features
        feature_frames: list[pd.DataFrame] = [feature_df.copy()]

        # Lag features for all numeric columns
        numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
        for lag in self.lag_windows:
            lagged = feature_df[numeric_cols].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in numeric_cols]
            feature_frames.append(lagged)
        self._log.append(f"Added lag features: {self.lag_windows} for {len(numeric_cols)} cols")

        # Rolling statistics
        for window in self.rolling_windows:
            roll = feature_df[numeric_cols].rolling(window, min_periods=1)
            roll_mean = roll.mean()
            roll_mean.columns = [f"{c}_roll{window}_mean" for c in numeric_cols]
            roll_std = roll.std().fillna(0)
            roll_std.columns = [f"{c}_roll{window}_std" for c in numeric_cols]
            feature_frames.extend([roll_mean, roll_std])
        self._log.append(f"Added rolling features: windows {self.rolling_windows}")

        # Cyclic encoding for identified cyclic columns
        cyclic_cols = [c for c in self.schema.columns if c.is_cyclic and c.name in feature_df.columns]
        for col_meta in cyclic_cols:
            col = col_meta.name
            max_val = feature_df[col].max()
            if max_val > 0:
                sin_df = pd.DataFrame(
                    {f"{col}_sin": np.sin(2 * np.pi * feature_df[col] / max_val)}
                )
                cos_df = pd.DataFrame(
                    {f"{col}_cos": np.cos(2 * np.pi * feature_df[col] / max_val)}
                )
                feature_frames.extend([sin_df, cos_df])
                self._log.append(f"Added cyclic encoding for {col}")

        # If DatetimeIndex available, extract time features
        if isinstance(df.index, pd.DatetimeIndex):
            time_features = pd.DataFrame({
                "hour_sin": np.sin(2 * np.pi * df.index.hour / 24),
                "hour_cos": np.cos(2 * np.pi * df.index.hour / 24),
                "day_of_week_sin": np.sin(2 * np.pi * df.index.dayofweek / 7),
                "day_of_week_cos": np.cos(2 * np.pi * df.index.dayofweek / 7),
            }, index=df.index)
            feature_frames.append(time_features)
            self._log.append("Added datetime cyclic features (hour, day_of_week)")

        # Concatenate all feature frames
        X = pd.concat(feature_frames, axis=1)
        X = X.dropna()
        y = y_raw.loc[X.index]

        # Remove any remaining NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        self._log.append(
            f"Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features"
        )

        return FeatureSet(
            X=X,
            y=y,
            feature_names=list(X.columns),
            target_name=target_col,
            n_samples=len(X),
            n_features=X.shape[1],
            extraction_log=list(self._log),
        )
