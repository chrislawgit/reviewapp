"""
Data Preprocessing Pipeline for Polymarket 15-Minute Market Data
Prepares collected data for machine learning analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sqlite3
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketDataPreprocessor:
    """Preprocesses Polymarket price data for ML model training"""

    def __init__(self, db_path: str = "polymarket_data.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load price data from database

        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            DataFrame with price data
        """
        query = """
            SELECT
                timestamp,
                market_id,
                token_id,
                price,
                bid,
                ask,
                spread,
                volume,
                window_start,
                window_end
            FROM price_data
            WHERE 1=1
        """

        params = []
        if start_date:
            query += " AND datetime(timestamp, 'unixepoch') >= ?"
            params.append(start_date)

        if end_date:
            query += " AND datetime(timestamp, 'unixepoch') <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp, market_id, token_id"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        logger.info(f"Loaded {len(df)} price records")
        return df

    def create_15min_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Organize data into 15-minute trading windows

        Args:
            df: Raw price data

        Returns:
            DataFrame with window assignments
        """
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Create 15-minute window identifier
        df['window_id'] = df['datetime'].dt.floor('15min')

        # Calculate time within window (0-900 seconds)
        df['time_in_window'] = (df['datetime'] - df['window_id']).dt.total_seconds()

        logger.info(f"Created {df['window_id'].nunique()} unique 15-minute windows")
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading features for ML model

        Features include:
        - Price momentum and velocity
        - Price volatility
        - Bid-ask spread metrics
        - Volume patterns
        - Time-based features
        """
        df = df.copy()

        # Sort by market, token, and time
        df = df.sort_values(['market_id', 'token_id', 'timestamp'])

        # Group by market and token for feature calculation
        for group_name, group_df in df.groupby(['market_id', 'token_id']):
            idx = group_df.index

            # Price change features
            df.loc[idx, 'price_change'] = group_df['price'].diff()
            df.loc[idx, 'price_change_pct'] = group_df['price'].pct_change()

            # Rolling statistics (over 60 seconds = 12 observations at 5-sec intervals)
            df.loc[idx, 'price_mean_60s'] = group_df['price'].rolling(window=12, min_periods=1).mean()
            df.loc[idx, 'price_std_60s'] = group_df['price'].rolling(window=12, min_periods=1).std()
            df.loc[idx, 'price_min_60s'] = group_df['price'].rolling(window=12, min_periods=1).min()
            df.loc[idx, 'price_max_60s'] = group_df['price'].rolling(window=12, min_periods=1).max()

            # Momentum indicators
            df.loc[idx, 'momentum_30s'] = group_df['price'].diff(6)  # 30 seconds = 6 * 5sec
            df.loc[idx, 'momentum_60s'] = group_df['price'].diff(12)  # 60 seconds = 12 * 5sec

            # Velocity (rate of price change)
            df.loc[idx, 'velocity'] = df.loc[idx, 'price_change'] / 5.0  # per second

            # Acceleration (rate of velocity change)
            df.loc[idx, 'acceleration'] = df.loc[idx, 'velocity'].diff()

            # Spread metrics (if available)
            if 'spread' in df.columns:
                df.loc[idx, 'spread_mean_60s'] = group_df['spread'].rolling(window=12, min_periods=1).mean()
                df.loc[idx, 'spread_change'] = group_df['spread'].diff()

        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Cyclical time features (for neural networks)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        logger.info(f"Calculated {df.shape[1]} features")
        return df

    def create_window_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregate features for each 15-minute window

        Args:
            df: DataFrame with window assignments

        Returns:
            DataFrame with one row per window containing aggregate metrics
        """
        window_aggs = df.groupby(['window_id', 'market_id', 'token_id']).agg({
            'price': ['first', 'last', 'min', 'max', 'mean', 'std'],
            'price_change': ['sum', 'mean', 'std'],
            'velocity': ['mean', 'std', 'min', 'max'],
            'spread': ['mean', 'std', 'min', 'max'],
            'volume': ['sum', 'mean'],
            'timestamp': 'count'
        }).reset_index()

        # Flatten column names
        window_aggs.columns = ['_'.join(col).strip('_') for col in window_aggs.columns]

        # Calculate additional window-level metrics
        window_aggs['price_range'] = window_aggs['price_max'] - window_aggs['price_min']
        window_aggs['price_change_total'] = window_aggs['price_last'] - window_aggs['price_first']
        window_aggs['price_change_pct_total'] = (
            (window_aggs['price_last'] - window_aggs['price_first']) / window_aggs['price_first']
        )

        # Determine outcome (up or down)
        window_aggs['outcome'] = (window_aggs['price_change_total'] > 0).astype(int)

        logger.info(f"Created {len(window_aggs)} window aggregates")
        return window_aggs

    def detect_anomalies_statistical(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods (Z-score)

        Args:
            df: DataFrame with features
            threshold: Z-score threshold for anomaly detection

        Returns:
            DataFrame with anomaly flags
        """
        df = df.copy()

        # Calculate Z-scores for key metrics
        numeric_cols = ['price_change_pct', 'velocity', 'acceleration', 'spread']
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                df[f'{col}_zscore'] = (df[col] - mean) / std
                df[f'{col}_anomaly'] = (np.abs(df[f'{col}_zscore']) > threshold).astype(int)

        # Overall anomaly flag (if any metric is anomalous)
        anomaly_cols = [col for col in df.columns if col.endswith('_anomaly')]
        df['is_anomaly'] = df[anomaly_cols].max(axis=1)

        anomaly_count = df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")

        return df

    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 36) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences for LSTM/RNN models

        Args:
            df: DataFrame with features
            sequence_length: Number of time steps per sequence (36 = 3 minutes at 5-sec intervals)

        Returns:
            Tuple of (X, y) numpy arrays
        """
        sequences = []
        targets = []

        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'market_id', 'token_id', 'datetime', 'window_id', 'is_anomaly'
        ]]

        # Group by market and token
        for (market_id, token_id), group_df in df.groupby(['market_id', 'token_id']):
            group_df = group_df.sort_values('timestamp')
            values = group_df[feature_cols].values

            # Create sequences
            for i in range(len(values) - sequence_length):
                seq = values[i:i + sequence_length]
                target = group_df.iloc[i + sequence_length]['price_change_pct']

                sequences.append(seq)
                targets.append(target)

        X = np.array(sequences)
        y = np.array(targets)

        logger.info(f"Created {len(sequences)} sequences of length {sequence_length}")
        return X, y

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler

        Args:
            df: DataFrame with features
            fit: Whether to fit the scaler (True for training data)

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()

        # Select numerical columns to normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude certain columns
        exclude_cols = ['timestamp', 'is_anomaly', 'outcome', 'hour', 'day_of_week', 'is_weekend']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols].fillna(0))
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols].fillna(0))

        return df

    def prepare_ml_dataset(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Complete preprocessing pipeline

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary containing processed data and metadata
        """
        # Load data
        df = self.load_data(start_date, end_date)

        if len(df) == 0:
            logger.warning("No data loaded")
            return None

        # Create windows
        df = self.create_15min_windows(df)

        # Calculate features
        df = self.calculate_features(df)

        # Detect anomalies
        df = self.detect_anomalies_statistical(df)

        # Create window aggregates
        window_df = self.create_window_aggregates(df)

        # Normalize
        df_normalized = self.normalize_features(df.copy())

        # Create sequences
        X, y = self.create_sequences(df_normalized)

        return {
            'raw_data': df,
            'window_aggregates': window_df,
            'sequences_X': X,
            'sequences_y': y,
            'feature_columns': df.columns.tolist(),
            'scaler': self.scaler
        }

    def export_for_ml(self, output_dir: str = "ml_data"):
        """
        Export processed data for ML training

        Args:
            output_dir: Directory to save processed data
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        dataset = self.prepare_ml_dataset()

        if dataset is None:
            return

        # Save dataframes
        dataset['raw_data'].to_csv(f"{output_dir}/processed_data.csv", index=False)
        dataset['window_aggregates'].to_csv(f"{output_dir}/window_aggregates.csv", index=False)

        # Save sequences as numpy arrays
        np.save(f"{output_dir}/sequences_X.npy", dataset['sequences_X'])
        np.save(f"{output_dir}/sequences_y.npy", dataset['sequences_y'])

        # Save metadata
        import pickle
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(dataset['scaler'], f)

        with open(f"{output_dir}/feature_columns.txt", 'w') as f:
            f.write('\n'.join(dataset['feature_columns']))

        logger.info(f"Exported ML dataset to {output_dir}/")


if __name__ == "__main__":
    preprocessor = PolymarketDataPreprocessor()
    preprocessor.export_for_ml()
