"""
Feature Engineering Module for Cloud Anomaly Detection
Builds unified node behavior representations with temporal features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class FeatureEngineer:
    """Feature engineering for cloud anomaly detection."""
    
    # Core resource utilization features
    RESOURCE_FEATURES = [
        'cpu_util', 'ram_util', 'bandwidth_util', 'disk_io_util'
    ]
    
    # Network behavior features
    NETWORK_FEATURES = [
        'active_connections', 'request_count', 'error_count'
    ]
    
    # Temporal window sizes for rolling statistics
    WINDOW_SIZES = [5, 10, 30]  # 5min, 10min, 30min windows
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize feature engineer.
        
        Args:
            scaler_type: 'standard' or 'minmax'
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_columns: List[str] = []
        self.fitted = False
        
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features based on timestamp.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Cyclical encoding for hour (captures 23:00 -> 00:00 continuity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics per node.
        
        Args:
            df: DataFrame sorted by timestamp
            
        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()
        df = df.sort_values(['node_id', 'timestamp'])
        
        for feature in self.RESOURCE_FEATURES:
            if feature not in df.columns:
                continue
                
            for window in self.WINDOW_SIZES:
                # Rolling mean
                df[f'{feature}_roll_mean_{window}'] = df.groupby('node_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Rolling std
                df[f'{feature}_roll_std_{window}'] = df.groupby('node_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                )
                # Rolling max
                df[f'{feature}_roll_max_{window}'] = df.groupby('node_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
        
        return df
    
    def add_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features measuring deviation from node's typical behavior.
        
        Args:
            df: DataFrame with rolling statistics
            
        Returns:
            DataFrame with deviation features
        """
        df = df.copy()
        
        for feature in self.RESOURCE_FEATURES:
            if feature not in df.columns:
                continue
            
            # Deviation from rolling mean (z-score like)
            roll_mean_col = f'{feature}_roll_mean_10'
            roll_std_col = f'{feature}_roll_std_10'
            
            if roll_mean_col in df.columns and roll_std_col in df.columns:
                df[f'{feature}_deviation'] = (
                    (df[feature] - df[roll_mean_col]) / 
                    (df[roll_std_col] + 1e-8)
                )
        
        return df
    
    def add_cross_node_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features comparing node to cluster-wide behavior.
        
        Args:
            df: DataFrame with node data
            
        Returns:
            DataFrame with cross-node features
        """
        df = df.copy()
        
        # Calculate cluster-wide statistics per timestamp
        for feature in self.RESOURCE_FEATURES:
            if feature not in df.columns:
                continue
            
            # Cluster mean and std at each timestamp
            cluster_stats = df.groupby('timestamp')[feature].agg(['mean', 'std']).reset_index()
            cluster_stats.columns = ['timestamp', f'{feature}_cluster_mean', f'{feature}_cluster_std']
            
            df = df.merge(cluster_stats, on='timestamp', how='left')
            
            # Node's deviation from cluster
            df[f'{feature}_cluster_deviation'] = (
                (df[feature] - df[f'{feature}_cluster_mean']) /
                (df[f'{feature}_cluster_std'] + 1e-8)
            )

        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between resource metrics.

        Args:
            df: DataFrame with resource features

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # CPU-RAM interaction (high both = potential issue)
        if 'cpu_util' in df.columns and 'ram_util' in df.columns:
            df['cpu_ram_product'] = df['cpu_util'] * df['ram_util']
            df['cpu_ram_ratio'] = df['cpu_util'] / (df['ram_util'] + 1e-8)

        # Bandwidth-connections ratio (bytes per connection)
        if 'bandwidth_util' in df.columns and 'active_connections' in df.columns:
            df['bandwidth_per_conn'] = df['bandwidth_util'] / (df['active_connections'] + 1)

        # Error rate
        if 'error_count' in df.columns and 'request_count' in df.columns:
            df['error_rate'] = df['error_count'] / (df['request_count'] + 1)

        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        include_temporal: bool = True,
        include_rolling: bool = True,
        include_deviation: bool = True,
        include_cross_node: bool = True,
        include_interaction: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Raw DataFrame
            fit: Whether to fit the scaler
            include_*: Flags to include specific feature types

        Returns:
            Engineered DataFrame
        """
        df = df.copy()

        if include_temporal:
            df = self.add_temporal_features(df)

        if include_rolling:
            df = self.add_rolling_statistics(df)

        if include_deviation:
            df = self.add_deviation_features(df)

        if include_cross_node:
            df = self.add_cross_node_features(df)

        if include_interaction:
            df = self.add_interaction_features(df)

        # Identify numeric feature columns for scaling
        exclude_cols = ['timestamp', 'node_id', 'node_type', 'label', 'attack_type', 'window_idx']
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
        ]

        # Scale features
        if fit:
            df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature matrix and labels.

        Args:
            df: Engineered DataFrame
            feature_cols: Specific columns to use (default: all feature columns)

        Returns:
            Tuple of (X, y) arrays
        """
        if feature_cols is None:
            feature_cols = self.feature_columns

        X = df[feature_cols].values
        y = df['label'].values if 'label' in df.columns else np.zeros(len(df))

        return X, y

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 10,
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for temporal models (LSTM, etc.).

        Args:
            df: Engineered DataFrame sorted by node_id and timestamp
            sequence_length: Length of each sequence
            feature_cols: Features to include

        Returns:
            Tuple of (X_sequences, y_labels, node_ids)
        """
        if feature_cols is None:
            feature_cols = self.feature_columns

        sequences = []
        labels = []
        node_ids = []

        for node_id in df['node_id'].unique():
            node_data = df[df['node_id'] == node_id].sort_values('timestamp')

            if len(node_data) < sequence_length:
                continue

            X_node = node_data[feature_cols].values
            y_node = node_data['label'].values if 'label' in node_data.columns else np.zeros(len(node_data))

            for i in range(len(node_data) - sequence_length + 1):
                sequences.append(X_node[i:i + sequence_length])
                labels.append(y_node[i + sequence_length - 1])  # Label of last timestep
                node_ids.append(node_id)

        return np.array(sequences), np.array(labels), np.array(node_ids)

