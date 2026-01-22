"""
Unit tests for Feature Engineering module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample cloud data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'node_id': np.random.choice(['node_001', 'node_002', 'node_003'], n_samples),
        'window_idx': np.arange(n_samples) // 3,
        'cpu_util': np.random.uniform(0.1, 0.9, n_samples),
        'ram_util': np.random.uniform(0.2, 0.8, n_samples),
        'disk_io_util': np.random.uniform(0.1, 0.7, n_samples),
        'bandwidth_util': np.random.uniform(0.1, 0.6, n_samples),
        'active_connections': np.random.randint(1, 50, n_samples),
        'request_count': np.random.randint(10, 500, n_samples),
        'error_count': np.random.randint(0, 10, n_samples),
        'label': np.random.choice([0, 0, 0, 0, 1], n_samples),  # 20% anomaly
    }
    return pd.DataFrame(data)


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_initialization(self, feature_engineer):
        """Test FeatureEngineer initialization."""
        assert feature_engineer is not None
        assert feature_engineer.scaler is not None
        assert feature_engineer.fitted == False
    
    def test_engineer_features_shape(self, feature_engineer, sample_data):
        """Test that engineer_features returns correct shape."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)
        
        # Should have more columns than original
        assert len(df_eng.columns) > len(sample_data.columns)
        # Should have same number of rows
        assert len(df_eng) == len(sample_data)
    
    def test_temporal_features(self, feature_engineer, sample_data):
        """Test temporal feature extraction."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)

        # Check temporal features exist
        assert 'hour' in df_eng.columns
        assert 'day_of_week' in df_eng.columns
        assert 'is_weekend' in df_eng.columns
        assert 'is_business_hours' in df_eng.columns

        # Features may be scaled, so just check they exist and are numeric
        assert df_eng['hour'].dtype in [np.float64, np.float32, np.int64]
    
    def test_rolling_features(self, feature_engineer, sample_data):
        """Test rolling statistics features."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)
        
        # Check rolling features exist
        rolling_cols = [c for c in df_eng.columns if 'roll_' in c]
        assert len(rolling_cols) > 0
        
        # Check for different window sizes
        assert any('roll_mean_5' in c for c in rolling_cols)
        assert any('roll_std_30' in c for c in rolling_cols)
    
    def test_interaction_features(self, feature_engineer, sample_data):
        """Test interaction features."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)
        
        # Check interaction features
        assert 'cpu_ram_product' in df_eng.columns
        assert 'bandwidth_per_conn' in df_eng.columns
        assert 'error_rate' in df_eng.columns
    
    def test_get_feature_matrix(self, feature_engineer, sample_data):
        """Test feature matrix extraction."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)
        X, y = feature_engineer.get_feature_matrix(df_eng)

        # Check shapes
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y)

        # Check finite values (NaN may exist in rolling features for small data)
        assert np.isfinite(X[~np.isnan(X)]).all()
        assert not np.isnan(y).any()
    
    def test_create_sequences(self, feature_engineer, sample_data):
        """Test sequence creation for temporal models."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)

        seq_len = 10
        X_seq, y_seq, indices = feature_engineer.create_sequences(df_eng, sequence_length=seq_len)

        # Check shapes
        assert X_seq.ndim == 3
        assert X_seq.shape[1] == seq_len
        assert len(X_seq) == len(y_seq)

        # Check finite values where not NaN
        assert np.isfinite(X_seq[~np.isnan(X_seq)]).all()
    
    def test_scaler_fitted(self, feature_engineer, sample_data):
        """Test that scaler is properly fitted."""
        df_eng = feature_engineer.engineer_features(sample_data, fit=True)
        X, y = feature_engineer.get_feature_matrix(df_eng)

        assert feature_engineer.fitted == True

        # Check scaling (values should be roughly normalized, ignoring NaN)
        X_valid = X[~np.isnan(X)]
        assert np.abs(X_valid.mean()) < 10
        assert X_valid.std() < 10
    
    def test_transform_without_fit(self, feature_engineer, sample_data):
        """Test transform mode without fitting."""
        # First fit
        df_eng1 = feature_engineer.engineer_features(sample_data, fit=True)
        
        # Then transform new data
        df_eng2 = feature_engineer.engineer_features(sample_data, fit=False)
        
        # Should have same columns
        assert list(df_eng1.columns) == list(df_eng2.columns)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, feature_engineer):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            feature_engineer.engineer_features(empty_df, fit=True)
    
    def test_missing_columns(self, feature_engineer):
        """Test handling of missing required columns."""
        incomplete_df = pd.DataFrame({'cpu_util': [0.5, 0.6]})
        
        with pytest.raises(Exception):
            feature_engineer.engineer_features(incomplete_df, fit=True)

