"""
Unit tests for Anomaly Detection Models.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_models import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from src.models.temporal_autoencoder import TemporalAnomalyDetector


@pytest.fixture
def normal_data():
    """Generate normal data for training."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    return np.random.randn(n_samples, n_features) * 0.5


@pytest.fixture
def anomaly_data():
    """Generate anomaly data for testing."""
    np.random.seed(43)
    n_samples = 100
    n_features = 20
    return np.random.randn(n_samples, n_features) * 2 + 3


@pytest.fixture
def sequence_data():
    """Generate sequence data for temporal models."""
    np.random.seed(42)
    n_samples = 500
    seq_len = 10
    n_features = 20
    return np.random.randn(n_samples, seq_len, n_features) * 0.5


class TestIsolationForest:
    """Test cases for Isolation Forest detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
        assert detector is not None
        assert detector.contamination == 0.05
    
    def test_fit(self, normal_data):
        """Test model fitting."""
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(normal_data)
        assert detector.fitted == True
    
    def test_predict_shape(self, normal_data, anomaly_data):
        """Test prediction output shape."""
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(normal_data)
        
        predictions = detector.predict(anomaly_data)
        assert predictions.shape == (len(anomaly_data),)
        assert set(predictions).issubset({0, 1})
    
    def test_score_samples(self, normal_data, anomaly_data):
        """Test anomaly scoring."""
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(normal_data)
        
        scores = detector.score_samples(anomaly_data)
        assert scores.shape == (len(anomaly_data),)
        assert not np.isnan(scores).any()
    
    def test_anomaly_detection(self, normal_data, anomaly_data):
        """Test that anomalies are detected."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(normal_data)
        
        # Anomaly data should have higher scores
        normal_scores = detector.score_samples(normal_data)
        anomaly_scores = detector.score_samples(anomaly_data)
        
        assert anomaly_scores.mean() > normal_scores.mean()


class TestOneClassSVM:
    """Test cases for One-Class SVM detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = OneClassSVMDetector(nu=0.05)
        assert detector is not None
    
    def test_fit_predict(self, normal_data, anomaly_data):
        """Test fit and predict."""
        detector = OneClassSVMDetector(nu=0.05)
        detector.fit(normal_data[:500])  # Smaller for speed
        
        predictions = detector.predict(anomaly_data)
        assert predictions.shape == (len(anomaly_data),)


class TestLOF:
    """Test cases for LOF detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = LOFDetector(contamination=0.05, n_neighbors=20)
        assert detector is not None
    
    def test_fit_predict(self, normal_data, anomaly_data):
        """Test fit and predict."""
        detector = LOFDetector(contamination=0.05, n_neighbors=20)
        detector.fit(normal_data)
        
        predictions = detector.predict(anomaly_data)
        assert predictions.shape == (len(anomaly_data),)


class TestTemporalAutoencoder:
    """Test cases for Temporal Autoencoder."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = TemporalAnomalyDetector(
            input_dim=20,
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            seq_len=10
        )
        assert detector is not None
        assert detector.fitted == False
    
    def test_fit(self, sequence_data):
        """Test model fitting."""
        detector = TemporalAnomalyDetector(
            input_dim=sequence_data.shape[2],
            hidden_dim=32,
            latent_dim=16,
            num_layers=1,
            seq_len=sequence_data.shape[1]
        )
        
        history = detector.fit(
            sequence_data[:400],
            X_val=sequence_data[400:],
            epochs=5,
            batch_size=32,
            verbose=False
        )
        
        assert detector.fitted == True
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_predict(self, sequence_data):
        """Test prediction."""
        detector = TemporalAnomalyDetector(
            input_dim=sequence_data.shape[2],
            hidden_dim=32,
            latent_dim=16,
            num_layers=1,
            seq_len=sequence_data.shape[1]
        )
        
        detector.fit(sequence_data[:400], epochs=3, verbose=False)
        predictions = detector.predict(sequence_data[400:])
        
        assert predictions.shape == (100,)
        assert set(predictions).issubset({0, 1})
    
    def test_score_samples(self, sequence_data):
        """Test anomaly scoring."""
        detector = TemporalAnomalyDetector(
            input_dim=sequence_data.shape[2],
            hidden_dim=32,
            latent_dim=16,
            num_layers=1,
            seq_len=sequence_data.shape[1]
        )
        
        detector.fit(sequence_data[:400], epochs=3, verbose=False)
        scores = detector.score_samples(sequence_data[400:])
        
        assert scores.shape == (100,)
        assert not np.isnan(scores).any()
        assert (scores >= 0).all()  # Reconstruction error is non-negative

