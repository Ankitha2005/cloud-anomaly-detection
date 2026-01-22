"""
Baseline Anomaly Detection Models
Implements Isolation Forest, One-Class SVM, and Local Outlier Factor.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Any, Optional, Tuple
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BaseAnomalyDetector:
    """Base class for anomaly detection models."""
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.model = None
        self.fitted = False
        
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the model on training data."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal)."""
        raise NotImplementedError
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.fitted = True


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        random_state: int = 42
    ):
        super().__init__(contamination)
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit Isolation Forest on training data."""
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal."""
        predictions = self.model.predict(X)
        # sklearn returns -1 for anomaly, 1 for normal; convert to 1/0
        return (predictions == -1).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (negated so higher = more anomalous)."""
        return -self.model.score_samples(X)


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        nu: float = None
    ):
        super().__init__(contamination)
        # nu approximates the fraction of outliers
        if nu is None:
            nu = contamination
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu
        )
        
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit One-Class SVM on training data."""
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal."""
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (negated decision function)."""
        return -self.model.decision_function(X)


class LOFDetector(BaseAnomalyDetector):
    """Local Outlier Factor anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        metric: str = 'minkowski',
        novelty: bool = True
    ):
        super().__init__(contamination)
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            metric=metric,
            novelty=novelty,
            n_jobs=-1
        )
        
    def fit(self, X: np.ndarray) -> 'LOFDetector':
        """Fit LOF on training data."""
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal."""
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (negated LOF scores)."""
        return -self.model.score_samples(X)

