"""
Ensemble Anomaly Detector
Combines multiple anomaly detection models with weighted voting.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baseline_models import (
    IsolationForestDetector,
    OneClassSVMDetector,
    LOFDetector
)


class EnsembleAnomalyDetector:
    """
    Ensemble detector combining multiple anomaly detection methods.
    
    Supports:
    - Weighted average of anomaly scores
    - Majority voting on predictions
    - Stacking with meta-learner
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        use_isolation_forest: bool = True,
        use_ocsvm: bool = True,
        use_lof: bool = True,
        use_autoencoder: bool = True,
        weights: Dict[str, float] = None
    ):
        """
        Initialize ensemble detector.
        
        Args:
            contamination: Expected anomaly ratio
            use_*: Flags to include specific detectors
            weights: Custom weights for each detector
        """
        self.contamination = contamination
        self.detectors = {}
        self.weights = weights or {}
        self.fitted = False
        
        # Initialize detectors
        if use_isolation_forest:
            self.detectors['isolation_forest'] = IsolationForestDetector(
                contamination=contamination,
                n_estimators=100
            )
            self.weights.setdefault('isolation_forest', 1.0)
            
        if use_ocsvm:
            self.detectors['ocsvm'] = OneClassSVMDetector(
                contamination=contamination
            )
            self.weights.setdefault('ocsvm', 0.8)
            
        if use_lof:
            self.detectors['lof'] = LOFDetector(
                contamination=contamination,
                n_neighbors=20
            )
            self.weights.setdefault('lof', 0.6)
        
        self.use_autoencoder = use_autoencoder
        self.autoencoder = None
        self.weights.setdefault('autoencoder', 1.2)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.threshold = None
        
    def fit(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None,
        ocsvm_subset_size: int = 10000
    ) -> 'EnsembleAnomalyDetector':
        """
        Fit all detectors on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            X_sequences: Sequence data for autoencoder (n_samples, seq_len, n_features)
            ocsvm_subset_size: Subset size for OCSVM (computational efficiency)
        """
        print("Training ensemble detectors...")
        
        for name, detector in self.detectors.items():
            print(f"  Training {name}...")
            if name == 'ocsvm' and len(X) > ocsvm_subset_size:
                # Use subset for OCSVM
                subset_idx = np.random.choice(len(X), ocsvm_subset_size, replace=False)
                detector.fit(X[subset_idx])
            else:
                detector.fit(X)
        
        # Train autoencoder if enabled and sequences provided
        if self.use_autoencoder and X_sequences is not None:
            print("  Training autoencoder...")
            from src.models.temporal_autoencoder import TemporalAnomalyDetector
            
            self.autoencoder = TemporalAnomalyDetector(
                input_dim=X_sequences.shape[2],
                hidden_dim=64,
                latent_dim=32,
                num_layers=2,
                seq_len=X_sequences.shape[1]
            )
            self.autoencoder.fit(
                X_sequences,
                epochs=30,
                batch_size=64,
                verbose=False
            )
        
        # Set ensemble threshold based on combined scores
        print("  Calibrating ensemble threshold...")
        scores = self._get_combined_scores(X, X_sequences)
        self.threshold = np.percentile(scores, 100 - self.contamination * 100)
        
        self.fitted = True
        print(f"Ensemble training complete. Threshold: {self.threshold:.4f}")
        return self
    
    def _get_combined_scores(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None
    ) -> np.ndarray:
        """Get weighted combined anomaly scores."""
        n_samples = len(X)
        all_scores = []
        all_weights = []

        for name, detector in self.detectors.items():
            scores = detector.score_samples(X)
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            all_scores.append(scores)
            all_weights.append(self.weights[name])

        # Add autoencoder scores if available and lengths match
        if self.autoencoder is not None and X_sequences is not None:
            ae_scores = self.autoencoder.score_samples(X_sequences)
            ae_scores = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)

            # Handle length mismatch - truncate or pad
            if len(ae_scores) >= n_samples:
                ae_scores = ae_scores[:n_samples]
            else:
                # Pad with mean score
                pad_size = n_samples - len(ae_scores)
                ae_scores = np.concatenate([ae_scores, np.full(pad_size, ae_scores.mean())])

            all_scores.append(ae_scores)
            all_weights.append(self.weights['autoencoder'])

        # Weighted average
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)
        combined = np.average(all_scores, axis=0, weights=all_weights)

        return combined

    def predict(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None
    ) -> np.ndarray:
        """
        Predict anomaly labels using ensemble.

        Args:
            X: Feature matrix
            X_sequences: Sequence data for autoencoder

        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        scores = self.score_samples(X, X_sequences)
        return (scores > self.threshold).astype(int)

    def score_samples(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None
    ) -> np.ndarray:
        """
        Get ensemble anomaly scores.

        Args:
            X: Feature matrix
            X_sequences: Sequence data for autoencoder

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        return self._get_combined_scores(X, X_sequences)

    def predict_with_voting(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict using majority voting.

        Args:
            X: Feature matrix
            X_sequences: Sequence data
            threshold: Fraction of detectors that must agree

        Returns:
            Binary predictions
        """
        votes = []

        for name, detector in self.detectors.items():
            votes.append(detector.predict(X))

        if self.autoencoder is not None and X_sequences is not None:
            votes.append(self.autoencoder.predict(X_sequences))

        votes = np.array(votes)
        agreement = np.mean(votes, axis=0)

        return (agreement >= threshold).astype(int)

    def get_detector_scores(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Get individual detector scores for analysis.

        Returns:
            Dict mapping detector name to scores
        """
        scores = {}

        for name, detector in self.detectors.items():
            scores[name] = detector.score_samples(X)

        if self.autoencoder is not None and X_sequences is not None:
            scores['autoencoder'] = self.autoencoder.score_samples(X_sequences)

        return scores

    def get_detector_predictions(
        self,
        X: np.ndarray,
        X_sequences: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Get individual detector predictions for analysis.

        Returns:
            Dict mapping detector name to predictions
        """
        predictions = {}

        for name, detector in self.detectors.items():
            predictions[name] = detector.predict(X)

        if self.autoencoder is not None and X_sequences is not None:
            predictions['autoencoder'] = self.autoencoder.predict(X_sequences)

        return predictions

