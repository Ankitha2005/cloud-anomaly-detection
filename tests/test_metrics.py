"""
Unit tests for Evaluation Metrics module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_metrics, compare_models


class TestComputeMetrics:
    """Test cases for compute_metrics function."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        metrics = compute_metrics(y_true, y_pred, y_scores)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['auc_roc'] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        
        metrics = compute_metrics(y_true, y_pred, y_scores)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
    
    def test_partial_predictions(self):
        """Test metrics with partial correct predictions."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_scores = np.array([0.2, 0.6, 0.7, 0.8, 0.4, 0.3])
        
        metrics = compute_metrics(y_true, y_pred, y_scores)
        
        # Check all metrics are in valid range
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1
        assert 0 <= metrics['fpr'] <= 1
    
    def test_confusion_matrix_values(self):
        """Test confusion matrix values."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])

        metrics = compute_metrics(y_true, y_pred, y_scores)

        assert metrics['true_positives'] == 2
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['true_negatives'] == 2

    def test_all_zeros(self):
        """Test with all negative labels."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = compute_metrics(y_true, y_pred, y_scores)

        # Should handle edge case gracefully
        assert metrics['true_negatives'] == 4
        assert metrics['fpr'] == 0.0

    def test_all_ones(self):
        """Test with all positive labels."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_scores = np.array([0.7, 0.8, 0.9, 0.95])

        metrics = compute_metrics(y_true, y_pred, y_scores)

        assert metrics['true_positives'] == 4
        assert metrics['recall'] == 1.0


class TestCompareModels:
    """Test cases for compare_models function."""

    def test_compare_two_models(self):
        """Test comparing two models."""
        results = {
            'Model A': {
                'precision': 0.8, 'recall': 0.7, 'f1': 0.75,
                'auc_roc': 0.85, 'auc_pr': 0.80, 'fpr': 0.1
            },
            'Model B': {
                'precision': 0.9, 'recall': 0.6, 'f1': 0.72,
                'auc_roc': 0.88, 'auc_pr': 0.82, 'fpr': 0.05
            }
        }

        comparison = compare_models(results)

        # compare_models returns a formatted string
        assert comparison is not None
        assert 'Model A' in comparison
        assert 'Model B' in comparison

    def test_empty_results(self):
        """Test with empty results."""
        results = {}
        comparison = compare_models(results)

        # Should return header even if empty
        assert comparison is not None


class TestMetricsEdgeCases:
    """Test edge cases for metrics computation."""
    
    def test_single_sample(self):
        """Test with single sample."""
        y_true = np.array([1])
        y_pred = np.array([1])
        y_scores = np.array([0.9])
        
        metrics = compute_metrics(y_true, y_pred, y_scores)
        assert metrics is not None
    
    def test_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(42)
        n = 10000
        y_true = np.random.choice([0, 1], n, p=[0.95, 0.05])
        y_pred = np.random.choice([0, 1], n, p=[0.95, 0.05])
        y_scores = np.random.uniform(0, 1, n)

        metrics = compute_metrics(y_true, y_pred, y_scores)

        # Check key metrics are in valid range
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['fpr'] <= 1

    def test_imbalanced_dataset(self):
        """Test with highly imbalanced dataset."""
        y_true = np.array([0] * 99 + [1])
        y_pred = np.array([0] * 100)
        y_scores = np.array([0.1] * 99 + [0.2])

        metrics = compute_metrics(y_true, y_pred, y_scores)

        assert metrics['recall'] == 0.0  # Missed the only positive
        assert metrics['true_negatives'] == 99

