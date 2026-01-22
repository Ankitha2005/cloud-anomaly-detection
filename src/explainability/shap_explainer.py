"""
SHAP-based Explainability for Anomaly Detection
Provides feature importance and local explanations for anomaly predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class AnomalyExplainer:
    """
    Explainability module for anomaly detection models.
    Uses permutation importance and local feature analysis.
    """
    
    def __init__(self, detector, feature_names: List[str] = None):
        """
        Initialize explainer.
        
        Args:
            detector: Trained anomaly detector with score_samples method
            feature_names: List of feature names
        """
        self.detector = detector
        self.feature_names = feature_names
        self.global_importance = None
        
    def compute_permutation_importance(
        self,
        X: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Compute feature importance using permutation.
        
        Args:
            X: Feature matrix
            n_repeats: Number of permutation repeats
            random_state: Random seed
            
        Returns:
            Dict mapping feature names to importance scores
        """
        np.random.seed(random_state)
        
        # Baseline scores
        baseline_scores = self.detector.score_samples(X)
        baseline_mean = np.mean(baseline_scores)
        
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        for feat_idx in range(n_features):
            feat_importance = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feat_idx])
                
                permuted_scores = self.detector.score_samples(X_permuted)
                permuted_mean = np.mean(permuted_scores)
                
                # Importance = change in mean anomaly score
                feat_importance.append(abs(permuted_mean - baseline_mean))
            
            importance[feat_idx] = np.mean(feat_importance)
        
        # Normalize
        importance = importance / (importance.sum() + 1e-8)
        
        # Create dict with feature names
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.global_importance = dict(zip(self.feature_names, importance))
        
        return self.global_importance
    
    def explain_instance(
        self,
        x: np.ndarray,
        X_background: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Explain a single anomaly instance.
        
        Args:
            x: Single instance to explain (1D array)
            X_background: Background data for comparison
            top_k: Number of top features to return
            
        Returns:
            Dict of feature contributions
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Get anomaly score for this instance
        instance_score = self.detector.score_samples(x)[0]
        
        # Compare to background
        bg_mean = X_background.mean(axis=0)
        bg_std = X_background.std(axis=0) + 1e-8
        
        # Z-score of instance features
        z_scores = (x[0] - bg_mean) / bg_std
        
        # Weight by global importance if available
        if self.global_importance is not None:
            importance_weights = np.array([
                self.global_importance.get(name, 0) 
                for name in self.feature_names
            ])
            contributions = z_scores * importance_weights
        else:
            contributions = z_scores
        
        # Create explanation dict
        explanation = dict(zip(self.feature_names, contributions))
        
        # Sort by absolute contribution
        sorted_explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        )
        
        return sorted_explanation
    
    def plot_global_importance(
        self,
        top_k: int = 20,
        save_path: str = None
    ):
        """Plot global feature importance."""
        if self.global_importance is None:
            raise ValueError("Run compute_permutation_importance first")
        
        # Sort and get top k
        sorted_imp = sorted(
            self.global_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        features, values = zip(*sorted_imp)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), values, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Global Feature Importance for Anomaly Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved importance plot to {save_path}")
        plt.close()

    def plot_instance_explanation(
        self,
        explanation: Dict[str, float],
        instance_score: float = None,
        save_path: str = None
    ):
        """Plot local explanation for a single instance."""
        features = list(explanation.keys())
        values = list(explanation.values())

        colors = ['red' if v > 0 else 'green' for v in values]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), values, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Contribution')

        title = 'Anomaly Explanation'
        if instance_score is not None:
            title += f' (Score: {instance_score:.4f})'
        plt.title(title)

        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        top_anomalies: int = 5,
        output_dir: str = None
    ) -> str:
        """
        Generate a comprehensive explainability report.

        Args:
            X: Feature matrix
            y_pred: Predictions
            y_scores: Anomaly scores
            top_anomalies: Number of top anomalies to explain
            output_dir: Directory to save plots

        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("ANOMALY DETECTION EXPLAINABILITY REPORT")
        report_lines.append("=" * 60)

        # Summary statistics
        n_anomalies = y_pred.sum()
        report_lines.append(f"\nTotal samples: {len(X)}")
        report_lines.append(f"Detected anomalies: {n_anomalies} ({n_anomalies/len(X)*100:.2f}%)")
        report_lines.append(f"Mean anomaly score: {y_scores.mean():.4f}")
        report_lines.append(f"Max anomaly score: {y_scores.max():.4f}")

        # Global importance
        if self.global_importance is not None:
            report_lines.append("\n" + "-" * 40)
            report_lines.append("TOP 10 MOST IMPORTANT FEATURES")
            report_lines.append("-" * 40)

            sorted_imp = sorted(
                self.global_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for i, (feat, imp) in enumerate(sorted_imp, 1):
                report_lines.append(f"  {i}. {feat}: {imp:.4f}")

        # Top anomalies
        report_lines.append("\n" + "-" * 40)
        report_lines.append(f"TOP {top_anomalies} ANOMALIES")
        report_lines.append("-" * 40)

        top_indices = np.argsort(y_scores)[-top_anomalies:][::-1]

        for rank, idx in enumerate(top_indices, 1):
            report_lines.append(f"\n  Anomaly #{rank} (Index: {idx})")
            report_lines.append(f"    Score: {y_scores[idx]:.4f}")

            # Get explanation
            explanation = self.explain_instance(X[idx], X, top_k=5)
            report_lines.append("    Top contributing features:")
            for feat, contrib in explanation.items():
                direction = "↑" if contrib > 0 else "↓"
                report_lines.append(f"      - {feat}: {contrib:.4f} {direction}")

            # Save plot if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self.plot_instance_explanation(
                    explanation,
                    y_scores[idx],
                    os.path.join(output_dir, f"anomaly_{rank}_explanation.png")
                )

        report_lines.append("\n" + "=" * 60)

        report = "\n".join(report_lines)

        # Save report
        if output_dir:
            report_path = os.path.join(output_dir, "explainability_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")

        return report

