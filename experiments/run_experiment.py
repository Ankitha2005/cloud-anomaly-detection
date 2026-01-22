"""
Main Experiment Script for Cloud Anomaly Detection
Runs the complete pipeline: data loading, feature engineering, model training, and evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.models.baseline_models import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from src.models.temporal_autoencoder import TemporalAnomalyDetector
from src.models.ensemble_detector import EnsembleAnomalyDetector
from src.evaluation.metrics import compute_metrics, print_metrics, compare_models, plot_roc_curves, plot_precision_recall_curves
from src.utils.config_loader import get_project_root


def load_and_prepare_data(data_path: str, sequence_length: int = 10):
    """Load data and prepare features."""
    print("=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    # Feature engineering
    fe = FeatureEngineer()
    df_engineered = fe.engineer_features(df, fit=True)
    
    # Get feature matrix
    X, y = fe.get_feature_matrix(df_engineered)
    print(f"Feature matrix: {X.shape}")
    print(f"Anomaly ratio: {y.mean()*100:.2f}%")
    
    # Create sequences for temporal models
    X_seq, y_seq, node_ids = fe.create_sequences(df_engineered, sequence_length=sequence_length)
    print(f"Sequences: {X_seq.shape}")
    
    return X, y, X_seq, y_seq, fe


def split_data(X, y, X_seq, y_seq, test_size=0.3, random_state=42):
    """Split data into train/test sets."""
    # For point-based models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # For sequence-based models - align with point data
    # Use indices to maintain correspondence
    n_train = len(X_train)
    n_seq = len(X_seq)
    
    # Simple split for sequences
    seq_train_size = int(n_seq * (1 - test_size))
    X_seq_train = X_seq[:seq_train_size]
    X_seq_test = X_seq[seq_train_size:]
    y_seq_train = y_seq[:seq_train_size]
    y_seq_test = y_seq[seq_train_size:]
    
    print(f"\nData split:")
    print(f"  Train (point): {X_train.shape}, Test: {X_test.shape}")
    print(f"  Train (seq): {X_seq_train.shape}, Test: {X_seq_test.shape}")
    
    return (X_train, X_test, y_train, y_test, 
            X_seq_train, X_seq_test, y_seq_train, y_seq_test)


def evaluate_baselines(X_train, X_test, y_test):
    """Train and evaluate baseline models."""
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE MODELS")
    print("=" * 60)
    
    results = {}
    predictions = {}
    
    # Isolation Forest
    print("\n[1/3] Training Isolation Forest...")
    if_detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
    if_detector.fit(X_train)
    y_pred = if_detector.predict(X_test)
    y_scores = if_detector.score_samples(X_test)
    results['Isolation Forest'] = compute_metrics(y_test, y_pred, y_scores)
    predictions['Isolation Forest'] = (y_test, y_pred, y_scores)
    print_metrics(results['Isolation Forest'], 'Isolation Forest')
    
    # One-Class SVM
    print("[2/3] Training One-Class SVM...")
    subset_size = min(10000, len(X_train))
    subset_idx = np.random.choice(len(X_train), subset_size, replace=False)
    ocsvm_detector = OneClassSVMDetector(contamination=0.05)
    ocsvm_detector.fit(X_train[subset_idx])
    y_pred = ocsvm_detector.predict(X_test)
    y_scores = ocsvm_detector.score_samples(X_test)
    results['One-Class SVM'] = compute_metrics(y_test, y_pred, y_scores)
    predictions['One-Class SVM'] = (y_test, y_pred, y_scores)
    print_metrics(results['One-Class SVM'], 'One-Class SVM')
    
    # LOF
    print("[3/3] Training LOF...")
    lof_detector = LOFDetector(contamination=0.05, n_neighbors=20)
    lof_detector.fit(X_train)
    y_pred = lof_detector.predict(X_test)
    y_scores = lof_detector.score_samples(X_test)
    results['LOF'] = compute_metrics(y_test, y_pred, y_scores)
    predictions['LOF'] = (y_test, y_pred, y_scores)
    print_metrics(results['LOF'], 'LOF')
    
    return results, predictions


def evaluate_autoencoder(X_seq_train, X_seq_test, y_seq_test):
    """Train and evaluate temporal autoencoder."""
    print("\n" + "=" * 60)
    print("EVALUATING TEMPORAL AUTOENCODER")
    print("=" * 60)
    
    # Train only on normal sequences
    normal_mask = np.zeros(len(X_seq_train), dtype=bool)
    # Assume first 95% are mostly normal for semi-supervised training
    normal_mask[:int(len(X_seq_train) * 0.95)] = True
    X_train_normal = X_seq_train[normal_mask]
    
    print(f"Training on {len(X_train_normal)} sequences...")
    
    detector = TemporalAnomalyDetector(
        input_dim=X_seq_train.shape[2],
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        seq_len=X_seq_train.shape[1]
    )
    
    detector.fit(
        X_train_normal[:15000],  # Limit for speed
        epochs=30,
        batch_size=64,
        threshold_percentile=95.0,
        verbose=True
    )
    
    y_pred = detector.predict(X_seq_test)
    y_scores = detector.score_samples(X_seq_test)

    metrics = compute_metrics(y_seq_test, y_pred, y_scores)
    print_metrics(metrics, 'Temporal Autoencoder')

    return metrics, (y_seq_test, y_pred, y_scores), detector


def evaluate_ensemble(X_train, X_test, y_test, X_seq_train, X_seq_test, y_seq_test):
    """Train and evaluate ensemble detector."""
    print("\n" + "=" * 60)
    print("EVALUATING ENSEMBLE DETECTOR")
    print("=" * 60)

    # Train only on normal data
    normal_mask_train = np.random.rand(len(X_train)) < 0.95  # Approximate
    X_train_subset = X_train[normal_mask_train][:20000]
    X_seq_train_subset = X_seq_train[:15000]

    ensemble = EnsembleAnomalyDetector(
        contamination=0.05,
        use_isolation_forest=True,
        use_ocsvm=True,
        use_lof=True,
        use_autoencoder=True
    )

    ensemble.fit(X_train_subset, X_seq_train_subset)

    # Evaluate on test set
    y_pred = ensemble.predict(X_test, X_seq_test[:len(X_test)] if len(X_seq_test) >= len(X_test) else None)
    y_scores = ensemble.score_samples(X_test, X_seq_test[:len(X_test)] if len(X_seq_test) >= len(X_test) else None)

    metrics = compute_metrics(y_test, y_pred, y_scores)
    print_metrics(metrics, 'Ensemble')

    return metrics, (y_test, y_pred, y_scores), ensemble


def save_results(results: dict, output_dir: str):
    """Save experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")

    # Convert numpy types to Python types
    serializable_results = {}
    for model, metrics in results.items():
        serializable_results[model] = {k: float(v) for k, v in metrics.items()}

    with open(metrics_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {metrics_path}")
    return metrics_path


def main():
    """Run the complete experiment."""
    print("\n" + "=" * 60)
    print("CLOUD ANOMALY DETECTION EXPERIMENT")
    print("=" * 60)

    project_root = get_project_root()
    data_path = project_root / "data" / "raw" / "cloudsim_logs.csv"
    output_dir = project_root / "results"

    # Load and prepare data
    X, y, X_seq, y_seq, fe = load_and_prepare_data(str(data_path))

    # Split data
    (X_train, X_test, y_train, y_test,
     X_seq_train, X_seq_test, y_seq_train, y_seq_test) = split_data(X, y, X_seq, y_seq)

    # Evaluate all models
    all_results = {}
    all_predictions = {}

    # Baselines
    baseline_results, baseline_preds = evaluate_baselines(X_train, X_test, y_test)
    all_results.update(baseline_results)
    all_predictions.update(baseline_preds)

    # Temporal Autoencoder
    ae_metrics, ae_preds, ae_model = evaluate_autoencoder(X_seq_train, X_seq_test, y_seq_test)
    all_results['Temporal Autoencoder'] = ae_metrics
    all_predictions['Temporal Autoencoder'] = ae_preds

    # Ensemble
    ens_metrics, ens_preds, ens_model = evaluate_ensemble(
        X_train, X_test, y_test, X_seq_train, X_seq_test, y_seq_test
    )
    all_results['Ensemble'] = ens_metrics
    all_predictions['Ensemble'] = ens_preds

    # Print comparison
    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)
    print(compare_models(all_results))

    # Save results
    save_results(all_results, str(output_dir))

    # Plot curves
    plot_roc_curves(all_predictions, str(output_dir / "roc_curves.png"))
    plot_precision_recall_curves(all_predictions, str(output_dir / "pr_curves.png"))

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = main()

