"""
Ablation Study for Cloud Anomaly Detection
Evaluates the contribution of each component to overall performance.
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
from src.models.baseline_models import IsolationForestDetector
from src.evaluation.metrics import compute_metrics, compare_models
from src.utils.config_loader import get_project_root


def run_feature_ablation(X_train, X_test, y_test, feature_names, feature_groups):
    """
    Ablation study on feature groups.
    
    Args:
        X_train, X_test, y_test: Data splits
        feature_names: List of all feature names
        feature_groups: Dict mapping group name to list of feature indices
    """
    print("\n" + "=" * 60)
    print("FEATURE GROUP ABLATION STUDY")
    print("=" * 60)
    
    results = {}
    
    # Baseline with all features
    print("\n[Baseline] Training with all features...")
    detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
    detector.fit(X_train)
    y_pred = detector.predict(X_test)
    y_scores = detector.score_samples(X_test)
    results['All Features'] = compute_metrics(y_test, y_pred, y_scores)
    print(f"  F1: {results['All Features']['f1']:.4f}, AUC-ROC: {results['All Features']['auc_roc']:.4f}")
    
    # Ablate each feature group
    for group_name, group_indices in feature_groups.items():
        print(f"\n[Ablation] Removing {group_name}...")
        
        # Create mask for remaining features
        remaining_indices = [i for i in range(X_train.shape[1]) if i not in group_indices]
        
        X_train_ablated = X_train[:, remaining_indices]
        X_test_ablated = X_test[:, remaining_indices]
        
        detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
        detector.fit(X_train_ablated)
        y_pred = detector.predict(X_test_ablated)
        y_scores = detector.score_samples(X_test_ablated)
        
        results[f'Without {group_name}'] = compute_metrics(y_test, y_pred, y_scores)
        
        # Calculate drop
        f1_drop = results['All Features']['f1'] - results[f'Without {group_name}']['f1']
        print(f"  F1: {results[f'Without {group_name}']['f1']:.4f} (drop: {f1_drop:.4f})")
    
    return results


def run_model_ablation(X_train, X_test, y_test, X_seq_train, X_seq_test, y_seq_test):
    """
    Ablation study on ensemble components.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPONENT ABLATION STUDY")
    print("=" * 60)
    
    from src.models.ensemble_detector import EnsembleAnomalyDetector
    
    results = {}
    
    # Full ensemble
    print("\n[Full Ensemble] Training all components...")
    ensemble = EnsembleAnomalyDetector(
        contamination=0.05,
        use_isolation_forest=True,
        use_ocsvm=True,
        use_lof=True,
        use_autoencoder=True
    )
    ensemble.fit(X_train[:15000], X_seq_train[:10000])
    y_pred = ensemble.predict(X_test, X_seq_test[:len(X_test)] if len(X_seq_test) >= len(X_test) else None)
    y_scores = ensemble.score_samples(X_test, X_seq_test[:len(X_test)] if len(X_seq_test) >= len(X_test) else None)
    results['Full Ensemble'] = compute_metrics(y_test, y_pred, y_scores)
    print(f"  F1: {results['Full Ensemble']['f1']:.4f}")
    
    # Without each component
    ablation_configs = [
        ('Without IF', {'use_isolation_forest': False, 'use_ocsvm': True, 'use_lof': True, 'use_autoencoder': True}),
        ('Without OCSVM', {'use_isolation_forest': True, 'use_ocsvm': False, 'use_lof': True, 'use_autoencoder': True}),
        ('Without LOF', {'use_isolation_forest': True, 'use_ocsvm': True, 'use_lof': False, 'use_autoencoder': True}),
        ('Without AE', {'use_isolation_forest': True, 'use_ocsvm': True, 'use_lof': True, 'use_autoencoder': False}),
    ]
    
    for name, config in ablation_configs:
        print(f"\n[{name}] Training...")
        ensemble = EnsembleAnomalyDetector(contamination=0.05, **config)
        
        seq_data = X_seq_train[:10000] if config.get('use_autoencoder', True) else None
        ensemble.fit(X_train[:15000], seq_data)
        
        test_seq = X_seq_test[:len(X_test)] if config.get('use_autoencoder', True) and len(X_seq_test) >= len(X_test) else None
        y_pred = ensemble.predict(X_test, test_seq)
        y_scores = ensemble.score_samples(X_test, test_seq)
        
        results[name] = compute_metrics(y_test, y_pred, y_scores)
        
        f1_drop = results['Full Ensemble']['f1'] - results[name]['f1']
        print(f"  F1: {results[name]['f1']:.4f} (drop: {f1_drop:.4f})")
    
    return results


def run_contamination_sensitivity(X_train, X_test, y_test):
    """Study sensitivity to contamination parameter."""
    print("\n" + "=" * 60)
    print("CONTAMINATION SENSITIVITY STUDY")
    print("=" * 60)
    
    contamination_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    results = {}
    
    for cont in contamination_values:
        print(f"\n[Contamination={cont}] Training...")
        detector = IsolationForestDetector(contamination=cont, n_estimators=100)
        detector.fit(X_train)
        y_pred = detector.predict(X_test)
        y_scores = detector.score_samples(X_test)
        
        results[f'cont={cont}'] = compute_metrics(y_test, y_pred, y_scores)
        print(f"  Precision: {results[f'cont={cont}']['precision']:.4f}, "
              f"Recall: {results[f'cont={cont}']['recall']:.4f}, "
              f"F1: {results[f'cont={cont}']['f1']:.4f}")

    return results


def main():
    """Run complete ablation study."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY FOR CLOUD ANOMALY DETECTION")
    print("=" * 60)

    project_root = get_project_root()

    # Load data
    print("\nLoading and preparing data...")
    df = pd.read_csv(project_root / "data" / "raw" / "cloudsim_logs.csv")

    fe = FeatureEngineer()
    df_eng = fe.engineer_features(df, fit=True)

    X, y = fe.get_feature_matrix(df_eng)
    X_seq, y_seq, _ = fe.create_sequences(df_eng, sequence_length=10)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    seq_train_size = int(len(X_seq) * 0.7)
    X_seq_train = X_seq[:seq_train_size]
    X_seq_test = X_seq[seq_train_size:]
    y_seq_train = y_seq[:seq_train_size]
    y_seq_test = y_seq[seq_train_size:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Define feature groups for ablation
    feature_groups = {
        'Temporal Features': [i for i, name in enumerate(fe.feature_columns)
                              if 'hour' in name or 'dow' in name or 'weekend' in name or 'business' in name],
        'Rolling Statistics': [i for i, name in enumerate(fe.feature_columns)
                               if 'roll_' in name],
        'Deviation Features': [i for i, name in enumerate(fe.feature_columns)
                               if 'deviation' in name],
        'Cluster Features': [i for i, name in enumerate(fe.feature_columns)
                             if 'cluster' in name],
        'Interaction Features': [i for i, name in enumerate(fe.feature_columns)
                                 if 'product' in name or 'ratio' in name or 'per_conn' in name or 'error_rate' in name],
    }

    all_results = {}

    # 1. Feature ablation
    feature_results = run_feature_ablation(X_train, X_test, y_test, fe.feature_columns, feature_groups)
    all_results['feature_ablation'] = feature_results

    # 2. Model ablation
    model_results = run_model_ablation(X_train, X_test, y_test, X_seq_train, X_seq_test, y_seq_test)
    all_results['model_ablation'] = model_results

    # 3. Contamination sensitivity
    cont_results = run_contamination_sensitivity(X_train, X_test, y_test)
    all_results['contamination_sensitivity'] = cont_results

    # Save results
    output_dir = project_root / "results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"ablation_results_{timestamp}.json"

    # Convert to serializable format
    serializable = {}
    for study, results in all_results.items():
        serializable[study] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    print("\nFeature Group Importance (by F1 drop when removed):")
    baseline_f1 = feature_results['All Features']['f1']
    for name, metrics in feature_results.items():
        if name != 'All Features':
            drop = baseline_f1 - metrics['f1']
            print(f"  {name}: {drop:+.4f}")

    print("\nModel Component Importance (by F1 drop when removed):")
    baseline_f1 = model_results['Full Ensemble']['f1']
    for name, metrics in model_results.items():
        if name != 'Full Ensemble':
            drop = baseline_f1 - metrics['f1']
            print(f"  {name}: {drop:+.4f}")

    return all_results


if __name__ == "__main__":
    results = main()

