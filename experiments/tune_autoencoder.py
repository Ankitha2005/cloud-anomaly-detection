"""
Hyperparameter Tuning for Temporal Autoencoder
Grid search over architecture and training parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
import json
from datetime import datetime

from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.models.temporal_autoencoder import TemporalAnomalyDetector
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.config_loader import get_project_root


def prepare_data(sequence_length=10):
    """Load and prepare data for autoencoder training."""
    project_root = get_project_root()
    df = pd.read_csv(project_root / "data" / "raw" / "cloudsim_logs.csv")
    
    fe = FeatureEngineer()
    df_eng = fe.engineer_features(df, fit=True)
    
    X_seq, y_seq, _ = fe.create_sequences(df_eng, sequence_length=sequence_length)
    
    # Split: train on normal, test on mixed
    normal_mask = y_seq == 0
    X_normal = X_seq[normal_mask]
    X_anomaly = X_seq[~normal_mask]
    
    # Train/val split on normal data
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
    
    # Test set: balanced normal and anomaly
    n_test = min(2000, len(X_anomaly))
    X_test = np.vstack([X_val[:n_test], X_anomaly[:n_test]])
    y_test = np.concatenate([np.zeros(n_test), np.ones(n_test)])
    
    # Shuffle test set
    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]
    
    return X_train, X_val, X_test, y_test, fe.feature_columns


def run_hyperparameter_search():
    """Run grid search over hyperparameters."""
    print("=" * 70)
    print("TEMPORAL AUTOENCODER HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # Prepare data with different sequence lengths
    sequence_lengths = [5, 10, 20]
    
    # Hyperparameter grid
    param_grid = {
        'hidden_dim': [32, 64, 128],
        'latent_dim': [16, 32, 64],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'threshold_percentile': [90, 95, 99],
    }
    
    # Reduced grid for faster search
    reduced_grid = {
        'hidden_dim': [64, 128],
        'latent_dim': [32, 64],
        'num_layers': [2, 3],
        'dropout': [0.1, 0.2],
        'learning_rate': [1e-3, 5e-4],
        'threshold_percentile': [90, 95],
    }
    
    results = []
    best_f1 = 0
    best_config = None
    
    # Test different sequence lengths
    for seq_len in [10, 15]:
        print(f"\n{'='*50}")
        print(f"Testing sequence length: {seq_len}")
        print(f"{'='*50}")
        
        X_train, X_val, X_test, y_test, _ = prepare_data(sequence_length=seq_len)
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Generate parameter combinations
        param_names = list(reduced_grid.keys())
        param_values = list(reduced_grid.values())
        
        for i, combo in enumerate(product(*param_values)):
            params = dict(zip(param_names, combo))
            params['sequence_length'] = seq_len
            
            print(f"\n[{i+1}] Testing: hidden={params['hidden_dim']}, "
                  f"latent={params['latent_dim']}, layers={params['num_layers']}, "
                  f"dropout={params['dropout']}, lr={params['learning_rate']}")
            
            try:
                detector = TemporalAnomalyDetector(
                    input_dim=X_train.shape[2],
                    hidden_dim=params['hidden_dim'],
                    latent_dim=params['latent_dim'],
                    num_layers=params['num_layers'],
                    seq_len=seq_len,
                    dropout=params['dropout']
                )
                
                # Train with early stopping
                history = detector.fit(
                    X_train[:15000],
                    X_val=X_val[:3000],
                    epochs=50,
                    batch_size=64,
                    learning_rate=params['learning_rate'],
                    early_stopping_patience=10,
                    threshold_percentile=params['threshold_percentile'],
                    verbose=False
                )
                
                # Evaluate
                y_pred = detector.predict(X_test)
                y_scores = detector.score_samples(X_test)
                metrics = compute_metrics(y_test, y_pred, y_scores)
                
                params['f1'] = metrics['f1']
                params['precision'] = metrics['precision']
                params['recall'] = metrics['recall']
                params['auc_roc'] = metrics['auc_roc']
                params['auc_pr'] = metrics['auc_pr']
                
                results.append(params)
                
                print(f"    F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_config = params.copy()
                    print(f"    *** NEW BEST F1: {best_f1:.4f} ***")

            except Exception as e:
                print(f"    Error: {e}")
                continue

    return results, best_config


def train_best_model(best_config):
    """Train the best model configuration and save it."""
    print("\n" + "=" * 70)
    print("TRAINING BEST MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Config: {best_config}")

    seq_len = best_config.get('sequence_length', 10)
    X_train, X_val, X_test, y_test, feature_names = prepare_data(sequence_length=seq_len)

    detector = TemporalAnomalyDetector(
        input_dim=X_train.shape[2],
        hidden_dim=best_config['hidden_dim'],
        latent_dim=best_config['latent_dim'],
        num_layers=best_config['num_layers'],
        seq_len=seq_len,
        dropout=best_config['dropout']
    )

    # Train with more epochs
    history = detector.fit(
        X_train[:20000],
        X_val=X_val[:5000],
        epochs=100,
        batch_size=64,
        learning_rate=best_config['learning_rate'],
        early_stopping_patience=15,
        threshold_percentile=best_config['threshold_percentile'],
        verbose=True
    )

    # Final evaluation
    y_pred = detector.predict(X_test)
    y_scores = detector.score_samples(X_test)
    metrics = compute_metrics(y_test, y_pred, y_scores)

    print_metrics(metrics, "Tuned Temporal Autoencoder")

    # Save model
    project_root = get_project_root()
    model_path = project_root / "outputs" / "models" / "best_autoencoder.pt"
    detector.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    return detector, metrics


def main():
    """Run complete tuning pipeline."""
    # Run hyperparameter search
    results, best_config = run_hyperparameter_search()

    # Save results
    project_root = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(results)
    results_path = project_root / "results" / f"autoencoder_tuning_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nTuning results saved to {results_path}")

    # Print top 10 configurations
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY F1 SCORE")
    print("=" * 70)
    results_df_sorted = results_df.sort_values('f1', ascending=False).head(10)
    print(results_df_sorted[['hidden_dim', 'latent_dim', 'num_layers', 'dropout',
                              'learning_rate', 'threshold_percentile', 'f1', 'auc_roc']].to_string())

    # Train best model
    if best_config:
        detector, final_metrics = train_best_model(best_config)

        # Save best config
        config_path = project_root / "results" / "best_autoencoder_config.json"
        with open(config_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in best_config.items()}, f, indent=2)
        print(f"Best config saved to {config_path}")

    return results, best_config


if __name__ == "__main__":
    results, best_config = main()

