"""
Generate Publication-Quality Figures for Research Paper
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.models.baseline_models import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from src.models.temporal_autoencoder import TemporalAnomalyDetector
from src.evaluation.metrics import compute_metrics
from src.utils.config_loader import get_project_root


def load_data():
    """Load and prepare data."""
    project_root = get_project_root()
    df = pd.read_csv(project_root / "data" / "raw" / "cloudsim_logs.csv")
    
    fe = FeatureEngineer()
    df_eng = fe.engineer_features(df, fit=True)
    X, y = fe.get_feature_matrix(df_eng)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, fe


def figure1_model_comparison_roc(X_train, X_test, y_test, output_dir):
    """Figure 1: ROC curves for all models."""
    print("Generating Figure 1: ROC Comparison...")
    
    models = {
        'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
        'One-Class SVM': OneClassSVMDetector(nu=0.05),
        'LOF': LOFDetector(contamination=0.05, n_neighbors=20),
    }
    
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train[:15000])
        y_scores = model.score_samples(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        metrics = compute_metrics(y_test, model.predict(X_test), y_scores)
        ax.plot(fpr, tpr, color=color, lw=2.5, 
                label=f'{name} (AUC = {metrics["auc_roc"]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Model Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.savefig(output_dir / 'fig1_roc_comparison.png')
    plt.savefig(output_dir / 'fig1_roc_comparison.pdf')
    plt.close()
    print("  Saved fig1_roc_comparison.png/pdf")


def figure2_precision_recall(X_train, X_test, y_test, output_dir):
    """Figure 2: Precision-Recall curves."""
    print("Generating Figure 2: Precision-Recall Curves...")
    
    models = {
        'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
        'One-Class SVM': OneClassSVMDetector(nu=0.05),
    }
    
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#3498db']
    
    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train[:15000])
        y_scores = model.score_samples(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        metrics = compute_metrics(y_test, model.predict(X_test), y_scores)
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f'{name} (AP = {metrics["auc_pr"]:.3f})')
    
    ax.axhline(y=y_test.mean(), color='gray', linestyle='--', lw=1.5, 
               label=f'Baseline ({y_test.mean():.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.savefig(output_dir / 'fig2_precision_recall.png')
    plt.savefig(output_dir / 'fig2_precision_recall.pdf')
    plt.close()
    print("  Saved fig2_precision_recall.png/pdf")


def figure3_feature_importance(X_train, X_test, y_test, fe, output_dir):
    """Figure 3: Feature importance bar chart."""
    print("Generating Figure 3: Feature Importance...")
    
    from src.explainability.shap_explainer import AnomalyExplainer
    
    detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
    detector.fit(X_train[:15000])
    
    explainer = AnomalyExplainer(detector, feature_names=fe.feature_columns)
    importance = explainer.compute_permutation_importance(X_test[:3000], n_repeats=5)
    
    # Top 15 features
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features, values = zip(*sorted_imp)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    
    bars = ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Top 15 Most Important Features')
    
    plt.savefig(output_dir / 'fig3_feature_importance.png')
    plt.savefig(output_dir / 'fig3_feature_importance.pdf')
    plt.close()
    print("  Saved fig3_feature_importance.png/pdf")


def figure4_confusion_matrix(X_train, X_test, y_test, output_dir):
    """Figure 4: Confusion matrix heatmap."""
    print("Generating Figure 4: Confusion Matrix...")

    detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
    detector.fit(X_train[:15000])
    y_pred = detector.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Isolation Forest')

    plt.savefig(output_dir / 'fig4_confusion_matrix.png')
    plt.savefig(output_dir / 'fig4_confusion_matrix.pdf')
    plt.close()
    print("  Saved fig4_confusion_matrix.png/pdf")


def figure5_attack_detection_rates(output_dir):
    """Figure 5: Attack detection rates bar chart."""
    print("Generating Figure 5: Attack Detection Rates...")

    import json
    project_root = get_project_root()

    with open(project_root / 'results' / 'adversarial_results.json', 'r') as f:
        results = json.load(f)

    attacks = list(results.keys())
    rates = [results[a]['detection_rate'] * 100 for a in attacks]

    # Sort by detection rate
    sorted_data = sorted(zip(attacks, rates), key=lambda x: x[1], reverse=True)
    attacks, rates = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#2ecc71' if r >= 80 else '#f39c12' if r >= 50 else '#e74c3c' for r in rates]

    bars = ax.bar(range(len(attacks)), rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([a.replace('_', '\n') for a in attacks], rotation=0, ha='center')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Attack Detection Rates by Type')
    ax.set_ylim([0, 110])
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='High (≥80%)')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Medium (≥50%)')
    ax.legend(loc='upper right')

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10)

    plt.savefig(output_dir / 'fig5_attack_detection.png')
    plt.savefig(output_dir / 'fig5_attack_detection.pdf')
    plt.close()
    print("  Saved fig5_attack_detection.png/pdf")


def figure6_model_comparison_bar(X_train, X_test, y_test, output_dir):
    """Figure 6: Model comparison bar chart."""
    print("Generating Figure 6: Model Comparison Bar Chart...")

    models = {
        'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
        'One-Class SVM': OneClassSVMDetector(nu=0.05),
        'LOF': LOFDetector(contamination=0.05, n_neighbors=20),
    }

    metrics_list = []
    for name, model in models.items():
        model.fit(X_train[:15000])
        y_pred = model.predict(X_test)
        y_scores = model.score_samples(X_test)
        m = compute_metrics(y_test, y_pred, y_scores)
        m['model'] = name
        metrics_list.append(m)

    df_metrics = pd.DataFrame(metrics_list)

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(models))
    width = 0.2

    metrics_to_plot = ['precision', 'recall', 'f1', 'auc_roc']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        values = df_metrics[metric].values
        bars = ax.bar(x + i*width, values, width, label=metric.upper().replace('_', '-'), color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_metrics['model'])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.1])

    plt.savefig(output_dir / 'fig6_model_comparison.png')
    plt.savefig(output_dir / 'fig6_model_comparison.pdf')
    plt.close()
    print("  Saved fig6_model_comparison.png/pdf")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60)

    project_root = get_project_root()
    output_dir = project_root / "results" / "figures"
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, fe = load_data()

    figure1_model_comparison_roc(X_train, X_test, y_test, output_dir)
    figure2_precision_recall(X_train, X_test, y_test, output_dir)
    figure3_feature_importance(X_train, X_test, y_test, fe, output_dir)
    figure4_confusion_matrix(X_train, X_test, y_test, output_dir)
    figure5_attack_detection_rates(output_dir)
    figure6_model_comparison_bar(X_train, X_test, y_test, output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

