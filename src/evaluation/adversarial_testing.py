"""
Adversarial Testing Module
Injects synthetic attack patterns to evaluate model robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class AdversarialInjector:
    """
    Injects various types of adversarial/attack patterns into cloud data.
    """
    
    # Attack type definitions - comprehensive set
    ATTACK_TYPES = {
        # Original attacks
        'cryptomining': {
            'description': 'High CPU usage with periodic patterns',
            'cpu_range': (0.85, 0.99),
            'ram_range': (0.3, 0.5),
            'pattern': 'sustained'
        },
        'data_exfiltration': {
            'description': 'High bandwidth with low CPU',
            'bandwidth_range': (0.80, 0.99),
            'cpu_range': (0.1, 0.3),
            'pattern': 'burst'
        },
        'ddos': {
            'description': 'Massive connection spike',
            'connections_range': (100, 500),
            'request_range': (1000, 5000),
            'pattern': 'spike'
        },
        'insider_threat': {
            'description': 'Unusual access during off-hours',
            'disk_io_range': (0.7, 0.95),
            'hour_range': (0, 5),
            'pattern': 'stealth'
        },
        'resource_exhaustion': {
            'description': 'Gradual resource consumption',
            'cpu_range': (0.6, 0.99),
            'ram_range': (0.7, 0.99),
            'pattern': 'gradual'
        },
        # New sophisticated attacks
        'slowloris': {
            'description': 'Slow HTTP attack - many connections, low bandwidth',
            'connections_range': (200, 400),
            'bandwidth_range': (0.05, 0.15),
            'request_range': (50, 100),
            'pattern': 'slow'
        },
        'ransomware': {
            'description': 'High disk I/O with encryption patterns',
            'disk_io_range': (0.90, 0.99),
            'cpu_range': (0.70, 0.90),
            'ram_range': (0.60, 0.80),
            'pattern': 'burst'
        },
        'lateral_movement': {
            'description': 'Unusual inter-node communication',
            'bandwidth_range': (0.40, 0.60),
            'connections_range': (30, 60),
            'pattern': 'stealth'
        },
        'privilege_escalation': {
            'description': 'Sudden resource access spike',
            'cpu_range': (0.50, 0.70),
            'ram_range': (0.50, 0.70),
            'disk_io_range': (0.60, 0.80),
            'pattern': 'spike'
        },
        'covert_channel': {
            'description': 'Low-bandwidth data exfiltration via timing',
            'bandwidth_range': (0.02, 0.08),
            'cpu_range': (0.20, 0.35),
            'pattern': 'periodic'
        },
        'vm_escape': {
            'description': 'Hypervisor-level attack patterns',
            'cpu_range': (0.95, 0.99),
            'ram_range': (0.90, 0.99),
            'disk_io_range': (0.85, 0.95),
            'pattern': 'spike'
        },
        'botnet_c2': {
            'description': 'Command and control communication',
            'bandwidth_range': (0.10, 0.25),
            'connections_range': (5, 15),
            'request_range': (10, 30),
            'pattern': 'periodic'
        },
        'memory_scraping': {
            'description': 'Memory-based data theft',
            'ram_range': (0.85, 0.99),
            'cpu_range': (0.30, 0.50),
            'disk_io_range': (0.10, 0.20),
            'pattern': 'sustained'
        }
    }
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def inject_attack(
        self,
        df: pd.DataFrame,
        attack_type: str,
        target_nodes: List[str] = None,
        injection_ratio: float = 0.1,
        duration_windows: int = 10
    ) -> pd.DataFrame:
        """
        Inject a specific attack pattern into the dataset.
        
        Args:
            df: Original DataFrame
            attack_type: Type of attack to inject
            target_nodes: Specific nodes to target (random if None)
            injection_ratio: Fraction of data to affect
            duration_windows: How many time windows the attack lasts
            
        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()
        
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        attack_config = self.ATTACK_TYPES[attack_type]
        
        # Select target nodes
        if target_nodes is None:
            all_nodes = df['node_id'].unique()
            n_targets = max(1, int(len(all_nodes) * injection_ratio))
            target_nodes = np.random.choice(all_nodes, n_targets, replace=False)
        
        # Select time windows for attack
        all_windows = df['window_idx'].unique()
        start_window = np.random.choice(all_windows[:-duration_windows])
        attack_windows = range(start_window, start_window + duration_windows)
        
        # Apply attack pattern
        mask = (df['node_id'].isin(target_nodes)) & (df['window_idx'].isin(attack_windows))
        
        df = self._apply_attack_pattern(df, mask, attack_type, attack_config)
        
        # Mark as anomaly
        df.loc[mask, 'label'] = 1
        df.loc[mask, 'attack_type'] = attack_type
        
        n_injected = mask.sum()
        print(f"Injected {n_injected} {attack_type} attack records")
        
        return df
    
    def _apply_attack_pattern(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        attack_type: str,
        config: dict
    ) -> pd.DataFrame:
        """Apply specific attack pattern to masked records."""

        n_records = mask.sum()
        if n_records == 0:
            return df

        # Original attacks
        if attack_type == 'cryptomining':
            df.loc[mask, 'cpu_util'] = np.random.uniform(*config['cpu_range'], n_records)
            df.loc[mask, 'ram_util'] = np.random.uniform(*config['ram_range'], n_records)

        elif attack_type == 'data_exfiltration':
            df.loc[mask, 'bandwidth_util'] = np.random.uniform(*config['bandwidth_range'], n_records)
            df.loc[mask, 'cpu_util'] = np.random.uniform(*config['cpu_range'], n_records)
            df.loc[mask, 'request_count'] = df.loc[mask, 'request_count'] * 5

        elif attack_type == 'ddos':
            df.loc[mask, 'active_connections'] = np.random.randint(*config['connections_range'], n_records)
            df.loc[mask, 'request_count'] = np.random.randint(*config['request_range'], n_records)
            df.loc[mask, 'error_count'] = df.loc[mask, 'error_count'] * 10

        elif attack_type == 'insider_threat':
            df.loc[mask, 'disk_io_util'] = np.random.uniform(*config['disk_io_range'], n_records)

        elif attack_type == 'resource_exhaustion':
            indices = df.loc[mask].index
            for i, idx in enumerate(indices):
                progress = i / max(1, len(indices) - 1)
                df.loc[idx, 'cpu_util'] = config['cpu_range'][0] + progress * (config['cpu_range'][1] - config['cpu_range'][0])
                df.loc[idx, 'ram_util'] = config['ram_range'][0] + progress * (config['ram_range'][1] - config['ram_range'][0])

        # New sophisticated attacks
        elif attack_type == 'slowloris':
            df.loc[mask, 'active_connections'] = np.random.randint(*config['connections_range'], n_records)
            df.loc[mask, 'bandwidth_util'] = np.random.uniform(*config['bandwidth_range'], n_records)
            df.loc[mask, 'request_count'] = np.random.randint(*config['request_range'], n_records)

        elif attack_type == 'ransomware':
            df.loc[mask, 'disk_io_util'] = np.random.uniform(*config['disk_io_range'], n_records)
            df.loc[mask, 'cpu_util'] = np.random.uniform(*config['cpu_range'], n_records)
            df.loc[mask, 'ram_util'] = np.random.uniform(*config['ram_range'], n_records)
            df.loc[mask, 'error_count'] = df.loc[mask, 'error_count'] * 3

        elif attack_type == 'lateral_movement':
            df.loc[mask, 'bandwidth_util'] = np.random.uniform(*config['bandwidth_range'], n_records)
            df.loc[mask, 'active_connections'] = np.random.randint(*config['connections_range'], n_records)

        elif attack_type == 'privilege_escalation':
            df.loc[mask, 'cpu_util'] = np.random.uniform(*config['cpu_range'], n_records)
            df.loc[mask, 'ram_util'] = np.random.uniform(*config['ram_range'], n_records)
            df.loc[mask, 'disk_io_util'] = np.random.uniform(*config['disk_io_range'], n_records)

        elif attack_type == 'covert_channel':
            # Periodic low-bandwidth pattern
            indices = df.loc[mask].index
            for i, idx in enumerate(indices):
                phase = np.sin(i * 0.5) * 0.5 + 0.5
                df.loc[idx, 'bandwidth_util'] = config['bandwidth_range'][0] + phase * (config['bandwidth_range'][1] - config['bandwidth_range'][0])
                df.loc[idx, 'cpu_util'] = config['cpu_range'][0] + phase * (config['cpu_range'][1] - config['cpu_range'][0])

        elif attack_type == 'vm_escape':
            df.loc[mask, 'cpu_util'] = np.random.uniform(*config['cpu_range'], n_records)
            df.loc[mask, 'ram_util'] = np.random.uniform(*config['ram_range'], n_records)
            df.loc[mask, 'disk_io_util'] = np.random.uniform(*config['disk_io_range'], n_records)
            df.loc[mask, 'error_count'] = df.loc[mask, 'error_count'] * 5

        elif attack_type == 'botnet_c2':
            # Periodic beaconing pattern
            indices = df.loc[mask].index
            for i, idx in enumerate(indices):
                if i % 5 == 0:  # Beacon every 5th record
                    df.loc[idx, 'bandwidth_util'] = np.random.uniform(*config['bandwidth_range'])
                    df.loc[idx, 'active_connections'] = np.random.randint(*config['connections_range'])
                    df.loc[idx, 'request_count'] = np.random.randint(*config['request_range'])

        elif attack_type == 'memory_scraping':
            df.loc[mask, 'ram_util'] = np.random.uniform(*config['ram_range'], n_records)
            df.loc[mask, 'cpu_util'] = np.random.uniform(*config['cpu_range'], n_records)
            df.loc[mask, 'disk_io_util'] = np.random.uniform(*config['disk_io_range'], n_records)

        return df
    
    def inject_multiple_attacks(
        self,
        df: pd.DataFrame,
        attack_types: List[str] = None,
        injection_ratio: float = 0.05
    ) -> pd.DataFrame:
        """Inject multiple attack types."""
        if attack_types is None:
            attack_types = list(self.ATTACK_TYPES.keys())

        for attack_type in attack_types:
            df = self.inject_attack(
                df, attack_type,
                injection_ratio=injection_ratio / len(attack_types),
                duration_windows=np.random.randint(5, 20)
            )

        return df


def evaluate_robustness(
    detector,
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_adversarial: np.ndarray,
    y_adversarial: np.ndarray,
    attack_types: np.ndarray = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate detector robustness against different attack types.

    Args:
        detector: Trained anomaly detector
        X_clean: Clean test data
        y_clean: Clean labels
        X_adversarial: Adversarial test data
        y_adversarial: Adversarial labels
        attack_types: Array of attack type labels

    Returns:
        Dict with robustness metrics per attack type
    """
    from src.evaluation.metrics import compute_metrics

    results = {}

    # Overall performance on clean data
    y_pred_clean = detector.predict(X_clean)
    y_scores_clean = detector.score_samples(X_clean)
    results['clean'] = compute_metrics(y_clean, y_pred_clean, y_scores_clean)

    # Overall performance on adversarial data
    y_pred_adv = detector.predict(X_adversarial)
    y_scores_adv = detector.score_samples(X_adversarial)
    results['adversarial'] = compute_metrics(y_adversarial, y_pred_adv, y_scores_adv)

    # Per-attack-type performance
    if attack_types is not None:
        # Convert to string and handle NaN
        attack_types_str = pd.Series(attack_types).fillna('normal').astype(str).values
        unique_attacks = [a for a in np.unique(attack_types_str) if a not in ['normal', 'nan', 'None']]

        for attack in unique_attacks:
            mask = attack_types_str == attack
            if mask.sum() > 0:
                y_true_attack = y_adversarial[mask]
                y_pred_attack = y_pred_adv[mask]
                y_scores_attack = y_scores_adv[mask]
                results[f'attack_{attack}'] = compute_metrics(
                    y_true_attack, y_pred_attack, y_scores_attack
                )

    return results


def print_robustness_report(results: Dict[str, Dict[str, float]]):
    """Print formatted robustness report."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION REPORT")
    print("=" * 70)

    print(f"\n{'Scenario':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-" * 70)

    for scenario, metrics in results.items():
        print(f"{scenario:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics.get('auc_roc', 0):>10.4f}")

    print("=" * 70)

