"""
CloudSim Python Simulator
Generates realistic cloud node resource metrics for anomaly detection research.
Simulates a private cloud environment (VIT-AP use case).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CloudNode:
    """Represents a single cloud node (VM/Host) with resource metrics."""
    
    def __init__(self, node_id: str, node_type: str = "vm", base_load: float = 0.3):
        """
        Initialize a cloud node.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node ('vm', 'host', 'server')
            base_load: Base resource utilization level (0-1)
        """
        self.node_id = node_id
        self.node_type = node_type
        self.base_load = base_load
        
        # Node characteristics (randomized for realism)
        self.cpu_cores = np.random.choice([2, 4, 8, 16])
        self.ram_gb = np.random.choice([4, 8, 16, 32, 64])
        self.bandwidth_mbps = np.random.choice([100, 1000, 10000])
        self.disk_iops = np.random.choice([1000, 5000, 10000, 20000])
        
    def generate_normal_metrics(self, timestamp: datetime, hour_of_day: int) -> Dict:
        """
        Generate normal behavior metrics for a single time window.
        Includes realistic daily patterns (higher load during work hours).
        """
        # Daily pattern: higher load during work hours (9-17)
        if 9 <= hour_of_day <= 17:
            load_multiplier = 1.0 + np.random.uniform(0.1, 0.3)
        elif 0 <= hour_of_day <= 6:
            load_multiplier = 0.3 + np.random.uniform(0, 0.2)
        else:
            load_multiplier = 0.6 + np.random.uniform(0, 0.2)
        
        # Generate metrics with realistic noise
        cpu_util = np.clip(
            self.base_load * load_multiplier + np.random.normal(0, 0.05),
            0.01, 0.95
        )
        ram_util = np.clip(
            self.base_load * load_multiplier * 0.8 + np.random.normal(0, 0.03),
            0.05, 0.95
        )
        bandwidth_util = np.clip(
            self.base_load * load_multiplier * 0.5 + np.random.normal(0, 0.08),
            0.01, 0.90
        )
        disk_io_util = np.clip(
            self.base_load * load_multiplier * 0.4 + np.random.normal(0, 0.05),
            0.01, 0.85
        )
        
        return {
            'timestamp': timestamp,
            'node_id': self.node_id,
            'node_type': self.node_type,
            'cpu_util': cpu_util,
            'ram_util': ram_util,
            'bandwidth_util': bandwidth_util,
            'disk_io_util': disk_io_util,
            'cpu_cores': self.cpu_cores,
            'ram_gb': self.ram_gb,
            'bandwidth_mbps': self.bandwidth_mbps,
            'active_connections': int(np.random.poisson(10 * load_multiplier)),
            'request_count': int(np.random.poisson(50 * load_multiplier)),
            'error_count': int(np.random.poisson(0.5)),
            'label': 0  # 0 = normal
        }


class CloudSimSimulator:
    """Main simulator class for generating cloud environment data."""
    
    def __init__(self, num_nodes: int = 50, seed: int = 42):
        """
        Initialize the cloud simulator.
        
        Args:
            num_nodes: Number of cloud nodes to simulate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.nodes: List[CloudNode] = []
        self._create_nodes()
        
    def _create_nodes(self):
        """Create cloud nodes with varied characteristics."""
        node_types = ['vm'] * int(self.num_nodes * 0.7) + \
                     ['host'] * int(self.num_nodes * 0.2) + \
                     ['server'] * int(self.num_nodes * 0.1)
        
        for i in range(self.num_nodes):
            node_type = node_types[i] if i < len(node_types) else 'vm'
            base_load = np.random.uniform(0.2, 0.5)
            node = CloudNode(
                node_id=f"node_{i:03d}",
                node_type=node_type,
                base_load=base_load
            )
            self.nodes.append(node)
    
    def generate_dataset(
        self,
        duration_hours: int = 48,
        window_size_seconds: int = 60,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate complete dataset for all nodes over specified duration.
        
        Args:
            duration_hours: Total simulation duration in hours
            window_size_seconds: Time window size in seconds
            start_time: Starting timestamp (defaults to now)
            
        Returns:
            DataFrame with all node metrics over time
        """
        if start_time is None:
            start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        total_windows = (duration_hours * 3600) // window_size_seconds
        all_records = []
        
        print(f"Generating {total_windows} time windows for {self.num_nodes} nodes...")
        
        for window_idx in range(total_windows):
            current_time = start_time + timedelta(seconds=window_idx * window_size_seconds)
            hour_of_day = current_time.hour
            
            for node in self.nodes:
                record = node.generate_normal_metrics(current_time, hour_of_day)
                record['window_idx'] = window_idx
                all_records.append(record)
            
            if (window_idx + 1) % 500 == 0:
                print(f"  Generated {window_idx + 1}/{total_windows} windows...")
        
        df = pd.DataFrame(all_records)
        print(f"Generated {len(df)} total records.")
        return df

    def inject_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_ratio: float = 0.05,
        attack_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Inject synthetic anomalies into the dataset.

        Args:
            df: Original dataset
            anomaly_ratio: Fraction of records to make anomalous
            attack_types: List of attack types to inject

        Returns:
            DataFrame with injected anomalies
        """
        if attack_types is None:
            attack_types = ['data_exfiltration', 'resource_abuse', 'insider_misuse']

        df = df.copy()
        n_anomalies = int(len(df) * anomaly_ratio)

        # Select random indices for anomaly injection
        anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

        for idx in anomaly_indices:
            attack_type = np.random.choice(attack_types)
            df = self._inject_single_anomaly(df, idx, attack_type)

        print(f"Injected {n_anomalies} anomalies ({anomaly_ratio*100:.1f}%)")
        return df

    def _inject_single_anomaly(
        self,
        df: pd.DataFrame,
        idx: int,
        attack_type: str
    ) -> pd.DataFrame:
        """Inject a single anomaly at the specified index."""

        if attack_type == 'data_exfiltration':
            # High bandwidth, normal CPU - data being stolen
            df.loc[idx, 'bandwidth_util'] = np.random.uniform(0.85, 0.99)
            df.loc[idx, 'cpu_util'] = df.loc[idx, 'cpu_util'] * 0.8  # Slightly lower
            df.loc[idx, 'request_count'] = int(df.loc[idx, 'request_count'] * 3)

        elif attack_type == 'resource_abuse':
            # Very high CPU and RAM - cryptomining or DoS
            df.loc[idx, 'cpu_util'] = np.random.uniform(0.90, 0.99)
            df.loc[idx, 'ram_util'] = np.random.uniform(0.85, 0.98)
            df.loc[idx, 'disk_io_util'] = np.random.uniform(0.70, 0.95)

        elif attack_type == 'insider_misuse':
            # Unusual access patterns - high disk IO at odd hours
            df.loc[idx, 'disk_io_util'] = np.random.uniform(0.75, 0.95)
            df.loc[idx, 'active_connections'] = int(np.random.uniform(50, 100))
            df.loc[idx, 'error_count'] = int(np.random.uniform(5, 20))

        df.loc[idx, 'label'] = 1  # Mark as anomaly
        df.loc[idx, 'attack_type'] = attack_type

        return df


def generate_cloudsim_data(
    output_path: str = None,
    num_nodes: int = 50,
    duration_hours: int = 48,
    inject_anomalies: bool = True,
    anomaly_ratio: float = 0.05
) -> pd.DataFrame:
    """
    Main function to generate CloudSim data.

    Args:
        output_path: Path to save CSV (optional)
        num_nodes: Number of cloud nodes
        duration_hours: Simulation duration
        inject_anomalies: Whether to inject anomalies
        anomaly_ratio: Fraction of anomalies

    Returns:
        Generated DataFrame
    """
    simulator = CloudSimSimulator(num_nodes=num_nodes, seed=42)
    df = simulator.generate_dataset(duration_hours=duration_hours)

    if inject_anomalies:
        df = simulator.inject_anomalies(df, anomaly_ratio=anomaly_ratio)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    # Generate sample data
    from src.utils.config_loader import get_project_root

    output_path = get_project_root() / "data" / "raw" / "cloudsim_logs.csv"
    df = generate_cloudsim_data(
        output_path=str(output_path),
        num_nodes=50,
        duration_hours=48,
        inject_anomalies=True,
        anomaly_ratio=0.05
    )

    print("\nDataset Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Normal records: {len(df[df['label'] == 0])}")
    print(f"  Anomaly records: {len(df[df['label'] == 1])}")
    print(f"  Columns: {list(df.columns)}")

