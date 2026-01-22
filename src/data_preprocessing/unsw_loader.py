"""
UNSW-NB15 Dataset Loader and Preprocessor
Loads and preprocesses the UNSW-NB15 network intrusion dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# UNSW-NB15 column names (49 features + label)
UNSW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
    'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'attack_cat', 'label'
]

# Features to use for anomaly detection
NUMERIC_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'smeansz', 'dmeansz',
    'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm'
]

CATEGORICAL_FEATURES = ['proto', 'state', 'service']


class UNSWLoader:
    """Loader and preprocessor for UNSW-NB15 dataset."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory containing UNSW-NB15 CSV files
        """
        if data_dir is None:
            from src.utils.config_loader import get_project_root
            data_dir = str(get_project_root() / "data" / "raw")
        self.data_dir = data_dir
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
    def load_data(self, train_file: str = None, test_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and testing datasets.
        
        Args:
            train_file: Path to training CSV
            test_file: Path to testing CSV
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if train_file is None:
            train_file = os.path.join(self.data_dir, "UNSW_NB15_training-set.csv")
        if test_file is None:
            test_file = os.path.join(self.data_dir, "UNSW_NB15_testing-set.csv")
        
        print(f"Loading training data from {train_file}...")
        train_df = pd.read_csv(train_file, low_memory=False)
        
        print(f"Loading testing data from {test_file}...")
        test_df = pd.read_csv(test_file, low_memory=False)
        
        print(f"Loaded {len(train_df)} training and {len(test_df)} testing records.")
        return train_df, test_df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess the dataset.
        
        Args:
            df: Raw DataFrame
            fit: Whether to fit encoders/scalers (True for training data)
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    known = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else 'unknown')
                    if 'unknown' not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(
                            self.label_encoders[col].classes_, 'unknown'
                        )
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Scale numeric features
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def get_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract network behavior features for integration with CloudSim.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with network features per source IP
        """
        # Aggregate by source IP to get per-node network behavior
        agg_features = df.groupby('srcip').agg({
            'sbytes': ['sum', 'mean', 'std'],
            'dbytes': ['sum', 'mean', 'std'],
            'Spkts': ['sum', 'mean'],
            'Dpkts': ['sum', 'mean'],
            'dur': ['mean', 'std'],
            'label': ['sum', 'max']  # Number of attacks, any attack
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['_'.join(col).strip('_') for col in agg_features.columns]
        
        return agg_features


def check_unsw_data_exists(data_dir: str = None) -> bool:
    """Check if UNSW-NB15 data files exist."""
    if data_dir is None:
        from src.utils.config_loader import get_project_root
        data_dir = str(get_project_root() / "data" / "raw")
    
    train_exists = os.path.exists(os.path.join(data_dir, "UNSW_NB15_training-set.csv"))
    test_exists = os.path.exists(os.path.join(data_dir, "UNSW_NB15_testing-set.csv"))
    
    return train_exists and test_exists

