"""
Unit tests for Adversarial Testing module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.adversarial_testing import AdversarialInjector


@pytest.fixture
def sample_cloud_data():
    """Create sample cloud data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'node_id': np.random.choice(['node_001', 'node_002', 'node_003'], n_samples),
        'window_idx': np.arange(n_samples),
        'cpu_util': np.random.uniform(0.1, 0.5, n_samples),
        'ram_util': np.random.uniform(0.2, 0.5, n_samples),
        'disk_io_util': np.random.uniform(0.1, 0.4, n_samples),
        'bandwidth_util': np.random.uniform(0.1, 0.3, n_samples),
        'active_connections': np.random.randint(1, 20, n_samples),
        'request_count': np.random.randint(10, 100, n_samples),
        'error_count': np.random.randint(0, 5, n_samples),
        'label': np.zeros(n_samples, dtype=int),
    }
    return pd.DataFrame(data)


@pytest.fixture
def injector():
    """Create AdversarialInjector instance."""
    return AdversarialInjector(seed=42)


class TestAdversarialInjector:
    """Test cases for AdversarialInjector class."""
    
    def test_initialization(self, injector):
        """Test injector initialization."""
        assert injector is not None
        assert len(injector.ATTACK_TYPES) > 0
    
    def test_attack_types_defined(self, injector):
        """Test that all attack types are properly defined."""
        expected_attacks = [
            'cryptomining', 'data_exfiltration', 'ddos', 
            'insider_threat', 'resource_exhaustion'
        ]
        
        for attack in expected_attacks:
            assert attack in injector.ATTACK_TYPES
            assert 'description' in injector.ATTACK_TYPES[attack]
            assert 'pattern' in injector.ATTACK_TYPES[attack]
    
    def test_inject_cryptomining(self, injector, sample_cloud_data):
        """Test cryptomining attack injection."""
        df = sample_cloud_data.copy()
        original_labels = df['label'].sum()
        
        df_attacked = injector.inject_attack(
            df, 'cryptomining', 
            injection_ratio=0.1, 
            duration_windows=10
        )
        
        # Should have more anomalies
        assert df_attacked['label'].sum() > original_labels
        
        # Attacked records should have high CPU
        attacked_mask = df_attacked['label'] == 1
        assert df_attacked.loc[attacked_mask, 'cpu_util'].mean() > 0.8
    
    def test_inject_data_exfiltration(self, injector, sample_cloud_data):
        """Test data exfiltration attack injection."""
        df = sample_cloud_data.copy()
        
        df_attacked = injector.inject_attack(
            df, 'data_exfiltration',
            injection_ratio=0.1,
            duration_windows=10
        )
        
        attacked_mask = df_attacked['label'] == 1
        assert df_attacked.loc[attacked_mask, 'bandwidth_util'].mean() > 0.7
    
    def test_inject_ddos(self, injector, sample_cloud_data):
        """Test DDoS attack injection."""
        df = sample_cloud_data.copy()
        
        df_attacked = injector.inject_attack(
            df, 'ddos',
            injection_ratio=0.1,
            duration_windows=10
        )
        
        attacked_mask = df_attacked['label'] == 1
        assert df_attacked.loc[attacked_mask, 'active_connections'].mean() > 50
    
    def test_inject_multiple_attacks(self, injector, sample_cloud_data):
        """Test multiple attack injection."""
        df = sample_cloud_data.copy()
        
        df_attacked = injector.inject_multiple_attacks(
            df,
            attack_types=['cryptomining', 'ddos'],
            injection_ratio=0.1
        )
        
        # Should have anomalies
        assert df_attacked['label'].sum() > 0
    
    def test_attack_type_column(self, injector, sample_cloud_data):
        """Test that attack_type column is added."""
        df = sample_cloud_data.copy()
        df['attack_type'] = 'normal'
        
        df_attacked = injector.inject_attack(
            df, 'ransomware',
            injection_ratio=0.1,
            duration_windows=10
        )
        
        assert 'attack_type' in df_attacked.columns
        assert (df_attacked['attack_type'] == 'ransomware').sum() > 0
    
    def test_invalid_attack_type(self, injector, sample_cloud_data):
        """Test handling of invalid attack type."""
        with pytest.raises(ValueError):
            injector.inject_attack(sample_cloud_data, 'invalid_attack')
    
    def test_injection_ratio(self, injector, sample_cloud_data):
        """Test that injection ratio is approximately correct."""
        df = sample_cloud_data.copy()
        
        df_attacked = injector.inject_attack(
            df, 'cryptomining',
            injection_ratio=0.1,
            duration_windows=10
        )
        
        # Injection should affect roughly the expected number of records
        n_attacked = df_attacked['label'].sum()
        assert n_attacked > 0
        assert n_attacked < len(df) * 0.5  # Not too many


class TestNewAttackTypes:
    """Test cases for newly added attack types."""
    
    def test_slowloris_attack(self, injector, sample_cloud_data):
        """Test slowloris attack injection."""
        df = sample_cloud_data.copy()
        df_attacked = injector.inject_attack(df, 'slowloris', injection_ratio=0.1, duration_windows=10)
        
        attacked_mask = df_attacked['label'] == 1
        # Slowloris: many connections, low bandwidth
        assert df_attacked.loc[attacked_mask, 'active_connections'].mean() > 100
    
    def test_ransomware_attack(self, injector, sample_cloud_data):
        """Test ransomware attack injection."""
        df = sample_cloud_data.copy()
        df_attacked = injector.inject_attack(df, 'ransomware', injection_ratio=0.1, duration_windows=10)
        
        attacked_mask = df_attacked['label'] == 1
        # Ransomware: high disk I/O
        assert df_attacked.loc[attacked_mask, 'disk_io_util'].mean() > 0.8
    
    def test_vm_escape_attack(self, injector, sample_cloud_data):
        """Test VM escape attack injection."""
        df = sample_cloud_data.copy()
        df_attacked = injector.inject_attack(df, 'vm_escape', injection_ratio=0.1, duration_windows=10)
        
        attacked_mask = df_attacked['label'] == 1
        # VM escape: very high resource usage
        assert df_attacked.loc[attacked_mask, 'cpu_util'].mean() > 0.9

