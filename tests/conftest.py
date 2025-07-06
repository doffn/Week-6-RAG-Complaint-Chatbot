"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil


@pytest.fixture(scope="session")
def sample_complaint_data():
    """Create sample complaint data for testing"""
    np.random.seed(42)
    
    products = ['Credit card', 'Personal loan', 'Buy Now, Pay Later (BNPL)', 'Savings account', 'Money transfers']
    issues = ['Billing dispute', 'Fraud', 'Customer service', 'Payment issues', 'Account access']
    companies = ['Bank A', 'Bank B', 'Bank C']
    states = ['CA', 'NY', 'TX', 'FL', 'IL']
    
    narratives = [
        "I was charged an unauthorized fee on my credit card without any notification.",
        "The personal loan terms were not clearly explained during application.",
        "My BNPL payment failed multiple times despite having sufficient funds.",
        "I cannot access my savings account online due to system errors.",
        "Money transfer has been pending for over 5 business days.",
        "Customer service was unhelpful when I called about fraudulent charges.",
        "Interest rate increased without proper notice being provided.",
        "Late fees were charged even though payment was made on time.",
        "Account was closed without notification and funds were held.",
        "Transfer was cancelled but fees were not refunded."
    ]
    
    n_samples = 100
    
    data = pd.DataFrame({
        'Date received': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'Product': np.random.choice(products, n_samples),
        'Issue': np.random.choice(issues, n_samples),
        'Consumer complaint narrative': np.random.choice(narratives, n_samples),
        'Company': np.random.choice(companies, n_samples),
        'State': np.random.choice(states, n_samples),
        'ZIP code': np.random.randint(10000, 99999, n_samples),
        'Submitted via': np.random.choice(['Web', 'Phone', 'Email'], n_samples)
    })
    
    return data


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing"""
    return np.random.rand(10, 384)  # 10 samples, 384 dimensions


@pytest.fixture
def mock_chunks():
    """Create mock text chunks for testing"""
    return [
        {
            'id': f'{i}_0',
            'text': f'Sample complaint text {i}',
            'complaint_id': i,
            'product': 'Credit card',
            'issue': 'Billing',
            'company': 'Bank A',
            'date_received': '2023-01-01',
            'chunk_index': 0,
            'original_length': 100,
            'chunk_length': 50
        }
        for i in range(5)
    ]


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during testing"""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ['integration', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
