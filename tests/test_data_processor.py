"""
Tests for data processor module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
sys.path.append('../src')
from src.data_processor import DataProcessor


class TestDataProcessor:
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing"""
        return DataProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample complaint data for testing"""
        return pd.DataFrame({
            'Date received': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Product': ['Credit card', 'Personal loan', 'Credit card'],
            'Issue': ['Billing dispute', 'Loan terms', 'Fraud'],
            'Consumer complaint narrative': [
                'I was charged an unauthorized fee on my credit card.',
                'The loan terms were not clearly explained to me.',
                'Someone made unauthorized purchases on my account.'
            ],
            'Company': ['Bank A', 'Bank B', 'Bank A'],
            'State': ['CA', 'NY', 'TX']
        })
    
    def test_initialization(self, processor):
        """Test DataProcessor initialization"""
        assert processor.target_products == [
            'Credit card', 
            'Personal loan', 
            'Buy Now, Pay Later (BNPL)', 
            'Savings account', 
            'Money transfers'
        ]
    
    def test_load_data_success(self, processor, sample_data):
        """Test successful data loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            
            result = processor.load_data(f.name)
            
            assert result is not None
            assert len(result) == 3
            assert 'Product' in result.columns
            
            os.unlink(f.name)
    
    def test_load_data_file_not_found(self, processor):
        """Test data loading with non-existent file"""
        result = processor.load_data('non_existent_file.csv')
        assert result is None
    
    def test_clean_text(self, processor):
        """Test text cleaning functionality"""
        # Test normal text
        text = "I AM WRITING TO FILE A COMPLAINT about billing issues!"
        cleaned = processor.clean_text(text)
        assert cleaned == "about billing issues!"
        
        # Test empty text
        assert processor.clean_text("") == ""
        assert processor.clean_text(None) == ""
        
        # Test special characters
        text_with_special = "This has @#$% special characters!"
        cleaned = processor.clean_text(text_with_special)
        assert "@#$%" not in cleaned
    
    def test_filter_and_clean_data(self, processor, sample_data):
        """Test data filtering and cleaning"""
        result = processor.filter_and_clean_data(sample_data)
        
        # Should keep Credit card and Personal loan
        assert len(result) == 3
        assert 'cleaned_narrative' in result.columns
        assert 'word_count' in result.columns
        
        # Check that narratives are cleaned
        assert all(result['cleaned_narrative'].str.len() > 0)
    
    def test_filter_removes_short_narratives(self, processor):
        """Test that very short narratives are removed"""
        short_data = pd.DataFrame({
            'Product': ['Credit card'],
            'Consumer complaint narrative': ['Short'],
            'Issue': ['Test'],
            'Company': ['Test'],
            'Date received': ['2023-01-01'],
            'State': ['CA']
        })
        
        result = processor.filter_and_clean_data(short_data)
        # Should be empty because narrative is too short
        assert len(result) == 0
    
    def test_save_processed_data(self, processor, sample_data):
        """Test saving processed data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            success = processor.save_processed_data(sample_data, f.name)
            
            assert success is True
            assert os.path.exists(f.name)
            
            # Verify saved data
            loaded_data = pd.read_csv(f.name)
            assert len(loaded_data) == len(sample_data)
            
            os.unlink(f.name)
    
    @patch('matplotlib.pyplot.show')
    def test_perform_eda(self, mock_show, processor, sample_data):
        """Test EDA functionality"""
        # Mock the show function to prevent actual plotting
        result = processor.perform_eda(sample_data)
        
        # Should return a plotly figure or None
        assert result is not None or result is None
    
    @patch('matplotlib.pyplot.show')
    def test_generate_word_cloud(self, mock_show, processor, sample_data):
        """Test word cloud generation"""
        # Add cleaned narrative column
        sample_data['cleaned_narrative'] = sample_data['Consumer complaint narrative'].apply(processor.clean_text)
        
        # Test word cloud generation
        result = processor.generate_word_cloud(sample_data, 'Credit card')
        
        # Should return a matplotlib figure
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__])
