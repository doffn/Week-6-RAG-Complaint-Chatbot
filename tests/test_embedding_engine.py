"""
Tests for embedding engine module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
sys.path.append('../src')
from src.embedding_engine import EmbeddingEngine


class TestEmbeddingEngine:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample processed data for testing"""
        return pd.DataFrame({
            'Product': ['Credit card', 'Personal loan', 'Credit card'],
            'Issue': ['Billing dispute', 'Loan terms', 'Fraud'],
            'cleaned_narrative': [
                'I was charged an unauthorized fee on my credit card account without any prior notification.',
                'The personal loan terms were not clearly explained during the application process.',
                'Someone made unauthorized purchases on my credit card account without my permission.'
            ],
            'Company': ['Bank A', 'Bank B', 'Bank A'],
            'Date received': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'State': ['CA', 'NY', 'TX']
        })
    
    @pytest.fixture
    def temp_vector_store(self):
        """Create temporary vector store directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.embedding_engine.SentenceTransformer')
    @patch('src.embedding_engine.chromadb.PersistentClient')
    def test_initialization(self, mock_chroma, mock_transformer, temp_vector_store):
        """Test EmbeddingEngine initialization"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        
        engine = EmbeddingEngine(vector_store_path=temp_vector_store)
        
        assert engine.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
        assert engine.vector_store_path == temp_vector_store
        assert engine.chunk_size == 500
        assert engine.chunk_overlap == 50
        
        # Verify initialization calls
        mock_transformer.assert_called_once()
        mock_chroma.assert_called_once()
    
    def test_create_text_chunks(self, sample_data, temp_vector_store):
        """Test text chunking functionality"""
        with patch('src.embedding_engine.SentenceTransformer'), \
             patch('src.embedding_engine.chromadb.PersistentClient'):
            
            engine = EmbeddingEngine(vector_store_path=temp_vector_store)
            chunks = engine.create_text_chunks(sample_data)
            
            # Should create chunks for each narrative
            assert len(chunks) >= len(sample_data)
            
            # Check chunk structure
            for chunk in chunks:
                assert 'id' in chunk
                assert 'text' in chunk
                assert 'product' in chunk
                assert 'issue' in chunk
                assert 'chunk_length' in chunk
                assert len(chunk['text']) > 20  # Minimum chunk length
    
    @patch('src.embedding_engine.SentenceTransformer')
    def test_create_embeddings(self, mock_transformer, temp_vector_store):
        """Test embedding creation"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384)  # 3 chunks, 384 dimensions
        mock_transformer.return_value = mock_model
        
        with patch('src.embedding_engine.chromadb.PersistentClient'):
            engine = EmbeddingEngine(vector_store_path=temp_vector_store)
            
            # Create sample chunks
            chunks = [
                {'text': 'Sample text 1'},
                {'text': 'Sample text 2'},
                {'text': 'Sample text 3'}
            ]
            
            embeddings = engine.create_embeddings(chunks)
            
            assert embeddings.shape == (3, 384)
            assert not np.isnan(embeddings).any()
    
    @patch('src.embedding_engine.SentenceTransformer')
    @patch('src.embedding_engine.chromadb.PersistentClient')
    def test_create_vector_store(self, mock_chroma, mock_transformer, temp_vector_store):
        """Test vector store creation"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        engine = EmbeddingEngine(vector_store_path=temp_vector_store)
        
        # Sample data
        chunks = [
            {
                'id': '0_0',
                'text': 'Sample complaint text',
                'complaint_id': 0,
                'product': 'Credit card',
                'issue': 'Billing',
                'company': 'Bank A',
                'date_received': '2023-01-01',
                'chunk_index': 0,
                'original_length': 100,
                'chunk_length': 50
            }
        ]
        embeddings = np.random.rand(1, 384)
        
        engine.create_vector_store(chunks, embeddings)
        
        # Verify collection creation and data insertion
        mock_client.create_collection.assert_called_once()
        mock_collection.add.assert_called()
    
    @patch('src.embedding_engine.SentenceTransformer')
    @patch('src.embedding_engine.chromadb.PersistentClient')
    def test_search_similar(self, mock_chroma, mock_transformer, temp_vector_store):
        """Test similarity search"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['0_0']],
            'documents': [['Sample complaint text']],
            'metadatas': [[{'product': 'Credit card', 'issue': 'Billing'}]],
            'distances': [[0.1]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        engine = EmbeddingEngine(vector_store_path=temp_vector_store)
        engine.collection = mock_collection
        
        results = engine.search_similar("billing issues", n_results=1)
        
        assert 'query' in results
        assert 'results' in results
        assert len(results['results']) == 1
        assert results['results'][0]['similarity_score'] > 0
    
    def test_process_full_pipeline(self, sample_data, temp_vector_store):
        """Test the complete embedding pipeline"""
        with patch('src.embedding_engine.SentenceTransformer') as mock_transformer, \
             patch('src.embedding_engine.chromadb.PersistentClient') as mock_chroma:
            
            # Mock the transformer
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(5, 384)  # Assume 5 chunks
            mock_transformer.return_value = mock_model
            
            # Mock ChromaDB
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            mock_client.create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client
            
            engine = EmbeddingEngine(vector_store_path=temp_vector_store)
            
            # This should run without errors
            engine.process_full_pipeline(sample_data)
            
            # Verify that key methods were called
            mock_model.encode.assert_called()
            mock_collection.add.assert_called()


if __name__ == '__main__':
    pytest.main([__file__])
