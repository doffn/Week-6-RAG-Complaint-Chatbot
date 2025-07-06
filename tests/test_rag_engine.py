"""
Tests for RAG engine module
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
from src.rag_engine import RAGEngine


class TestRAGEngine:
    
    @pytest.fixture
    def temp_vector_store(self):
        """Create temporary vector store directory with config"""
        temp_dir = tempfile.mkdtemp()
        
        # Create a mock config file
        config = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'collection_name': 'complaint_embeddings'
        }
        
        import json
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.rag_engine.SentenceTransformer')
    @patch('src.rag_engine.chromadb.PersistentClient')
    def test_initialization(self, mock_chroma, mock_transformer, temp_vector_store):
        """Test RAGEngine initialization"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        engine = RAGEngine(vector_store_path=temp_vector_store)
        
        assert engine.vector_store_path == temp_vector_store
        assert engine.max_context_length == 2000
        assert engine.max_response_length == 500
        
        # Verify initialization calls
        mock_transformer.assert_called_once()
        mock_chroma.assert_called_once()
    
    @patch('src.rag_engine.SentenceTransformer')
    @patch('src.rag_engine.chromadb.PersistentClient')
    def test_retrieve_relevant_chunks(self, mock_chroma, mock_transformer, temp_vector_store):
        """Test chunk retrieval functionality"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['0_0', '1_0']],
            'documents': [['Sample complaint 1', 'Sample complaint 2']],
            'metadatas': [[
                {'product': 'Credit card', 'issue': 'Billing'},
                {'product': 'Personal loan', 'issue': 'Terms'}
            ]],
            'distances': [[0.1, 0.2]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        engine = RAGEngine(vector_store_path=temp_vector_store)
        
        chunks = engine.retrieve_relevant_chunks("billing issues", n_results=2)
        
        assert len(chunks) == 2
        assert chunks[0]['similarity_score'] > chunks[1]['similarity_score']
        assert 'text' in chunks[0]
        assert 'metadata' in chunks[0]
    
    def test_create_context(self, temp_vector_store):
        """Test context creation from chunks"""
        with patch('src.rag_engine.SentenceTransformer'), \
             patch('src.rag_engine.chromadb.PersistentClient'):
            
            engine = RAGEngine(vector_store_path=temp_vector_store)
            
            chunks = [
                {
                    'text': 'Sample complaint about billing',
                    'metadata': {'product': 'Credit card', 'issue': 'Billing'}
                },
                {
                    'text': 'Another complaint about fees',
                    'metadata': {'product': 'Savings account', 'issue': 'Fees'}
                }
            ]
            
            context = engine.create_context(chunks)
            
            assert 'Source 1' in context
            assert 'Source 2' in context
            assert 'Credit card' in context
            assert 'Savings account' in context
    
    def test_create_prompt(self, temp_vector_store):
        """Test prompt creation"""
        with patch('src.rag_engine.SentenceTransformer'), \
             patch('src.rag_engine.chromadb.PersistentClient'):
            
            engine = RAGEngine(vector_store_path=temp_vector_store)
            
            chunks = [
                {
                    'text': 'Sample complaint',
                    'metadata': {'product': 'Credit card', 'issue': 'Billing'}
                }
            ]
            
            prompt = engine.create_prompt("What are billing issues?", chunks)
            
            assert "What are billing issues?" in prompt
            assert "Sample complaint" in prompt
            assert "financial analyst assistant" in prompt.lower()
    
    def test_generate_fallback_response(self, temp_vector_store):
        """Test fallback response generation"""
        with patch('src.rag_engine.SentenceTransformer'), \
             patch('src.rag_engine.chromadb.PersistentClient'):
            
            engine = RAGEngine(vector_store_path=temp_vector_store)
            
            # Create a prompt with context
            prompt = """
            CONTEXT:
            [Source 1 - Credit card]: I was charged unauthorized fees for billing issues.
            [Source 2 - Personal loan]: Customer service was unhelpful with payment problems.
            
            QUESTION: What are the main issues?
            
            INSTRUCTIONS:
            Answer based on context.
            """
            
            response = engine._generate_fallback_response(prompt)
            
            assert "2 relevant complaint" in response
            assert "billing" in response.lower() or "payment" in response.lower()
    
    @patch('src.rag_engine.SentenceTransformer')
    @patch('src.rag_engine.chromadb.PersistentClient')
    def test_query_complete_pipeline(self, mock_chroma, mock_transformer, temp_vector_store):
        """Test complete query pipeline"""
        # Mock the transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['0_0']],
            'documents': [['Sample complaint about billing issues']],
            'metadatas': [[{'product': 'Credit card', 'issue': 'Billing', 'company': 'Bank A'}]],
            'distances': [[0.1]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        engine = RAGEngine(vector_store_path=temp_vector_store)
        
        result = engine.query("What are billing issues?")
        
        assert 'question' in result
        assert 'answer' in result
        assert 'sources' in result
        assert len(result['sources']) > 0
        assert result['sources'][0]['product'] == 'Credit card'
    
    def test_query_no_results(self, temp_vector_store):
        """Test query with no results"""
        with patch('src.rag_engine.SentenceTransformer'), \
             patch('src.rag_engine.chromadb.PersistentClient') as mock_chroma:
            
            # Mock empty results
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            mock_client.get_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client
            
            engine = RAGEngine(vector_store_path=temp_vector_store)
            
            result = engine.query("Nonexistent query")
            
            assert "couldn't find any relevant" in result['answer']
            assert len(result['sources']) == 0
    
    def test_evaluate_system(self, temp_vector_store):
        """Test system evaluation"""
        with patch('src.rag_engine.SentenceTransformer'), \
             patch('src.rag_engine.chromadb.PersistentClient'):
            
            engine = RAGEngine(vector_store_path=temp_vector_store)
            
            # Mock the query method to return consistent results
            def mock_query(question, product_filter=None):
                return {
                    'question': question,
                    'answer': 'Sample answer about billing issues',
                    'sources': [
                        {
                            'product': 'Credit card',
                            'similarity_score': 0.8,
                            'text': 'Sample source text'
                        }
                    ],
                    'product_filter': product_filter
                }
            
            engine.query = mock_query
            
            test_questions = [
                {'question': 'What are billing issues?', 'expected_theme': 'billing'},
                {'question': 'What are fraud issues?', 'expected_theme': 'fraud'}
            ]
            
            results_df = engine.evaluate_system(test_questions)
            
            assert len(results_df) == 2
            assert 'Question' in results_df.columns
            assert 'Generated Answer' in results_df.columns
            assert 'Number of Sources' in results_df.columns


if __name__ == '__main__':
    pytest.main([__file__])
