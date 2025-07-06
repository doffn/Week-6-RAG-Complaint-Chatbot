"""
Notebook 2: Text Chunking, Embedding, and Vector Store Creation
This script creates embeddings and builds the vector store for the RAG system.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('../src')
from src.embedding_engine import EmbeddingEngine

def main():
    print("=== EMBEDDING AND VECTOR STORE CREATION ===\n")
    
    # Check if processed data exists
    data_path = '../data/filtered_complaints.csv'
    if not os.path.exists(data_path):
        print("❌ Processed data not found. Please run data preprocessing first.")
        print("Expected file: ../data/filtered_complaints.csv")
        return
    
    # Load processed data
    print("Loading processed complaint data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} complaint records")
    
    # Initialize embedding engine
    print("\nInitializing embedding engine...")
    embedding_engine = EmbeddingEngine(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        vector_store_path='../vector_store'
    )
    
    # Display chunking strategy
    print(f"\n=== CHUNKING STRATEGY ===")
    print(f"Chunk size: {embedding_engine.chunk_size} characters")
    print(f"Chunk overlap: {embedding_engine.chunk_overlap} characters")
    print(f"Embedding model: {embedding_engine.model_name}")
    
    # Run the complete embedding pipeline
    try:
        embedding_engine.process_full_pipeline(df)
        
        # Display final statistics
        stats = embedding_engine.get_collection_stats()
        if stats:
            print(f"\n=== VECTOR STORE STATISTICS ===")
            print(f"Total chunks created: {stats['total_chunks']:,}")
            print(f"Average chunks per complaint: {stats['total_chunks']/len(df):.1f}")
            
            if 'product_distribution' in stats:
                print("\nChunk distribution by product:")
                for product, count in stats['product_distribution'].items():
                    print(f"  - {product}: {count:,}")
        
        # Test the search functionality
        print(f"\n=== TESTING SEARCH FUNCTIONALITY ===")
        test_queries = [
            "billing issues with credit cards",
            "unauthorized transactions",
            "customer service problems",
            "payment failures"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            results = embedding_engine.search_similar(query, n_results=3)
            
            if results['results']:
                print(f"Found {len(results['results'])} relevant chunks:")
                for i, result in enumerate(results['results'], 1):
                    print(f"  {i}. Similarity: {result['similarity_score']:.3f} | Product: {result['metadata']['product']}")
                    print(f"     Text: {result['text'][:100]}...")
            else:
                print("No results found")
        
        print(f"\n✅ Embedding pipeline completed successfully!")
        print(f"Vector store saved to: ../vector_store/")
        
    except Exception as e:
        print(f"❌ Error in embedding pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
