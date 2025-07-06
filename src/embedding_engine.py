import numpy as np
import pandas as pd
import langchain
import sentence_transformers
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingEngine:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 vector_store_path='./vector_store'):
        """
        Initialize the embedding engine with specified model and vector store path
        """
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Text chunking parameters
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and vector store"""
        print("Initializing embedding engine...")
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✓ Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"✗ Error loading embedding model: {e}")
            raise
        
        # Initialize ChromaDB
        try:
            # Create vector store directory if it doesn't exist
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=self.vector_store_path)
            print("✓ Initialized ChromaDB client")
        except Exception as e:
            print(f"✗ Error initializing ChromaDB: {e}")
            raise
    
    def create_text_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Split complaint narratives into chunks for better embedding
        """
        print("Creating text chunks...")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = []
        
        for idx, row in df.iterrows():
            narrative = row['cleaned_narrative']
            
            # Split text into chunks
            text_chunks = text_splitter.split_text(narrative)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                if len(chunk.strip()) > 20:  # Only keep meaningful chunks
                    chunk_data = {
                        'id': f"{row.name}_{chunk_idx}",
                        'text': chunk.strip(),
                        'complaint_id': row.name,
                        'product': row['Product'],
                        'issue': row.get('Issue', 'Unknown'),
                        'company': row.get('Company', 'Unknown'),
                        'date_received': str(row.get('Date received', 'Unknown')),
                        'chunk_index': chunk_idx,
                        'original_length': len(narrative),
                        'chunk_length': len(chunk)
                    }
                    chunks.append(chunk_data)
        
        print(f"✓ Created {len(chunks)} text chunks from {len(df)} complaints")
        print(f"  - Average chunks per complaint: {len(chunks)/len(df):.1f}")
        print(f"  - Average chunk length: {np.mean([c['chunk_length'] for c in chunks]):.0f} characters")
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        """
        print("Generating embeddings...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            # Generate embeddings in batches to manage memory
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + len(batch_texts)}/{len(texts)} chunks")
            
            embeddings = np.vstack(embeddings)
            print(f"✓ Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
            
            return embeddings
            
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
            raise
    
    def create_vector_store(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Create and populate ChromaDB vector store
        """
        print("Creating vector store...")
        
        try:
            # Create or get collection
            collection_name = "complaint_embeddings"
            
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(name=collection_name)
                print("  Deleted existing collection")
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Financial complaint embeddings"}
            )
            
            # Prepare data for ChromaDB
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    'complaint_id': str(chunk['complaint_id']),
                    'product': chunk['product'],
                    'issue': chunk['issue'],
                    'company': chunk['company'],
                    'date_received': chunk['date_received'],
                    'chunk_index': chunk['chunk_index'],
                    'original_length': chunk['original_length'],
                    'chunk_length': chunk['chunk_length']
                }
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                
                self.collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx].tolist(),
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Added {end_idx}/{len(ids)} chunks to vector store")
            
            print(f"✓ Successfully created vector store with {len(ids)} embeddings")
            
            # Save chunking parameters
            self._save_config()
            
        except Exception as e:
            print(f"✗ Error creating vector store: {e}")
            raise
    
    def _save_config(self):
        """Save configuration for later use"""
        config = {
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'vector_store_path': self.vector_store_path
        }
        
        config_path = os.path.join(self.vector_store_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Saved configuration to {config_path}")
    
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            collection_name = "complaint_embeddings"
            self.collection = self.chroma_client.get_collection(name=collection_name)
            
            # Load configuration
            config_path = os.path.join(self.vector_store_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.chunk_size = config.get('chunk_size', self.chunk_size)
                    self.chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)
            
            print("✓ Loaded existing vector store")
            return True
            
        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            return False
    
    def search_similar(self, query: str, n_results: int = 5, 
                      product_filter: str = None) -> Dict[str, Any]:
        """
        Search for similar complaint chunks
        """
        if not self.collection:
            raise ValueError("Vector store not initialized. Please create or load a vector store first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            # Prepare where clause for filtering
            where_clause = None
            if product_filter:
                where_clause = {"product": product_filter}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = {
                'query': query,
                'results': []
            }
            
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                }
                formatted_results['results'].append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"✗ Error searching vector store: {e}")
            raise
    
    def get_collection_stats(self):
        """Get statistics about the vector store"""
        if not self.collection:
            return None
        
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])
            
            if sample_results['metadatas']:
                products = [meta['product'] for meta in sample_results['metadatas']]
                product_counts = pd.Series(products).value_counts()
                
                return {
                    'total_chunks': count,
                    'product_distribution': product_counts.to_dict(),
                    'sample_size': len(sample_results['metadatas'])
                }
            
            return {'total_chunks': count}
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return None
    
    def process_full_pipeline(self, df: pd.DataFrame):
        """
        Run the complete embedding pipeline
        """
        print("=== EMBEDDING PIPELINE ===\n")
        
        # Step 1: Create text chunks
        chunks = self.create_text_chunks(df)
        
        # Step 2: Generate embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Step 3: Create vector store
        self.create_vector_store(chunks, embeddings)
        
        # Step 4: Display statistics
        stats = self.get_collection_stats()
        if stats:
            print(f"\n=== VECTOR STORE STATISTICS ===")
            print(f"Total chunks: {stats['total_chunks']:,}")
            if 'product_distribution' in stats:
                print("Product distribution:")
                for product, count in stats['product_distribution'].items():
                    print(f"  - {product}: {count:,}")
        
        print("\n✓ Embedding pipeline completed successfully!")
