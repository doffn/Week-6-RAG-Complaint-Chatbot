import os
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import json
import pandas as pd

class RAGEngine:
    def __init__(self, vector_store_path='./vector_store', 
                 model_name='microsoft/DialoGPT-medium'):
        """
        Initialize RAG engine with vector store and language model
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        
        # Components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.llm_pipeline = None
        
        # Configuration
        self.max_context_length = 2000
        self.max_response_length = 500
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        print("Initializing RAG engine...")
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✓ Loaded embedding model")
        except Exception as e:
            print(f"✗ Error loading embedding model: {e}")
            raise
        
        # Load vector store
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.vector_store_path)
            self.collection = self.chroma_client.get_collection(name="complaint_embeddings")
            print("✓ Loaded vector store")
        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            raise
        
        # Initialize LLM pipeline
        try:
            # Use a more suitable model for text generation
            device = 0 if torch.cuda.is_available() else -1
            self.llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",  # Smaller model for better compatibility
                device=device,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            print("✓ Loaded language model")
        except Exception as e:
            print(f"Warning: Could not load advanced LLM, using fallback: {e}")
            # Fallback to a simpler approach
            self.llm_pipeline = None
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5, 
                               product_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant complaint chunks for a given query
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            # Prepare filter
            where_clause = None
            if product_filter and product_filter != "All Products":
                where_clause = {"product": product_filter}
            
            # Search vector store
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            retrieved_chunks = []
            for i in range(len(results['ids'][0])):
                chunk = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],
                    'distance': results['distances'][0][i]
                }
                retrieved_chunks.append(chunk)
            
            return retrieved_chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def create_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Create context string from retrieved chunks
        """
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = f"[Source {i+1} - {chunk['metadata']['product']}]: {chunk['text']}"
            
            if current_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a well-structured prompt for the LLM
        """
        prompt_template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to analyze customer complaints and provide helpful insights.

Based on the following customer complaint excerpts, please answer the user's question. Use only the information provided in the context. If the context doesn't contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, concise answer based on the context
- Highlight key patterns or trends if relevant
- If specific numbers or statistics are mentioned, include them
- If the context is insufficient, say "Based on the available complaints, I don't have enough information to fully answer this question."

Answer:"""
        
        return prompt_template.format(context=context, question=query)
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using the language model
        """
        if self.llm_pipeline is None:
            # Fallback to rule-based response generation
            return self._generate_fallback_response(prompt)
        
        try:
            # Generate response using the LLM
            response = self.llm_pipeline(
                prompt,
                max_length=len(prompt) + self.max_response_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I apologize, but I couldn't generate a proper response based on the available information."
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a simple rule-based response when LLM is not available
        """
        # Extract context and question from prompt
        if "Context:" in prompt and "Question:" in prompt:
            context_start = prompt.find("Context:") + len("Context:")
            question_start = prompt.find("Question:") + len("Question:")
            
            context = prompt[context_start:prompt.find("Question:")].strip()
            question = prompt[question_start:prompt.find("Instructions:")].strip()
            
            # Simple analysis based on context
            if context:
                # Count sources
                source_count = context.count("[Source")
                
                # Extract products mentioned
                products = []
                if "Credit card" in context:
                    products.append("Credit card")
                if "Personal loan" in context:
                    products.append("Personal loan")
                if "BNPL" in context:
                    products.append("BNPL")
                if "Savings account" in context:
                    products.append("Savings account")
                if "Money transfers" in context:
                    products.append("Money transfers")
                
                response = f"Based on {source_count} relevant complaint(s)"
                if products:
                    response += f" related to {', '.join(products)}"
                response += f", here are the key insights:\n\n"
                
                # Extract key themes
                if "billing" in context.lower() or "charge" in context.lower():
                    response += "• Billing and charging issues are prominent concerns\n"
                if "customer service" in context.lower() or "support" in context.lower():
                    response += "• Customer service quality is a recurring theme\n"
                if "fraud" in context.lower() or "unauthorized" in context.lower():
                    response += "• Fraud and unauthorized transactions are reported\n"
                if "payment" in context.lower():
                    response += "• Payment-related issues are frequently mentioned\n"
                
                response += f"\nThe complaints show various customer concerns that may require attention from the relevant product teams."
                
                return response
            else:
                return "I don't have enough relevant complaint information to answer this question."
        
        return "I apologize, but I couldn't process your question properly."
    
    def query(self, question: str, product_filter: Optional[str] = None, 
              n_results: int = 5) -> Dict[str, Any]:
        """
        Main query method that combines retrieval and generation
        """
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retrieve_relevant_chunks(
                question, n_results=n_results, product_filter=product_filter
            )
            
            if not retrieved_chunks:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant complaint information to answer your question.",
                    'sources': [],
                    'context_used': "",
                    'product_filter': product_filter
                }
            
            # Step 2: Create context
            context = self.create_context(retrieved_chunks)
            
            # Step 3: Create prompt
            prompt = self.create_prompt(question, context)
            
            # Step 4: Generate response
            answer = self.generate_response(prompt)
            
            # Step 5: Format sources for display
            sources = []
            for i, chunk in enumerate(retrieved_chunks):
                source = {
                    'index': i + 1,
                    'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'full_text': chunk['text'],
                    'product': chunk['metadata']['product'],
                    'issue': chunk['metadata']['issue'],
                    'similarity_score': round(chunk['similarity_score'], 3),
                    'company': chunk['metadata'].get('company', 'Unknown')
                }
                sources.append(source)
            
            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'context_used': context,
                'product_filter': product_filter,
                'num_sources': len(sources)
            }
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {
                'question': question,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'context_used': "",
                'product_filter': product_filter
            }
    
    def evaluate_system(self, test_questions: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate the RAG system with a set of test questions
        """
        print("Evaluating RAG system...")
        
        evaluation_results = []
        
        for i, test_case in enumerate(test_questions):
            question = test_case['question']
            expected_theme = test_case.get('expected_theme', '')
            product_filter = test_case.get('product_filter', None)
            
            print(f"Evaluating question {i+1}/{len(test_questions)}: {question[:50]}...")
            
            # Get RAG response
            result = self.query(question, product_filter=product_filter)
            
            # Simple evaluation metrics
            has_sources = len(result['sources']) > 0
            answer_length = len(result['answer'])
            mentions_product = any(product.lower() in result['answer'].lower() 
                                 for product in ['credit card', 'loan', 'bnpl', 'savings', 'transfer'])
            
            evaluation_results.append({
                'Question': question,
                'Product Filter': product_filter or 'All',
                'Generated Answer': result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'],
                'Full Answer': result['answer'],
                'Number of Sources': len(result['sources']),
                'Has Sources': has_sources,
                'Answer Length': answer_length,
                'Mentions Product': mentions_product,
                'Top Source Product': result['sources'][0]['product'] if result['sources'] else 'None',
                'Top Source Similarity': result['sources'][0]['similarity_score'] if result['sources'] else 0,
                'Expected Theme': expected_theme
            })
        
        return pd.DataFrame(evaluation_results)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system
        """
        try:
            collection_count = self.collection.count()
            
            # Get sample for analysis
            sample_results = self.collection.get(limit=100, include=['metadatas'])
            
            if sample_results['metadatas']:
                products = [meta['product'] for meta in sample_results['metadatas']]
                product_dist = pd.Series(products).value_counts()
                
                return {
                    'total_chunks': collection_count,
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'llm_model': self.model_name,
                    'product_distribution': product_dist.to_dict(),
                    'max_context_length': self.max_context_length,
                    'vector_store_path': self.vector_store_path
                }
            
            return {
                'total_chunks': collection_count,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'llm_model': self.model_name
            }
            
        except Exception as e:
            print(f"Error getting system stats: {e}")
            return {}
