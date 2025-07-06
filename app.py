import gradio as gr
import pandas as pd
import chromadb
import os
import sys
from typing import List, Dict, Any, Optional
import json

# Add src to path
sys.path.append('src')

from src.data_processor import DataProcessor
from src.embedding_engine import EmbeddingEngine
from src.rag_engine import RAGEngine
from src.evaluation import RAGEvaluator

class ComplaintChatbot:
    def __init__(self):
        self.data_processor = None
        self.embedding_engine = None
        self.rag_engine = None
        self.is_initialized = False
        
        # Available products for filtering
        self.products = [
            "All Products",
            "Credit card",
            "Personal loan", 
            "Buy Now, Pay Later (BNPL)",
            "Savings account",
            "Money transfers"
        ]
        
        # Initialize components if vector store exists
        self._try_initialize()
    
    def _try_initialize(self):
        """Try to initialize RAG engine if vector store exists"""
        try:
            if os.path.exists('./vector_store') and os.path.exists('./vector_store/chroma.sqlite3'):
                self.rag_engine = RAGEngine()
                self.is_initialized = True
                print("✓ RAG engine initialized successfully")
            else:
                print("Vector store not found. Please process data first.")
        except Exception as e:
            print(f"Could not initialize RAG engine: {e}")
            self.is_initialized = False
    
    def process_data(self, file_path: str, progress=gr.Progress()) -> str:
        """Process uploaded data file"""
        if not file_path:
            return "Please upload a CSV file first."
        
        try:
            progress(0.1, desc="Loading data...")
            
            # Initialize data processor
            self.data_processor = DataProcessor()
            
            # Load data
            df = self.data_processor.load_data(file_path)
            if df is None:
                return "Error loading data file. Please check the file format."
            
            progress(0.2, desc="Performing EDA...")
            
            # Perform EDA
            self.data_processor.perform_eda(df)
            
            progress(0.4, desc="Filtering and cleaning data...")
            
            # Filter and clean data
            df_filtered = self.data_processor.filter_and_clean_data(df)
            
            if len(df_filtered) == 0:
                return "No data remaining after filtering. Please check your dataset."
            
            progress(0.5, desc="Saving processed data...")
            
            # Save processed data
            os.makedirs('data', exist_ok=True)
            self.data_processor.save_processed_data(df_filtered, 'data/filtered_complaints.csv')
            
            progress(0.6, desc="Creating embeddings...")
            
            # Initialize embedding engine
            self.embedding_engine = EmbeddingEngine()
            
            # Process embedding pipeline
            self.embedding_engine.process_full_pipeline(df_filtered)
            
            progress(0.9, desc="Initializing RAG engine...")
            
            # Initialize RAG engine
            self.rag_engine = RAGEngine()
            self.is_initialized = True
            
            progress(1.0, desc="Complete!")
            
            return f"""✅ Data processing completed successfully!

📊 **Processing Summary:**
- Original records: {len(df):,}
- Filtered records: {len(df_filtered):,}
- Products included: {', '.join(df_filtered['Product'].unique())}
- Average narrative length: {df_filtered['word_count'].mean():.0f} words

🔧 **System Status:**
- Vector store created with embeddings
- RAG engine initialized and ready
- You can now ask questions about the complaints!

💡 **Try asking questions like:**
- "What are the main issues with credit cards?"
- "Why are customers unhappy with BNPL?"
- "What problems do people report with savings accounts?"
"""
            
        except Exception as e:
            return f"❌ Error processing data: {str(e)}"
    
    def chat_response(self, message: str, history: List[List[str]], 
                     product_filter: str) -> tuple:
        """Generate chat response"""
        if not self.is_initialized:
            error_msg = "❌ System not initialized. Please upload and process data first."
            history.append([message, error_msg])
            return history, ""
        
        if not message.strip():
            error_msg = "Please enter a question about customer complaints."
            history.append([message, error_msg])
            return history, ""
        
        try:
            # Get RAG response
            result = self.rag_engine.query(
                message, 
                product_filter=product_filter if product_filter != "All Products" else None,
                n_results=5
            )
            
            # Format response
            response = f"**Answer:** {result['answer']}\n\n"
            
            if result['sources']:
                response += "**📚 Sources:**\n"
                for i, source in enumerate(result['sources'][:3], 1):
                    response += f"\n**Source {i}** ({source['product']}) - Similarity: {source['similarity_score']:.3f}\n"
                    response += f"*{source['text'][:200]}{'...' if len(source['text']) > 200 else ''}*\n"
            else:
                response += "\n*No relevant sources found for this query.*"
            
            history.append([message, response])
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            history.append([message, error_msg])
            return history, ""
    
    def clear_chat(self):
        """Clear chat history"""
        return []
    
    def run_evaluation(self) -> str:
        """Run system evaluation"""
        if not self.is_initialized:
            return "❌ System not initialized. Please process data first."
        
        try:
            evaluator = RAGEvaluator(self.rag_engine)
            report = evaluator.run_comprehensive_evaluation()
            
            # Create summary
            metrics = report['metrics']
            analysis = report['analysis']
            
            summary = f"""# 📊 RAG System Evaluation Report

## Key Metrics
- **Total Questions Evaluated:** {metrics['total_questions']}
- **Source Retrieval Rate:** {metrics['source_retrieval_rate']:.1f}%
- **Average Sources per Question:** {metrics['avg_sources_per_question']:.1f}
- **Average Answer Length:** {metrics['avg_answer_length']:.0f} characters
- **Product Mention Rate:** {metrics['product_mention_rate']:.1f}%
- **Average Similarity Score:** {metrics['avg_top_source_similarity']:.3f}

## ✅ Strengths
"""
            for strength in analysis['strengths']:
                summary += f"- {strength}\n"
            
            summary += "\n## ⚠️ Areas for Improvement\n"
            for weakness in analysis['weaknesses']:
                summary += f"- {weakness}\n"
            
            summary += "\n## 💡 Recommendations\n"
            for rec in analysis['recommendations']:
                summary += f"- {rec}\n"
            
            # Save detailed report
            evaluator.save_evaluation_report(report, 'evaluation_report.json')
            
            return summary
            
        except Exception as e:
            return f"❌ Error running evaluation: {str(e)}"
    
    def get_system_info(self) -> str:
        """Get system information"""
        if not self.is_initialized:
            return "❌ System not initialized."
        
        try:
            stats = self.rag_engine.get_system_stats()
            
            info = f"""# 🔧 System Information

## Vector Store
- **Total Chunks:** {stats.get('total_chunks', 'Unknown'):,}
- **Embedding Model:** {stats.get('embedding_model', 'Unknown')}
- **Vector Store Path:** {stats.get('vector_store_path', 'Unknown')}

## Product Distribution
"""
            
            if 'product_distribution' in stats:
                for product, count in stats['product_distribution'].items():
                    info += f"- **{product}:** {count:,} chunks\n"
            
            info += f"""
## Configuration
- **Max Context Length:** {stats.get('max_context_length', 'Unknown')} characters
- **LLM Model:** {stats.get('llm_model', 'Unknown')}
"""
            
            return info
            
        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"

# Initialize chatbot
chatbot = ComplaintChatbot()

# Create Gradio interface
with gr.Blocks(title="CrediTrust Financial - Complaint Analysis Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 CrediTrust Financial - Intelligent Complaint Analysis
    
    **Transform customer feedback into actionable insights with AI-powered analysis**
    
    This RAG-powered chatbot helps product managers, support teams, and compliance officers understand customer complaints across:
    - 💳 Credit Cards
    - 💰 Personal Loans  
    - 🛒 Buy Now, Pay Later (BNPL)
    - 🏦 Savings Accounts
    - 💸 Money Transfers
    """)
    
    with gr.Tabs():
        # Data Processing Tab
        with gr.Tab("📊 Data Processing"):
            gr.Markdown("### Upload and Process Complaint Data")
            gr.Markdown("Upload your CFPB complaint dataset (CSV format) to initialize the system.")
            
            with gr.Row():
                file_input = gr.File(
                    label="Upload CFPB Complaint Dataset (CSV)",
                    file_types=[".csv"],
                    type="filepath"
                )
                process_btn = gr.Button("🚀 Process Data", variant="primary", size="lg")
            
            processing_output = gr.Markdown(label="Processing Status")
            
            process_btn.click(
                fn=chatbot.process_data,
                inputs=[file_input],
                outputs=[processing_output]
            )
        
        # Chat Interface Tab
        with gr.Tab("💬 Chat Interface"):
            gr.Markdown("### Ask Questions About Customer Complaints")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot_interface = gr.Chatbot(
                        label="Complaint Analysis Assistant",
                        height=500,
                        show_label=True
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the main issues customers face with credit cards?",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                
                with gr.Column(scale=1):
                    product_filter = gr.Dropdown(
                        choices=chatbot.products,
                        value="All Products",
                        label="Filter by Product",
                        info="Narrow down search to specific products"
                    )
                    
                    gr.Markdown("""
                    ### 💡 Example Questions:
                    - What are the main issues with credit cards?
                    - Why are people unhappy with BNPL?
                    - Which product has the most fraud complaints?
                    - What do customers say about customer service?
                    - Are there patterns in billing disputes?
                    """)
            
            # Chat functionality
            msg_input.submit(
                fn=chatbot.chat_response,
                inputs=[msg_input, chatbot_interface, product_filter],
                outputs=[chatbot_interface, msg_input]
            )
            
            submit_btn.click(
                fn=chatbot.chat_response,
                inputs=[msg_input, chatbot_interface, product_filter],
                outputs=[chatbot_interface, msg_input]
            )
            
            clear_btn.click(
                fn=chatbot.clear_chat,
                outputs=[chatbot_interface]
            )
        
        # Evaluation Tab
        with gr.Tab("📈 System Evaluation"):
            gr.Markdown("### Evaluate RAG System Performance")
            gr.Markdown("Run comprehensive evaluation to assess system quality and identify areas for improvement.")
            
            eval_btn = gr.Button("🔍 Run Evaluation", variant="primary", size="lg")
            eval_output = gr.Markdown(label="Evaluation Results")
            
            eval_btn.click(
                fn=chatbot.run_evaluation,
                outputs=[eval_output]
            )
        
        # System Info Tab
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("### System Status and Configuration")
            
            info_btn = gr.Button("📋 Get System Info", variant="primary")
            info_output = gr.Markdown(label="System Information")
            
            info_btn.click(
                fn=chatbot.get_system_info,
                outputs=[info_output]
            )
    
    gr.Markdown("""
    ---
    ### 🚀 Getting Started:
    1. **Upload Data**: Go to the "Data Processing" tab and upload your CFPB complaint CSV file
    2. **Process**: Click "Process Data" and wait for initialization to complete
    3. **Chat**: Switch to "Chat Interface" tab and start asking questions
    4. **Evaluate**: Use "System Evaluation" tab to assess performance
    
    ### 📝 Tips for Better Results:
    - Be specific in your questions
    - Use product filters to narrow down results
    - Check the sources provided with each answer
    - Try different phrasings if you don't get good results
    
    **Built with ❤️ for CrediTrust Financial**
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
