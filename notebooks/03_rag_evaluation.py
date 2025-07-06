"""
Notebook 3: RAG System Evaluation
This script evaluates the RAG system performance and generates detailed reports.
"""

import pandas as pd
import sys
import os
import json

# Add src to path
sys.path.append('../src')
from src.rag_engine import RAGEngine
from src.evaluation import RAGEvaluator

def main():
    print("=== RAG SYSTEM EVALUATION ===\n")
    
    # Check if vector store exists
    if not os.path.exists('../vector_store'):
        print("❌ Vector store not found. Please run embedding creation first.")
        return
    
    # Initialize RAG engine
    print("Initializing RAG engine...")
    try:
        rag_engine = RAGEngine(vector_store_path='../vector_store')
        print("✓ RAG engine initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG engine: {e}")
        return
    
    # Display system statistics
    print("\n=== SYSTEM CONFIGURATION ===")
    stats = rag_engine.get_system_stats()
    for key, value in stats.items():
        if key == 'product_distribution':
            print(f"{key}:")
            for product, count in value.items():
                print(f"  - {product}: {count:,}")
        else:
            print(f"{key}: {value}")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_engine)
    
    # Run comprehensive evaluation
    print(f"\n=== RUNNING EVALUATION ===")
    print(f"Testing {len(evaluator.test_questions)} questions...")
    
    report = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    evaluator.print_evaluation_summary(report)
    
    # Create detailed results table
    results_df = pd.DataFrame(report['detailed_results'])
    
    # Save results
    os.makedirs('../results', exist_ok=True)
    
    # Save evaluation report
    evaluator.save_evaluation_report(report, '../results/evaluation_report.json')
    
    # Save detailed results CSV
    results_df.to_csv('../results/evaluation_results.csv', index=False)
    print(f"✓ Detailed results saved to ../results/evaluation_results.csv")
    
    # Create markdown evaluation table
    markdown_table = evaluator.create_evaluation_table(results_df)
    with open('../results/evaluation_table.md', 'w') as f:
        f.write("# RAG System Evaluation Results\n\n")
        f.write(markdown_table)
    print(f"✓ Evaluation table saved to ../results/evaluation_table.md")
    
    # Test individual queries interactively
    print(f"\n=== INTERACTIVE TESTING ===")
    print("You can now test individual queries. Type 'quit' to exit.")
    
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Ask for product filter
        print("Available products: All, Credit card, Personal loan, BNPL, Savings account, Money transfers")
        product_filter = input("Filter by product (or press Enter for all): ").strip()
        if not product_filter or product_filter.lower() == 'all':
            product_filter = None
        
        # Get response
        result = rag_engine.query(query, product_filter=product_filter)
        
        print(f"\n📝 Question: {result['question']}")
        if result['product_filter']:
            print(f"🔍 Filter: {result['product_filter']}")
        print(f"💬 Answer: {result['answer']}")
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"\n{i}. Product: {source['product']} | Similarity: {source['similarity_score']:.3f}")
                print(f"   Text: {source['text'][:200]}{'...' if len(source['text']) > 200 else ''}")
        else:
            print("\n📚 No sources found")
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()
