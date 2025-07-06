import pandas as pd
from typing import List, Dict, Any
from src.rag_engine import RAGEngine
import json
import os

class RAGEvaluator:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        
        # Define comprehensive test questions
        self.test_questions = [
            {
                'question': 'What are the main issues customers face with credit cards?',
                'expected_theme': 'billing, fees, fraud',
                'product_filter': 'Credit card'
            },
            {
                'question': 'Why are people unhappy with BNPL services?',
                'expected_theme': 'payment issues, terms',
                'product_filter': 'Buy Now, Pay Later (BNPL)'
            },
            {
                'question': 'What problems do customers report with personal loans?',
                'expected_theme': 'approval, terms, payments',
                'product_filter': 'Personal loan'
            },
            {
                'question': 'What are common complaints about savings accounts?',
                'expected_theme': 'fees, access, interest',
                'product_filter': 'Savings account'
            },
            {
                'question': 'What issues do customers have with money transfers?',
                'expected_theme': 'delays, fees, failed transfers',
                'product_filter': 'Money transfers'
            },
            {
                'question': 'Which financial product has the most fraud-related complaints?',
                'expected_theme': 'fraud comparison across products',
                'product_filter': None
            },
            {
                'question': 'What are customers saying about customer service quality?',
                'expected_theme': 'service quality issues',
                'product_filter': None
            },
            {
                'question': 'Are there any patterns in billing disputes across products?',
                'expected_theme': 'billing issues patterns',
                'product_filter': None
            },
            {
                'question': 'What are the most frequent complaint types?',
                'expected_theme': 'complaint frequency analysis',
                'product_filter': None
            },
            {
                'question': 'How do customers describe unauthorized transactions?',
                'expected_theme': 'unauthorized transaction descriptions',
                'product_filter': None
            }
        ]
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the RAG system
        """
        print("=== RAG SYSTEM EVALUATION ===\n")
        
        # Run evaluation
        results_df = self.rag_engine.evaluate_system(self.test_questions)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results_df)
        
        # Generate detailed analysis
        analysis = self._generate_analysis(results_df)
        
        # Create evaluation report
        report = {
            'metrics': metrics,
            'analysis': analysis,
            'detailed_results': results_df.to_dict('records'),
            'system_stats': self.rag_engine.get_system_stats()
        }
        
        return report
    
    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate evaluation metrics
        """
        total_questions = len(results_df)
        
        metrics = {
            'total_questions': total_questions,
            'questions_with_sources': results_df['Has Sources'].sum(),
            'avg_sources_per_question': results_df['Number of Sources'].mean(),
            'avg_answer_length': results_df['Answer Length'].mean(),
            'questions_mentioning_products': results_df['Mentions Product'].sum(),
            'avg_top_source_similarity': results_df['Top Source Similarity'].mean(),
            'source_retrieval_rate': results_df['Has Sources'].mean() * 100,
            'product_mention_rate': results_df['Mentions Product'].mean() * 100
        }
        
        return metrics
    
    def _generate_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate detailed analysis of results
        """
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Analyze strengths
        if results_df['Has Sources'].mean() > 0.8:
            analysis['strengths'].append("High source retrieval rate - system finds relevant information consistently")
        
        if results_df['Top Source Similarity'].mean() > 0.7:
            analysis['strengths'].append("High similarity scores indicate good semantic matching")
        
        if results_df['Mentions Product'].mean() > 0.6:
            analysis['strengths'].append("Answers frequently mention relevant products, showing context awareness")
        
        # Analyze weaknesses
        if results_df['Number of Sources'].min() == 0:
            no_source_questions = results_df[results_df['Number of Sources'] == 0]['Question'].tolist()
            analysis['weaknesses'].append(f"Some questions returned no sources: {no_source_questions}")
        
        if results_df['Answer Length'].min() < 50:
            short_answers = results_df[results_df['Answer Length'] < 50]['Question'].tolist()
            analysis['weaknesses'].append(f"Some answers are very short: {short_answers}")
        
        # Generate recommendations
        if results_df['Top Source Similarity'].mean() < 0.6:
            analysis['recommendations'].append("Consider improving embedding model or chunking strategy")
        
        if results_df['Answer Length'].std() > 200:
            analysis['recommendations'].append("Standardize answer length for consistency")
        
        analysis['recommendations'].append("Implement user feedback mechanism to continuously improve responses")
        analysis['recommendations'].append("Add confidence scores to help users assess answer reliability")
        
        return analysis
    
    def create_evaluation_table(self, results_df: pd.DataFrame) -> str:
        """
        Create a formatted evaluation table for the report
        """
        table_data = []
        
        for _, row in results_df.iterrows():
            quality_score = self._calculate_quality_score(row)
            
            table_row = {
                'Question': row['Question'][:60] + "..." if len(row['Question']) > 60 else row['Question'],
                'Product Filter': row['Product Filter'],
                'Generated Answer': row['Generated Answer'],
                'Sources Found': row['Number of Sources'],
                'Top Similarity': f"{row['Top Source Similarity']:.3f}",
                'Quality Score': f"{quality_score}/5",
                'Comments': self._generate_comments(row)
            }
            table_data.append(table_row)
        
        # Convert to markdown table
        markdown_table = "| Question | Product Filter | Answer Preview | Sources | Similarity | Quality | Comments |\n"
        markdown_table += "|----------|----------------|----------------|---------|------------|---------|----------|\n"
        
        for row in table_data:
            markdown_table += f"| {row['Question']} | {row['Product Filter']} | {row['Generated Answer'][:50]}... | {row['Sources Found']} | {row['Top Similarity']} | {row['Quality Score']} | {row['Comments']} |\n"
        
        return markdown_table
    
    def _calculate_quality_score(self, row: pd.Series) -> int:
        """
        Calculate a quality score from 1-5 for each response
        """
        score = 1
        
        # Has sources
        if row['Has Sources']:
            score += 1
        
        # Good similarity
        if row['Top Source Similarity'] > 0.7:
            score += 1
        
        # Reasonable answer length
        if 50 <= row['Answer Length'] <= 300:
            score += 1
        
        # Mentions relevant products
        if row['Mentions Product']:
            score += 1
        
        return min(score, 5)
    
    def _generate_comments(self, row: pd.Series) -> str:
        """
        Generate comments for each evaluation result
        """
        comments = []
        
        if not row['Has Sources']:
            comments.append("No sources found")
        elif row['Number of Sources'] < 3:
            comments.append("Few sources")
        
        if row['Top Source Similarity'] < 0.5:
            comments.append("Low similarity")
        elif row['Top Source Similarity'] > 0.8:
            comments.append("High similarity")
        
        if row['Answer Length'] < 50:
            comments.append("Very short answer")
        elif row['Answer Length'] > 300:
            comments.append("Long answer")
        
        if not row['Mentions Product']:
            comments.append("No product mention")
        
        return "; ".join(comments) if comments else "Good response"
    
    def save_evaluation_report(self, report: Dict[str, Any], output_path: str = "evaluation_report.json"):
        """
        Save evaluation report to file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✓ Evaluation report saved to {output_path}")
        except Exception as e:
            print(f"✗ Error saving evaluation report: {e}")
    
    def print_evaluation_summary(self, report: Dict[str, Any]):
        """
        Print a summary of the evaluation results
        """
        metrics = report['metrics']
        analysis = report['analysis']
        
        print("=== EVALUATION SUMMARY ===\n")
        
        print("📊 Key Metrics:")
        print(f"  • Total questions evaluated: {metrics['total_questions']}")
        print(f"  • Source retrieval rate: {metrics['source_retrieval_rate']:.1f}%")
        print(f"  • Average sources per question: {metrics['avg_sources_per_question']:.1f}")
        print(f"  • Average answer length: {metrics['avg_answer_length']:.0f} characters")
        print(f"  • Product mention rate: {metrics['product_mention_rate']:.1f}%")
        print(f"  • Average similarity score: {metrics['avg_top_source_similarity']:.3f}")
        
        print("\n✅ Strengths:")
        for strength in analysis['strengths']:
            print(f"  • {strength}")
        
        print("\n⚠️ Areas for Improvement:")
        for weakness in analysis['weaknesses']:
            print(f"  • {weakness}")
        
        print("\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
        
        print("\n" + "="*50)
