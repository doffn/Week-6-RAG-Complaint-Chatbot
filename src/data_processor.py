import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
import plotly
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataProcessor:
    def __init__(self):
        self.target_products = [
            'Credit card', 
            'Personal loan', 
            'Buy Now, Pay Later (BNPL)', 
            'Savings account', 
            'Money transfers'
        ]
        
    def load_data(self, file_path):
        """Load the CFPB complaint dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def perform_eda(self, df):
        """Perform comprehensive exploratory data analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        
        # Basic info
        print("1. Dataset Overview:")
        print(f"   - Total records: {len(df):,}")
        print(f"   - Total columns: {len(df.columns)}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        # Column info
        print("2. Column Information:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"   - {col}: {null_pct:.1f}% missing")
        print()
        
        # Product distribution
        print("3. Product Distribution:")
        product_counts = df['Product'].value_counts()
        for product, count in product_counts.head(10).items():
            print(f"   - {product}: {count:,} ({count/len(df)*100:.1f}%)")
        print()
        
        # Narrative analysis
        print("4. Consumer Complaint Narrative Analysis:")
        narrative_col = 'Consumer complaint narrative'
        
        if narrative_col in df.columns:
            # Non-null narratives
            non_null_narratives = df[narrative_col].notna().sum()
            print(f"   - Records with narratives: {non_null_narratives:,} ({non_null_narratives/len(df)*100:.1f}%)")
            print(f"   - Records without narratives: {len(df) - non_null_narratives:,}")
            
            # Word count analysis
            df['narrative_word_count'] = df[narrative_col].fillna('').apply(lambda x: len(str(x).split()))
            
            word_stats = df[df[narrative_col].notna()]['narrative_word_count'].describe()
            print(f"   - Average words per narrative: {word_stats['mean']:.1f}")
            print(f"   - Median words per narrative: {word_stats['50%']:.1f}")
            print(f"   - Min words: {word_stats['min']:.0f}")
            print(f"   - Max words: {word_stats['max']:.0f}")
        
        return self.create_eda_visualizations(df)
    
    def create_eda_visualizations(self, df):
        """Create comprehensive visualizations for EDA"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Product Distribution', 'Narrative Word Count Distribution', 
                          'Complaints Over Time', 'Issue Distribution'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Product distribution
        product_counts = df['Product'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=product_counts.values, y=product_counts.index, orientation='h'),
            row=1, col=1
        )
        
        # Word count distribution
        if 'narrative_word_count' in df.columns:
            word_counts = df[df['Consumer complaint narrative'].notna()]['narrative_word_count']
            fig.add_trace(
                go.Histogram(x=word_counts, nbinsx=50),
                row=1, col=2
            )
        
        # Complaints over time (if date column exists)
        if 'Date received' in df.columns:
            df['Date received'] = pd.to_datetime(df['Date received'], errors='coerce')
            monthly_complaints = df.groupby(df['Date received'].dt.to_period('M')).size()
            fig.add_trace(
                go.Scatter(x=monthly_complaints.index.astype(str), y=monthly_complaints.values, mode='lines'),
                row=2, col=1
            )
        
        # Issue distribution
        if 'Issue' in df.columns:
            issue_counts = df['Issue'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=issue_counts.values, y=issue_counts.index, orientation='h'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="CFPB Complaints - Exploratory Data Analysis")
        return fig
    
    def clean_text(self, text):
        """Clean and normalize text narratives"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Remove common boilerplate text
        boilerplate_patterns = [
            r'i am writing to file a complaint',
            r'dear sir/madam',
            r'to whom it may concern',
            r'i would like to file a complaint',
            r'this is a complaint about'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def filter_and_clean_data(self, df):
        """Filter data for target products and clean narratives"""
        print("=== DATA FILTERING AND CLEANING ===\n")
        
        # Filter for target products
        print("1. Filtering for target products...")
        initial_count = len(df)
        
        # Create a mapping for product name variations
        product_mapping = {
            'Credit card': ['Credit card', 'Credit card or prepaid card'],
            'Personal loan': ['Personal loan'],
            'Buy Now, Pay Later (BNPL)': ['Buy Now, Pay Later (BNPL)', 'BNPL'],
            'Savings account': ['Savings account', 'Checking or savings account'],
            'Money transfers': ['Money transfers', 'Money transfer', 'Money transfer, virtual currency, or money service']
        }
        
        # Create filter condition
        filter_condition = df['Product'].isin([item for sublist in product_mapping.values() for item in sublist])
        df_filtered = df[filter_condition].copy()
        
        print(f"   - Records before filtering: {initial_count:,}")
        print(f"   - Records after product filtering: {len(df_filtered):,}")
        print(f"   - Records removed: {initial_count - len(df_filtered):,}\n")
        
        # Remove records without narratives
        print("2. Removing records without narratives...")
        narrative_col = 'Consumer complaint narrative'
        
        before_narrative_filter = len(df_filtered)
        df_filtered = df_filtered[df_filtered[narrative_col].notna() & 
                                 (df_filtered[narrative_col] != '') & 
                                 (df_filtered[narrative_col].str.strip() != '')].copy()
        
        print(f"   - Records before narrative filtering: {before_narrative_filter:,}")
        print(f"   - Records after narrative filtering: {len(df_filtered):,}")
        print(f"   - Records removed: {before_narrative_filter - len(df_filtered):,}\n")
        
        # Clean narratives
        print("3. Cleaning text narratives...")
        df_filtered['cleaned_narrative'] = df_filtered[narrative_col].apply(self.clean_text)
        
        # Remove very short narratives (less than 10 words)
        df_filtered['word_count'] = df_filtered['cleaned_narrative'].apply(lambda x: len(str(x).split()))
        before_length_filter = len(df_filtered)
        df_filtered = df_filtered[df_filtered['word_count'] >= 10].copy()
        
        print(f"   - Records before length filtering: {before_length_filter:,}")
        print(f"   - Records after length filtering (>=10 words): {len(df_filtered):,}")
        print(f"   - Records removed: {before_length_filter - len(df_filtered):,}\n")
        
        # Final statistics
        print("4. Final Dataset Statistics:")
        final_product_dist = df_filtered['Product'].value_counts()
        for product, count in final_product_dist.items():
            print(f"   - {product}: {count:,} complaints")
        
        print(f"\n   - Total final records: {len(df_filtered):,}")
        print(f"   - Average narrative length: {df_filtered['word_count'].mean():.1f} words")
        print(f"   - Median narrative length: {df_filtered['word_count'].median():.1f} words")
        
        return df_filtered
    
    def save_processed_data(self, df, output_path):
        """Save the processed dataset"""
        try:
            df.to_csv(output_path, index=False)
            print(f"\nProcessed data saved to: {output_path}")
            print(f"Saved {len(df):,} records with {len(df.columns)} columns")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def generate_word_cloud(self, df, product=None):
        """Generate word cloud for narratives"""
        if product:
            text_data = df[df['Product'] == product]['cleaned_narrative'].str.cat(sep=' ')
            title = f"Word Cloud - {product}"
        else:
            text_data = df['cleaned_narrative'].str.cat(sep=' ')
            title = "Word Cloud - All Products"
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=100,
                             colormap='viridis').generate(text_data)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return plt.gcf()
