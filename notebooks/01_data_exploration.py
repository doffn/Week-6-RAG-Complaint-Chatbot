"""
Notebook 1: Data Exploration and Preprocessing
This script performs comprehensive EDA and data preprocessing for the CFPB complaint dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append('../src')
from src.data_processor import DataProcessor

def main():
    print("=== CFPB COMPLAINT DATA EXPLORATION ===\n")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load data (you'll need to provide the path to your CFPB dataset)
    data_path = input("Enter path to CFPB complaint dataset (CSV): ")
    
    if not os.path.exists(data_path):
        print("File not found. Please check the path.")
        return
    
    # Load and explore data
    df = processor.load_data(data_path)
    if df is None:
        return
    
    # Perform comprehensive EDA
    print("Performing Exploratory Data Analysis...")
    fig = processor.perform_eda(df)
    
    # Save EDA visualization
    if fig:
        fig.write_html("eda_visualization.html")
        print("✓ EDA visualization saved to eda_visualization.html")
    
    # Filter and clean data
    print("\nFiltering and cleaning data...")
    df_filtered = processor.filter_and_clean_data(df)
    
    # Generate word clouds for each product
    print("\nGenerating word clouds...")
    products = df_filtered['Product'].unique()
    
    for product in products:
        try:
            fig = processor.generate_word_cloud(df_filtered, product)
            plt.savefig(f"wordcloud_{product.replace(' ', '_').replace(',', '').lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Word cloud saved for {product}")
        except Exception as e:
            print(f"Could not generate word cloud for {product}: {e}")
    
    # Save processed data
    os.makedirs('../data', exist_ok=True)
    success = processor.save_processed_data(df_filtered, '../data/filtered_complaints.csv')
    
    if success:
        print(f"\n✅ Data processing completed successfully!")
        print(f"Processed dataset saved with {len(df_filtered):,} records")
        
        # Display final statistics
        print("\n=== FINAL DATASET STATISTICS ===")
        print(f"Total complaints: {len(df_filtered):,}")
        print(f"Products covered: {len(df_filtered['Product'].unique())}")
        print(f"Average narrative length: {df_filtered['word_count'].mean():.1f} words")
        print(f"Date range: {df_filtered['Date received'].min()} to {df_filtered['Date received'].max()}")
        
        print("\nProduct distribution:")
        product_dist = df_filtered['Product'].value_counts()
        for product, count in product_dist.items():
            print(f"  - {product}: {count:,} ({count/len(df_filtered)*100:.1f}%)")

if __name__ == "__main__":
    main()
