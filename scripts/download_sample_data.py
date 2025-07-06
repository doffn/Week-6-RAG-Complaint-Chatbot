"""
Script to download sample CFPB complaint data for testing
"""

import pandas as pd
import requests
import os
from io import StringIO

def download_cfpb_data():
    """Download sample CFPB complaint data"""
    print("Downloading sample CFPB complaint data...")
    
    # CFPB API endpoint for complaints
    url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    
    # Parameters for API request
    params = {
        'size': 1000,  # Number of records
        'format': 'csv',
        'field': ['complaint_id', 'date_received', 'product', 'issue', 'consumer_complaint_narrative', 'company', 'state'],
        'has_narrative': 'true'  # Only get complaints with narratives
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Save the data
        os.makedirs('data', exist_ok=True)
        with open('data/sample_cfpb_complaints.csv', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print("✓ Sample data downloaded successfully")
        print("✓ Saved to: data/sample_cfpb_complaints.csv")
        
        # Load and display basic info
        df = pd.read_csv('data/sample_cfpb_complaints.csv')
        print(f"✓ Downloaded {len(df)} complaint records")
        print(f"✓ Products included: {', '.join(df['Product'].unique()[:5])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("\n💡 Alternative: You can manually download CFPB data from:")
        print("https://www.consumerfinance.gov/data-research/consumer-complaints/")
        return False

def create_sample_data():
    """Create a small sample dataset for testing"""
    print("Creating sample dataset for testing...")
    
    sample_data = {
        'Date received': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'] * 20,
        'Product': ['Credit card', 'Personal loan', 'Buy Now, Pay Later (BNPL)', 'Savings account', 'Money transfers'] * 20,
        'Issue': ['Billing dispute', 'Loan terms', 'Payment issues', 'Account access', 'Transfer delays'] * 20,
        'Consumer complaint narrative': [
            'I was charged an unauthorized fee on my credit card statement. The charge appeared without any prior notification.',
            'The personal loan terms were not clearly explained during the application process. Hidden fees were added.',
            'My BNPL payment failed multiple times despite having sufficient funds in my account.',
            'I cannot access my savings account online. The website keeps showing error messages.',
            'My money transfer has been pending for over 5 business days without any explanation.',
            'Credit card customer service was unhelpful when I called about fraudulent charges on my account.',
            'The interest rate on my personal loan increased without proper notice being provided to me.',
            'BNPL service charged me late fees even though I made the payment on time through their app.',
            'Savings account was closed without notification and my funds were held for weeks.',
            'Money transfer was cancelled but the fees were not refunded to my account.',
        ] * 10,
        'Company': ['CrediTrust Financial'] * 100,
        'State': ['CA', 'NY', 'TX', 'FL', 'IL'] * 20
    }
    
    df = pd.DataFrame(sample_data)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_complaints.csv', index=False)
    
    print("✓ Sample dataset created")
    print("✓ Saved to: data/sample_complaints.csv")
    print(f"✓ Created {len(df)} sample records")
    
    return True

def main():
    print("=== SAMPLE DATA DOWNLOAD ===\n")
    
    # Try to download real CFPB data first
    if not download_cfpb_data():
        print("\nFalling back to creating sample data...")
        create_sample_data()
    
    print("\n✅ Data preparation completed!")
    print("\n🚀 You can now run the data processing pipeline:")
    print("python notebooks/01_data_exploration.py")

if __name__ == "__main__":
    main()
