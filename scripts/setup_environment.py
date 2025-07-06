"""
Setup script to prepare the environment and download required models
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'vector_store', 'results', 'notebooks']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'sentence_transformers', 'chromadb', 'transformers', 
        'torch', 'gradio', 'sklearn', 'nltk'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✓ All packages imported successfully")
    return True

def main():
    print("=== ENVIRONMENT SETUP ===\n")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download NLTK data
    if not download_nltk_data():
        return
    
    # Test imports
    if not test_imports():
        return
    
    print("\n✅ Environment setup completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Place your CFPB complaint dataset in the 'data' folder")
    print("2. Run: python notebooks/01_data_exploration.py")
    print("3. Run: python notebooks/02_embedding_creation.py")
    print("4. Run: python notebooks/03_rag_evaluation.py")
    print("5. Launch the app: python app.py")

if __name__ == "__main__":
    main()
