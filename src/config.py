"""
Configuration settings for the AI vs Human essay classification project.
"""
import os
import nltk

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, 'nltk_data')

# Ensure NLTK data directory exists
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Add NLTK data path
nltk.data.path.append(NLTK_DATA_DIR)

# Download required NLTK resources
def download_nltk_resources():
    """Download required NLTK resources to project directory."""
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

# Auto-download resources when config is imported
download_nltk_resources()

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Text preprocessing settings
MAX_FEATURES = 10000
MIN_DF = 2
MAX_DF = 0.95
