"""
Dataset loading and preprocessing utilities.
"""
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE

def load_data(file_path=None):
    """
    Load dataset from CSV file. Defaults to Kaggle dataset path.
    
    Args:
        file_path (str): Path to the CSV file. If None, uses Kaggle dataset path.
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Default to Kaggle dataset if no path provided
    if file_path is None:
        file_path = 'balanced_ai_human_prompts.csv'
    
    data = pd.read_csv(file_path)
    
    # Ensure column names match your Kaggle dataset
    if 'text' not in data.columns or 'generated' not in data.columns:
        # Try to rename columns if they exist with different names
        if len(data.columns) >= 2:
            data.columns = ['text', 'generated']
        else:
            print("Available columns:", data.columns.tolist())
            print("Please ensure your dataset has 'text' and 'generated' columns")
    
    return data

def clean_text(text):
    """
    Basic text cleaning function.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_data(X, y):
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target labels
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )