"""
Feature engineering and text preprocessing.
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import MAX_FEATURES, MIN_DF, MAX_DF, NLTK_DATA_DIR

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    # Fallback if stopwords not downloaded
    stop_words = set()

def preprocess_text(text):
    """
    Advanced text preprocessing with lemmatization.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Split into words and process
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_tfidf_features(texts, fit=True):
    """
    Create TF-IDF features from text data.
    
    Args:
        texts: List or Series of text documents
        fit (bool): Whether to fit the vectorizer (True for training, False for prediction)
        
    Returns:
        tuple: (tfidf_matrix, vectorizer) if fit=True, else (tfidf_matrix,)
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Match your working script
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5,  # Match your working script
        max_df=0.7,  # Match your working script
        stop_words='english'
    )
    
    if fit:
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix, vectorizer
    else:
        tfidf_matrix = vectorizer.transform(texts)
        return tfidf_matrix