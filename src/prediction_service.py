"""
Clean prediction service for AI vs Human essay classification.
Separates business logic from UI concerns.
"""
import re
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .cache_manager import CacheManager


class PredictionService:
    """
    Service class for AI vs Human text classification.
    Handles all prediction logic with clean separation of concerns.
    """
    
    def __init__(self, model_path: str = "model.pkl", vectorizer_path: str = "vectorizer.pkl"):
        """
        Initialize the prediction service.
        
        Args:
            model_path (str): Path to the trained model file
            vectorizer_path (str): Path to the vectorizer file
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self._lemmatizer = None
        self._stop_words = None
        self._ensure_nltk_resources()
    
    def _ensure_nltk_resources(self):
        """Ensure NLTK resources are available."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            pass  # Resources might already be downloaded
        
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self._lemmatizer.lemmatize(word) for word in words 
                if word not in self._stop_words]
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _load_model(self) -> Tuple[Any, Any]:
        """
        Load the trained model and vectorizer.
        
        Returns:
            tuple: (model, vectorizer) or (None, None) if loading fails
        """
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {e}. Please train a model first.")
    
    def _ensure_model_loaded(self):
        """Ensure model and vectorizer are loaded."""
        if self.model is None or self.vectorizer is None:
            self.model, self.vectorizer = self._load_model()
    
    def predict_single(self, text: str, cache_manager: Optional[CacheManager] = None) -> Dict[str, Any]:
        """
        Predict if a single text is AI-generated or human-written.
        
        Args:
            text (str): Text to classify
            cache_manager (CacheManager, optional): Cache manager for saving results
            
        Returns:
            dict: Prediction result with confidence
            
        Raises:
            FileNotFoundError: If model files are not found
        """
        self._ensure_model_loaded()
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Transform text
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        confidence = self.model.predict_proba(text_vectorized).max()
        
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": "AI-generated" if prediction == 1 else "Human-written",
            "confidence": round(confidence * 100, 2),
            "raw_prediction": int(prediction)
        }
        
        # Save result to cache if cache manager is provided
        if cache_manager is not None:
            cache_manager.save_prediction_result(result)
        
        return result
    
    def predict_batch(self, texts: List[str], cache_manager: Optional[CacheManager] = None) -> List[Dict[str, Any]]:
        """
        Predict multiple texts at once.
        
        Args:
            texts (list): List of texts to classify
            cache_manager (CacheManager, optional): Cache manager for saving results
            
        Returns:
            list: List of prediction results
            
        Raises:
            FileNotFoundError: If model files are not found
        """
        self._ensure_model_loaded()
        
        results = []
        for text in texts:
            result = self.predict_single(text, cache_manager)
            results.append(result)
        
        # Save batch results to cache if cache manager is provided
        if cache_manager is not None and results:
            cache_manager.save_multiple_predictions(results)
        
        return results
    
    def is_model_available(self) -> bool:
        """
        Check if model files are available.
        
        Returns:
            bool: True if model files exist, False otherwise
        """
        return (Path(self.model_path).exists() and 
                Path(self.vectorizer_path).exists())
