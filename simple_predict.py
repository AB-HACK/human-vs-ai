"""
Simple prediction script - just run this to classify texts!
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_text(text):
    """
    Classify a single text as AI-generated or human-written.
    
    Args:
        text (str): Text to classify
        
    Returns:
        str: "AI-generated" or "Human-written"
    """
    # Load model and vectorizer
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        return "❌ Model not found! Please train a model first using run_analysis.py"
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform and predict
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]
    
    return "🤖 AI-generated" if prediction == 1 else "👤 Human-written"

def classify_with_confidence(text):
    """
    Classify text with confidence score.
    
    Args:
        text (str): Text to classify
        
    Returns:
        dict: Prediction with confidence
    """
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        return {"error": "Model not found! Please train a model first."}
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform and predict
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]
    confidence = model.predict_proba(text_vectorized).max()
    
    return {
        "prediction": "AI-generated" if prediction == 1 else "Human-written",
        "confidence": round(confidence * 100, 2)
    }

# Example usage
if __name__ == "__main__":
    print("🤖 AI vs Human Essay Classifier")
    print("=" * 40)
    
    # Test with sample texts
    sample_texts = [
        "The economy, a complex system of production, distribution, and consumption of goods and services, plays a crucial role in shaping society.",
        "I love spending time with my family on weekends. We usually go to the park and have a picnic together.",
        "Artificial intelligence represents a paradigm shift in computational capabilities, enabling machines to process information and make decisions with unprecedented efficiency."
    ]
    
    print("🧪 Testing with sample texts:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts, 1):
        result = classify_text(text)
        print(f"{i}. {result}")
        print(f"   Text: {text[:60]}...")
        print()
    
    # Interactive mode
    print("🚀 Interactive mode - Enter texts to classify:")
    print("(Type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        text = input("\n📝 Enter text: ").strip()
        
        if text.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not text:
            print("❌ Please enter some text.")
            continue
        
        # Get prediction with confidence
        result = classify_with_confidence(text)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            continue
        
        print(f"🎯 Result: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']}%")
