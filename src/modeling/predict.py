"""
Simple prediction script for AI vs Human essay classification.
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download NLTK resources if not already downloaded
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Text preprocessing function"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model(model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """Load trained model and vectorizer"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        print(f"Model files not found. Please train a model first.")
        return None, None

def predict_single_text(text, model=None, vectorizer=None):
    """
    Predict if a single text is AI-generated or human-written.
    
    Args:
        text (str): Text to classify
        model: Trained model (optional, will load from files if not provided)
        vectorizer: Fitted vectorizer (optional, will load from files if not provided)
    
    Returns:
        dict: Prediction result with confidence
    """
    # Load model if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
        if model is None:
            return {"error": "Model not found. Please train a model first."}
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    confidence = model.predict_proba(text_vectorized).max()
    
    result = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "prediction": "AI-generated" if prediction == 1 else "Human-written",
        "confidence": round(confidence * 100, 2),
        "raw_prediction": int(prediction)
    }
    
    return result

def predict_multiple_texts(texts, model=None, vectorizer=None):
    """
    Predict multiple texts at once.
    
    Args:
        texts (list): List of texts to classify
        model: Trained model (optional)
        vectorizer: Fitted vectorizer (optional)
    
    Returns:
        list: List of prediction results
    """
    # Load model if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
        if model is None:
            return [{"error": "Model not found. Please train a model first."}]
    
    results = []
    for text in texts:
        result = predict_single_text(text, model, vectorizer)
        results.append(result)
    
    return results

def interactive_prediction():
    """Interactive command-line prediction interface"""
    print("🤖 AI vs Human Essay Classifier")
    print("=" * 40)
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    print("✅ Model loaded successfully!")
    print("\nEnter texts to classify (type 'quit' to exit):")
    print("-" * 40)
    
    while True:
        text = input("\n📝 Enter text: ").strip()
        
        if text.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not text:
            print("❌ Please enter some text.")
            continue
        
        # Make prediction
        result = predict_single_text(text, model, vectorizer)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            continue
        
        # Display result
        print(f"\n🎯 Result: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']}%")
        
        # Add some emoji based on prediction
        if result['prediction'] == "AI-generated":
            print("🤖 This text appears to be AI-generated")
        else:
            print("👤 This text appears to be human-written")

# Example usage
if __name__ == "__main__":
    # Example texts for testing
    sample_texts = [
        "The economy, a complex system of production, distribution, and consumption of goods and services, plays a crucial role in shaping society.",
        "Artificial intelligence represents a paradigm shift in computational capabilities, enabling machines to process information and make decisions with unprecedented efficiency.",
        "I love spending time with my family on weekends. We usually go to the park and have a picnic together.",
        "The implementation of machine learning algorithms requires careful consideration of data preprocessing, feature engineering, and model validation techniques."
    ]
    
    print("🧪 Testing with sample texts...")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        result = predict_single_text(text)
        if "error" not in result:
            print(f"\n{i}. {result['text']}")
            print(f"   Result: {result['prediction']} ({result['confidence']}%)")
    
    print("\n" + "=" * 50)
    print("🚀 Starting interactive mode...")
    interactive_prediction()