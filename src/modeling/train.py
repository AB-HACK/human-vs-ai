"""
Model training pipeline for AI vs Human essay classification.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from ..dataset import load_data, clean_text, split_data
from ..features import preprocess_text, create_tfidf_features
from ..plots import plot_class_distribution, plot_confusion_matrix, create_wordcloud, plot_feature_importance
from ..config import RANDOM_STATE
try:
    from ..cache_manager import CacheManager
except ImportError:
    # Fallback for when running as script
    from cache_manager import CacheManager

def train_model(file_path=None, cache_manager=None):
    """
    Complete training pipeline.
    
    Args:
        file_path (str): Path to the dataset CSV file. If None, uses Kaggle dataset.
        cache_manager (CacheManager, optional): Cache manager for saving results
        
    Returns:
        tuple: (model, vectorizer, X_test, y_test, y_pred)
    """
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Ensure the dataset has 'text' and 'generated' columns
    if 'text' not in df.columns or 'generated' not in df.columns:
        print("Available columns:", df.columns.tolist())
        print("Please ensure your dataset has 'text' and 'generated' columns")
        return None, None, None, None, None
    
    # Plot class distribution
    plot_class_distribution(df, 'generated', "Class Distribution (0: Human, 1: AI)")
    
    # Clean and preprocess text
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Create word clouds for each class
    human_texts = df[df['generated'] == 0]['cleaned_text']
    ai_texts = df[df['generated'] == 1]['cleaned_text']
    
    create_wordcloud(' '.join(human_texts), 'Human Texts')
    create_wordcloud(' '.join(ai_texts), 'AI Texts')
    
    # Prepare features and target
    X = df['cleaned_text']
    y = df['generated']
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    X_train_tfidf, vectorizer = create_tfidf_features(X_train, fit=True)
    X_test_tfidf = create_tfidf_features(X_test, fit=False)[0]
    
    # Train model
    print("Training model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate model
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    plot_feature_importance(model, vectorizer)
    
    # Save training results to cache if cache manager is provided
    if cache_manager is not None:
        training_results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "model_type": "LogisticRegression",
            "random_state": RANDOM_STATE,
            "test_size": len(y_test),
            "train_size": len(y_train)
        }
        cache_manager.save_training_results(training_results)
        
        # Save model artifacts to cache
        cache_manager.save_model_artifacts(model, vectorizer)
    
    return model, vectorizer, X_test, y_test, y_pred

def classify_text(text, model, vectorizer):
    """
    Classify a single text.
    
    Args:
        text (str): Text to classify
        model: Trained model
        vectorizer: Fitted vectorizer
        
    Returns:
        str: Classification result
    """
    from ..features import preprocess_text
    
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    return "AI-generated" if prediction[0] == 1 else "Human-written"

def save_model(model, vectorizer, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """
    Save trained model and vectorizer.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        model_path (str): Path to save model
        vectorizer_path (str): Path to save vectorizer
    """
    import pickle
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
