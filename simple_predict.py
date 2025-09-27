"""
Simple prediction script - just run this to classify texts!
Refactored to use clean, organized components.
"""
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.prediction_service import PredictionService
    from src.ui import TextClassifierUI
    from src.cache_manager import CacheManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))
    from prediction_service import PredictionService
    from ui import TextClassifierUI
    from cache_manager import CacheManager

def get_sample_texts():
    """Get sample texts for demonstration."""
    return [
        "The economy, a complex system of production, distribution, and consumption of goods and services, plays a crucial role in shaping society.",
        "I love spending time with my family on weekends. We usually go to the park and have a picnic together.",
        "Artificial intelligence represents a paradigm shift in computational capabilities, enabling machines to process information and make decisions with unprecedented efficiency."
    ]


def classify_text(text):
    """
    Simple classification function for backward compatibility.
    
    Args:
        text (str): Text to classify
        
    Returns:
        str: "AI-generated" or "Human-written"
    """
    try:
        prediction_service = PredictionService()
        result = prediction_service.predict_single(text)
        return "🤖 AI-generated" if result['prediction'] == "AI-generated" else "👤 Human-written"
    except FileNotFoundError:
        return "❌ Model not found! Please train a model first using run_analysis.py"


def classify_with_confidence(text):
    """
    Simple classification with confidence for backward compatibility.
    
    Args:
        text (str): Text to classify
        
    Returns:
        dict: Prediction with confidence
    """
    try:
        prediction_service = PredictionService()
        result = prediction_service.predict_single(text)
        return {
            "prediction": result['prediction'],
            "confidence": result['confidence']
        }
    except FileNotFoundError:
        return {"error": "Model not found! Please train a model first."}

def main():
    """Main application entry point."""
    # Configuration
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Initialize services
    prediction_service = PredictionService(model_path, vectorizer_path)
    
    # Initialize cache manager with automatic cleanup
    with CacheManager("simple_predictions", cleanup_on_exit=True) as cache_manager:
        # Initialize UI
        ui = TextClassifierUI(prediction_service, cache_manager)
        
        # Get sample texts
        sample_texts = get_sample_texts()
        
        # Run the full demonstration
        ui.run_full_demo(sample_texts)


if __name__ == "__main__":
    main()
