"""
Main entry point for the AI vs Human essay classification project.
Clean, organized code with proper separation of concerns.
"""
from pathlib import Path
import sys

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

try:
    from .prediction_service import PredictionService
    from .ui import TextClassifierUI
    from .cache_manager import CacheManager
except ImportError:
    # Fallback for when running as script
    from prediction_service import PredictionService
    from ui import TextClassifierUI
    from cache_manager import CacheManager


def get_sample_texts() -> list:
    """
    Get sample texts for demonstration.
    
    Returns:
        list: List of sample texts
    """
    return [
        "The economy, a complex system of production, distribution, and consumption of goods and services, plays a crucial role in shaping society.",
        "Artificial intelligence represents a paradigm shift in computational capabilities, enabling machines to process information and make decisions with unprecedented efficiency.",
        "I love spending time with my family on weekends. We usually go to the park and have a picnic together.",
        "The implementation of machine learning algorithms requires careful consideration of data preprocessing, feature engineering, and model validation techniques."
    ]


def main():
    """Main application entry point."""
    # Configuration
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Initialize services
    prediction_service = PredictionService(model_path, vectorizer_path)
    
    # Initialize cache manager with automatic cleanup
    with CacheManager("ai_human_predictions", cleanup_on_exit=True) as cache_manager:
        # Initialize UI
        ui = TextClassifierUI(prediction_service, cache_manager)
        
        # Get sample texts
        sample_texts = get_sample_texts()
        
        # Run the full demonstration
        ui.run_full_demo(sample_texts)


if __name__ == "__main__":
    main()
