"""
User interface components for the AI vs Human essay classification project.
Separates UI logic from business logic.
"""
from typing import Optional, List
try:
    from .prediction_service import PredictionService
    from .cache_manager import CacheManager
except ImportError:
    # Fallback for when running as script
    from prediction_service import PredictionService
    from cache_manager import CacheManager


class TextClassifierUI:
    """
    User interface for the text classifier.
    Handles all user interactions and display logic.
    """
    
    def __init__(self, prediction_service: PredictionService, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the UI.
        
        Args:
            prediction_service (PredictionService): Service for making predictions
            cache_manager (CacheManager, optional): Cache manager for saving results
        """
        self.prediction_service = prediction_service
        self.cache_manager = cache_manager
    
    def display_header(self):
        """Display the application header."""
        print("🤖 AI vs Human Essay Classifier")
        print("=" * 40)
        
        if not self.prediction_service.is_model_available():
            print("❌ Model not found! Please train a model first.")
            return False
        
        print("✅ Model loaded successfully!")
        if self.cache_manager:
            print(f"📁 Cache enabled: {self.cache_manager.cache_dir}")
        return True
    
    def display_prediction_result(self, result: dict):
        """
        Display a single prediction result.
        
        Args:
            result (dict): Prediction result dictionary
        """
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        print(f"\n🎯 Result: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']}%")
        
        # Add emoji based on prediction
        if result['prediction'] == "AI-generated":
            print("🤖 This text appears to be AI-generated")
        else:
            print("👤 This text appears to be human-written")
    
    def display_sample_predictions(self, sample_texts: List[str]):
        """
        Display predictions for sample texts.
        
        Args:
            sample_texts (list): List of sample texts to classify
        """
        print("🧪 Testing with sample texts...")
        print("=" * 50)
        
        for i, text in enumerate(sample_texts, 1):
            try:
                result = self.prediction_service.predict_single(text, self.cache_manager)
                if "error" not in result:
                    print(f"\n{i}. {result['text']}")
                    print(f"   Result: {result['prediction']} ({result['confidence']}%)")
            except FileNotFoundError as e:
                print(f"❌ {e}")
                break
    
    def display_batch_predictions(self, texts: List[str]):
        """
        Display batch predictions.
        
        Args:
            texts (list): List of texts to classify
        """
        print("\n" + "=" * 50)
        print("📦 Testing batch predictions...")
        
        try:
            batch_results = self.prediction_service.predict_batch(texts, self.cache_manager)
            print(f"✅ Processed {len(batch_results)} texts in batch")
        except FileNotFoundError as e:
            print(f"❌ {e}")
    
    def display_cache_info(self):
        """Display cache information."""
        if self.cache_manager:
            print("\n" + "=" * 50)
            cache_info = self.cache_manager.get_cache_info()
            print(f"📊 Cache Statistics:")
            print(f"   Files created: {cache_info['file_count']}")
            print(f"   Total size: {cache_info['total_size']} bytes")
            print(f"   Cached files: {len(self.cache_manager.list_cached_files())}")
    
    def interactive_mode(self):
        """Run interactive prediction mode."""
        print("\nEnter texts to classify (type 'quit' to exit):")
        print("-" * 40)
        
        while True:
            try:
                text = input("\n🔍 Enter text: ").strip()
                
                if text.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("❌ Please enter some text.")
                    continue
                
                result = self.prediction_service.predict_single(text, self.cache_manager)
                self.display_prediction_result(result)
                
            except FileNotFoundError as e:
                print(f"❌ {e}")
                break
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                continue
    
    def run_full_demo(self, sample_texts: List[str]):
        """
        Run a complete demonstration of the classifier.
        
        Args:
            sample_texts (list): Sample texts to use in the demonstration
        """
        print("🚀 AI vs Human Essay Classifier with Caching")
        print("=" * 60)
        
        if self.cache_manager:
            print(f"📁 Temporary cache: {self.cache_manager.cache_dir}")
            print("ℹ️  All results will be saved temporarily and cleaned up on exit")
            print()
        
        if not self.display_header():
            return
        
        # Test individual predictions
        self.display_sample_predictions(sample_texts)
        
        # Test batch predictions
        self.display_batch_predictions(sample_texts)
        
        # Show cache info
        self.display_cache_info()
        
        # Start interactive mode
        print("\n" + "=" * 50)
        print("🚀 Starting interactive mode...")
        self.interactive_mode()
        
        print("\n" + "=" * 50)
        print("🏁 Program finished - cache will be cleaned up automatically")
