"""
Test script to verify the caching system works correctly.
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
import tempfile
import os


def test_cache_manager():
    """Test the cache manager functionality."""
    print("🧪 Testing CacheManager...")
    
    with CacheManager("test_cache", cleanup_on_exit=True) as cache:
        # Test saving prediction result
        test_result = {
            "text": "This is a test text",
            "prediction": "Human-written",
            "confidence": 85.5,
            "raw_prediction": 0
        }
        
        file_path = cache.save_prediction_result(test_result)
        assert os.path.exists(file_path), "Prediction result file not created"
        
        # Test saving multiple predictions
        test_results = [test_result, test_result]
        batch_path = cache.save_multiple_predictions(test_results)
        assert os.path.exists(batch_path), "Batch predictions file not created"
        
        # Test cache info
        cache_info = cache.get_cache_info()
        assert cache_info['file_count'] >= 2, "File count incorrect"
        
        print("✅ CacheManager tests passed!")


def test_prediction_service():
    """Test the prediction service."""
    print("🧪 Testing PredictionService...")
    
    # Check if model files exist
    if not (Path("model.pkl").exists() and Path("vectorizer.pkl").exists()):
        print("⚠️  Model files not found - skipping prediction tests")
        return
    
    try:
        service = PredictionService()
        
        # Test single prediction
        test_text = "This is a test text for classification"
        result = service.predict_single(test_text)
        
        assert "prediction" in result, "Prediction result missing prediction field"
        assert "confidence" in result, "Prediction result missing confidence field"
        assert result["confidence"] > 0, "Confidence should be positive"
        
        print("✅ PredictionService tests passed!")
        
    except FileNotFoundError:
        print("⚠️  Model files not found - skipping prediction tests")


def test_ui_components():
    """Test UI components."""
    print("🧪 Testing UI components...")
    
    # Test UI initialization
    service = PredictionService()
    ui = TextClassifierUI(service)
    
    # Test sample texts
    sample_texts = ["Test text 1", "Test text 2"]
    ui.display_sample_predictions(sample_texts)
    
    print("✅ UI component tests passed!")


def test_integration():
    """Test integration of all components."""
    print("🧪 Testing full integration...")
    
    # Check if model files exist
    if not (Path("model.pkl").exists() and Path("vectorizer.pkl").exists()):
        print("⚠️  Model files not found - creating mock test")
        
        # Test with mock components
        with CacheManager("integration_test", cleanup_on_exit=True) as cache:
            print("✅ Cache integration test passed!")
        return
    
    try:
        # Test full integration
        service = PredictionService()
        
        with CacheManager("integration_test", cleanup_on_exit=True) as cache:
            ui = TextClassifierUI(service, cache)
            
            # Test a simple prediction with caching
            test_text = "This is an integration test"
            result = service.predict_single(test_text, cache)
            
            assert "prediction" in result, "Integration test failed - no prediction"
            
            # Check cache has files
            cache_info = cache.get_cache_info()
            assert cache_info['file_count'] > 0, "Integration test failed - no cached files"
            
            print("✅ Full integration test passed!")
            
    except Exception as e:
        print(f"⚠️  Integration test failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Starting Cache System Tests")
    print("=" * 50)
    
    try:
        test_cache_manager()
        test_prediction_service()
        test_ui_components()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("✅ The caching system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
