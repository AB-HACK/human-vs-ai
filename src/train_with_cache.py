"""
Training script with caching system for AI vs Human essay classification.
"""
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from modeling.train import train_model, save_model
from cache_manager import CacheManager


def main():
    """Main training entry point with caching."""
    # Configuration
    dataset_path = "balanced_ai_human_prompts.csv"
    
    # Initialize cache manager with automatic cleanup
    with CacheManager("training_cache", cleanup_on_exit=True) as cache_manager:
        print("🚀 AI vs Human Essay Classifier - Training with Caching")
        print("=" * 60)
        print(f"📁 Temporary cache: {cache_manager.cache_dir}")
        print("ℹ️  All training results will be saved temporarily and cleaned up on exit")
        print()
        
        try:
            # Train the model with caching
            print("🔧 Starting model training...")
            model, vectorizer, X_test, y_test, y_pred = train_model(
                file_path=dataset_path, 
                cache_manager=cache_manager
            )
            
            if model is not None:
                print("\n✅ Training completed successfully!")
                
                # Save model to permanent location
                save_model(model, vectorizer, "model.pkl", "vectorizer.pkl")
                
                # Show cache info
                print("\n" + "=" * 50)
                cache_info = cache_manager.get_cache_info()
                print(f"📊 Cache Statistics:")
                print(f"   Files created: {cache_info['file_count']}")
                print(f"   Total size: {cache_info['total_size']} bytes")
                print(f"   Cached files: {len(cache_manager.list_cached_files())}")
                
                print("\n🏁 Training finished - cache will be cleaned up automatically")
            else:
                print("❌ Training failed!")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise


if __name__ == "__main__":
    main()
