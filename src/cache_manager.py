"""
Temporary caching system for AI vs Human essay classification project.
Handles saving results temporarily and automatic cleanup.
"""
import os
import tempfile
import shutil
import json
import pickle
import atexit
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd


class CacheManager:
    """
    Manages temporary files and caching for the AI vs Human classification project.
    Automatically creates a temporary directory and cleans up on exit.
    """
    
    def __init__(self, cache_name: str = "ai_human_cache", cleanup_on_exit: bool = True):
        """
        Initialize the cache manager.
        
        Args:
            cache_name (str): Name for the cache directory
            cleanup_on_exit (bool): Whether to clean up on program exit
        """
        self.cache_name = cache_name
        self.cleanup_on_exit = cleanup_on_exit
        self.cache_dir = None
        self.created_files = []
        self._setup_cache()
        
        if cleanup_on_exit:
            atexit.register(self.cleanup)
    
    def _setup_cache(self):
        """Set up the temporary cache directory."""
        # Create a unique temporary directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_base = tempfile.gettempdir()
        self.cache_dir = Path(temp_base) / f"{self.cache_name}_{timestamp}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Cache directory created: {self.cache_dir}")
    
    def save_prediction_result(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save a prediction result to cache.
        
        Args:
            result (dict): Prediction result dictionary
            filename (str, optional): Custom filename. If None, generates one.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"prediction_{timestamp}.json"
        
        file_path = self.cache_dir / filename
        
        # Add metadata
        result_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "cache_dir": str(self.cache_dir),
            "result": result
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_with_meta, f, indent=2, ensure_ascii=False)
        
        self.created_files.append(file_path)
        print(f"💾 Prediction result saved: {file_path}")
        return str(file_path)
    
    def save_multiple_predictions(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save multiple prediction results to cache.
        
        Args:
            results (list): List of prediction result dictionaries
            filename (str, optional): Custom filename. If None, generates one.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"predictions_batch_{timestamp}.json"
        
        file_path = self.cache_dir / filename
        
        # Add metadata
        results_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "cache_dir": str(self.cache_dir),
            "count": len(results),
            "results": results
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
        
        self.created_files.append(file_path)
        print(f"💾 Batch predictions saved: {file_path} ({len(results)} results)")
        return str(file_path)
    
    def save_model_artifacts(self, model: Any, vectorizer: Any, 
                           model_filename: Optional[str] = None, 
                           vectorizer_filename: Optional[str] = None) -> Dict[str, str]:
        """
        Save model and vectorizer to cache.
        
        Args:
            model: Trained model object
            vectorizer: Fitted vectorizer object
            model_filename (str, optional): Custom model filename
            vectorizer_filename (str, optional): Custom vectorizer filename
            
        Returns:
            dict: Paths to saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_filename is None:
            model_filename = f"model_{timestamp}.pkl"
        if vectorizer_filename is None:
            vectorizer_filename = f"vectorizer_{timestamp}.pkl"
        
        model_path = self.cache_dir / model_filename
        vectorizer_path = self.cache_dir / vectorizer_filename
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        self.created_files.extend([model_path, vectorizer_path])
        
        paths = {
            "model": str(model_path),
            "vectorizer": str(vectorizer_path)
        }
        
        print(f"💾 Model artifacts saved: {paths}")
        return paths
    
    def save_training_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save training results to cache.
        
        Args:
            results (dict): Training results dictionary
            filename (str, optional): Custom filename. If None, generates one.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"training_results_{timestamp}.json"
        
        file_path = self.cache_dir / filename
        
        # Add metadata
        results_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "cache_dir": str(self.cache_dir),
            "results": results
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
        
        self.created_files.append(file_path)
        print(f"💾 Training results saved: {file_path}")
        return str(file_path)
    
    def save_plot(self, fig, filename: Optional[str] = None) -> str:
        """
        Save a matplotlib figure to cache.
        
        Args:
            fig: Matplotlib figure object
            filename (str, optional): Custom filename. If None, generates one.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"plot_{timestamp}.png"
        
        file_path = self.cache_dir / filename
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        
        self.created_files.append(file_path)
        print(f"💾 Plot saved: {file_path}")
        return str(file_path)
    
    def save_dataframe(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save a pandas DataFrame to cache.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str, optional): Custom filename. If None, generates one.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"dataframe_{timestamp}.csv"
        
        file_path = self.cache_dir / filename
        df.to_csv(file_path, index=False)
        
        self.created_files.append(file_path)
        print(f"💾 DataFrame saved: {file_path}")
        return str(file_path)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache.
        
        Returns:
            dict: Cache information
        """
        return {
            "cache_dir": str(self.cache_dir),
            "created_files": [str(f) for f in self.created_files],
            "file_count": len(self.created_files),
            "total_size": sum(f.stat().st_size for f in self.created_files if f.exists())
        }
    
    def list_cached_files(self) -> List[str]:
        """
        List all cached files.
        
        Returns:
            list: List of cached file paths
        """
        return [str(f) for f in self.created_files if f.exists()]
    
    def cleanup(self):
        """
        Clean up the cache directory and all created files.
        """
        if self.cache_dir and self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                print(f"🗑️ Cache directory cleaned up: {self.cache_dir}")
                self.created_files.clear()
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up cache directory {self.cache_dir}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if configured."""
        if self.cleanup_on_exit:
            self.cleanup()
    
    def __del__(self):
        """Destructor - cleanup if configured."""
        if self.cleanup_on_exit and hasattr(self, 'cache_dir'):
            self.cleanup()


# Global cache manager instance
_global_cache = None

def get_cache_manager(cache_name: str = "ai_human_cache", cleanup_on_exit: bool = True) -> CacheManager:
    """
    Get or create a global cache manager instance.
    
    Args:
        cache_name (str): Name for the cache directory
        cleanup_on_exit (bool): Whether to clean up on program exit
        
    Returns:
        CacheManager: Global cache manager instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(cache_name, cleanup_on_exit)
    return _global_cache

def cleanup_global_cache():
    """Manually clean up the global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.cleanup()
        _global_cache = None

