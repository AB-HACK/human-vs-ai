# AI vs Human Essay Classification 🤖👤

sophisticated machine learning system for classifying essays as AI-generated or human-written, featuring a clean architecture with automatic caching and comprehensive analysis capabilities.

## ✨ Features

- **Clean Architecture**: Separated concerns with business logic, UI, and caching
- **Automatic Caching**: Temporary storage of all results with automatic cleanup
- **Interactive & Programmatic**: Both command-line and API interfaces
- **Comprehensive Analysis**: Visualizations, metrics, and detailed reports
- **Modular Design**: Reusable components for easy extension
- **Error Handling**: Robust error handling throughout the system

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd human-vs-ai

# Install dependencies
pip install -r requirements.txt

# Optional: Use virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the dataset from [Kaggle: Human vs AI Generated Essays](https://www.kaggle.com/datasets/ai-detector/human-vs-ai-generated-essays) and place `balanced_ai_human_prompts.csv` in the project root.

Expected format:
- `text`: Essay content
- `generated`: Label (0 = Human, 1 = AI)

### 3. Train the Model

**Option A: Quick Training with Caching**
```bash
python src/train_with_cache.py
```

**Option B: Complete Analysis Pipeline**
```bash
python kaggle_analysis.py
```

**Option C: Modular Training**
```bash
python -c "from src.modeling.train import train_model, save_model; model, vect, Xt, yt, yp = train_model('balanced_ai_human_prompts.csv'); save_model(model, vect)"
```

### 4. Make Predictions

**Interactive Mode with Caching**
```bash
python src/main.py
```

**Simple Interactive Mode**
```bash
python simple_predict.py
```

**Programmatic Usage**
```python
from src.prediction_service import PredictionService

service = PredictionService()
result = service.predict_single("Your text here")
print(f"Prediction: {result['prediction']} ({result['confidence']}%)")
```

## 🏗️ Architecture

### Core Components

```
src/
├── prediction_service.py     # Business logic for predictions
├── ui.py                     # User interface components  
├── cache_manager.py          # Temporary file management
├── main.py                   # Clean main entry point
├── modeling/
│   ├── train.py             # Training pipeline
│   └── predict.py           # Prediction utilities
└── train_with_cache.py      # Training with caching
```

### Design Principles

- **Separation of Concerns**: Business logic, UI, and caching are separate
- **Single Responsibility**: Each class has one clear purpose
- **Dependency Injection**: Components are loosely coupled
- **Error Handling**: Comprehensive exception handling
- **Clean Code**: No spaghetti code, maintainable structure

## 📊 Caching System

### Automatic Caching Features

- **Temporary Storage**: All results saved during execution
- **Automatic Cleanup**: Files deleted when program finishes
- **Multiple Formats**: JSON, pickle, CSV, PNG support
- **Context Manager**: Safe cleanup even on errors
- **Cache Statistics**: Monitor file counts and sizes

### Usage Example

```python
from src.cache_manager import CacheManager
from src.prediction_service import PredictionService

# Automatic caching and cleanup
with CacheManager("my_session", cleanup_on_exit=True) as cache:
    service = PredictionService()
    result = service.predict_single("Text to classify", cache_manager=cache)
    # Results automatically cached
# Cache automatically cleaned up here
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_cache_system.py
```

Tests include:
- Cache manager functionality
- Prediction service integration
- UI components
- Full system integration

## 📁 Project Structure

```
human-vs-ai/
├── README.md                     # This file
├── CACHING_SYSTEM_README.md      # Detailed caching documentation
├── requirements.txt               # Python dependencies
├── balanced_ai_human_prompts.csv  # Dataset (download from Kaggle)
├── kaggle_analysis.py            # Complete analysis pipeline
├── run_analysis.py               # Original reference script
├── simple_predict.py             # Simplified prediction script
├── test_cache_system.py          # Test suite
├── src/
│   ├── __init__.py
│   ├── main.py                   # Main entry point
│   ├── config.py                 # Configuration settings
│   ├── dataset.py                # Data loading utilities
│   ├── features.py               # Text preprocessing
│   ├── plots.py                  # Visualization functions
│   ├── prediction_service.py     # Core prediction logic
│   ├── ui.py                     # User interface
│   ├── cache_manager.py          # Caching system
│   ├── train_with_cache.py       # Training with caching
│   └── modeling/
│       ├── __init__.py
│       ├── train.py              # Training pipeline
│       └── predict.py            # Prediction utilities
└── venv/                         # Virtual environment (if used)
```

## 🔧 Configuration

### Model Settings
- **Algorithm**: Logistic Regression with TF-IDF features
- **Features**: Bigrams, max 10,000 features
- **Preprocessing**: Lowercase, lemmatization, stopword removal
- **Split**: 80/20 train/test with random state 42

### Cache Settings
- **Location**: System temporary directory
- **Naming**: Timestamped directories
- **Cleanup**: Automatic on program exit
- **Formats**: JSON, pickle, CSV, PNG

## 📈 Performance

### Model Performance
- **Accuracy**: ~85-90% on test set
- **Features**: TF-IDF with bigrams
- **Preprocessing**: NLTK-based text cleaning
- **Training Time**: ~2-5 minutes on typical hardware

### Cache Performance
- **Overhead**: Minimal impact on runtime
- **Storage**: Efficient temporary file management
- **Cleanup**: Fast directory removal
- **Safety**: Context manager ensures cleanup

## 🚨 Troubleshooting

### Common Issues

**Model files not found:**
```bash
# Train the model first
python src/train_with_cache.py
```

**NLTK download issues:**
```bash
# Manual download
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

**Permission errors on Windows:**
```powershell
# Set execution policy
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

**Import errors:**
```bash
# Ensure you're in the project directory
cd human-vs-ai
python src/main.py
```

### Dataset Issues

**Wrong column names:**
- Ensure your CSV has `text` and `generated` columns
- Or modify the loader in `src/dataset.py`

**File not found:**
- Place `balanced_ai_human_prompts.csv` in project root
- Or specify full path when calling training functions

## 🔄 Migration from Old Code

The project has been refactored from spaghetti code to clean architecture:

### Before (Issues)
- Mixed responsibilities in single files
- Hardcoded configurations
- Duplicate code
- Poor error handling
- Global state management

### After (Improvements)
- Clean separation of concerns
- Configurable parameters
- Reusable components
- Comprehensive error handling
- Dependency injection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_cache_system.py`
5. Submit a pull request

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Human vs AI Generated Essays](https://www.kaggle.com/datasets/ai-detector/human-vs-ai-generated-essays)
- Libraries: scikit-learn, NLTK, pandas, matplotlib, seaborn
- Architecture: Clean Code principles and SOLID design patterns

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite
3. Review the caching system documentation
4. Open an issue with detailed error information
5. you can find me on http://trybenode.space and 09029252005

---

**Happy Classifying! 🎯**

*This project demonstrates clean software architecture principles while solving a practical machine learning problem.*