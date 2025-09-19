# AI vs Human Essay Classification

This project classifies essays as either AI-generated or human-written using machine learning techniques.

## 📊 Dataset

**Kaggle Dataset**: [human-vs-ai-generated-essays](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays)
- **File**: `balanced_ai_human_prompts.csv`
- **Columns**: 
  - `text`: Essay content
  - `generated`: Label (0 = Human, 1 = AI)
- **Size**: Balanced dataset with equal human and AI samples

## 🚀 Quick Start

### Option 1: Run Complete Analysis (Recommended)
```python
# In Kaggle or Colab
exec(open('kaggle_analysis.py').read())
```

### Option 2: Use Modular Approach
```python
# Install dependencies
!pip install -r requirements.txt

# Train model
from src.modeling.train import train_model
model, vectorizer, X_test, y_test, y_pred = train_model()

# Make predictions
from simple_predict import classify_text
result = classify_text("Your text here")
print(result)
```

## 📁 Project Structure

```
human-vs-ai/
├── kaggle_analysis.py          # Complete analysis script (use this!)
├── simple_predict.py           # Simple prediction script
├── run_analysis.py             # Original working script
├── requirements.txt            # Dependencies
├── src/                        # Modular code
│   ├── config.py              # Configuration & NLTK setup
│   ├── dataset.py             # Data loading utilities
│   ├── features.py            # Text preprocessing
│   ├── plots.py               # Visualizations
│   └── modeling/
│       ├── train.py           # Training pipeline
│       └── predict.py         # Prediction utilities
└── README.md                  # This file
```

## 🎯 Usage Examples

### 1. Complete Analysis
```python
# Run the full analysis with visualizations
exec(open('kaggle_analysis.py').read())
```

### 2. Simple Prediction
```python
# After training, classify new texts
from simple_predict import classify_text, classify_with_confidence

# Simple classification
result = classify_text("Your essay text here")
print(result)  # "🤖 AI-generated" or "👤 Human-written"

# With confidence
result = classify_with_confidence("Your essay text here")
print(f"Result: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
```

### 3. Interactive Prediction
```python
# Run interactive mode
exec(open('simple_predict.py').read())
```

## 📈 What the Analysis Includes

1. **Data Loading**: Loads Kaggle dataset automatically
2. **Visualizations**: 
   - Class distribution plot
   - Word clouds for human vs AI texts
   - Confusion matrix
   - Feature importance plots
3. **Model Training**: Logistic Regression with TF-IDF features
4. **Evaluation**: Accuracy, classification report, confusion matrix
5. **Prediction**: Ready-to-use classification functions

## 🔧 Technical Details

- **Preprocessing**: Text cleaning, lemmatization, stopword removal
- **Features**: TF-IDF with unigrams and bigrams (5000 features)
- **Model**: Logistic Regression with balanced class weights
- **Evaluation**: 80/20 train-test split with stratification

## 📦 Dependencies

All dependencies are in `requirements.txt`:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy, statsmodels
- nltk, wordcloud
- kaggle (for dataset access)

## 🎉 Results

The model typically achieves:
- **Accuracy**: 85-95% on test set
- **Features**: Identifies key linguistic patterns
- **Visualizations**: Clear insights into human vs AI writing patterns

## 💡 Tips

1. **Use `kaggle_analysis.py`** for the complete experience
2. **Run in Kaggle** for direct dataset access
3. **Use `simple_predict.py`** for quick predictions
4. **Check the visualizations** to understand model behavior

## 🔗 Links

- **Dataset**: https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays
- **Kaggle Notebook**: Upload and run `kaggle_analysis.py`
