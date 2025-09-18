"""
Visualization utilities for the AI vs Human essay classification project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import numpy as np

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_class_distribution(data, column='generated', title="Class Distribution (0: Human, 1: AI)"):
    """
    Plot the distribution of classes.
    
    Args:
        data: DataFrame with the data
        column (str): Column name for the target variable
        title (str): Plot title
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column, data=data)
    plt.title(title)
    plt.xlabel('Text Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Human', 'AI']):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, 
                yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def create_wordcloud(text, title="Word Cloud", max_words=100):
    """
    Create and display a word cloud.
    
    Args:
        text: Text data (string or list of strings)
        title (str): Title for the plot
        max_words (int): Maximum number of words to display
    """
    if isinstance(text, list):
        text = ' '.join(text)
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=max_words
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(clf, vectorizer, top_n=20):
    """
    Plot feature importance for logistic regression.
    
    Args:
        clf: Trained classifier
        vectorizer: Fitted vectorizer
        top_n (int): Number of top features to show
    """
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    
    # Top positive features (AI-predictive)
    top_positive = pd.DataFrame({'feature': feature_names, 'coef': coef}
                              ).sort_values('coef', ascending=False).head(top_n)
    
    # Top negative features (Human-predictive)
    top_negative = pd.DataFrame({'feature': feature_names, 'coef': coef}
                              ).sort_values('coef', ascending=True).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_positive['feature'], top_positive['coef'], color='green')
    plt.title(f'Top {top_n} AI-predictive Features')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_negative['feature'], top_negative['coef'], color='red')
    plt.title(f'Top {top_n} Human-predictive Features')
    plt.tight_layout()
    plt.show()