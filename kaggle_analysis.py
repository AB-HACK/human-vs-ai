"""
Kaggle AI vs Human Essay Classification
balanced_ai_human_prompts.csv
"""
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud

print("AI vs Human Essay Classifier")
print("Dataset: human-vs-ai-generated-essays/balanced_ai_human_prompts.csv")
print("=" * 60)

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Text preprocessing function"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data from Kaggle dataset
print("Loading Kaggle dataset...")
try:
    data = pd.read_csv('balanced_ai_human_prompts.csv')
    print(f"Dataset loaded successfully! Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
except FileNotFoundError:
    print("Dataset not found! Make sure you're running this in Kaggle or have the dataset uploaded.")
    print("Dataset URL: https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays")
    exit()

# Ensure column names match your data
data.columns = ['text', 'generated']
print(f"Class distribution:")
print(data['generated'].value_counts())
print(f"   - 0: Human-written ({data[data['generated']==0].shape[0]} samples)")
print(f"   - 1: AI-generated ({data[data['generated']==1].shape[0]} samples)")

# 1. Visualize class distribution
print("\n Creating visualizations...")
plt.figure(figsize=(8, 5))
sns.countplot(x='generated', data=data)
plt.title('Class Distribution (0: Human, 1: AI)')
plt.xlabel('Text Type')
plt.ylabel('Count')
plt.show()

# Apply preprocessing
print("Preprocessing text...")
data['cleaned_text'] = data['text'].apply(preprocess_text)

# 2. Word Clouds
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

print("Creating word clouds...")
plot_wordcloud(' '.join(data[data['generated'] == 0]['cleaned_text']), 'Human Texts')
plot_wordcloud(' '.join(data[data['generated'] == 1]['cleaned_text']), 'AI Texts')

# Prepare data
X = data['cleaned_text']
y = data['generated']

# Vectorize text
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
classifier.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 3. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'], 
                yticklabels=['Human', 'AI'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# 4. Feature Importance
def plot_feature_importance(clf, vectorizer, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    top_positive = pd.DataFrame({'feature': feature_names, 'coef': coef}
                              ).sort_values('coef', ascending=False).head(top_n)
    top_negative = pd.DataFrame({'feature': feature_names, 'coef': coef}
                              ).sort_values('coef', ascending=True).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_positive['feature'], top_positive['coef'], color='green')
    plt.title(f'Top {top_n} AI-predictive Features')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_negative['feature'], top_negative['coef'], color='red')
    plt.title(f'Top {top_n} Human-predictive Features')
    plt.show()

plot_feature_importance(classifier, vectorizer)

# Save model for prediction
print("Saving model...")
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Model saved as 'model.pkl' and 'vectorizer.pkl'")

# Classification function
def classify_text(text):
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(text_vectorized)
    return "AI-generated" if prediction[0] == 1 else "Human-written"

# Test with sample texts
print("\nTesting with sample texts:")
sample_texts = [
    "The economy, a complex system of production, distribution, and consumption of goods and services, plays a crucial role in shaping society.",
    "I love spending time with my family on weekends. We usually go to the park and have a picnic together.",
    "Artificial intelligence represents a paradigm shift in computational capabilities, enabling machines to process information and make decisions with unprecedented efficiency."
]

for i, text in enumerate(sample_texts, 1):
    result = classify_text(text)
    print(f"{i}. {result}")
    print(f"   Text: {text[:60]}...")
    print()

print("Analysis complete! Use 'simple_predict.py' to classify new texts.")
