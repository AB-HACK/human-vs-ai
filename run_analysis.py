"""
Complete AI vs Human essay classification script.
This matches your working Kaggle notebook code.
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

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
data = pd.read_csv('/kaggle/input/human-vs-ai-generated-essays/balanced_ai_human_prompts.csv')  

# Ensure column names match your data
data.columns = ['text', 'generated']  

# 1. Visualize class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='generated', data=data)
plt.title('Class Distribution (0: Human, 1: AI)')
plt.xlabel('Text Type')
plt.ylabel('Count')
plt.show()

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# 2. Word Clouds
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

plot_wordcloud(' '.join(data[data['generated'] == 0]['cleaned_text']), 'Human Texts')
plot_wordcloud(' '.join(data[data['generated'] == 1]['cleaned_text']), 'AI Texts')

# Prepare data
X = data['cleaned_text']
y = data['generated']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
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

# Classification function
def classify_text(text):
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(text_vectorized)
    return "AI-generated" if prediction[0] == 1 else "Human-written"

# Test
sample_text = "The economy, a complex system of production, distribution, and consumption of goods and services,"
print("\nSample classification:", classify_text(sample_text))
