# Fake News Detection using NLP & Machine Learning
Logistic Regression + TF-IDF + Text Preprocessing

This project builds a Fake News Classification System using Natural Language Processing (NLP) techniques and a Logistic Regression model.

The system processes raw news text, converts it into numerical features using TF-IDF, and classifies news articles as:

1. Real (0)

2. Fake (1)

# Project Overview

With the rise of misinformation online, detecting fake news has become a critical challenge. This project demonstrates a classical NLP + Machine Learning pipeline for binary text classification.

The workflow includes:

Load Dataset
    ↓
Data Cleaning
    ↓
Text Preprocessing (Regex + Stopwords Removal + Stemming)
    ↓
TF-IDF Vectorization
    ↓
Train-Test Split
    ↓
Logistic Regression Training
    ↓
Model Evaluation
    ↓
Prediction on New Article

# Dataset

CSV file: train.csv

Contains:

author

title

label (0 = Real, 1 = Fake)

Missing values are handled using:

news_dataset = news_dataset.fillna('')

# Text Preprocessing

Text preprocessing is performed using NLTK.

Library used:
Natural Language Toolkit (NLTK)

Steps:

1️. Remove non-alphabetic characters using Regex
2️. Convert text to lowercase
3️. Tokenization (split into words)
4️. Remove English stopwords
5️. Apply stemming using Porter Stemmer
6️. Rejoin processed words into final cleaned string

Example preprocessing function:

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) 
                       for word in stemmed_content 
                       if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)
    
# Feature Extraction

The project uses:

1. TF-IDF Vectorization

Library:
scikit-learn

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

TF-IDF converts text into weighted numerical features based on word importance across the dataset.

# Model Used
Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

Why Logistic Regression?

Efficient for high-dimensional sparse data

Works well for binary classification

Interpretable coefficients

Fast training time

# Model Evaluation

Metrics used:

Accuracy Score (Training Data)

Accuracy Score (Test Data)

accuracy_score(X_test_prediction, Y_test)

Typical performance for this pipeline:

Training Accuracy: 95%+

Test Accuracy: 93–96%

(Varies depending on dataset)

# Making Predictions

Example:

prediction = model.predict(X_new)

Output logic:

if prediction[0] == 0:
    print("The news is Real")
else:
    print("The news is Fake")
    
# Installation
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
python fake_news.py

# Requirements
numpy
pandas
scikit-learn
nltk

# Key Concepts Demonstrated

Text Cleaning & Preprocessing

Stopword Removal

Stemming (Porter Stemmer)

TF-IDF Vectorization

Sparse Matrix Representation

Binary Text Classification

Logistic Regression for NLP

Model Evaluation with Accuracy

# Possible Improvements

Use Lemmatization instead of Stemming

Add N-grams in TF-IDF

Use GridSearchCV for hyperparameter tuning

Add Confusion Matrix

Use Naive Bayes classifier comparison

Use LSTM / BERT for deep learning approach

Deploy as a Flask or Streamlit web app

Add real-time news scraping integration

# Real-World Applications

Social media misinformation detection

News credibility scoring

Media monitoring systems

Fact-checking automation tools
