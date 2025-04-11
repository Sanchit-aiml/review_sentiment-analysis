# Amazon Review Sentiment Analysis

This project classifies Amazon product reviews into Positive, Negative, or Neutral sentiments using Natural Language Processing and Machine Learning.

## Features
- Text preprocessing
- Model comparison: SVM, Naive Bayes, Logistic Regression
- Evaluation metrics: Accuracy and F1 Score
- Streamlit app for real-time predictions

## Setup

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset
Uses a subset of the Amazon product reviews (Books category).

## Outputs
- Trained SVM model and TF-IDF vectorizer
- Model performance bar plot
