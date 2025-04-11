# 📘 Amazon Review Sentiment Analysis

This project is a machine learning application that analyzes the sentiment of Amazon product reviews. It uses Natural Language Processing (NLP) techniques and multiple classification models to determine whether a review expresses a **positive**, **negative**, or **neutral** sentiment.

---

## 🔍 Project Overview

The main goal of this project is to build and evaluate sentiment classification models using Amazon product reviews data. The models are trained using TF-IDF vectorization and compared using accuracy and F1 score metrics. A Streamlit web app is also included for interactive prediction.

---

## 📂 Project Structure

```
review_sentiment_analysis/
│
├── app/
│   └── app.py                      # Streamlit application for sentiment prediction
│
├── src/
│   ├── data_preprocessing.py       # Functions for preprocessing and balancing data
│   ├── model_training.py           # Model training and evaluation logic
│   └── utils.py                    # Utility functions (e.g., clean_text)
│
├── models/
│   ├── review_sentiment_classifier_model.pkl
│   └── review_sentiment_classifier_vectorizer.pkl
│
├── notebooks/
│   └── analysis_visualizations.ipynb  # Visual comparison of different ML models
│
├── visualizations/
│   ├── model_comparison.png       # Accuracy and F1 bar plot
│   └── svm_boundary.png           # SVM decision boundary on synthetic data
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Files to ignore by Git
```

---

## 🧠 Models Used

- **Support Vector Machine (SVM)** – RBF kernel
- **Naive Bayes**
- **Logistic Regression**

> ✅ Decision Tree was trained but excluded from the final comparison due to overfitting tendencies.

---

## 📊 Model Evaluation

| Model              | Accuracy | F1 Score (Macro Avg) |
|-------------------|----------|----------------------|
| SVM               | ✅ Best  | ✅ Best              |
| Naive Bayes       | Good     | Good                 |
| Logistic Regression | Moderate | Moderate           |

Evaluation metrics used:
- Accuracy
- F1 Score (macro average across Positive, Negative, Neutral)

All metrics were visualized in a bar chart.

---

## 🧪 Input/Output Demo

```
Input:  "The product quality was amazing and delivery was quick."
Output: "Predicted Sentiment: Positive"
```

---

## 🌐 Streamlit Web App

To launch the interactive sentiment analyzer:

```bash
cd app
streamlit run app.py
```

Paste your review and click “Analyze” to see the predicted sentiment.

---

## 🧹 Preprocessing Steps

- Text cleaning (removing punctuation, lowering case)
- TF-IDF Vectorization
- Balanced dataset using equal class distribution
- Train/test split

---

## 📂 How to Use

### 1. Clone the Repo

```bash
git clone https://github.com/Sanchit-aiml/review_sentiment-analysis.git
cd review_sentiment-analysis
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
cd app
streamlit run app.py
```

---

## 🛠 Tools & Libraries

- Python 3.x
- Scikit-learn
- Streamlit
- Numpy, Pandas, Matplotlib
- Pickle (for saving model/vectorizer)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤛🏼‍♂️ Author

**Sanchit**  
🔗 [GitHub Profile](https://github.com/Sanchit-aiml)

---





