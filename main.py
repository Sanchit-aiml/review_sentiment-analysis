# main.py - Core training, evaluation, and visualization

import json
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score < 3:
            return "Negative"
        elif self.score > 3:
            return "Positive"
        else:
            return "Neutral"

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == "Negative", self.reviews))
        positive = list(filter(lambda x: x.sentiment == "Positive", self.reviews))
        neutral = list(filter(lambda x: x.sentiment == "Neutral", self.reviews))
        size = min(len(negative), len(positive), len(neutral))
        self.reviews = negative[:size] + positive[:size] + neutral[:size]
        random.shuffle(self.reviews)

# Load and preprocess data
file_name = "Books_small_10000.json"
reviews = []
with open(file_name) as f:
    for line in f:
        data = json.loads(line)
        reviews.append(Review(data["reviewText"], data["overall"]))

training, testing = train_test_split(reviews, test_size=0.33, random_state=42)
train_cont = ReviewContainer(training)
test_cont = ReviewContainer(testing)
train_cont.evenly_distribute()
test_cont.evenly_distribute()

train_x, train_y = train_cont.get_text(), train_cont.get_sentiment()
test_x, test_y = test_cont.get_text(), test_cont.get_sentiment()

vectorizer = TfidfVectorizer()
train_x_vec = vectorizer.fit_transform(train_x)
test_x_vec = vectorizer.transform(test_x)

# Train classifiers
clf_svm = svm.SVC(kernel='rbf', C=1)
clf_svm.fit(train_x_vec, train_y)
clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vec.toarray(), train_y)
clf_log = LogisticRegression()
clf_log.fit(train_x_vec, train_y)

# Scores
score_svm = clf_svm.score(test_x_vec, test_y)
score_gnb = clf_gnb.score(test_x_vec.toarray(), test_y)
score_log = clf_log.score(test_x_vec, test_y)

f1_svm = f1_score(test_y, clf_svm.predict(test_x_vec), average='macro', labels=['Positive', 'Negative', 'Neutral'])
f1_gnb = f1_score(test_y, clf_gnb.predict(test_x_vec.toarray()), average='macro', labels=['Positive', 'Negative', 'Neutral'])
f1_log = f1_score(test_y, clf_log.predict(test_x_vec), average='macro', labels=['Positive', 'Negative', 'Neutral'])

# Plot comparison
model_names = ['SVM', 'Naive Bayes', 'Logistic Regression']
accuracies = [score_svm, score_gnb, score_log]
f1_scores = [f1_svm, f1_gnb, f1_log]

x = range(len(model_names))
bar_width = 0.4
plt.figure(figsize=(10, 6))
plt.bar(x, accuracies, width=bar_width, label='Accuracy', color='skyblue')
plt.bar([i + bar_width for i in x], f1_scores, width=bar_width, label='F1 Score', color='salmon')
plt.xticks([i + bar_width / 2 for i in x], model_names)
plt.ylabel("Score")
plt.title("Model Comparison: Accuracy vs F1 Score")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# Save best model and vectorizer
with open("models/review_sentiment_classifier_model.pkl", "wb") as f:
    pickle.dump(clf_svm, f)
with open("models/review_sentiment_classifier_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
