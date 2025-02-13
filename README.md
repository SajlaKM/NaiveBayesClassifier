# NaiveBayesClassifier
Text classification using Naive Bayes Classifier
# Naïve Bayes Text Classification

## Overview
This project demonstrates the implementation of a **Naïve Bayes classifier** for **sentiment analysis**, classifying text messages as either **positive** or **negative**. The model is trained using the **Multinomial Naïve Bayes** algorithm, which is well-suited for text classification tasks.

## Dataset Information
The dataset consists of a small set of manually labeled text messages:
- **Positive (pos)**: Messages expressing positive sentiment.
- **Negative (neg)**: Messages expressing negative sentiment.

Labels are converted into numerical values:
- **pos** → `1`
- **neg** → `0`

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas scikit-learn
```

## Implementation Steps
### **1. Load and Prepare Data**
```python
import pandas as pd

# Sample dataset: Positive vs. Negative sentiment classification
data = {
    'message': [
        "I love this movie, it's amazing!",
        "The product is terrible, I hate it.",
        "Such a wonderful experience!",
        "This was the worst decision ever.",
        "I highly recommend this!",
        "Absolutely awful, never buying again."
    ],
    'label': ['pos', 'neg', 'pos', 'neg', 'pos', 'neg']
}

# Convert dataset to DataFrame
msg = pd.DataFrame(data)

# Convert labels to numerical values (pos=1, neg=0)
msg['labelnum'] = msg['label'].map({'pos': 1, 'neg': 0})
```

### **2. Split Dataset**
```python
from sklearn.model_selection import train_test_split

X = msg['message']
y = msg['labelnum']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **3. Convert Text to Features**
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
```

### **4. Train Naïve Bayes Classifier**
```python
from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_transformed, y_train)
```

### **5. Make Predictions and Evaluate Model**
```python
from sklearn import metrics

y_pred = nb_classifier.predict(X_test_transformed)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred, target_names=['neg', 'pos']))
```

## Expected Output
- Accuracy, Precision, and Recall scores.
- Classification report showing precision, recall, and f1-score for both classes (pos, neg).

## Customization
- Replace the sample dataset with **real-world text data** (e.g., IMDB reviews, Twitter sentiment analysis, etc.).
- Use **TF-IDF Vectorization** (`TfidfVectorizer`) for improved feature extraction.
- Experiment with **n-grams** and **stopword removal** for better performance.

## Conclusion
This project provides a simple yet effective implementation of **Naïve Bayes text classification**. It is useful for sentiment analysis tasks such as **spam detection**, **product reviews classification**, and **customer feedback analysis**.

---
**Author:** Sajla KM 
**License:** MIT

