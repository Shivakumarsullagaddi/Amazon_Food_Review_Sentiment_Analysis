# Phase 4: Model Building - Naive Bayes Classifier

Documentation for [`4_model_building_naive_bayes_classifier_prod.ipynb`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/4_model_building_naive_bayes_classifier_prod.ipynb).

---

## 1. Objective
Train, evaluate, and test a supervised probabilistic text classification model ([`MultinomialNB`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/4_model_building_naive_bayes_classifier_prod.ipynb)) using unigram and bigram TF-IDF features to categorize customer reviews into positive or negative sentiment.

---

## 2. Methodology & Implementation

### 2.1 Dataset Partitioning
The dataset of 365,331 reviews is split into training (80%) and testing (20%) sets. Stratification ensures both sets maintain identical class distributions:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("preprocessed_reviews.csv")
X = df["CleanedText"]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- Training partition: 292,264 reviews
- Testing partition: 73,067 reviews

### 2.2 TF-IDF Feature Transformation
Feature transformation strictly enforces training isolation to eliminate data leakage:
- Vocabulary and IDF statistics are fitted exclusively on `X_train`.
- `X_test` is transformed using learned parameters without refitting.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = X_train.fillna("").astype(str)
X_test = X_test.fillna("").astype(str)

tfidf_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

- Matrix dimension (train): `(292264, 8000)`
- Matrix dimension (test): `(73067, 8000)`

### 2.3 Model Training & Prediction
A Multinomial Naive Bayes classifier is trained on the TF-IDF feature space:

$$P(c \mid d) \propto P(c) \prod_{i=1}^{n} P(w_i \mid c)$$

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred = nb_model.predict(X_test_tfidf)
```

---

## 3. Quantitative Evaluation

### 3.1 Overall Metrics
- **Model Accuracy**: **90.10%**

### 3.2 Confusion Matrix
```
                  Predicted Negative    Predicted Positive
Actual Negative          4,727                 6,746
Actual Positive            484                61,110
```

### 3.3 Detailed Classification Report

| Sentiment Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | **0.91** | **0.41** | **0.57** | 11,473 |
| **Positive** | **0.90** | **0.99** | **0.94** | 61,594 |
| **Accuracy** | | | **0.90** | 73,067 |
| **Macro Average** | 0.90 | 0.70 | 0.76 | 73,067 |
| **Weighted Average** | 0.90 | 0.90 | 0.88 | 73,067 |

### 3.4 Confusion Matrix Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes Sentiment Classification - TF-IDF Features")
plt.show()
```

---

## 4. Inference on Unseen Sample Reviews

```python
new_reviews = [
    "I absolutely love this chips! Its taste is soo good.",
    "This is the worst experience I have ever had.",
    "It is okay, not too bad but not great either.",
    "I am very satisfied with the quality and performance.",
    "Totally disappointed, waste of money."
]

new_tfidf = tfidf_vectorizer.transform(new_reviews)
predictions = nb_model.predict(new_tfidf)

for review, sentiment in zip(new_reviews, predictions):
    print(review, "->", sentiment)
```

### Inference Results
- `"I absolutely love this chips! Its taste is soo good."` -> **Positive**
- `"This is the worst experience I have ever had."` -> **Negative**
- `"It is okay, not too bad but not great either."` -> **Positive**
- `"I am very satisfied with the quality and performance."` -> **Positive**
- `"Totally disappointed, waste of money."` -> **Negative**

---

## 5. Mentor Insights & Evaluation Discrepancies
- **Imbalance Distortion**: While overall accuracy is 90.10%, the macro recall is 70% due to class imbalance (84.3% positive). 
- **Precision vs Recall for Negative Class**: Negative reviews exhibit high precision (0.91) but lower recall (0.41). When the model declares a review negative, it is correct 91% of the time; however, it misclassifies 6,746 negative reviews as positive.
- **Root Cause**: Prior probabilities derived from training class frequencies bias probability thresholds toward the positive class in ambiguous samples. Setting `fit_prior=False` or adjusting decision probability thresholds can improve negative recall.
