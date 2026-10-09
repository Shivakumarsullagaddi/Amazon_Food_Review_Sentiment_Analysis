# Phase 3: Feature Extraction and Vectorization

Documentation for [`3_feature_extraction_prod.ipynb`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/3_feature_extraction_prod.ipynb).

---

## 1. Objective
Convert preprocessed review text into quantitative numerical representations suitable for machine learning algorithms, comparing Bag-of-Words (BoW), N-grams, and Term Frequency - Inverse Document Frequency (TF-IDF).

---

## 2. Feature Extraction Architectures

### 2.1 Bag-of-Words (BoW)
Bag-of-Words converts each review into a count vector of words present in the corpus vocabulary. It discards word order and syntax while preserving word frequency.

- Implementation: [`CountVectorizer`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/3_feature_extraction_prod.ipynb)
- Vocabulary cutoff: `max_features=50000`
- Representation: Compressed Sparse Row matrix (`scipy.sparse._csr.csr_matrix`) to conserve memory.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("preprocessed_reviews.csv")
preprocessed_reviews = df["CleanedText"].astype(str).values

count_vect = CountVectorizer(max_features=50000)
count_vect.fit(preprocessed_reviews)
final_counts = count_vect.transform(preprocessed_reviews)
```

#### Matrix Sparsity
Because each review contains only a tiny fraction of the global 50,000 vocabulary words, the vast majority of cells in the document-term matrix contain zeros. Inspecting a sample slice demonstrates how only relevant tokens populate values:

```python
word_freq = np.array(final_counts.sum(axis=0)).flatten()
top_cols = word_freq.argsort()[-10:]
sample = final_counts[:5][:, top_cols].toarray()
pd.DataFrame(sample, columns=count_vect.get_feature_names_out()[top_cols])
```

---

### 2.2 N-Grams (Unigrams & Bigrams)
Single-word representations lose local phrase context (for example, `good` vs `not good`). N-grams retain contiguous sequences of $N$ words:
- 1-gram (unigram): Individual word frequency (`good`).
- 2-gram (bigram): Pair-wise sequential phrase dependencies (`not good`, `highly recommend`).

```python
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=10, max_features=5000)
ngram_vectorizer.fit(preprocessed_reviews)
final_ngram_counts = ngram_vectorizer.transform(preprocessed_reviews)
```

---

### 2.3 TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF balances local word prominence against global prevalence. High frequency in an individual document increases weight, while high occurrence across the entire corpus decreases weight:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit(df["CleanedText"].astype(str).values)
final_tfidf = tfidf_vectorizer.transform(df["CleanedText"].astype(str).values)
```

#### Evaluation: TF-IDF Weighted Word Cloud
A corpus-level TF-IDF score is computed by summing weights across all reviews. This generates a Word Cloud reflecting term significance rather than raw word counts:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

tfidf_scores = np.asarray(final_tfidf.sum(axis=0)).flatten()
feature_names = tfidf_vectorizer.get_feature_names_out()
word_scores = dict(zip(feature_names, tfidf_scores))

wc = WordCloud(width=1200, height=700, background_color="white")
wc.generate_from_frequencies(word_scores)

plt.figure(figsize=(12, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("TF-IDF Weighted Word Cloud")
plt.show()
```

---

### 2.4 TF-IDF with Bigrams & Sparsity Analysis
Combining TF-IDF with unigrams and bigrams (`ngram_range=(1, 2)`) provides both single-token signals and multi-word semantic cues.

- Configuration: `max_features=8000`, `ngram_range=(1, 2)`
- Target Matrix: `(365331, 8000)`

```python
tfidf_bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
tfidf_bigram_vectorizer.fit(df["CleanedText"].astype(str).values)
final_tfidf_bigram = tfidf_bigram_vectorizer.transform(df["CleanedText"].astype(str).values)
```

#### Top Bigrams Visualization
Filtering the vocabulary for bigrams (`r'^\w+\s\w+$'`) surfaces key customer phrases:
- Positive drivers: `highly recommend`, `great taste`, `good product`
- Domain attributes: `gluten free`, `green tea`, `grocery store`, `olive oil`

```python
bigram_scores = np.asarray(final_tfidf_bigram.sum(axis=0)).flatten()
bigram_features = tfidf_bigram_vectorizer.get_feature_names_out()

bigram_df = pd.DataFrame({"ngram": bigram_features, "score": bigram_scores})
bigram_df = bigram_df[bigram_df["ngram"].str.contains(r"^\w+\s\w+$")]
top_bigrams = bigram_df.sort_values(by="score", ascending=False).head(25)

plt.figure(figsize=(12, 7))
bars = plt.barh(top_bigrams["ngram"], top_bigrams["score"])
plt.gca().invert_yaxis()
plt.title("Top 25 TF-IDF Bigrams", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()
```

#### Sparsity Heatmap (120x120 Slice)
Visualizing a 120-document by 120-feature slice highlights matrix sparsity. Because each document uses a distinct, narrow subset of phrases, most values remain zero:

```python
sample_bigram = final_tfidf_bigram[:120, :120].toarray()

plt.figure(figsize=(7, 7))
plt.imshow(sample_bigram, aspect="auto")
plt.title("TF-IDF Bigram Sparsity Heatmap (120 x 120 sample)")
plt.xlabel("Bigram Feature Index")
plt.ylabel("Document Index")
plt.show()
```

---

## 3. Comparison of Vectorization Techniques

| Feature Representation | Dimension | Retains Phrase Context? | Weights Common Words Down? | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Bag of Words (BoW)** | 50,000 | No | No | Baseline token frequency counting |
| **N-Grams (1, 2)** | 5,000 | Yes | No | Phrase pattern frequency detection |
| **TF-IDF (Unigrams)** | 5,000 | No | Yes | Keyword discrimination |
| **TF-IDF (Unigrams + Bigrams)** | 8,000 | Yes | Yes | Supervised Sentiment Classification |

---

## 4. Mentor Insights & Method Selection
- **Why TF-IDF + Bigrams for Classification?**: Plain BoW treats `not good` as independent occurrences of `not` and `good`, skewing sentiment scoring. TF-IDF with bigrams captures the composite phrase with high weighting while discounting generic stopwords.
- **Data Leakage Prevention**: In production, vectorizers must be fit exclusively on training splits (`fit_transform`) and subsequently applied to test splits (`transform`) without refitting. This design is strictly followed in Phase 4.
