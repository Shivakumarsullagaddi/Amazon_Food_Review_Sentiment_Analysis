# Phase 5: Semantic Similarity and Clustering

Documentation for [`5_similarity_and_clustering_prod.ipynb`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/5_similarity_and_clustering_prod.ipynb).

---

## 1. Objective
Extract latent thematic representations from customer reviews using continuous semantic embeddings ([`Word2Vec`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/5_similarity_and_clustering_prod.ipynb)), partition reviews into unsupervised topical clusters via [`KMeans`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/5_similarity_and_clustering_prod.ipynb), perform vector-based document retrieval using cosine similarity, and visualize the semantic space with Principal Component Analysis ([`PCA`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/5_similarity_and_clustering_prod.ipynb)).

---

## 2. Architecture & Pipeline Stages

### 2.1 Tokenization
Cleaned review texts are split into discrete word tokens using NLTK:

```python
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

final = pd.read_csv("preprocessed_reviews.csv")
tokenized_reviews = [
    word_tokenize(text)
    for text in final["CleanedText"].dropna()
    if isinstance(text, str)
]
```

### 2.2 Word2Vec Continuous Embedding Training
A Skip-Gram Word2Vec model is trained over the tokenized corpus:
- `vector_size=50`: 50-dimensional continuous latent space.
- `window=5`: Context window spanning 5 tokens to the left and right.
- `min_count=3`: Filters out infrequent vocabulary appearing fewer than 3 times.
- `sg=1`: Skip-Gram architecture (predicts context words given target word).
- `workers=4`: Multi-core parallel training.

```python
from gensim.models import Word2Vec

w2v_model = Word2Vec(
    sentences=tokenized_reviews,
    vector_size=50,
    window=5,
    min_count=3,
    workers=4,
    sg=1
)
```

### 2.3 Sentence Vector Aggregation (Average Word Embeddings)
To represent an entire review as a single vector, all constituent word embeddings within the review are averaged:

$$\vec{v}_{\text{review}} = \frac{1}{|R|} \sum_{w \in R} \vec{v}_w$$

```python
import numpy as np

def get_sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

review_vectors = np.array([get_sentence_vector(tokens, w2v_model) for tokens in tokenized_reviews])
```

- Resulting Matrix Shape: `(294156, 50)`

---

## 3. Unsupervised Clustering with K-Means
K-Means partitions the 50-dimensional document vectors into 5 thematic clusters:

```python
from sklearn.cluster import KMeans

num_clusters = 5
kmeans_w2v = KMeans(n_clusters=num_clusters, random_state=42)
clusters_w2v = kmeans_w2v.fit_predict(review_vectors)

non_null_cleaned_text_indices = final["CleanedText"].dropna().index
cluster_series_aligned = pd.Series(clusters_w2v, index=non_null_cleaned_text_indices)
final["Cluster_Word2Vec"] = cluster_series_aligned
```

### Cluster Volume Distribution
- **Cluster 1**: 85,216 reviews
- **Cluster 2**: 73,186 reviews
- **Cluster 4**: 51,847 reviews
- **Cluster 0**: 45,003 reviews
- **Cluster 3**: 38,904 reviews

---

## 4. Semantic Search via Cosine Similarity
Cosine similarity evaluates the angular alignment between a query review vector $\vec{q}$ and all corpus vectors $\vec{d}_i$, independent of text length:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}_i) = \frac{\vec{q} \cdot \vec{d}_i}{\|\vec{q}\| \|\vec{d}_i\|}$$

```python
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_reviews(review_index, top_n=5):
    sims = cosine_similarity(
        review_vectors[review_index].reshape(1, -1),
        review_vectors
    )[0]
    top_indices = sims.argsort()[-(top_n + 1):-1][::-1]

    print(final["Text"].iloc[review_index])
    for idx in top_indices:
        print(sims[idx], final["Text"].iloc[idx])
```

- Demonstrates high semantic correspondence (similarity scores reaching 0.970) based on conceptual meaning rather than exact word overlap.

---

## 5. 2D Cluster Visualization via PCA
Principal Component Analysis reduces the 50-dimensional review vectors to 2 orthogonal components for planar visualization:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sample_size = 10000
sample_idx = np.random.choice(len(review_vectors), sample_size, replace=False)

pca = PCA(n_components=2, random_state=42)
w2v_pca = pca.fit_transform(review_vectors[sample_idx])
sample_clusters = clusters_w2v[sample_idx]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=w2v_pca[:, 0],
    y=w2v_pca[:, 1],
    hue=sample_clusters,
    palette="tab10",
    s=15
)
plt.title("Word2Vec KMeans Clusters (PCA Projection)")
plt.show()
```

---

## 6. Mentor Insights & Discrepancies Resolved
- **Sparse vs Dense Semantics**: BoW and TF-IDF create sparse, high-dimensional matrices (e.g., 8,000 dimensions) where synonyms are orthogonal. Word2Vec creates dense, low-dimensional vectors (50 dimensions) capturing semantic similarity (e.g., `delicious` and `tasty` have close cosine proximity).
- **Averaging Limitations**: Simple mean aggregation weights all words equally, which can wash out salient keywords in longer reviews. Incorporating TF-IDF weighted sentence embeddings can further enhance cluster boundaries.
