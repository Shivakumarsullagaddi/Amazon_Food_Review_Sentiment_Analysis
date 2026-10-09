# Phase 2: Text Preprocessing and Normalization

Documentation for [`2_text_prerprossing_prod.ipynb`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/2_text_prerprossing_prod.ipynb).

---

## 1. Objective
Transform raw, noisy customer review text into standardized, tokenized, and normalized representations suitable for feature vectorization and machine learning models.

---

## 2. Preprocessing Architecture & Pipeline Steps

### 2.1 HTML Stripping and URL Removal
Reviews often contain HTML markup fragments and hyperlinks. [`BeautifulSoup`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/2_text_prerprossing_prod.ipynb) strips all DOM tags, and regular expressions purge any remaining web URLs.

```python
from bs4 import BeautifulSoup
import re

def remove_tags(sent):
    soup = BeautifulSoup(sent, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return text
```

### 2.2 Contraction Expansion
Informal contractions obscure word meaning. Expanding contractions standardizes negated terms and helper verbs.

```python
def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"o\'clock", "of the clock", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
```

### 2.3 Digit, Special Character & Punctuation Removal
Alphanumeric terms containing numbers (e.g., product codes `ABC123`) and punctuation symbols are stripped to keep only alphabetic tokens.

```python
sent = re.sub(r"\S*\d\S*", "", sent).strip()
sent = re.sub(r"[^A-Za-z]+", " ", sent)
```

### 2.4 Case Folding and Stopword Filtering
All tokens are converted to lowercase, and standard English stopwords from NLTK are filtered out.

```python
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stopwords_set = set(stopwords.words("english"))

res = " ".join(word.lower() for word in sent.split() if word.lower() not in stopwords_set)
```

### 2.5 Stemming vs Lemmatization
Both morphological normalization techniques are computed across the entire corpus:
- **Porter Stemmer**: Heuristic suffix stripping (`playing` -> `play`, `studies` -> `studi`). Faster, but can produce non-dictionary roots.
- **WordNet Lemmatizer**: Vocabulary-grounded morphological reduction (`studies` -> `study`). Slower, but preserves grammatical validity.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed = " ".join(stemmer.stem(word) for word in res.split())
lemmatized = " ".join(lemmatizer.lemmatize(word) for word in res.split())
```

---

## 3. Visual Quality Evaluation (Word Clouds)
To assess normalization impact, Word Clouds are generated across:
1. **Raw Text (Before Preprocessing)**: Dominated by high-frequency non-informative stopwords (`the`, `and`, `to`, `this`, `was`, `for`).
2. **Cleaned Text (After Preprocessing)**: High-signal domain keywords emerge prominently (`flavor`, `taste`, `coffee`, `tea`, `product`, `good`, `love`).
3. **Lemmatized Text**: Suffix variants merge into single semantic roots (`tastes`/`tasting` -> `taste`).

```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud_after = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=STOPWORDS,
    max_words=200,
    colormap="plasma"
).generate(text_after)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud_after, interpolation="bilinear")
plt.axis("off")
plt.show()
```

---

## 4. Pipeline Execution & Output Dataset
The processed strings are added back to the DataFrame and exported to disk:
- `CleanedText`: Decontracted, lowercased, punctuation-free, stopword-filtered text.
- `StemmedText`: Porter-stemmed sequence.
- `LemmatizedText`: WordNet-lemmatized sequence.
- Target File: [`preprocessed_reviews.csv`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/preprocessed_reviews.csv)

```python
df["CleanedText"] = preprocessed_reviews
df["StemmedText"] = stemmed_reviews
df["LemmatizedText"] = lemmatized_reviews
df.to_csv("preprocessed_reviews.csv", index=False)
```

---

## 5. Mentor Insights & Engineering Trade-offs
- **Execution Overhead**: Running both `PorterStemmer` and `WordNetLemmatizer` across 365,331 records sequentially takes ~9-10 minutes. In production pipelines, selecting one normalization method based on latency requirements is recommended.
- **Negation Handling Caution**: Standard stopword lists include words like `not`, `no`, `nor`. Stripping them naively can invert sentence sentiment (e.g., `not good` becoming `good`). In Phase 3, this is mitigated by combining unigrams with bigrams in feature extraction.
