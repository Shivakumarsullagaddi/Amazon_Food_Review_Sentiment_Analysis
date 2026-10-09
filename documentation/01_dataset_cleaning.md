# Phase 1: Dataset Loading and Cleaning

Documentation for [`1_cleaning.ipynb`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/1_cleaning.ipynb).

---

## 1. Objective
Prepare the raw Amazon Fine Food Reviews dataset for sentiment modeling by removing ambiguous neutral feedback, eliminating duplicate records that introduce statistical bias, and pruning physically impossible helpfulness score entries.

---

## 2. Source Data Details
- Source format: SQLite database ([`database.sqlite`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/prod_code/database.sqlite))
- Table: `Reviews`
- Raw records evaluated: 525,814 rows (after filtering out ambiguous ratings where `Score == 3`)

---

## 3. Data Transformations & Methodology

### 3.1 Binary Sentiment Mapping
Customer ratings range from 1 to 5. Rating 3 represents neutral sentiment, which creates ambiguity for binary classification. Neutral reviews are omitted, and remaining ratings are mapped as:
- Scores 1 and 2: `negative`
- Scores 4 and 5: `positive`

```python
import sqlite3
import pandas as pd
import os

connection = sqlite3.connect("database.sqlite")
df = pd.read_sql_query(
    """
    SELECT *
    FROM Reviews
    WHERE Score != 3
    ORDER BY ProductId
    """,
    connection
)

def partition(x):
    if x < 3:
        return "negative"
    return "positive"

df["Score"] = df["Score"].map(partition)
```

### 3.2 Deduplication
Multiple reviews frequently share identical user IDs, profile names, timestamps, summaries, and text bodies due to product cross-listing across variations (e.g., flavors, pack sizes).

- Records before deduplication: 525,814
- Deduplication key: `{'UserId', 'ProfileName', 'Time', 'Summary', 'Text'}`
- Records after deduplication: 365,333
- Data retention rate: 69.48%

```python
sorted_data = df.sort_values("ProductId", axis=0, ascending=True)
df2 = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Summary", "Text"})
```

### 3.3 Logical Anomaly Filtering
A helpfulness vote cannot exceed total votes cast. Any record where `HelpfulnessNumerator > HelpfulnessDenominator` represents corrupt telemetry.

- Condition applied: `HelpfulnessNumerator <= HelpfulnessDenominator`
- Records dropped: 2
- Final valid dataset size: 365,331 rows

```python
final = df2[df2.HelpfulnessNumerator <= df2.HelpfulnessDenominator]
```

---

## 4. Output Summary & Distribution

| Metric | Value |
| :--- | :--- |
| Initial Records (`Score != 3`) | 525,814 |
| Records after Deduplication | 365,333 |
| Final Clean Records | 365,331 |
| Positive Class Count | 307,967 (84.30%) |
| Negative Class Count | 57,364 (15.70%) |
| Missing / Null Values | 0 |
| Export File | [`reviews_clean_file`](file:///D:/nxtwave/NLP/amazon_food_review_sentimental_analysis/reviews_clean_file) |

```python
final.to_csv("./reviews_clean_file")
```

---

## 5. Mentor Insights & Discrepancies Resolved
- **Duplicate Impact**: Retaining identical text entries artificially inflates word frequencies and contaminates train/test splits through data leakage. Deduplication eliminated ~30.5% redundant rows.
- **Class Imbalance**: Positive reviews comprise 84.3% of the dataset. Stratified sampling is essential during downstream model evaluation to preserve class representation.
