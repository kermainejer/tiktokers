# README.md

# Google Maps Restaurant Reviews Classification – Colab Version

---

## Project Overview

This Google Colab pipeline allows you to clean, preprocess, and classify Google Maps restaurant reviews from Kaggle. The workflow includes:

1. Downloading the Kaggle dataset of restaurant reviews.
2. Cleaning and preprocessing text:
   - Converting text to lowercase
   - Removing punctuation, numbers, and special characters
   - Removing stopwords and lemmatizing words
3. Renaming columns for clarity:
   - `business_name` → `location_name`
   - `text` → `review_text`
   - `author_name` → `review_id`
4. Optional: Fetching Google store categories via Google Places API.
5. Classifying reviews using FLAN-T5 into:
   - Spam
   - Advertisement
   - Irrelevant
   - Rant/Fake Complaint
   - Genuine Review
6. Flagging policy violations using a RandomForest classifier.
7. Computing sentiment scores using VADER.
8. Ranking reviews based on a quality score.
9. Saving the final processed dataset for analysis.

---

## Setup Instructions

1. Open Google Colab and ensure GPU is enabled (optional but recommended for LLM inference).
2. Install required packages:

```python
!pip install pandas requests tqdm scikit-learn transformers torch nltk kagglehub pillow
```

3. Import packages and download NLTK resources:

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from kagglehub import KaggleDatasetAdapter, kagglehub
from huggingface_hub import notebook_login

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

notebook_login()
```

4. Download Kaggle dataset:

```python
file_path = "reviews.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "denizbilginn/google-maps-restaurant-reviews",
    file_path
)
```

5. Clean and preprocess text:

```python
stop = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)
```

6. Rename columns:

```python
df.rename(columns={
    "business_name": "location_name",
    "text": "review_text",
    "author_name": "review_id"
}, inplace=True)
```

---

## How to Reproduce Results

1. Optional: Sample a subset of reviews for testing:

```python
TEST_MODE = True
SAMPLE_SIZE = 100
if TEST_MODE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
```

2. Optional: Fetch Google store categories:

```python
API_KEY = "YOUR_GOOGLE_PLACES_API_KEY"

def get_store_category(store_name, retries=3):
    if not isinstance(store_name, str) or store_name.strip() == "":
        return []
    import requests, time
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": store_name, "key": API_KEY}
    for _ in range(retries):
        try:
            resp = requests.get(url, params=params).json()
            if "results" in resp and len(resp["results"]) > 0:
                return resp["results"][0].get("types", [])
        except:
            time.sleep(1)
    return []

df["store_category"] = df["location_name"].apply(get_store_category)
```

3. Classify reviews using FLAN-T5:

```python
classifier = pipeline("text2text-generation", model="google/flan-t5-large", device_map="auto")
CATEGORIES = ["Spam", "Advertisement", "Irrelevant", "Rant/Fake Complaint", "Genuine Review"]

# Implement batch_classify function as in the notebook
df["llm_category"] = batch_classify(df, CATEGORIES)
```

4. Compute policy violations, sentiment, and quality score as per the notebook.

5. Save the final dataset:

```python
df.to_csv("kaggle_reviews_classified.csv", index=False)
```

This reproduces the cleaned, classified, and ranked review dataset for further analysis.

