# CS5246 Text Mining Final Project

## Introduction

This project analyzes posts and comments from the Singapore subreddit to surface public opinions, trending topics, and sentiment patterns. Raw social media text is processed through a structured multi-stage pipeline — from scraping to search and recommendation.

## Project Structure

```
CS5246Project/
├── dashboard-ui/          # Code for the dashboard UI
├── data-scrape/          # Code for scraping Reddit Data
├── sentiment_plots/          # Sentiment Distribution Scripts
├── utilities/                # Helper scripts and dictionaries
├── Stage_0_Introduction.ipynb
├── Stage_1_Data_Collection_and_Data_Cleaning.ipynb
├── Stage_2_POS_and_NER_Tagging.ipynb
├── Stage_3_Singlish_Normalisation.ipynb
├── Stage_4_Singlish_to_English_Conversion.ipynb
├── Stage_5_Common_Normalisation.ipynb
├── Stage_6_Vector_Space_Model_and_Inverted_Index.ipynb
├── Stage_7_Sentiment_Analysis.ipynb
├── Stage_8_Clustering_and_Visualization.ipynb
├── Stage_9_Document_Search.ipynb
└── Step_Appendix_1_Sentiment Labelling.ipynb    # Evaluation of 6 models for Sentiment Labelling
```

## Setup

### Special Instructions
1. For incremental scraping via PRAW, create a `.env` file in the root directory:
   ```env
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

## Pipeline

Run each notebook in order. Outputs from earlier stages are consumed by later stages.

### Stage 0 — Data Scraping

**Historical scraping** (Arctic Shift API):
```bash
python data_scrape/scrape.py --subreddit singapore --year 2025
```
- `--year`: Year to scrape (default: `2025`)
- `--download-media`: Also download media files

**Incremental scraping** (PRAW, designed for cron jobs):
```bash
python data_scrape/scrape_incremental.py --subreddit singapore --limit 1000
```
- `--limit`: Max posts to fetch (max 1000)
- `--output-dir`: Output directory for CSV files (default: `.`)

---

### Stage 1 — Data Collection and Cleaning

Cleans the raw Reddit dataset by removing noise (URLs, mentions, bots, duplicates, deleted posts) and normalizing text (punctuation, contractions, emojis).

**Output:** `PostVault.csv`, `CommentVault.csv`

---

### Stage 2 — POS and NER Tagging

Applies Part-of-Speech and Named Entity Recognition tagging to the cleaned text for downstream linguistic analysis.

---

### Stage 3 — Singlish Normalisation

Standardizes Singlish expressions to reduce lexical variation (e.g., consistent spelling and form).

---

### Stage 4 — Singlish to English Conversion

Converts Singlish terms to standard English using a custom dictionary. Tracks the number of converted terms per post/comment in a `singlish_count` column.

---

### Stage 5 — Text Normalisation

Applies slang expansion, spelling correction, stop word removal, and lemmatization. Produces the final clean text columns used for all downstream modelling.

---

### Stage 6 — Vector Space Model and Inverted Index

Builds TF-IDF, BM25, and Sentence-BERT representations for posts and comments, and constructs an inverted index for fast retrieval.

**Outputs:** `.npz` matrices, `.joblib` models, `.npy` BERT embeddings, `.json` inverted indices — all saved under `data/`.

---

### Stage 7 — Sentiment Analysis

Performs sentiment and emotion analysis on posts and comments.

---

### Stage 8 — Clustering and Visualization

Reduces feature dimensionality with SVD, selects the optimal cluster count via Silhouette Score, and runs K-means clustering. Visualizes clusters with t-SNE and word clouds.

**Output:** `tfidf_cluster`, `bm25_cluster`, `bert_cluster` columns added to `PostVault.csv` and `CommentVault.csv`.

---

### Stage 9 — Document Search and Recommendation

Implements a search engine and recommendation system using TF-IDF, BM25, and BERT embeddings, with heuristic ranking rules (title weighting, recency, upvotes, comment count).
