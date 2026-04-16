# CS5246 Text Mining Final Project

## Introduction
This project focuses on analyzing articles and posts from the Singapore subreddit to gain insights into public opinions, trending topics, and sentiment patterns. To ensure reliable and meaningful analysis, the raw dataset undergoes a structured preprocessing workflow, transforming noisy social media text into clean, standardized, and linguistically annotated data.

### Project Structure

```bash
CS5246Project/
    │
    ├─ data/
    │   ├─ inverted_index/
    │   │   ├─ inverted_index_tfidf_titles.json
    │   │   └─ inverted_index_tfidf_fulltext.json
    │   │
    │   ├─ models/
    │   │   ├─ sentence_bert_model/
    │   │   ├─ bm25_comments_model.joblib
    │   │   ├─ bm25_fulltext_model.joblib
    │   │   ├─ bm25_titles_model.joblib
    │   │   ├─ tfidf_comments_vectorizer.joblib
    │   │   └─ tfidf_posts_vectorizer.joblib
    │   │
    │   ├─ vector_database/
    │   │   ├─ bert_comments_embeddings.npy
    │   │   ├─ bert_contents_embeddings.npy
    │   │   ├─ bert_fulltext_embeddings.npy
    │   │   ├─ bert_titles_embeddings.npy
    │   │   ├─ bm25_comments.npz
    │   │   ├─ bm25_contents.npz
    │   │   ├─ bm25_fulltext.npz
    │   │   ├─ bm25_titles.npz
    │   │   ├─ tfidf_comments.npz
    │   │   ├─ tfidf_contents.npz
    │   │   ├─ tfidf_fulltext.npz
    │   │   └─ tfidf_titles.npz
    │   │
    │   ├─ PostVault.csv
    │   ├─ CommentVault.csv
    │   └─ raw_data/
    │
    ├─ sentiment_plots/
    │   ├─ emotion_plots/
    │   ├─ emotion_dashboard.py
    │   └─ plot_emotion_summary.py
    │
-----------------------------------------------------------------
│   ├─ Stage_0_Introduction.ipynb                               │
-----------------------------------------------------------------            
    ├─ Stage_1_Data_Collection_and_Data_Cleaning.ipynb
    ├─ Stage_2_POS_and_NER_Tagging.ipynb
    ├─ Stage_3_Singlish_Normalisation.ipynb
    ├─ Stage_4_Singlish_to_English_Conversion.ipynb     
    ├─ Stage_5_Common_Normalisation.ipynb
    ├─ Stage_6_Vector_Space_Model_and_Inverted_Index.ipynb
    ├─ Stage_7_Sentiment_Analysis.ipynb
    ├─ Stage_8_Clustering_and_Visualization.ipynb       
    ├─ Stage_9_Document_Search.ipynb
    └─ utilities/
        │
        ├─ pp_class.py
        ├─ singlish_dictionary.json
        ├─ singlish_regex_to_text.txt
        └─ slang_dictionary.csv
```

### Workflow
```

[Stage 0: Data Scraping]

#### Historical Scraping (Arctic Shift API)
Scrape historical posts and comments for a specific year and subreddit:
```bash
python data_scrape/scrape.py --subreddit singapore --year 2025
# OR with uv
uv run python data_scrape/scrape.py --subreddit singapore --year 2025
```
*Options:*
- `--subreddit`: Subreddit name (default: `singapore`).
- `--year`: Year to scrape (default: `2025`).
- `--download-media`: Download media files for non-self posts.

#### Incremental Scraping (PRAW)
Fetch new posts and comments since the last scrape. This is designed for cron jobs:
```bash
python data_scrape/scrape_incremental.py --subreddit singapore --limit 1000
# OR with uv
uv run python data_scrape/scrape_incremental.py --subreddit singapore --limit 1000
```
*Options:*
- `--subreddit`: Subreddit name (default: `singapore`).
- `--limit`: Max posts to fetch from `.new()` (max 1000).
- `--output-dir`: Directory for CSV files (default: `.`).
- `--download-media`: Download media files.

[Stage 1: Data Collection and Data Cleaning]
  - Input:
    * Raw dataset

  - Operations:
    1. Remove URLs
    2. Stripping user mentions
    3. Reducing character elongation
    4. Normalizing punctuation
    5. Remove bots and mods
    6. De-deduplicate
    7. Remove deleted posts
    8. Remove 3 word posts
    9. Expand contractions
    10. Convert emoji to text

  - Output:
    * PostVault.csv
    * CommentVault.csv

  - Notes:
    * Posts:
      1. Stored separately as `title` and `content`
      2. Combined version stored as `fulltext`
      3. Cleaned outputs stored in `cleaned_title` and `cleaned_content`
    * Comments:
      1. Original text stored as `text`
      2. Cleaned version stored as `cleaned_text`

        │
        │
        │
        ▼

[Stage 2: POS and NER Tagging]
  - Operations:
    1. Create POS and NER Tagger

        │    
        │
        │
        ▼

[Stage 3: Singlish Normalisation]
  - Input:
    * PostVault.csv
    * CommentVault.csv

  - Operations:
    1. Normalize Singlish expressions into standardized Singlish forms to reduce lexical variation and ensure consistency

  - Note:
    * Posts:
      1. Output stored in `singlish_normalized_title` and `singlish_normalized_content`
    * Comments:
      1. Output stored in `singlish_normalized_text`

        │
        │
        │
        ▼

[Stage 4: Singlish to English Conversion]
  - Input:
    * PostVault.csv
    * CommentVault.csv

  - Operations:
    1. Convert Singlish expressions into standard English to support advanced NLP processing  
    2. Track the number of converted Singlish terms and store in the `singlish_count` column (for downstream ranking or analysis)  

  - Note:
    * Posts:
      - Output stored in `english_converted_title` and `english_converted_content`
    * Comments:
      - Output stored in `english_converted_text`

        │
        │
        │
        ▼

[Stage 5: Text / Short Message Normalisation]
  - Input:
    * PostVault.csv
    * CommentVault.csv

  - Operations:
    1. Expand Shortforms and Slang: Expand short forms and slang expressions into their full forms
    2. Handling Emojis: Handle emojis by converting them into meaningful textual representations or removing them if irrelevant
    3. Spelling Correction: Perform spelling correction to standardize words
    4. Stop Word Removal: Remove stop words to reduce noise in the text
    5. Lemmatization: Apply lemmatization to reduce words to their base or root form
    6. Final Cleaning and Title–Content Concatenation: Perform final text cleaning and concatenate the title and content fields into a unified text representation for downstream processing

  - Output:
    * Posts:
      - Output stored in `expanded_title`, `demojized_title`, `spellchecked_title`, `lemmatized_title`, `expanded_content`, `demojized_content`, `spellchecked_content`, `lemmatized_content`, and `lemmatized_full_text`
    * Comments:
      - Output stored in `expanded_text`, `demojized_text`, `spellchecked_text`, `lemmatized_text`

        │
        │
        │
        ▼

[Stage 6: Vector Space Model and Inverted Index]
  - Input:
    * PostVault.csv
    * CommentVault.csv
    - Note: The processed text columns from Stage 5 (e.g., lemmatized_title, lemmatized_content, lemmatized_full_text, cleaned_text)

  - Operations:
    1. TF-IDF Vectorization:
      a. Creates a TfidfVectorizer for posts and fits it to lemmatized_full_text.
      b. Transforms lemmatized_title, lemmatized_content, and lemmatized_full_text into TF-IDF matrices.
      c. Creates and fits a separate TfidfVectorizer for comments to cleaned_text.
    2. Save TF-IDF Matrices: Saves the generated TF-IDF sparse matrices (tfidf_titles_matrix, tfidf_contents_matrix, tfidf_full_text_matrix, tfidf_comments_matrix) to .npz files.
    3. Save TF-IDF Vectorizers: Saves the fitted TfidfVectorizer models (tfidf_posts_vectorizer, tfidf_comments_vectorizer) to .joblib files.
    4. BM25 Tokenization: Tokenizes the lemmatized_title, lemmatized_content, lemmatized_full_text, and cleaned_text fields for BM25 processing.
    5. BM25 Model Initialization: Initializes BM25Okapi models for tokenized titles, full text, and comments.
    6. Save BM25 Models: Saves the initialized BM25Okapi models (bm25_titles, bm25_fulltext, bm25_comments) to .joblib files.
    7. Generate BM25 Matrices: Builds sparse BM25 score matrices (bm25_titles_matrix, bm25_contents_matrix, bm25_full_text_matrix, bm25_comments_matrix) from the tokenized documents and BM25 models.
    8. Save BM25 Matrices: Saves the generated BM25 sparse matrices to .npz files.
    9. Load Pre-trained BERT Model: Loads a SentenceTransformer model (all-MiniLM-L6-v2).
    10. Generate BERT Embeddings: Generates dense BERT embeddings for title, content, and fulltext fields of posts.
    11. Save BERT Embeddings: Saves the generated BERT embeddings (bert_titles_embeddings, bert_contents_embeddings, bert_full_text_embeddings) to .npy files.
    12. Save BERT Model: Saves the loaded SentenceTransformer model to a directory.
    13. Build Inverted Index Function: Defines a function to build an inverted index from a TF-IDF matrix and its feature names.
    14. Generate Inverted Indices: Uses the defined function to create inverted indices for TF-IDF processed titles (inverted_index_titles) and full text (inverted_index_full_text).
    15. Save Inverted Indices: Saves the generated inverted indices to .json files.

  - Output:
    * Posts:
      - tfidf_titles.npz: TF-IDF matrix for post titles.
      - tfidf_contents.npz: TF-IDF matrix for post content.
      - tfidf_fulltext.npz: TF-IDF matrix for post full text.
      - bm25_titles.npz: BM25 score matrix for post titles.
      - bm25_contents.npz: BM25 score matrix for post content.
      - bm25_fulltext.npz: BM25 score matrix for post full text.
      - bert_titles_embeddings.npy: BERT embeddings for post titles.
      - bert_contents_embeddings.npy: BERT embeddings for post content.
      - bert_fulltext_embeddings.npy: BERT embeddings for post full text.
      - inverted_index_tfidf_titles.json: Inverted index for TF-IDF post titles.
      - inverted_index_tfidf_fulltext.json: Inverted index for TF-IDF post full text.
      - tfidf_posts_vectorizer.joblib: Trained TF-IDF vectorizer for posts.
      - bm25_titles_model.joblib: Trained BM25 model for post titles.
      - bm25_fulltext_model.joblib: Trained BM25 model for post full text.
    * Comments:
      - tfidf_comments.npz: TF-IDF matrix for comments.
      - bm25_comments.npz: BM25 score matrix for comments.
      - tfidf_comments_vectorizer.joblib: Trained TF-IDF vectorizer for comments.
      - bm25_comments_model.joblib: Trained BM25 model for comments.
    * General Models:
      - sentence_bert_model/: Directory containing the saved Sentence-BERT model.

        │
        │
        │
        ▼

[Stage 7: Sentiment Analysis]
  - Input:
    * PostVault.csv
    * CommentVault.csv

  - Operations:

  - Output:

        │
        │
        │
        ▼

[Stage 8: Clustering and Visualization]
  - Input:
    * PostVault.csv
    * CommentVault.csv
    * bert_comments_embeddings.npy
    * bert_contents_embeddings.npy
    * bert_fulltext_embeddings.npy
    * bert_titles_embeddings.npy
    * bm25_comments.npz
    * bm25_contents.npz
    * bm25_fulltext.npz
    * bm25_titles.npz
    * tfidf_comments.npz
    * tfidf_contents.npz
    * tfidf_fulltext.npz
    * tfidf_titles.npz
    * tfidf_comments_vectorizer.joblib
    * tfidf_posts_vectorizer.joblib

  - Operations:
    1. Load TF-IDF, BM25, and BERT feature matrices  
    2. Apply Truncated SVD to reduce the dimensionality of the high-dimensional feature space  
    3. Use the Silhouette Score to determine the optimal number of clusters (K)  
    4. Perform K-means clustering on the reduced feature set  
    5. Store the resulting cluster labels in `PostVault.csv` and `CommentVault.csv`  
    6. Perform stratified sampling based on cluster distribution  
    7. Apply t-SNE on the sampled data for visualization  
    8. Conduct word cloud analysis to identify prominent terms within clusters  
    9. Perform keyword extraction to identify representative terms for each cluster

  - Output:
    * Posts:
      - Output stored in `tfidf_cluster`, `bm25_cluster`, and `bert_cluster`
    * Comments:
      - Output stored in `tfidf_cluster`, `bm25_cluster`, and `bert_cluster`

        │
        │
        │
        ▼

[Stage 9: Document Search]
  - Input:
    * PostVault.csv
    * CommentVault.csv
    * bert_comments_embeddings.npy
    * bert_contents_embeddings.npy
    * bert_fulltext_embeddings.npy
    * bert_titles_embeddings.npy
    * bm25_comments.npz
    * bm25_contents.npz
    * bm25_fulltext.npz
    * bm25_titles.npz
    * tfidf_comments.npz
    * tfidf_contents.npz
    * tfidf_fulltext.npz
    * tfidf_titles.npz
    * inverted_index_tfidf_titles.json
    * inverted_index_tfidf_fulltext.json
    * sentence_bert_model/
    * bm25_comments_model.joblib
    * bm25_fulltext_model.joblib
    * bm25_titles_model.joblib
    * tfidf_comments_vectorizer.joblib
    * tfidf_posts_vectorizer.joblib

  - Operations:
    1. Build a search engine using scores stored in the inverted index and cosine similarity within a vector space model (TF-IDF, BM25, and BERT embeddings). Incorporate additional business and heuristic rules, such as assigning higher weights to titles over content, filtering out posts with removed content, and prioritizing posts based on recency, number of comments, and upvote ratio  
    2. Build a recommendation system using scores stored in the inverted index and cosine similarity within a vector space model (TF-IDF, BM25, and BERT embeddings). Incorporate additional business and heuristic rules, such as filtering out posts with removed content and prioritizing posts based on recency, number of comments, and upvote ratio
```

## Setup

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CS5246-Text-Mining.git
   cd CS5246-Text-Mining
   ```

2. Install dependencies:
   ```bash
   pip install pandas requests praw torch transformers python-dotenv
   ```
   *Alternatively, if using `uv`:*
   ```bash
   uv sync
   ```

### Configuration

For incremental scraping using PRAW, create a `.env` file in the root directory with your Reddit API credentials:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```
