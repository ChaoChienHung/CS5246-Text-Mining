# CS5246-Text-Mining

A repository for scraping Reddit data and performing emotion inference on the collected text. This project focuses on the `r/singapore` subreddit but is configurable for any subreddit.

## Project Structure

```text
CS5246-Text-Mining/
├── data_scrape/
│   ├── scrape.py              # Historical scraper via Arctic Shift API
│   └── scrape_incremental.py  # Incremental scraper via PRAW (Reddit API)
└── emotion_inference.py       # Emotion classification using Hugging Face models
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

## Usage

### 1. Data Scraping

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

### 2. Emotion Inference

Classify emotions in the scraped CSV files using a pretrained Hugging Face model (default: `j-hartmann/emotion-english-distilroberta-base`):

```bash
python emotion_inference.py --input intermediate_data/PostVault.csv
# OR with uv
uv run python emotion_inference.py --input intermediate_data/PostVault.csv
```
*Options:*
- `--input`: CSV file or directory of CSV files to classify.
- `--output-dir`: Directory for enriched CSV outputs (default: `emotion_output`).
- `--model-name`: Hugging Face model name or local path.
- `--batch-size`: Inference batch size (default: `16`).
- `--device`: Inference device (`auto`, `cpu`, `mps`, `cuda`).

## Data Format

Scraped data is stored in monthly CSV files (e.g., `singapore_posts_2025_01_january.csv`).
- **Posts CSV:** Contains fields like `id`, `title`, `selftext`, `author`, `score`, `created_utc`, etc.
- **Comments CSV:** Contains fields like `id`, `parent_id`, `post_id`, `body`, `author`, `score`, etc.

The emotion inference script adds:
- `predicted_emotion`: The label with the highest probability.
- `prob_<label>`: The probability for each emotion category (e.g., `prob_anger`, `prob_joy`).
