# Indonesia Ecotourism Recommender System

A content-based filtering (CBF) recommendation system for Indonesian ecotourism destinations, built as a Final Year Project (FYP). The app is powered by TF-IDF + Cosine Similarity with User Feedback Weighting (UFW) and served via a Streamlit interface.

## Features

- **Feed tab** — Personalized destination feed using CBF + MMR diversity selection
- **Search / KB tab** — Keyword-based search over destination descriptions (TF-IDF KNN index)
- **Bookmarks tab** — Save destinations for later review
- **User Feedback Weighting (UFW)** — Like/Skip interactions re-rank results in real time
- **Serendipity injection** — Randomly inserts popular items to increase discovery
- **Filters** — Filter by category, city, and max price

## Architecture

```
eco_recsys/
├── cbf.py       # Core CBF logic: TF-IDF, MMR, feed & search functions
├── ufw.py       # User Feedback Weighting: Like/Skip re-ranking
├── data.py      # Artifact loading (items.csv, TF-IDF matrix, KNN index)
├── state.py     # Streamlit session state management
├── text.py      # Text preprocessing for TF-IDF
├── ui.py        # Modular Streamlit UI components
└── utils.py     # Utilities (min-max normalization, etc.)

artifacts/       # Pre-built ML artifacts (vectorizer, TF-IDF matrix, KNN index)
dataset/         # Raw ecotourism dataset (eco_place.csv, 182 items)
benchmark/       # Offline evaluation results
```

## Recommendation Pipeline

### Feed (no query)

1. Filter items by category / city / price
2. Remove user-skipped items
3. Score items by rating; if the user has Liked items, blend in cosine similarity to liked-item centroid (70% similarity + 30% rating)
4. Select top-N with **MMR** (Maximal Marginal Relevance) for relevance + diversity
5. Inject serendipity items (random popular picks, up to 20% of feed)

### Search

1. Preprocess query text → TF-IDF vector
2. Approximate nearest-neighbour lookup (sklearn `NearestNeighbors`, cosine)
3. Apply similarity threshold filter
4. Re-rank with **MMR**
5. Optionally apply UFW on top

### UFW Formula

```
final_score = base_score
            + α × similarity_to_like_centroid
            - β × (1 if skipped, else 0)
            + γ × like_count_for_category
```

Default: α=0.6, β=0.7, γ=0.02 (tunable via sidebar sliders)

## Benchmark Results (Test Set)

| Method        | Recall@10 | Precision@10 | F1@10 | Latency (ms/q) |
| ------------- | --------- | ------------ | ----- | -------------- |
| TF-IDF Cosine | 0.764     | 0.076        | 0.139 | 1.16           |
| Jaccard       | 0.655     | 0.066        | 0.119 | 0.25           |

Dataset: 182 items, 364 queries (seed=42)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

**If you see a model compatibility error** (sklearn version mismatch), rebuild artifacts:

```bash
python rebuild_artifacts.py
streamlit run app.py
```

## Dependencies

- `streamlit` — UI framework
- `scikit-learn` — TF-IDF vectorizer, NearestNeighbors
- `pandas`, `numpy`, `scipy` — data handling
- `joblib` — artifact serialization
