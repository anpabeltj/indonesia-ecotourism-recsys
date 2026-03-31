# Indonesia Ecotourism Recommender System

A content-based filtering (CBF) recommendation system for Indonesian ecotourism destinations, built as a Final Year Project (FYP). Powered by TF-IDF + Cosine Similarity with User Feedback Weighting (UFW) and served via a Streamlit interface.

## Features

- **Feed tab** — Personalized destination feed using CBF + MMR diversity selection; adapts to liked items in real time
- **Search / KB tab** — Keyword-based search over destination descriptions (TF-IDF + ANN index)
- **Bookmarks tab** — Save destinations for later review
- **User Feedback Weighting (UFW)** — Like/Skip interactions re-rank results in real time
- **Serendipity injection** — Inserts popular items (up to 20% of feed) to increase discovery
- **Filters** — Filter by category, city, and max price
- **Tunable parameters** — Sidebar sliders to adjust MMR lambda, UFW weights (α, β, γ), feed size, and serendipity %

## Architecture

```
eco_recsys/
├── cbf.py       # Core CBF logic: TF-IDF, MMR, feed & search functions
├── ufw.py       # User Feedback Weighting: Like/Skip re-ranking
├── data.py      # Artifact loading (items.csv, TF-IDF matrix, KNN index)
├── state.py     # Streamlit session state management
├── text.py      # Text preprocessing for TF-IDF (Indonesian stopword removal)
├── ui.py        # Modular Streamlit UI components
└── utils.py     # Utilities (min-max normalization, etc.)

artifacts/
├── vectorizer.joblib        # Fitted TF-IDF vectorizer
├── tfidf_matrix.npz         # Pre-built TF-IDF sparse matrix
├── nbrs_cosine.joblib       # sklearn NearestNeighbors index (cosine)
├── items.csv                # Processed items with normalized fields
└── metadata.json            # Artifact metadata

dataset/
└── eco_place.csv            # Raw ecotourism dataset (182 destinations)

notebook/
└── indo_ecotourism_cbf_ufw.ipynb  # Exploratory analysis notebook
```

## Recommendation Pipeline

### Feed (no query)

1. Filter items by category / city / price
2. Remove user-skipped items
3. Score items:
   - **With likes:** 70% cosine similarity to liked-item centroid + 30% normalized rating
   - **Without likes:** normalized rating only
4. Select top-N with **MMR** (Maximal Marginal Relevance) for relevance + diversity
   - Soft per-category cap (penalty-based, not hard block)
5. Inject serendipity items (random popular picks, capped at 20% of feed)

### Search

1. Preprocess query (Indonesian stopword removal) → TF-IDF vector
2. Approximate nearest-neighbour lookup (`sklearn NearestNeighbors`, cosine); falls back to full cosine similarity if index unavailable
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

### MMR Formula

```
MMR(i) = λ × relevance_score(i) − (1 − λ) × max_sim(i, selected)
```

Default: λ=0.7 (0 = full diversity, 1 = full relevance)

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
