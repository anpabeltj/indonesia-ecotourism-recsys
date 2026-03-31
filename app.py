import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Suppress sklearn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

from eco_recsys.data import load_artifacts, ensure_min_columns
from eco_recsys.state import init_session, like_item, skip_item, toggle_bookmark, clear_feedback
from eco_recsys.ui import (
    sidebar_filters, sidebar_feed_knobs, sidebar_feedback_knobs, status_chips,
    render_cards, search_controls
)
from eco_recsys.cbf import build_feed_cbf, search_cbf
from eco_recsys.ufw import apply_ufw

st.set_page_config(page_title="EcoTourism CBF + UFW", page_icon="🌿", layout="wide")

BASE_DIR = Path(__file__).parent
ART_DIR  = BASE_DIR / "artifacts"

# ---------------- Init & Load ----------------
init_session()
with st.spinner("Loading artifacts..."):
    items, X, vectorizer, nbrs = load_artifacts(ART_DIR)
ensure_min_columns(items)

# --- Validate Artifacts (IMPORTANT for cross-OS clients) ---
if vectorizer is None or not hasattr(vectorizer, "idf_"):
    st.error("🚨 **CRITICAL ERROR: Machine Learning model is corrupted or incompatible!**")
    st.warning("""
    **Cause:** The library version (scikit-learn) on this computer differs from the model builder.

    **SOLUTION:**
    1. Open a terminal in this project folder.
    2. Run the command:
       `python rebuild_artifacts.py`
    3. Restart this application.
    """)
    st.stop()  # Stop the application to prevent 'idf not fitted' error


st.title("🌿 Tourism Recommender — CBF + UFW")
st.caption("Basic CBF + User Feedback Weighting | TF‑IDF + NearestNeighbors (cosine) | Like/Skip/Bookmark (session)")

# ---------------- Sidebar (modular UI) ----------------
st.sidebar.header("Filter")
filters = sidebar_filters(items)

st.sidebar.header("Feed Settings")
feed_knobs = sidebar_feed_knobs(items)

st.sidebar.header("Session Feedback (UFW)")
fb_knobs = sidebar_feedback_knobs()

if st.sidebar.button("Reset Like/Skip", type="secondary"):
    clear_feedback()
    st.sidebar.success("Session preferences reset.")

status_chips()

# ---------------- Tabs ----------------
tab_feed, tab_search, tab_book = st.tabs(["🏠 Feed", "🔎 Search / KB", "🔖 Bookmarks"])

# ---------------- FEED (Basic CBF + UFW) ----------------
with tab_feed:
    cbf_candidates = build_feed_cbf(
        items=items, X=X, filters=filters,
        top_n=feed_knobs.top_n,
        mmr_lambda=feed_knobs.mmr_lambda,
        per_category_cap=feed_knobs.per_category_cap,
        serendipity_pct=feed_knobs.serendipity_pct,
        blocked_gids=st.session_state.blocked_idx,
        liked_gids=st.session_state.liked_idx  
    )
    # Apply UFW reranking if enabled
    if fb_knobs.use_feedback and cbf_candidates:
        final_pairs = apply_ufw(
            gids=[cid.gid for cid in cbf_candidates],
            base_scores_map={cid.gid: cid.base_score for cid in cbf_candidates},
            X=X, items=items,
            alpha=fb_knobs.alpha, beta=fb_knobs.beta, gamma=fb_knobs.gamma
        )
    else:
        final_pairs = [(cid.gid, cid.base_score) for cid in cbf_candidates]

    render_cards(items, final_pairs, title_suffix="Feed")

# ---------------- SEARCH / KB (CBF query) ----------------
with tab_search:
    query, top_n_s, mmr_lambda_s, min_sim, is_search_active = search_controls(items)
    
    # If search is active and we have a query
    if is_search_active and query.strip():
        # Create search params signature to check if a new search is needed
        current_params = (
            query, top_n_s, mmr_lambda_s, min_sim,
            feed_knobs.per_category_cap,
            tuple(sorted(filters.get("categories", []))),
            tuple(sorted(filters.get("cities", []))),
            filters.get("max_price")
        )
        
        # Check if a new search is needed (parameters changed)
        need_new_search = (
            st.session_state.search_results_cache is None or
            st.session_state.last_search_params != current_params
        )
        
        if need_new_search:
            # Run search and save results to cache
            results = search_cbf(
                items=items, X=X, vectorizer=vectorizer, nbrs=nbrs,
                query=query, filters=filters,
                top_n=top_n_s, mmr_lambda=mmr_lambda_s,
                per_category_cap=feed_knobs.per_category_cap,
                similarity_threshold=min_sim
            )
            # Save base results to cache (before UFW)
            st.session_state.search_results_cache = results
            st.session_state.last_search_params = current_params
        else:
            # Use cached results
            results = st.session_state.search_results_cache
        
        # Apply UFW (always re-run since feedback may have changed)
        if fb_knobs.use_feedback and results:
            final_pairs = apply_ufw(
                gids=[gid for gid,_ in results],
                base_scores_map=dict(results),
                X=X, items=items,
                alpha=fb_knobs.alpha, beta=fb_knobs.beta, gamma=fb_knobs.gamma
            )
        else:
            final_pairs = results
        
        if final_pairs:
            render_cards(items, final_pairs, title_suffix="Search/KB")
        else:
            st.warning(f"No results for '{query}' with threshold {min_sim:.2f}. Try lowering 'Min. Similarity' or use different keywords.")
    else:
        st.info("Enter a query then click **Search** to see results.")

# ---------------- BOOKMARKS ----------------
with tab_book:
    bms = sorted(list(st.session_state.bookmarked_idx))
    if not bms:
        st.info("No bookmarked items yet.")
    else:
        render_cards(items, [(gid, 0.0) for gid in bms], show_score=False, title_suffix="Bookmarks")

st.write("---")
st.caption("Note: eco_recsys.cbf (CBF), eco_recsys.ufw (User Feedback Weighting), eco_recsys.ui (UI), eco_recsys.data (loader)")