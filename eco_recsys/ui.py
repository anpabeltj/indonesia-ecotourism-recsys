from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import streamlit as st
import numpy as np
import pandas as pd

from .state import like_item, skip_item, toggle_bookmark
from .utils import format_idr, get_description

@dataclass
class FeedKnobs:
    top_n: int
    mmr_lambda: float
    per_category_cap: int
    serendipity_pct: int

@dataclass
class FeedbackKnobs:
    use_feedback: bool
    alpha: float
    beta: float
    gamma: float

def sidebar_filters(items: pd.DataFrame):
    all_categories = sorted(set([c.strip()
                                 for s in items["category"].fillna("").tolist()
                                 for c in str(s).split(",") if c.strip()]))
    sel_cats   = st.sidebar.multiselect("Category", options=all_categories)
    all_cities = sorted([c for c in items["city"].fillna("").unique() if c])
    sel_cities = st.sidebar.multiselect("City/Regency", options=all_cities)
    use_price  = st.sidebar.checkbox("Limit maximum price", value=False)
    max_price_val = float(np.nanmax(items["price"].values)) if items["price"].notna().any() else 0.0
    price_cap  = st.sidebar.slider("Maximum Price (IDR)", 0.0, max_price_val,
                                   min(max_price_val, 100_000.0), 1_000.0) if use_price else None
    return {"categories": sel_cats or [], "cities": sel_cities or [], "max_price": price_cap}

def sidebar_feed_knobs(items: pd.DataFrame) -> FeedKnobs:
    top_n_feed   = st.sidebar.slider("Number of Feed Items (Top-N)", 5, 40, 12, 1)
    mmr_lambda_f = st.sidebar.slider("MMR λ (Feed)", 0.0, 1.0, 0.85, 0.05)
    per_cat_cap  = st.sidebar.slider("Limit per category", 0, 10, 0, 1)
    serendip     = st.sidebar.slider("Serendipity (%)", 0, 30, 5, 5)
    return FeedKnobs(top_n_feed, mmr_lambda_f, per_cat_cap, serendip)

def sidebar_feedback_knobs() -> FeedbackKnobs:
    use_fb = st.sidebar.toggle("Enable Like/Skip Reranking (UFW)", value=True)
    alpha = st.sidebar.slider("Like Boost (α)", 0.0, 2.0, 1.2, 0.05, disabled=not use_fb)
    beta  = st.sidebar.slider("Skip Penalty (β)", 0.0, 2.0, 0.7, 0.05, disabled=not use_fb)
    gamma = st.sidebar.slider("Liked Category Boost (γ)", 0.0, 0.5, 0.08, 0.01, disabled=not use_fb,)
    return FeedbackKnobs(use_fb, alpha, beta, gamma)

def status_chips():
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"**Liked (⭐):** {len(st.session_state.liked_idx)} item")
        with c2: st.markdown(f"**Skipped (🚫):** {len(st.session_state.blocked_idx)} item")
        with c3: st.markdown(f"**Bookmarks (🔖):** {len(st.session_state.bookmarked_idx)} item")

def search_controls(items: pd.DataFrame):
    st.markdown("**Knowledge Base Search (TF-IDF):** type a theme/keyword, e.g.: _pantai aceh_, _snorkeling_, _gunung camping_, etc.")
    
    # Initialize session state for search if not present
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "search_active" not in st.session_state:
        st.session_state.search_active = False
    if "search_results_cache" not in st.session_state:
        st.session_state.search_results_cache = None
    if "last_search_params" not in st.session_state:
        st.session_state.last_search_params = None

    def on_search_click():
        st.session_state.search_active = True
        # Clear cache when user clicks the Search button (force new search)
        st.session_state.search_results_cache = None

    query_input = st.text_input("Search query", value=st.session_state.search_query,
                                placeholder="example: pantai, snorkeling, family hiking, savana, gunung ...",
                                key="search_input_widget")
    
    # Sync input widget to session state query
    st.session_state.search_query = query_input

    top_n_s = st.slider("Number of results", 5, 40, 12, 1)
    mmr_lambda_s = st.slider("MMR λ (Search)", 0.0, 1.0, 0.7, 0.05)
    min_similarity = st.slider("Min. Similarity (Threshold)", 0.0, 0.5, 0.1, 0.05, 
                               help="Only show results with similarity above this value.")
    
    st.button("Search", type="primary", on_click=on_search_click)
    
    return st.session_state.search_query, top_n_s, mmr_lambda_s, min_similarity, st.session_state.search_active

def render_cards(items: pd.DataFrame, pairs: List[tuple[int,float]], show_score: bool=True, title_suffix: str=""):
    if not pairs:
        st.warning("No items found for the current configuration.")
        return
    for gid, sc in pairs:
        row = items.iloc[int(gid)]
        with st.container(border=True):
            # Check if the image exists and is valid
            img = row.get("place_img")
            has_valid_image = False
            
            if pd.notna(img) and isinstance(img, str):
                img_str = str(img).strip()
                if img_str and (img_str.startswith("http://") or img_str.startswith("https://")):
                    has_valid_image = True
            
            # Create layout based on image availability
            if has_valid_image:
                cols = st.columns([1, 3])
                with cols[0]:
                    try:
                        st.image(img_str, width='stretch')
                    except Exception:
                        has_valid_image = False  # Failed to load, treat as no image
                
                if has_valid_image:
                    content_col = cols[1]
                else:
                    # Image failed to load, use full width
                    content_col = st.container()
            else:
                # No valid image, use full width
                content_col = st.container()
            
            with content_col:
                st.subheader((row.get("place_name") or "-"))
                st.markdown(f"**Category:** {row.get('category') or '-'}  \n**City:** {row.get('city') or '-'}")
                rating = row.get("rating")
                price  = row.get("price")
                st.markdown(f"**Rating:** {'-' if pd.isna(rating) else round(float(rating), 2)}  \n**Price:** {format_idr(None if pd.isna(price) else float(price))}")
                link = row.get("place_map")
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    st.link_button("Open map", link, width='content')
                if show_score:
                    st.caption(f"Score: {float(sc):.4f}")
                with st.expander("View description"):
                    st.write(get_description(row))

                b1, b2, b3, _ = st.columns([1,1,1,6])
                gid = int(gid)
                with b1:
                    if st.button("⭐ Like", key=f"like_{title_suffix}_{gid}"):
                        like_item(gid); st.rerun()
                with b2:
                    if st.button("🚫 Skip", key=f"skip_{title_suffix}_{gid}"):
                        skip_item(gid); st.rerun()
                with b3:
                    label = "Remove 🔖" if gid in st.session_state.bookmarked_idx else "🔖 Bookmark"
                    if st.button(label, key=f"bm_{title_suffix}_{gid}"):
                        toggle_bookmark(gid); st.rerun()