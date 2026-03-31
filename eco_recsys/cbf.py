"""
cbf.py - Content-Based Filtering (CBF) Module
==============================================
This module contains content-based recommendation algorithms using TF-IDF and cosine similarity.

Key Concepts:
- TF-IDF (Term Frequency - Inverse Document Frequency): A technique to convert text into numerical values
- Cosine Similarity: Measures similarity between items based on vector angles
- MMR (Maximal Marginal Relevance): Balances relevance and diversity in results

Simple Analogy:
- CBF is like a system that recommends restaurants based on menu descriptions
- If you like "seafood fried rice", the system will find restaurants with similar menus
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError

from .text import preprocess_text
from .utils import normalize_minmax


# =============================================================================
# DATA CLASSES - Data structures for storing results
# =============================================================================

@dataclass
class Candidate:
    """
    Class for storing recommendation item candidates.

    Attributes:
        gid (int): Global ID / item index in the DataFrame
        base_score (float): Base score of the item (0.0 - 1.0)

    Example:
        candidate = Candidate(gid=42, base_score=0.85)
        # This means item 42 has a score of 0.85
    """
    gid: int
    base_score: float


# =============================================================================
# HELPER FUNCTIONS - Smaller utility functions
# =============================================================================

def _mask_by_filters(items: pd.DataFrame, filters: Dict) -> np.ndarray:
    """
    Create a mask (boolean array) to filter items based on user criteria.

    Analogy:
        Like a "sieve" that only lets through items matching the criteria.
        If the user selects category "Beach" and city "Aceh", only items
        matching both criteria will pass (True).

    Args:
        items: DataFrame containing all destination items
        filters: Dictionary containing filter criteria:
            - categories (list): List of selected categories
            - cities (list): List of selected cities
            - max_price (float): Maximum desired price

    Returns:
        np.ndarray: Boolean array, True = item passes the filter

    Example:
        filters = {"categories": ["Pantai"], "cities": ["Aceh"], "max_price": 50000}
        mask = _mask_by_filters(items, filters)
        # mask = [True, False, True, False, ...]
    """
    # Start with all items as True (all pass)
    mask = np.full(len(items), True, dtype=bool)

    # Filter 1: Category
    # If the user selected specific categories, filter by those
    if filters.get("categories"):
        # Convert all categories to lowercase for consistent comparison
        selected_cats = [c.lower() for c in filters["categories"]]

        # Check if the item's category contains one of the selected categories
        mask &= items["category"].fillna("").apply(
            lambda s: any(cat in s.lower() for cat in selected_cats)
        ).values

    # Filter 2: City/District
    if filters.get("cities"):
        selected_cities = [c.lower() for c in filters["cities"]]

        # Check if the item's city is in the list of selected cities
        mask &= items["city"].fillna("").apply(
            lambda s: s.lower() in selected_cities
        ).values

    # Filter 3: Maximum Price
    if filters.get("max_price") is not None:
        # Get the price column, fill NaN with infinity so they don't pass the filter
        prices = items["price"].fillna(np.inf).values.astype(float)

        # Only items with price <= max_price pass
        mask &= prices <= float(filters["max_price"])

    return mask


def _calculate_category_penalty(
    item_gid: int,
    items: pd.DataFrame,
    cat_count: Dict[str, int],
    per_category_cap: int
) -> float:
    """
    Calculate penalty for an item based on its category saturation.

    Purpose:
        Provide diversity with a "soft cap" instead of hard blocking.
        Items with categories that already appear frequently get a penalty,
        but can still appear if their similarity is high enough.

    Soft Cap vs Hard Cap:
        - Hard Cap: "Already 5 beaches? Block all subsequent beaches!"
        - Soft Cap: "Already 5 beaches? The next beach needs higher similarity
                     to compete with other categories"

    Args:
        item_gid: Index of the item to check
        items: DataFrame containing all items
        cat_count: Dictionary counter of already selected categories
        per_category_cap: Soft limit per category (0 = no penalty)

    Returns:
        float: Penalty score (0.0 = no penalty, up to max 0.3)

    Example:
        - New category (count=0): penalty = 0.0
        - Category at cap (count=5, cap=5): penalty = 0.0
        - Category over cap by 1 (count=6, cap=5): penalty = 0.1
        - Category over cap by 3+ (count=8+, cap=5): penalty = 0.3 (max)
    """
    if per_category_cap <= 0:
        return 0.0  # No penalty if cap is disabled

    # Get the first category from the item (split by comma)
    category = str(items.iloc[item_gid]["category"]).split(",")[0].strip()

    if not category:
        return 0.0

    count = cat_count.get(category, 0)

    # No penalty until reaching the cap
    if count < per_category_cap:
        return 0.0

    # Calculate excess: how many items of this category exceed the cap
    excess = count - per_category_cap

    # Soft penalty: increases by 0.1 per excess item, max 0.3
    # This is enough to lower the ranking but not block entirely
    penalty = min(0.3, 0.1 * excess)

    return penalty


def _calculate_mmr_score(
    candidate_score: float,
    candidate_vector,
    selected_vectors,
    X,
    lambda_mmr: float
) -> float:
    """
    Calculate the MMR (Maximal Marginal Relevance) score for a single candidate.

    MMR Formula:
        MMR = lambda * (relevance_score) - (1 - lambda) * (similarity_to_selected_items)

    Analogy:
        Imagine you're choosing movies for a marathon:
        - High lambda (0.9): Pick highest-rated movies even if same genre
        - Low lambda (0.3): Pick diverse movies even if lower-rated

    Args:
        candidate_score: Base score of the candidate
        candidate_vector: TF-IDF vector of the candidate
        selected_vectors: TF-IDF vectors of already selected items
        X: Full TF-IDF matrix
        lambda_mmr: Balance parameter (0=diversity, 1=relevance)

    Returns:
        float: MMR score of the candidate
    """
    if selected_vectors is None or len(selected_vectors) == 0:
        # First candidate, nothing to compare against
        return candidate_score

    # Calculate similarity with all already selected items
    similarity_to_selected = cosine_similarity(candidate_vector, selected_vectors)
    max_similarity = float(similarity_to_selected.max())

    # MMR formula: high if relevant AND different from already selected items
    mmr_score = lambda_mmr * candidate_score - (1.0 - lambda_mmr) * max_similarity

    return mmr_score


def _update_category_count(
    item_gid: int,
    items: pd.DataFrame,
    cat_count: Dict[str, int]
) -> None:
    """
    Update the category counter after an item is selected.

    Args:
        item_gid: Index of the selected item
        items: DataFrame of items
        cat_count: Dictionary counter (will be modified in-place)
    """
    category = str(items.iloc[item_gid]["category"]).split(",")[0].strip()
    if category:
        cat_count[category] = cat_count.get(category, 0) + 1


# =============================================================================
# CORE FUNCTIONS - Main CBF algorithm functions
# =============================================================================

def mmr_select(
    idx_all: np.ndarray,
    X,
    base_scores: np.ndarray,
    top_n: int = 20,
    lambda_mmr: float = 0.7,
    per_category_cap: int = 0,
    items: Optional[pd.DataFrame] = None
) -> List[int]:
    """
    Select items using the MMR (Maximal Marginal Relevance) algorithm.

    Purpose:
        Select items that are both RELEVANT and DIVERSE.
        Avoids monotonous results (all beaches, all mountains, etc.).

    How It Works:
        1. Start with the candidate having the highest score
        2. For each subsequent candidate:
           - Calculate MMR score = relevance - similarity to already selected items
           - Pick the candidate with the highest MMR score
        3. Repeat until the desired number is reached

    Args:
        idx_all: Array containing candidate item indices
        X: TF-IDF matrix (sparse matrix) for computing similarity
        base_scores: Array of base scores for each candidate
        top_n: Number of items to select
        lambda_mmr: MMR parameter (0.0-1.0)
            - 1.0 = prioritize relevance (similar items allowed)
            - 0.0 = prioritize diversity (items must be different)
            - 0.7 = default, balanced
        per_category_cap: Maximum items per category (0 = unlimited)
        items: DataFrame for checking categories

    Returns:
        List[int]: List of selected item indices (global IDs)

    Example:
        >>> selected = mmr_select(idx_all, X, scores, top_n=10, lambda_mmr=0.7)
        >>> print(selected)  # [5, 12, 3, 27, ...]
    """
    selected_positions: List[int] = []  # Positions in idx_all that have been selected
    remaining_candidates = list(range(len(idx_all)))  # Positions still available
    category_count: Dict[str, int] = {}  # Counter per category

    # Maximum number of candidates to consider (3x top_n for efficiency)
    max_candidates = min(top_n * 3, len(idx_all))

    while remaining_candidates and len(selected_positions) < max_candidates:
        best_position = None
        best_mmr_score = -float('inf')  # Start from the lowest value

        # Evaluate each remaining candidate
        for position in remaining_candidates:
            item_gid = int(idx_all[position])

            # Calculate MMR score
            candidate_base_score = float(base_scores[position])

            if not selected_positions:
                # First candidate, use base score directly
                mmr_score = candidate_base_score
            else:
                # Calculate MMR considering already selected items
                selected_gids = idx_all[selected_positions]
                similarity = cosine_similarity(X[item_gid], X[selected_gids]).max()
                mmr_score = lambda_mmr * candidate_base_score - (1.0 - lambda_mmr) * float(similarity)

            # Apply soft category penalty instead of hard block
            # Items with categories that already appear frequently get a penalty,
            # but can still win if their MMR score is high enough
            if per_category_cap and items is not None:
                category_penalty = _calculate_category_penalty(
                    item_gid, items, category_count, per_category_cap
                )
                mmr_score -= category_penalty

            # Update best candidate
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_position = position

        # If no valid candidate found, stop
        if best_position is None:
            break

        # Add the best candidate to results
        selected_positions.append(best_position)

        # Update category counter
        if per_category_cap and items is not None:
            _update_category_count(int(idx_all[best_position]), items, category_count)

        # Remove from remaining candidates
        remaining_candidates.remove(best_position)

        # Stop if we've reached top_n
        if len(selected_positions) >= top_n:
            break

    # Convert positions to global IDs
    return [int(idx_all[pos]) for pos in selected_positions]


def _filter_blocked_items(idx_all: np.ndarray, blocked_gids: Optional[Set[int]]) -> np.ndarray:
    """
    Filter out items that have been skipped/blocked by the user.

    Args:
        idx_all: Array of all candidate indices
        blocked_gids: Set containing blocked item indices

    Returns:
        np.ndarray: Array of indices without blocked items
    """
    if not blocked_gids:
        return idx_all

    blocked_set = set(blocked_gids)
    return np.array([i for i in idx_all if i not in blocked_set])


def _calculate_base_scores(items: pd.DataFrame, idx_all: np.ndarray) -> np.ndarray:
    """
    Calculate base scores based on item ratings.

    How It Works:
        1. Get ratings from items that passed the filter
        2. Normalize to range 0-1 using min-max
        3. Add small noise to avoid ties (identical scores)

    Args:
        items: DataFrame of items
        idx_all: Indices of items that passed the filter

    Returns:
        np.ndarray: Normalized base scores
    """
    # Get ratings, fill NaN with 0, clip negative values
    ratings = items.iloc[idx_all]["rating"].fillna(0.0).clip(lower=0.0).values.astype(float)

    # Normalize to 0-1
    normalized_scores = normalize_minmax(ratings)

    # Add small noise (1e-4) for tie-breaking
    # RandomState(13) for reproducibility
    noise = np.random.RandomState(13).rand(len(idx_all)) * 1e-4

    return normalized_scores + noise


def _add_serendipity_items(
    selected_gids: List[int],
    idx_all: np.ndarray,
    items: pd.DataFrame,
    serendipity_pct: int,
    top_n: int
) -> List[int]:
    """
    Add serendipity (surprise) items to the recommendation results.

    Purpose:
        Increase diversity by inserting some popular items that may not
        have been selected by the MMR algorithm, but could be appealing
        to the user as "surprises".

    Analogy:
        Like when Spotify inserts a popular song into your playlist
        that may not be your favorite genre, but you might enjoy.

    Args:
        selected_gids: List of items already selected by MMR
        idx_all: All candidates that passed the filter
        items: DataFrame of items
        serendipity_pct: Percentage of serendipity items (0-30%)
        top_n: Target total number of items

    Returns:
        List[int]: List of items with serendipity additions
    """
    # Calculate the number of serendipity items to add
    min_serendipity = 1
    max_serendipity = top_n // 5  # Maximum 20% of top_n
    serendipity_count = max(0, min(max_serendipity, int(len(idx_all) * serendipity_pct / 100)))

    if serendipity_count <= 0:
        return selected_gids

    # Find items not yet selected
    already_selected = set(selected_gids)
    available_pool = [g for g in idx_all if g not in already_selected]

    if not available_pool:
        return selected_gids

    # Sort by rating (popularity)
    pool_items = items.iloc[available_pool].copy()
    sorted_by_rating = list(pool_items.sort_values(["rating"], ascending=False).index)

    # Shuffle to add surprise factor
    rng = np.random.RandomState(31)
    rng.shuffle(sorted_by_rating)

    # Take a few top items
    serendipity_picks = sorted_by_rating[:serendipity_count]

    # Combine with existing results
    result = selected_gids.copy()
    result.extend(serendipity_picks)

    return result


def build_feed_cbf(
    items: pd.DataFrame,
    X,
    filters: Dict,
    top_n: int = 12,
    mmr_lambda: float = 0.7,
    per_category_cap: int = 2,
    serendipity_pct: int = 15,
    blocked_gids: Optional[Set[int]] = None,
    liked_gids: Optional[Set[int]] = None
) -> List[Candidate]:
    """
    Build a recommendation feed using Content-Based Filtering.

    This is the MAIN function for the Feed tab that displays recommendations
    without a search query (based on filters and ratings only).

    **IMPROVEMENT:** Now considers user-liked items to provide
    better personalization.

    Workflow:
        1. Filter items by category/city/price
        2. Remove items skipped by the user
        3. Calculate base scores:
           - If user has liked items: combine rating + similarity to liked items
           - If no likes yet: use rating only
        4. Select items using MMR (relevant + diverse)
        5. Add serendipity (surprise) items
        6. Return results as a list of Candidates

    Args:
        items: DataFrame containing all tourist destinations
        X: TF-IDF matrix (scipy sparse matrix)
        filters: Dictionary of user filters (categories, cities, max_price)
        top_n: Number of items to display (default: 12)
        mmr_lambda: MMR parameter 0-1 (default: 0.7)
        per_category_cap: Item limit per category (default: 2)
        serendipity_pct: Percentage of surprise items (default: 15%)
        blocked_gids: Set of item indices skipped by the user
        liked_gids: Set of item indices liked by the user (NEW)

    Returns:
        List[Candidate]: List of candidates with gid and base_score

    Usage Example:
        >>> candidates = build_feed_cbf(
        ...     items=items, X=X,
        ...     filters={"categories": ["Pantai"], "cities": ["Aceh"]},
        ...     top_n=10,
        ...     liked_gids={5, 12, 23}  # User has liked these items
        ... )
        >>> for c in candidates:
        ...     print(f"Item {c.gid}: score {c.base_score:.2f}")
    """
    # Step 1: Filter items based on user criteria
    filter_mask = _mask_by_filters(items, filters)
    idx_all = np.arange(len(items))[filter_mask]

    # Step 2: Remove skipped/blocked items
    idx_all = _filter_blocked_items(idx_all, blocked_gids)

    # If no items pass the filter, return empty
    if idx_all.size == 0:
        return []

    # Step 3: Calculate base scores
    if liked_gids and len(liked_gids) > 0:
        # IMPROVEMENT: Use similarity to liked items if available
        # Calculate centroid of liked items
        liked_list = [g for g in liked_gids if g < len(items)]
        if liked_list:
            liked_vectors = X[liked_list]
            centroid = liked_vectors.mean(axis=0)
            centroid = np.asarray(getattr(centroid, "A", centroid)).reshape(1, -1)

            # Calculate similarity of each candidate to the centroid
            similarities = cosine_similarity(X[idx_all], centroid)[:, 0]

            # Normalize similarity
            sim_scores = normalize_minmax(similarities)

            # Combine with rating (50% similarity + 50% rating)
            rating_scores = normalize_minmax(items.iloc[idx_all]["rating"].fillna(0).values)
            base_scores = 0.7 * sim_scores + 0.3 * rating_scores  # Prioritize similarity
        else:
            # Fallback to rating only
            base_scores = _calculate_base_scores(items, idx_all)
    else:
        # No liked items, use rating only
        base_scores = _calculate_base_scores(items, idx_all)

    # Step 4: Select items using MMR
    selected_gids = mmr_select(
        idx_all=idx_all,
        X=X,
        base_scores=base_scores,
        top_n=top_n,
        lambda_mmr=mmr_lambda,
        per_category_cap=per_category_cap,
        items=items
    )

    # Step 5: Add serendipity items
    selected_gids = _add_serendipity_items(
        selected_gids=selected_gids,
        idx_all=idx_all,
        items=items,
        serendipity_pct=serendipity_pct,
        top_n=top_n
    )

    # Step 6: Create score mapping for output
    score_map = {int(idx_all[i]): float(base_scores[i]) for i in range(len(idx_all))}

    # Limit results to top_n and create Candidate list
    result = [
        Candidate(gid=int(g), base_score=float(score_map.get(int(g), 0.0)))
        for g in selected_gids[:top_n]
    ]

    return result


def _search_with_nn_index(
    query_vector,
    nbrs,
    idx_all: np.ndarray,
    top_n: int
) -> Optional[List[Tuple[int, float]]]:
    """
    Fast search using NearestNeighbors index.

    Advantage:
        Faster than full cosine_similarity, especially for large
        datasets because it uses approximate nearest neighbors algorithms.

    Args:
        query_vector: TF-IDF vector of the user query
        nbrs: NearestNeighbors index (from sklearn)
        idx_all: Indices of items that passed the filter
        top_n: Number of desired results

    Returns:
        List[(gid, similarity)] or None if failed/empty
    """
    # Find nearest neighbors
    n_neighbors = min(max(100, top_n * 5), nbrs.n_samples_fit_)
    distances, neighbor_indices = nbrs.kneighbors(query_vector, n_neighbors=n_neighbors)

    # Convert distance to similarity (cosine distance -> similarity)
    similarities = 1.0 - distances[0]
    candidates = neighbor_indices[0]

    # Filter only those in idx_all (passed user filter)
    valid_indices = set(idx_all.tolist())
    results = [
        (int(gid), float(sim))
        for gid, sim in zip(candidates, similarities)
        if gid in valid_indices
    ]

    return results if results else None


def _search_fallback(
    query_vector,
    X,
    idx_all: np.ndarray
) -> np.ndarray:
    """
    Fallback search using full cosine_similarity.

    Used when:
        - NearestNeighbors index is not available
        - Results from NN index are empty after filtering

    Args:
        query_vector: TF-IDF vector of the query
        X: Full TF-IDF matrix
        idx_all: Indices of items that passed the filter

    Returns:
        np.ndarray: Array of similarity scores
    """
    # Calculate cosine similarity between query and all items that passed the filter
    similarities = cosine_similarity(X[idx_all], query_vector)[:, 0]
    return similarities


def search_cbf(
    items: pd.DataFrame,
    X,
    vectorizer,
    nbrs,
    query: str,
    filters: Dict,
    top_n: int = 12,
    mmr_lambda: float = 0.7,
    per_category_cap: int = 3,
    similarity_threshold: float = 0.1
) -> List[Tuple[int, float]]:
    """
    Search for destinations using a text query (Knowledge Base search).

    This is the MAIN function for the Search tab that finds destinations
    based on user keywords.

    Workflow:
        1. Validate that the vectorizer is available
        2. Filter items based on user criteria
        3. Preprocess and transform the query into a TF-IDF vector
        4. Find similar items using the NN index (or fallback)
        5. Select results using MMR for diversity
        6. Return results as a list of (gid, similarity)

    How TF-IDF Search Works:
        - The user's query is converted into a numerical vector (TF-IDF)
        - Each item also has a TF-IDF vector from its description
        - Items with the most "similar" vectors (high cosine similarity)
          will appear in the search results

    Args:
        items: DataFrame of tourist destinations
        X: TF-IDF matrix
        vectorizer: Fitted TF-IDF vectorizer
        nbrs: NearestNeighbors index (optional, can be None)
        query: Search keywords from the user
        filters: Category/city/price filters
        top_n: Number of results (default: 12)
        mmr_lambda: MMR parameter (default: 0.7)
        per_category_cap: Limit per category (default: 3)

    Returns:
        List[(gid, score)]: List of tuples (item index, similarity score)

    Example:
        >>> results = search_cbf(
        ...     items, X, vectorizer, nbrs,
        ...     query="pantai snorkeling aceh",
        ...     filters={},
        ...     top_n=10
        ... )
        >>> for gid, score in results:
        ...     print(f"Item {gid}: similarity {score:.3f}")
    """
    # Validation: vectorizer must be available
    if vectorizer is None:
        return []  # Cannot search without a vectorizer

    # Step 1: Filter items based on criteria
    filter_mask = _mask_by_filters(items, filters)
    idx_all = np.arange(len(items))[filter_mask]

    if idx_all.size == 0:
        return []  # No items passed the filter

    # Step 2: Preprocess the query and transform to TF-IDF vector
    try:
        processed_query = preprocess_text(query)
        # Handle case where vectorizer might not be fitted properly
        if hasattr(vectorizer, 'idf_'):
            query_vector = vectorizer.transform([processed_query])
        else:
            # Fallback for some sklearn versions or if loaded incorrectly
            # Try to force check if it looks fitted, otherwise raise/return
            print("Warning: Vectorizer attribute 'idf_' missing. Attempting transform anyway.")
            query_vector = vectorizer.transform([processed_query])

    except (NotFittedError, AttributeError, ValueError) as e:
        print(f"Error in vectorizer: {e}")
        return []  # Return empty if vectorizer fails


    # Step 3: Find similar items
    if nbrs is not None:
        # Use NearestNeighbors index (faster)
        nn_results = _search_with_nn_index(query_vector, nbrs, idx_all, top_n)

        if nn_results:
            # Filter results below threshold
            nn_results = [(g, s) for g, s in nn_results if s >= similarity_threshold]

            if not nn_results:
                return []

            # Valid results from NN index
            sub_gids = np.array([g for g, _ in nn_results], dtype=int)
            sub_scores = normalize_minmax([s for _, s in nn_results])

            # Select with MMR for diversity
            selected_gids = mmr_select(
                idx_all=sub_gids,
                X=X,
                base_scores=sub_scores,
                top_n=top_n,
                lambda_mmr=mmr_lambda,
                per_category_cap=per_category_cap,
                items=items
            )

            # Create score mapping for output
            score_map = {int(g): float(s) for g, s in zip(sub_gids, sub_scores)}
            return [(int(g), float(score_map.get(int(g), 0.0))) for g in selected_gids]

    # Fallback: Use full cosine_similarity
    similarities = _search_fallback(query_vector, X, idx_all)

    # Filter by threshold (Raw Cosine Similarity)
    # Only take items with similarity >= threshold (e.g., 0.1)
    valid_mask = similarities >= similarity_threshold

    if not np.any(valid_mask):
        return []

    # Update idx_all and similarities to only valid items
    idx_all = idx_all[valid_mask]
    similarities = similarities[valid_mask]

    base_scores = normalize_minmax(similarities)

    # Select with MMR
    selected_gids = mmr_select(
        idx_all=idx_all,
        X=X,
        base_scores=base_scores,
        top_n=top_n,
        lambda_mmr=mmr_lambda,
        per_category_cap=per_category_cap,
        items=items
    )

    # Return results with scores
    idx_list = list(idx_all)
    return [(int(g), float(base_scores[idx_list.index(int(g))])) for g in selected_gids]
