"""
ufw.py - User Feedback Weighting (UFW) Module
==============================================
This module contains algorithms for adjusting recommendation rankings
based on user feedback (Like/Skip) during a session.

Core Concepts:
- UFW uses user preferences to personalize results
- Items similar to Liked items receive a score boost
- Skipped items receive a penalty
- Categories that are frequently Liked also receive a bonus

Simple Analogy:
    Imagine UFW as a smart restaurant waiter:
    - If you like dishes A and B (Like), the waiter will offer
      other dishes similar to A and B
    - If you say you don't want dish C (Skip), the waiter won't
      offer dish C again
    - If you like 3 seafood dishes, the waiter will offer more
      seafood options

UFW Formula:
    final_score = base_score
                  + α × similarity_to_like_centroid
                  - β × (1 if item is skipped, 0 otherwise)
                  + γ × like_count_for_this_category

    Where:
    - α (alpha): How much influence "similarity to Liked items" has
    - β (beta): How much penalty is applied to Skipped items
    - γ (gamma): How much bonus is given to favorite categories
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from .utils import normalize_minmax


# =============================================================================
# HELPER FUNCTIONS - Smaller utility functions
# =============================================================================

def _compute_centroid_dense(X, indices: List[int]) -> Optional[np.ndarray]:
    """
    Compute the centroid (center point) of Liked items.

    What is a Centroid?
        A centroid is the "average" of all Liked item vectors.
        Imagine you have 3 points on a map, the centroid is their midpoint.

    Analogy:
        If you Like 3 beaches: Beach A, Beach B, Beach C,
        the centroid is the "ideal representation" of your favorite beaches.
        New items similar to this centroid are likely ones you'd enjoy too.

    Args:
        X: TF-IDF matrix (all items)
        indices: List of indices of Liked items

    Returns:
        np.ndarray: Centroid vector (1 x n_features), or None if empty

    Example:
        >>> liked_items = [5, 12, 23]  # User has Liked items 5, 12, 23
        >>> centroid = _compute_centroid_dense(X, liked_items)
        >>> # centroid now represents the user's "average preference"
    """
    # If no items have been Liked, return None
    if not indices:
        return None

    # Get TF-IDF vectors for all Liked items
    liked_vectors = X[list(indices)]

    # Compute the mean of all vectors
    mean_vector = liked_vectors.mean(axis=0)

    # Convert to 2D numpy array (1 x n_features) for cosine_similarity
    # Handling sparse matrix with getattr for "A" attribute
    centroid = np.asarray(getattr(mean_vector, "A", mean_vector)).reshape(1, -1)

    return centroid


def _build_category_preference(
    items: pd.DataFrame,
    liked_indices: List[int]
) -> Dict[str, int]:
    """
    Build a category preference dictionary from Liked items.

    How It Works:
        - Look at all Liked items
        - Count how many times each category appears
        - Return as a dictionary {category: count}

    Example:
        If the user Likes 3 items with categories:
        - Item 1: "Beach"
        - Item 2: "Beach"
        - Item 3: "Mountain"

        The result would be: {"Beach": 2, "Mountain": 1}

        This means the user prefers beaches, so other Beach items
        will receive a larger score bonus.

    Args:
        items: Items DataFrame
        liked_indices: List of indices of Liked items

    Returns:
        Dict[str, int]: Dictionary {category_name: like_count}
    """
    category_preference: Dict[str, int] = {}

    if not liked_indices:
        return category_preference

    # Get categories from all Liked items
    liked_categories = (
        items.iloc[liked_indices]["category"]
        .fillna("")  # Handle NaN
        .apply(lambda s: str(s).split(",")[0].strip())  # Get the first category
    )

    # Count the frequency of each category
    for category in liked_categories:
        if category:  # Skip empty strings
            category_preference[category] = category_preference.get(category, 0) + 1

    return category_preference


def _compute_like_similarity(
    gids: List[int],
    X,
    centroid: Optional[np.ndarray]
) -> np.ndarray:
    """
    Compute the similarity of each item to the Like centroid.

    How It Works:
        1. If a centroid exists (user has Liked some items):
           - Compute cosine similarity of each candidate to the centroid
           - Normalize to the range 0-1
        2. If no centroid exists:
           - Return an array of zeros (no boost)

    Args:
        gids: List of candidate item indices
        X: TF-IDF matrix
        centroid: Centroid vector from Liked items (or None)

    Returns:
        np.ndarray: Array of similarity scores (0-1) for each candidate

    Example:
        >>> similarities = _compute_like_similarity([1, 2, 3], X, centroid)
        >>> print(similarities)  # [0.8, 0.3, 0.6]
        >>> # Item 1 is most similar to user preference (0.8)
    """
    # If no centroid exists or no candidates
    if centroid is None or len(gids) == 0:
        return np.zeros(len(gids), dtype=float)

    # Compute cosine similarity between each candidate and the centroid
    similarities = cosine_similarity(X[gids], centroid)[:, 0]

    # Normalize to the range 0-1 for consistency
    normalized_similarities = normalize_minmax(similarities)

    return normalized_similarities


def _calculate_final_scores(
    gids: List[int],
    base_scores_map: Dict[int, float],
    like_similarities: np.ndarray,
    blocked_gids: Set[int],
    category_preference: Dict[str, int],
    items: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    has_likes: bool
) -> Dict[int, float]:
    """
    Calculate the final UFW score for each item.

    Formula:
        final_score = base_score
                      + α × like_similarity (if Likes exist)
                      - β × 1 (if item is Skipped)
                      + γ × category_like_count

    Parameter Explanation:
        - α (alpha): Boost for items similar to Liked items
          High value = more personalization based on Likes

        - β (beta): Penalty for Skipped items
          High value = Skipped items are strongly avoided

        - γ (gamma): Boost for favorite categories
          High value = frequently Liked categories are highly prioritized

    Args:
        gids: List of candidate item indices
        base_scores_map: Dictionary {gid: base_score}
        like_similarities: Array of similarities to Like centroid
        blocked_gids: Set of indices of Skipped items
        category_preference: Category preference dictionary
        items: Items DataFrame
        alpha, beta, gamma: UFW parameters
        has_likes: Boolean indicating whether the user has Liked any items

    Returns:
        Dict[int, float]: Dictionary {gid: final_score}
    """
    final_scores: Dict[int, float] = {}

    for i, gid in enumerate(gids):
        # Start from the base score
        score = float(base_scores_map.get(gid, 0.0))

        # Component 1: Boost from similarity to Like centroid
        # Only applies if the user has Liked at least 1 item
        if has_likes:
            score += alpha * like_similarities[i]

        # Component 2: Penalty if item is Skipped
        # Skipped items will have their score reduced
        if gid in blocked_gids:
            score -= beta

        # Component 3: Boost from category preference
        # Categories that are frequently Liked receive a bonus
        category = str(items.iloc[gid]["category"]).split(",")[0].strip()
        if category and category_preference:
            like_count = category_preference.get(category, 0)
            score += gamma * like_count

        final_scores[gid] = score

    return final_scores


# =============================================================================
# MAIN FUNCTION - Primary UFW function
# =============================================================================

def apply_ufw(
    gids: List[int],
    base_scores_map: Dict[int, float],
    X,
    items: pd.DataFrame,
    alpha: float = 0.6,
    beta: float = 0.7,
    gamma: float = 0.02
) -> List[Tuple[int, float]]:
    """
    Apply User Feedback Weighting (UFW) to re-rank results.

    This is the MAIN function for personalizing recommendations based on
    user feedback during a session (Like/Skip).

    Workflow:
        1. Retrieve Like and Skip data from session_state
        2. Build category preferences from Liked items
        3. Compute the centroid (center point) of Liked items
        4. Compute the similarity of each candidate to the centroid
        5. Calculate the final score using the UFW formula
        6. Sort results by final score (descending)

    Full Analogy:
        Imagine you're in a bookstore:

        1. You pick up 3 books and say "I like these" (Like)
           → The system notes: "Oh, the user likes mystery novels"

        2. You see 1 book and say "not interested" (Skip)
           → The system notes: "Don't show this book again"

        3. The system then:
           - Finds other books SIMILAR to the 3 books you liked
           - Lowers the ranking of books you skipped
           - Raises the ranking of books in the same category

    Args:
        gids: List of candidate item indices to re-rank
        base_scores_map: Dictionary {gid: base_score} from CBF
        X: TF-IDF matrix for computing similarity
        items: Items metadata DataFrame
        alpha: Boost weight for similarity to Likes (default: 0.6)
            - 0.0 = no influence from Likes
            - 1.0 = strong influence from Likes
        beta: Penalty weight for Skipped items (default: 0.7)
            - 0.0 = no penalty for Skips
            - 1.0 = strong penalty for Skips
        gamma: Boost weight for favorite categories (default: 0.02)
            - 0.0 = no category bonus
            - 0.1 = significant bonus for favorite categories

    Returns:
        List[(gid, score)]: List of tuples (item index, final score),
                           sorted from highest to lowest score

    Usage Example:
        >>> # From CBF results
        >>> candidates = [(1, 0.8), (2, 0.7), (3, 0.9)]
        >>> gids = [1, 2, 3]
        >>> base_scores = {1: 0.8, 2: 0.7, 3: 0.9}
        >>>
        >>> # Apply UFW
        >>> reranked = apply_ufw(gids, base_scores, X, items)
        >>>
        >>> # Results may differ from CBF due to personalization
        >>> print(reranked)  # [(3, 1.2), (1, 0.9), (2, 0.3)]

    UFW Impact Example:
        Before UFW (from CBF):
            1. Beach A (score: 0.9)
            2. Mountain B (score: 0.85)
            3. Beach C (score: 0.8)

        User Likes: Beach A, Beach C
        User Skips: Mountain B

        After UFW:
            1. Beach C (score: 1.3) ← boosted due to similarity with Likes
            2. Beach A (score: 1.2) ← remains high
            3. Mountain B (score: 0.15) ← dropped significantly due to Skip
    """
    # Step 1: Retrieve feedback data from Streamlit session_state
    liked_indices: List[int] = list(st.session_state.liked_idx)
    blocked_indices: Set[int] = set(st.session_state.blocked_idx)

    # Step 2: Build category preferences from Liked items
    category_preference = _build_category_preference(items, liked_indices)

    # Step 3: Compute centroid from Liked items
    centroid = _compute_centroid_dense(X, liked_indices)

    # Step 4: Compute similarity of each candidate to the centroid
    like_similarities = _compute_like_similarity(gids, X, centroid)

    # Step 5: Calculate final UFW scores
    final_scores = _calculate_final_scores(
        gids=gids,
        base_scores_map=base_scores_map,
        like_similarities=like_similarities,
        blocked_gids=blocked_indices,
        category_preference=category_preference,
        items=items,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        has_likes=len(liked_indices) > 0
    )

    # Step 6: Sort by final score (highest first)
    sorted_results = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results
