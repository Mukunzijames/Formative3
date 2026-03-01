"""
Part 2: Bayesian Probability
Author: Favor

Implements Bayes' Theorem from scratch to compute P(Negative | keyword)
for sentiment analysis on the IMDb 50K Movie Reviews dataset.

No sklearn or NLTK — only standard Python, NumPy, Pandas, Matplotlib, Seaborn.

Dataset source
--------------
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Bayes' Theorem
--------------
P(Negative | keyword) = P(keyword | Negative) × P(Negative) / P(keyword)

Terms
-----
P(Negative)           — prior      : base rate of negative reviews
P(keyword | Negative) — likelihood : how often the keyword appears in negative reviews
P(keyword)            — marginal   : overall keyword frequency across all reviews
P(Negative | keyword) — posterior  : our updated belief after observing the keyword

Word-level (Multinomial) counting
----------------------------------
Each function counts word occurrences, not just presence per review.
A review that says "waste" five times contributes five counts to the likelihood.
This is multinomial Naive Bayes — differs from Bernoulli (binary has/hasn't),
but both approaches are valid. Multinomial is more common for text classification.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Private helpers ───────────────────────────────────────────────────────────

def _tokenize(text):
    """Split a string into lowercase word tokens using regex — no NLTK needed."""
    return re.findall(r"\b\w+\b", str(text).lower())


def _build_word_counts(df, sentiment=None):
    """
    Count word occurrences across the full dataset or a sentiment subset.

    Parameters
    ----------
    df        : pd.DataFrame  Must have 'tokens' (list[str]) and 'sentiment_clean'.
    sentiment : str or None   If given, restrict to rows matching that label.

    Returns
    -------
    (Counter, int)  word_counts, total_word_count
    """
    # Filter to the requested sentiment class, or keep all rows
    if sentiment is not None:
        subset = df[df["sentiment_clean"] == sentiment.lower().strip()]
    else:
        subset = df

    # Flatten all per-review token lists into a single list of words
    words  = [w for tokens in subset["tokens"] for w in tokens]
    counts = Counter(words)
    return counts, sum(counts.values())


# ── Preprocessing — call once after load_imdb_data() ─────────────────────────

def prepare_dataframe(df):
    """
    Add helper columns required by all Bayesian calculation functions.

    Must be called once on the raw dataframe before any probability calculations.

    Adds
    ----
    review_clean    : lowercase version of the review text
    sentiment_clean : lowercase, stripped sentiment label
    tokens          : list[str] of word tokens per review (no NLTK)

    Parameters
    ----------
    df : pd.DataFrame  Raw dataframe from load_imdb_data().

    Returns
    -------
    pd.DataFrame  Copy with three new columns added — original is not mutated.
    """
    df = df.copy()

    # Lowercase all review text for consistent tokenisation
    df["review_clean"]    = df["review"].str.lower()

    # Strip and lowercase the sentiment label to avoid 'Positive' vs 'positive' mismatches
    df["sentiment_clean"] = df["sentiment"].str.lower().str.strip()

    # Tokenise each review into a list of individual word strings
    df["tokens"]          = df["review_clean"].apply(_tokenize)

    total = sum(len(t) for t in df["tokens"])
    print(f"[Bayesian] {len(df):,} reviews | {total:,} total tokens")
    return df


# ── Core Bayesian functions ───────────────────────────────────────────────────

def calculate_prior(df, sentiment):
    """
    Compute P(sentiment) — the prior probability of a sentiment class.

        P(sentiment) = count(reviews with label) / count(all reviews)

    This is our baseline belief before observing any keyword.

    Parameters
    ----------
    df        : pd.DataFrame  Must have 'sentiment_clean' column.
    sentiment : str           'positive' or 'negative' (case-insensitive).

    Returns
    -------
    float  Prior probability in [0, 1].
    """
    sentiment = sentiment.lower().strip()

    # Count how many reviews carry the requested label
    count = (df["sentiment_clean"] == sentiment).sum()

    return count / len(df)


def calculate_likelihood(df, keyword, sentiment):
    """
    Compute P(keyword | sentiment) using word-level (multinomial) counting.

        P(keyword | sentiment) = occurrences of keyword in sentiment reviews
                                 / total words in sentiment reviews

    Intuition: of all words written in negative reviews, what fraction is 'waste'?

    Parameters
    ----------
    df        : pd.DataFrame  Must have 'tokens' and 'sentiment_clean'.
    keyword   : str           The word to look up (case-insensitive).
    sentiment : str           'positive' or 'negative'.

    Returns
    -------
    float  Likelihood in [0, 1]. Returns 0.0 if the sentiment subset has no words.
    """
    # Get word counts restricted to the target sentiment class
    counts, total = _build_word_counts(df, sentiment)

    if total == 0:
        return 0.0

    # What fraction of all words in this sentiment is the keyword?
    return counts[keyword.lower().strip()] / total


def calculate_marginal(df, keyword):
    """
    Compute P(keyword) — the keyword's overall frequency in the full corpus.

        P(keyword) = occurrences of keyword in all reviews / total words in corpus

    This equals the law of total probability result:
        P(kw) = P(kw|pos)×P(pos) + P(kw|neg)×P(neg)
    Both are equivalent; direct counting is used here for clarity.

    Parameters
    ----------
    df      : pd.DataFrame  Must have a 'tokens' column.
    keyword : str

    Returns
    -------
    float  Marginal probability in [0, 1]. Returns 0.0 if the corpus is empty.
    """
    # Count across the full dataset — no sentiment filter
    counts, total = _build_word_counts(df)

    if total == 0:
        return 0.0

    return counts[keyword.lower().strip()] / total


def calculate_posterior(likelihood, prior, marginal):
    """
    Apply Bayes' Theorem to compute the posterior probability.

        P(sentiment | keyword) = P(keyword | sentiment) × P(sentiment)
                                 / P(keyword)

    Parameters
    ----------
    likelihood : float  P(keyword | sentiment)
    prior      : float  P(sentiment)
    marginal   : float  P(keyword)

    Returns
    -------
    float  Posterior in [0, 1].
           Returns 0.0 if marginal is 0 (keyword not found in corpus).
    """
    # Guard against division by zero if the keyword doesn't appear at all
    if marginal == 0.0:
        return 0.0

    return (likelihood * prior) / marginal


# ── Convenience: run the full analysis for all keywords at once ───────────────

def analyze_keywords(df, positive_keywords, negative_keywords,
                     target_sentiment="negative"):
    """
    Run the complete Bayesian analysis for every keyword and return a tidy table.

    For each keyword the table shows all four required probabilities:
        Prior | Likelihood | Marginal | Posterior

    Parameters
    ----------
    df                : pd.DataFrame  Output of prepare_dataframe().
    positive_keywords : list[str]     Keywords expected to indicate positive sentiment.
    negative_keywords : list[str]     Keywords expected to indicate negative sentiment.
    target_sentiment  : str           Class to compute posterior for (default: 'negative').

    Returns
    -------
    pd.DataFrame  One row per keyword with columns:
        Keyword | Type | P(target) | P(keyword) | P(keyword|target) | P(target|keyword)
    """
    t     = target_sentiment.capitalize()
    prior = calculate_prior(df, target_sentiment)

    rows = []

    # Positive indicators — we expect posterior to DROP below the prior
    for kw in positive_keywords:
        lh = calculate_likelihood(df, kw, target_sentiment)
        mg = calculate_marginal(df, kw)
        po = calculate_posterior(lh, prior, mg)
        rows.append({
            "Keyword":          kw,
            "Type":             "Positive Indicator",
            f"P({t})":          round(prior, 6),
            "P(keyword)":       round(mg,    6),
            f"P(keyword|{t})":  round(lh,    6),
            f"P({t}|keyword)":  round(po,    6),
        })

    # Negative indicators — we expect posterior to RISE above the prior
    for kw in negative_keywords:
        lh = calculate_likelihood(df, kw, target_sentiment)
        mg = calculate_marginal(df, kw)
        po = calculate_posterior(lh, prior, mg)
        rows.append({
            "Keyword":          kw,
            "Type":             "Negative Indicator",
            f"P({t})":          round(prior, 6),
            "P(keyword)":       round(mg,    6),
            f"P(keyword|{t})":  round(lh,    6),
            f"P({t}|keyword)":  round(po,    6),
        })

    return pd.DataFrame(rows)


# Note: plot functions have moved to visualization.py
# Import from there: from formative3_utils.visualization import (
#     plot_sentiment_distribution, plot_prior_vs_posterior, plot_probability_heatmap
# )