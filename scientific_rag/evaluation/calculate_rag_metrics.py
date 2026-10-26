#!/usr/bin/env python3

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "rag_validation_retrievals.json"
)

QUESTION_METRICS_FILE = Path(
    "rag_validation_per_question.csv"
)

SUMMARY_FILE = Path(
    "rag_validation_metrics.csv"
)

CATEGORY_FILE = Path(
    "rag_validation_by_category.csv"
)

TOP_K = 5

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def reciprocal_rank(rank):

    if rank is None:
        return 0.0

    if rank < 1 or rank > TOP_K:
        return 0.0

    return 1.0 / rank


def hit_at(rank, k):

    if rank is None:
        return 0

    return int(rank <= k)


def normalize_source(source):

    return str(source or "").strip().lower()


# ============================================================
# SOURCE DIVERSITY
# ============================================================

def source_diversity(retrieved):

    sources = [
        normalize_source(x.get("source"))
        for x in retrieved[:TOP_K]
        if x.get("source")
    ]

    unique_sources = len(set(sources))

    denominator = TOP_K

    return (
        unique_sources,
        unique_sources / denominator,
    )


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_ci(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    means = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    n = len(values)

    for i in range(
        BOOTSTRAP_ITERATIONS
    ):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = np.mean(sample)

    mean = float(
        np.mean(values)
    )

    lower = float(
        np.percentile(means, 2.5)
    )

    upper = float(
        np.percentile(means, 97.5)
    )

    return mean, lower, upper


# ============================================================
# BUILD PER-QUESTION METRICS
# ============================================================

def build_question_metrics(records):

    rows = []

    for item in records:

        retrieved = item.get(
            "retrieved",
            [],
        )

        source_rank = item.get(
            "gold_source_first_rank"
        )

        chunk_rank = item.get(
            "gold_chunk_first_rank"
        )

        unique_sources, diversity = (
            source_diversity(retrieved)
        )

        rows.append({

            "query_id":
                item.get("id"),

            "category":
                item.get(
                    "category",
                    "Unknown",
                ),

            "question":
                item.get("question"),

            "gold_source":
                item.get("gold_source"),

            "gold_chunk_id":
                item.get(
                    "gold_chunk_id"
                ),

            "gold_source_rank":
                source_rank,

            "gold_chunk_rank":
                chunk_rank,

            # -------------------------
            # PAPER LEVEL
            # -------------------------

            "Gold-Paper-Hit@1":
                hit_at(
                    source_rank,
                    1,
                ),

            "Gold-Paper-Hit@3":
                hit_at(
                    source_rank,
                    3,
                ),

            "Gold-Paper-Hit@5":
                hit_at(
                    source_rank,
                    5,
                ),

            "Gold-Paper-MRR@5":
                reciprocal_rank(
                    source_rank
                ),

            # -------------------------
            # EXACT CHUNK LEVEL
            # -------------------------

            "Gold-Chunk-Hit@1":
                hit_at(
                    chunk_rank,
                    1,
                ),

            "Gold-Chunk-Hit@3":
                hit_at(
                    chunk_rank,
                    3,
                ),

            "Gold-Chunk-Hit@5":
                hit_at(
                    chunk_rank,
                    5,
                ),

            "Gold-Chunk-MRR@5":
                reciprocal_rank(
                    chunk_rank
                ),

            # -------------------------
            # MMR SOURCE DIVERSITY
            # -------------------------

            "Unique-Sources@5":
                unique_sources,

            "Source-Diversity@5":
                diversity,

            "retrieval_error":
                item.get(
                    "retrieval_error"
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY METRICS
# ============================================================

def summarize(df):

    metric_columns = [

        "Gold-Paper-Hit@1",
        "Gold-Paper-Hit@3",
        "Gold-Paper-Hit@5",
        "Gold-Paper-MRR@5",

        "Gold-Chunk-Hit@1",
        "Gold-Chunk-Hit@3",
        "Gold-Chunk-Hit@5",
        "Gold-Chunk-MRR@5",

        "Unique-Sources@5",
        "Source-Diversity@5",
    ]

    rows = []

    for metric in metric_columns:

        mean, lower, upper = (
            bootstrap_ci(
                df[metric].values
            )
        )

        rows.append({
            "metric": metric,
            "mean": mean,
            "ci_95_lower": lower,
            "ci_95_upper": upper,
            "n_queries": len(df),
        })

    return pd.DataFrame(rows)


# ============================================================
# CATEGORY METRICS
# ============================================================

def summarize_categories(df):

    rows = []

    for category, group in df.groupby(
        "category"
    ):

        rows.append({

            "category":
                category,

            "n_questions":
                len(group),

            "Gold-Paper-Hit@1":
                group[
                    "Gold-Paper-Hit@1"
                ].mean(),

            "Gold-Paper-Hit@3":
                group[
                    "Gold-Paper-Hit@3"
                ].mean(),

            "Gold-Paper-Hit@5":
                group[
                    "Gold-Paper-Hit@5"
                ].mean(),

            "Gold-Paper-MRR@5":
                group[
                    "Gold-Paper-MRR@5"
                ].mean(),

            "Gold-Chunk-Hit@5":
                group[
                    "Gold-Chunk-Hit@5"
                ].mean(),

            "Gold-Chunk-MRR@5":
                group[
                    "Gold-Chunk-MRR@5"
                ].mean(),

            "Source-Diversity@5":
                group[
                    "Source-Diversity@5"
                ].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        records = json.load(f)

    print(
        f"Loaded {len(records)} evaluated questions."
    )

    df = build_question_metrics(
        records
    )

    df.to_csv(
        QUESTION_METRICS_FILE,
        index=False,
    )

    summary = summarize(df)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    categories = summarize_categories(
        df
    )

    categories.to_csv(
        CATEGORY_FILE,
        index=False,
    )

    print("\n======================================")
    print("RAG RETRIEVAL VALIDATION")
    print("======================================")

    for _, row in summary.iterrows():

        print(
            f"{row['metric']:25s} "
            f"{row['mean']:.3f} "
            f"[{row['ci_95_lower']:.3f}, "
            f"{row['ci_95_upper']:.3f}]"
        )

    print("\n======================================")
    print("BENCHMARK")
    print("======================================")

    print(
        "Questions:",
        len(df),
    )

    print(
        "Categories:",
        df["category"].nunique(),
    )

    print(
        "Gold papers:",
        df["gold_source"].nunique(),
    )

    errors = df[
        "retrieval_error"
    ].notna().sum()

    print(
        "Retrieval errors:",
        errors,
    )

    print("\n======================================")
    print("FILES")
    print("======================================")

    print(
        QUESTION_METRICS_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        CATEGORY_FILE
    )


if __name__ == "__main__":
    main()