#!/usr/bin/env python3

import json
import math
import random
from pathlib import Path

import numpy as np


INPUT_FILE = Path("rag_semantic_relevance.json")

OUTPUT_QUESTION_FILE = Path(
    "rag_semantic_metrics_per_question.csv"
)

OUTPUT_SUMMARY_FILE = Path(
    "rag_semantic_metrics_summary.csv"
)

BOOTSTRAPS = 10000
SEED = 42


# ============================================================
# LOAD
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions.")


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def precision_at_k(grades, k):

    selected = grades[:k]

    if not selected:
        return 0.0

    relevant = sum(
        1 for g in selected
        if g >= 1
    )

    return relevant / len(selected)


def direct_precision_at_k(grades, k):

    selected = grades[:k]

    if not selected:
        return 0.0

    direct = sum(
        1 for g in selected
        if g == 2
    )

    return direct / len(selected)


def success_at_k(grades, k):

    """
    At least one partially or directly relevant passage.
    """

    return float(
        any(g >= 1 for g in grades[:k])
    )


def direct_success_at_k(grades, k):

    """
    At least one directly relevant passage.
    """

    return float(
        any(g == 2 for g in grades[:k])
    )


def reciprocal_rank(grades):

    """
    Reciprocal rank of first passage with grade >= 1.
    """

    for rank, grade in enumerate(
        grades,
        start=1,
    ):

        if grade >= 1:
            return 1.0 / rank

    return 0.0


def direct_reciprocal_rank(grades):

    """
    Reciprocal rank of first grade-2 passage.
    """

    for rank, grade in enumerate(
        grades,
        start=1,
    ):

        if grade == 2:
            return 1.0 / rank

    return 0.0


def dcg_at_k(grades, k):

    score = 0.0

    for rank, grade in enumerate(
        grades[:k],
        start=1,
    ):

        gain = (2 ** grade) - 1

        discount = math.log2(
            rank + 1
        )

        score += gain / discount

    return score


def retrieved_set_ndcg_at_k(grades, k):

    """
    Ranking quality within the judged retrieved set.

    IMPORTANT:
    This is NOT corpus-wide canonical nDCG because only the
    retrieved passages are judged.

    It asks:
    given the relevance grades of the retrieved passages,
    how well did the retriever order them?
    """

    actual = dcg_at_k(
        grades,
        k,
    )

    ideal_grades = sorted(
        grades,
        reverse=True,
    )

    ideal = dcg_at_k(
        ideal_grades,
        k,
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


# ============================================================
# PER-QUESTION METRICS
# ============================================================

rows = []

rank_grade_counts = {
    1: {0: 0, 1: 0, 2: 0},
    2: {0: 0, 1: 0, 2: 0},
    3: {0: 0, 1: 0, 2: 0},
    4: {0: 0, 1: 0, 2: 0},
    5: {0: 0, 1: 0, 2: 0},
}


for q in data:

    retrieved = sorted(
        q.get("retrieved", []),
        key=lambda x: x.get(
            "rank",
            999,
        ),
    )

    grades = []

    for result in retrieved:

        grade = result.get(
            "semantic_relevance_grade"
        )

        if grade is None:
            continue

        grade = int(grade)

        grades.append(grade)

        rank = int(
            result.get(
                "rank",
                len(grades),
            )
        )

        if (
            rank in rank_grade_counts
            and grade in (0, 1, 2)
        ):

            rank_grade_counts[
                rank
            ][grade] += 1

    row = {

        "id":
            q.get("id"),

        "question":
            q.get("question"),

        "Precision@1":
            precision_at_k(
                grades,
                1,
            ),

        "Precision@3":
            precision_at_k(
                grades,
                3,
            ),

        "Precision@5":
            precision_at_k(
                grades,
                5,
            ),

        "Direct-Precision@1":
            direct_precision_at_k(
                grades,
                1,
            ),

        "Direct-Precision@3":
            direct_precision_at_k(
                grades,
                3,
            ),

        "Direct-Precision@5":
            direct_precision_at_k(
                grades,
                5,
            ),

        "Success@1":
            success_at_k(
                grades,
                1,
            ),

        "Success@3":
            success_at_k(
                grades,
                3,
            ),

        "Success@5":
            success_at_k(
                grades,
                5,
            ),

        "Direct-Success@1":
            direct_success_at_k(
                grades,
                1,
            ),

        "Direct-Success@3":
            direct_success_at_k(
                grades,
                3,
            ),

        "Direct-Success@5":
            direct_success_at_k(
                grades,
                5,
            ),

        "Semantic-MRR@5":
            reciprocal_rank(
                grades
            ),

        "Direct-MRR@5":
            direct_reciprocal_rank(
                grades
            ),

        "RetrievedSet-nDCG@3":
            retrieved_set_ndcg_at_k(
                grades,
                3,
            ),

        "RetrievedSet-nDCG@5":
            retrieved_set_ndcg_at_k(
                grades,
                5,
            ),

        "Mean-Relevance":
            float(
                np.mean(grades)
            )
            if grades
            else 0.0,
    }

    rows.append(row)


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_ci(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        SEED
    )

    means = []

    n = len(values)

    for _ in range(BOOTSTRAPS):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means.append(
            np.mean(sample)
        )

    lower = np.percentile(
        means,
        2.5,
    )

    upper = np.percentile(
        means,
        97.5,
    )

    return (
        float(np.mean(values)),
        float(lower),
        float(upper),
    )


# ============================================================
# SUMMARY
# ============================================================

metric_names = [

    "Precision@1",
    "Precision@3",
    "Precision@5",

    "Direct-Precision@1",
    "Direct-Precision@3",
    "Direct-Precision@5",

    "Success@1",
    "Success@3",
    "Success@5",

    "Direct-Success@1",
    "Direct-Success@3",
    "Direct-Success@5",

    "Semantic-MRR@5",
    "Direct-MRR@5",

    "RetrievedSet-nDCG@3",
    "RetrievedSet-nDCG@5",

    "Mean-Relevance",
]


summary = {}

for metric in metric_names:

    values = [
        row[metric]
        for row in rows
    ]

    summary[metric] = (
        bootstrap_ci(values)
    )


# ============================================================
# SAVE CSV FILES
# ============================================================

import csv


with open(
    OUTPUT_QUESTION_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


with open(
    OUTPUT_SUMMARY_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Metric",
            "Mean",
            "CI95_Lower",
            "CI95_Upper",
        ]
    )

    for metric in metric_names:

        mean, low, high = (
            summary[metric]
        )

        writer.writerow(
            [
                metric,
                mean,
                low,
                high,
            ]
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n======================================"
)

print(
    "SEMANTIC RETRIEVAL VALIDATION"
)

print(
    "======================================"
)


for metric in metric_names:

    mean, low, high = (
        summary[metric]
    )

    print(
        f"{metric:<24} "
        f"{mean:.3f} "
        f"[{low:.3f}, {high:.3f}]"
    )


print(
    "\n======================================"
)

print(
    "RELEVANCE BY RETRIEVAL RANK"
)

print(
    "======================================"
)


for rank in range(1, 6):

    counts = rank_grade_counts[
        rank
    ]

    total = sum(
        counts.values()
    )

    print(
        f"Rank {rank}: "
        f"Direct={counts[2]:2d}  "
        f"Partial={counts[1]:2d}  "
        f"Irrelevant={counts[0]:2d}"
    )


print(
    "\n======================================"
)

print(
    "FILES"
)

print(
    "======================================"
)

print(OUTPUT_QUESTION_FILE)
print(OUTPUT_SUMMARY_FILE)
