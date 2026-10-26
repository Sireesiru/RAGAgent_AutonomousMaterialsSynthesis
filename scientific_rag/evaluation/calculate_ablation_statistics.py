#!/usr/bin/env python3

import json
import csv
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "rag_ablation_blinded_scores.json"
)

OUTPUT_PER_QUESTION = Path(
    "rag_ablation_per_question.csv"
)

OUTPUT_SUMMARY = Path(
    "rag_ablation_summary.csv"
)

BOOTSTRAPS = 10000
SEED = 42


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)


valid = [
    x for x in data
    if x.get("judge_scores") is not None
    and not x.get("judge_error")
]


print(
    f"Loaded {len(valid)} valid paired evaluations."
)


if len(valid) != 50:
    raise RuntimeError(
        f"Expected 50 valid pairs, found {len(valid)}."
    )


# ============================================================
# DECODE BLINDED SCORES
# ============================================================

rows = []

rag_wins = 0
gemma_wins = 0
ties = 0


for item in valid:

    scores = item["judge_scores"]

    condition_a = item["condition_a"]
    condition_b = item["condition_b"]

    # --------------------------------------------------------
    # Extract A/B scores
    # --------------------------------------------------------

    a = {
        "correctness":
            float(scores["answer_a_correctness"]),

        "completeness":
            float(scores["answer_a_completeness"]),

        "unsupported_claims":
            float(scores["answer_a_unsupported_claims"]),

        "overall":
            float(scores["answer_a_overall"]),
    }

    b = {
        "correctness":
            float(scores["answer_b_correctness"]),

        "completeness":
            float(scores["answer_b_completeness"]),

        "unsupported_claims":
            float(scores["answer_b_unsupported_claims"]),

        "overall":
            float(scores["answer_b_overall"]),
    }

    # --------------------------------------------------------
    # Decode conditions
    # --------------------------------------------------------

    if condition_a == "rag_gemma":

        rag = a
        gemma = b

    elif condition_b == "rag_gemma":

        rag = b
        gemma = a

    else:
        raise RuntimeError(
            f"No RAG condition found for {item['id']}"
        )


    # --------------------------------------------------------
    # Decode pairwise winner
    # --------------------------------------------------------

    winner = scores["winner"].upper()

    if winner == "TIE":

        decoded_winner = "tie"
        ties += 1

    elif (
        winner == "A"
        and condition_a == "rag_gemma"
    ) or (
        winner == "B"
        and condition_b == "rag_gemma"
    ):

        decoded_winner = "rag_gemma"
        rag_wins += 1

    else:

        decoded_winner = "gemma_only"
        gemma_wins += 1


    # --------------------------------------------------------
    # Paired differences
    #
    # Positive = RAG better for correctness/completeness/overall
    #
    # For unsupported claims:
    # positive improvement means RAG has FEWER unsupported claims
    # --------------------------------------------------------

    row = {
        "id":
            item["id"],

        "rag_correctness":
            rag["correctness"],

        "gemma_correctness":
            gemma["correctness"],

        "delta_correctness":
            rag["correctness"]
            - gemma["correctness"],

        "rag_completeness":
            rag["completeness"],

        "gemma_completeness":
            gemma["completeness"],

        "delta_completeness":
            rag["completeness"]
            - gemma["completeness"],

        "rag_unsupported_claims":
            rag["unsupported_claims"],

        "gemma_unsupported_claims":
            gemma["unsupported_claims"],

        "unsupported_claims_improvement":
            gemma["unsupported_claims"]
            - rag["unsupported_claims"],

        "rag_overall":
            rag["overall"],

        "gemma_overall":
            gemma["overall"],

        "delta_overall":
            rag["overall"]
            - gemma["overall"],

        "winner":
            decoded_winner,
    }

    rows.append(row)


# ============================================================
# BOOTSTRAP PAIRED DIFFERENCE
# ============================================================

rng = np.random.default_rng(SEED)


def paired_bootstrap(
    rag_values,
    gemma_values,
    higher_is_better=True,
):

    rag_values = np.asarray(
        rag_values,
        dtype=float,
    )

    gemma_values = np.asarray(
        gemma_values,
        dtype=float,
    )

    if higher_is_better:

        differences = (
            rag_values - gemma_values
        )

    else:

        # For unsupported claims,
        # lower score is better.
        differences = (
            gemma_values - rag_values
        )

    n = len(differences)

    bootstrap_means = np.empty(
        BOOTSTRAPS
    )

    for i in range(BOOTSTRAPS):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_means[i] = np.mean(
            differences[indices]
        )

    mean_difference = float(
        np.mean(differences)
    )

    lower = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    return (
        mean_difference,
        lower,
        upper,
        differences,
    )


# ============================================================
# WILCOXON PAIRED TEST
# ============================================================

def wilcoxon_test(
    rag_values,
    gemma_values,
):

    if wilcoxon is None:
        return float("nan")

    rag_values = np.asarray(
        rag_values,
        dtype=float,
    )

    gemma_values = np.asarray(
        gemma_values,
        dtype=float,
    )

    differences = (
        rag_values - gemma_values
    )

    if np.all(differences == 0):
        return 1.0

    try:

        result = wilcoxon(
            rag_values,
            gemma_values,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return float(
            result.pvalue
        )

    except Exception:
        return float("nan")


# ============================================================
# METRICS
# ============================================================

metric_specs = [
    (
        "Correctness",
        "rag_correctness",
        "gemma_correctness",
        True,
    ),
    (
        "Completeness",
        "rag_completeness",
        "gemma_completeness",
        True,
    ),
    (
        "Unsupported Claims",
        "rag_unsupported_claims",
        "gemma_unsupported_claims",
        False,
    ),
    (
        "Overall",
        "rag_overall",
        "gemma_overall",
        True,
    ),
]


summary_rows = []


print(
    "\n======================================"
)

print(
    "RAG ABLATION RESULTS"
)

print(
    "======================================"
)


for (
    metric_name,
    rag_key,
    gemma_key,
    higher_is_better,
) in metric_specs:

    rag_values = np.array(
        [
            row[rag_key]
            for row in rows
        ],
        dtype=float,
    )

    gemma_values = np.array(
        [
            row[gemma_key]
            for row in rows
        ],
        dtype=float,
    )

    rag_mean = float(
        np.mean(rag_values)
    )

    gemma_mean = float(
        np.mean(gemma_values)
    )

    (
        improvement,
        ci_low,
        ci_high,
        differences,
    ) = paired_bootstrap(
        rag_values,
        gemma_values,
        higher_is_better,
    )

    p_value = wilcoxon_test(
        rag_values,
        gemma_values,
    )


    # Standardized paired effect size
    # dz = mean paired difference / SD paired difference

    if higher_is_better:

        raw_difference = (
            rag_values - gemma_values
        )

    else:

        raw_difference = (
            gemma_values - rag_values
        )

    sd_difference = np.std(
        raw_difference,
        ddof=1,
    )

    if sd_difference > 0:

        cohens_dz = float(
            np.mean(raw_difference)
            / sd_difference
        )

    else:
        cohens_dz = 0.0


    print(
        f"\n{metric_name}"
    )

    print(
        f"  Gemma only mean: "
        f"{gemma_mean:.3f}"
    )

    print(
        f"  RAG + Gemma mean: "
        f"{rag_mean:.3f}"
    )

    if metric_name == "Unsupported Claims":

        print(
            f"  Improvement "
            f"(Gemma - RAG): "
            f"{improvement:.3f}"
        )

    else:

        print(
            f"  Improvement "
            f"(RAG - Gemma): "
            f"{improvement:.3f}"
        )

    print(
        f"  95% CI: "
        f"[{ci_low:.3f}, {ci_high:.3f}]"
    )

    print(
        f"  Wilcoxon p: "
        f"{p_value:.6f}"
    )

    print(
        f"  Cohen's dz: "
        f"{cohens_dz:.3f}"
    )


    summary_rows.append(
        {
            "metric":
                metric_name,

            "gemma_only_mean":
                gemma_mean,

            "rag_gemma_mean":
                rag_mean,

            "improvement":
                improvement,

            "ci95_lower":
                ci_low,

            "ci95_upper":
                ci_high,

            "wilcoxon_p":
                p_value,

            "cohens_dz":
                cohens_dz,
        }
    )


# ============================================================
# WIN / LOSS / TIE
# ============================================================

n = len(rows)

print(
    "\n======================================"
)

print(
    "PAIRWISE PREFERENCE"
)

print(
    "======================================"
)

print(
    f"RAG + Gemma wins: "
    f"{rag_wins}/{n} "
    f"({100 * rag_wins / n:.1f}%)"
)

print(
    f"Gemma only wins: "
    f"{gemma_wins}/{n} "
    f"({100 * gemma_wins / n:.1f}%)"
)

print(
    f"Ties: "
    f"{ties}/{n} "
    f"({100 * ties / n:.1f}%)"
)


# ============================================================
# SIMPLE SIGN TEST ON NON-TIES
# ============================================================

non_ties = (
    rag_wins + gemma_wins
)

if non_ties > 0:

    try:
        from scipy.stats import binomtest

        sign_test = binomtest(
            rag_wins,
            non_ties,
            p=0.5,
            alternative="two-sided",
        )

        sign_p = float(
            sign_test.pvalue
        )

    except Exception:
        sign_p = float("nan")

else:
    sign_p = float("nan")


print(
    f"Pairwise sign-test p: "
    f"{sign_p:.6f}"
)


# ============================================================
# SAVE PER-QUESTION CSV
# ============================================================

with open(
    OUTPUT_PER_QUESTION,
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


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "metric",
        "gemma_only_mean",
        "rag_gemma_mean",
        "improvement",
        "ci95_lower",
        "ci95_upper",
        "wilcoxon_p",
        "cohens_dz",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
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

print(
    OUTPUT_PER_QUESTION
)

print(
    OUTPUT_SUMMARY
)
