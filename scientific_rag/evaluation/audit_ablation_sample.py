#!/usr/bin/env python3

import json
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "rag_ablation_blinded_scores.json"
)

OUTPUT_FILE = Path(
    "rag_ablation_manual_audit_10.txt"
)

N_AUDIT = 10
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


if len(valid) != 50:
    raise RuntimeError(
        f"Expected 50 valid questions, found {len(valid)}"
    )


# ============================================================
# RANDOM SAMPLE
# ============================================================

rng = random.Random(SEED)

sample = rng.sample(
    valid,
    N_AUDIT,
)


# ============================================================
# DECODE A/B CONDITIONS
# ============================================================

def decode(item):

    scores = item["judge_scores"]

    if item["condition_a"] == "gemma_only":

        gemma_answer = item["answer_a"]
        rag_answer = item["answer_b"]

        gemma_scores = {
            "correctness":
                scores["answer_a_correctness"],
            "completeness":
                scores["answer_a_completeness"],
            "unsupported_claims":
                scores["answer_a_unsupported_claims"],
            "overall":
                scores["answer_a_overall"],
        }

        rag_scores = {
            "correctness":
                scores["answer_b_correctness"],
            "completeness":
                scores["answer_b_completeness"],
            "unsupported_claims":
                scores["answer_b_unsupported_claims"],
            "overall":
                scores["answer_b_overall"],
        }

    else:

        gemma_answer = item["answer_b"]
        rag_answer = item["answer_a"]

        gemma_scores = {
            "correctness":
                scores["answer_b_correctness"],
            "completeness":
                scores["answer_b_completeness"],
            "unsupported_claims":
                scores["answer_b_unsupported_claims"],
            "overall":
                scores["answer_b_overall"],
        }

        rag_scores = {
            "correctness":
                scores["answer_a_correctness"],
            "completeness":
                scores["answer_a_completeness"],
            "unsupported_claims":
                scores["answer_a_unsupported_claims"],
            "overall":
                scores["answer_a_overall"],
        }


    # Decode pairwise winner

    blinded_winner = scores["winner"].upper()

    if blinded_winner == "TIE":

        decoded_winner = "TIE"

    elif (
        blinded_winner == "A"
        and item["condition_a"] == "gemma_only"
    ) or (
        blinded_winner == "B"
        and item["condition_b"] == "gemma_only"
    ):

        decoded_winner = "GEMMA ONLY"

    else:

        decoded_winner = "RAG + GEMMA"


    return (
        gemma_answer,
        rag_answer,
        gemma_scores,
        rag_scores,
        decoded_winner,
    )


# ============================================================
# FORMAT AUDIT
# ============================================================

lines = []

lines.append(
    "RAG ABLATION MANUAL AUDIT"
)

lines.append(
    "=" * 80
)

lines.append(
    f"Random sample: {N_AUDIT}/50 questions"
)

lines.append(
    f"Random seed: {SEED}"
)

lines.append(
    ""
)


for number, item in enumerate(
    sample,
    start=1,
):

    (
        gemma_answer,
        rag_answer,
        gemma_scores,
        rag_scores,
        decoded_winner,
    ) = decode(item)


    lines.append(
        "=" * 80
    )

    lines.append(
        f"AUDIT ITEM {number}"
    )

    lines.append(
        f"Question ID: {item['id']}"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "QUESTION:"
    )

    lines.append(
        item["question"]
    )

    lines.append("")


    lines.append(
        "REFERENCE / GOLD ANSWER:"
    )

    lines.append(
        item["gold_answer"]
    )

    lines.append("")


    lines.append(
        "GEMMA-ONLY ANSWER:"
    )

    lines.append(
        gemma_answer
    )

    lines.append("")

    lines.append(
        "GEMMA-ONLY SCORES:"
    )

    lines.append(
        f"  Correctness:       "
        f"{gemma_scores['correctness']}/4"
    )

    lines.append(
        f"  Completeness:      "
        f"{gemma_scores['completeness']}/4"
    )

    lines.append(
        f"  Unsupported claims:"
        f" {gemma_scores['unsupported_claims']}/4"
    )

    lines.append(
        f"  Overall:           "
        f"{gemma_scores['overall']}/4"
    )

    lines.append("")


    lines.append(
        "RAG + GEMMA ANSWER:"
    )

    lines.append(
        rag_answer
    )

    lines.append("")

    lines.append(
        "RAG + GEMMA SCORES:"
    )

    lines.append(
        f"  Correctness:       "
        f"{rag_scores['correctness']}/4"
    )

    lines.append(
        f"  Completeness:      "
        f"{rag_scores['completeness']}/4"
    )

    lines.append(
        f"  Unsupported claims:"
        f" {rag_scores['unsupported_claims']}/4"
    )

    lines.append(
        f"  Overall:           "
        f"{rag_scores['overall']}/4"
    )

    lines.append("")


    lines.append(
        f"BLINDED JUDGE WINNER: "
        f"{decoded_winner}"
    )

    lines.append(
        "JUDGE REASON:"
    )

    lines.append(
        item["judge_scores"].get(
            "reason",
            ""
        )
    )

    lines.append("")
    lines.append("")


# ============================================================
# SAVE
# ============================================================

text = "\n".join(lines)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write(text)


print(text)

print(
    "\n======================================"
)

print(
    f"Saved audit to: {OUTPUT_FILE}"
)

print(
    "======================================"
)
