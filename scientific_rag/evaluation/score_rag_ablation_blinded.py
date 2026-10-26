#!/usr/bin/env python3

import json
import random
import re
import time
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("rag_ablation_results.json")

OUTPUT_FILE = Path(
    "rag_ablation_blinded_scores.json"
)

GEMMA_URL = "http://localhost:11434/api/chat"
GEMMA_MODEL = "gemma4:12b"

SEED = 42
TEMPERATURE = 0.0

random.seed(SEED)


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as f:

    data = json.load(f)


print(
    f"Loaded {len(data)} ablation questions."
)


# ============================================================
# GEMMA CALL
# ============================================================

def call_judge(prompt):

    payload = {
        "model": GEMMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 700,
        },
    }

    response = requests.post(
        GEMMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()

    content = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not content:
        raise RuntimeError(
            "Judge returned empty response."
        )

    return content


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text):

    # Remove markdown code fences if present.
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)

    except Exception:
        pass

    # Fallback: locate first JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        candidate = cleaned[
            start:end + 1
        ]

        return json.loads(candidate)

    raise ValueError(
        "Could not parse judge JSON."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_score(score):

    required = [
        "answer_a_correctness",
        "answer_a_completeness",
        "answer_a_unsupported_claims",
        "answer_a_overall",

        "answer_b_correctness",
        "answer_b_completeness",
        "answer_b_unsupported_claims",
        "answer_b_overall",

        "winner",
        "reason",
    ]

    for key in required:

        if key not in score:
            raise ValueError(
                f"Missing field: {key}"
            )

    numeric_fields = [
        "answer_a_correctness",
        "answer_a_completeness",
        "answer_a_unsupported_claims",
        "answer_a_overall",

        "answer_b_correctness",
        "answer_b_completeness",
        "answer_b_unsupported_claims",
        "answer_b_overall",
    ]

    for key in numeric_fields:

        value = score[key]

        if not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{key} is not numeric."
            )

        if value < 0 or value > 4:
            raise ValueError(
                f"{key} outside 0-4."
            )

    winner = str(
        score["winner"]
    ).strip().upper()

    if winner not in [
        "A",
        "B",
        "TIE",
    ]:

        raise ValueError(
            f"Invalid winner: {winner}"
        )

    score["winner"] = winner

    return score


# ============================================================
# JUDGE PROMPT
# ============================================================

def build_judge_prompt(
    question,
    gold_answer,
    answer_a,
    answer_b,
):

    return f"""
You are evaluating two answers to a scientific question.

You are given:

1. A QUESTION.
2. A SOURCE-DERIVED REFERENCE ANSWER.
3. ANSWER A.
4. ANSWER B.

The reference answer was derived from the scientific source
used to construct the benchmark question.

Evaluate each answer ONLY against the question and reference
answer.

Do not assume that a longer answer is better.

Do not reward stylistic detail.

Do not use outside scientific knowledge to repair an answer.

An answer should receive a high score only when its claims
are consistent with and supported by the reference answer.

A concise answer that exactly matches the reference should
receive a high score.

An answer that contains the correct information but adds
unsupported specific claims should be penalized.


SCORING
=======

CORRECTNESS: 0-4

4 = fully correct and consistent with reference
3 = mostly correct; minor error or imprecision
2 = partially correct; important error or missing component
1 = minimal correct information
0 = incorrect or does not answer


COMPLETENESS: 0-4

4 = contains all important information in reference
3 = contains most important information
2 = contains approximately half
1 = contains little
0 = contains none


UNSUPPORTED CLAIMS: 0-4

IMPORTANT: for this metric LOWER IS BETTER.

0 = no unsupported scientific claims
1 = minor unsupported elaboration
2 = several unsupported claims
3 = substantial unsupported content
4 = answer is dominated by unsupported/invented claims


OVERALL: 0-4

4 = excellent answer
3 = good answer
2 = partially satisfactory
1 = poor answer
0 = incorrect/unusable


PAIRWISE WINNER

Choose:

A
B
TIE

The winner should be the answer that best matches the
reference answer while minimizing unsupported claims.


QUESTION
========

{question}


REFERENCE ANSWER
================

{gold_answer}


ANSWER A
========

{answer_a}


ANSWER B
========

{answer_b}


Return ONLY valid JSON with exactly this structure:

{{
  "answer_a_correctness": 0,
  "answer_a_completeness": 0,
  "answer_a_unsupported_claims": 0,
  "answer_a_overall": 0,

  "answer_b_correctness": 0,
  "answer_b_completeness": 0,
  "answer_b_unsupported_claims": 0,
  "answer_b_overall": 0,

  "winner": "A",
  "reason": "brief explanation"
}}
""".strip()


# ============================================================
# SAVE
# ============================================================

def save_results(results):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

results = []

errors = 0

rag_positions = {
    "A": 0,
    "B": 0,
}


for i, item in enumerate(
    data,
    start=1,
):

    qid = item.get(
        "id",
        f"Q{i:02d}",
    )

    question = item.get(
        "question",
        "",
    )

    gold_answer = item.get(
        "gold_answer",
        "",
    )

    gemma_answer = item.get(
        "gemma_only_answer",
        "",
    )

    rag_answer = item.get(
        "rag_gemma_answer",
        "",
    )


    # ========================================================
    # RANDOMIZE A / B
    # ========================================================

    if random.random() < 0.5:

        answer_a = gemma_answer
        answer_b = rag_answer

        condition_a = "gemma_only"
        condition_b = "rag_gemma"

        rag_position = "B"

    else:

        answer_a = rag_answer
        answer_b = gemma_answer

        condition_a = "rag_gemma"
        condition_b = "gemma_only"

        rag_position = "A"


    rag_positions[
        rag_position
    ] += 1


    print(
        "\n======================================"
    )

    print(
        f"[{i}/{len(data)}] {qid}"
    )

    print(
        f"RAG hidden position: {rag_position}"
    )


    prompt = build_judge_prompt(
        question,
        gold_answer,
        answer_a,
        answer_b,
    )


    record = {
        "id": qid,
        "question": question,
        "gold_answer": gold_answer,

        # Stored for decoding after judging.
        # These labels are NOT shown to the judge.
        "condition_a": condition_a,
        "condition_b": condition_b,

        "answer_a": answer_a,
        "answer_b": answer_b,
    }


    try:

        raw = call_judge(
            prompt
        )

        score = parse_json_response(
            raw
        )

        score = validate_score(
            score
        )

        record[
            "judge_scores"
        ] = score

        record[
            "judge_error"
        ] = None

        print(
            "Judge completed."
        )


    except Exception as exc:

        errors += 1

        record[
            "judge_scores"
        ] = None

        record[
            "judge_error"
        ] = str(exc)

        print(
            f"ERROR: {exc}"
        )


    results.append(record)

    save_results(results)

    time.sleep(0.1)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n======================================"
)

print(
    "BLINDED ABLATION SCORING COMPLETE"
)

print(
    "======================================"
)

print(
    f"Questions: {len(results)}"
)

print(
    f"Errors: {errors}"
)

print(
    f"RAG assigned to A: "
    f"{rag_positions['A']}"
)

print(
    f"RAG assigned to B: "
    f"{rag_positions['B']}"
)

print(
    f"Saved: {OUTPUT_FILE}"
)
