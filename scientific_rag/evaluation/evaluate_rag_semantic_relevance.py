#!/usr/bin/env python3

import json
import re
import time
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("rag_validation_retrievals.json")
OUTPUT_FILE = Path("rag_semantic_relevance.json")

GEMMA_URL = "http://localhost:11434/api/chat"
GEMMA_MODEL = "gemma4:12b"

MAX_PASSAGE_LENGTH = 4000
TEMPERATURE = 0.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions.")

    return data


# ============================================================
# GEMMA
# ============================================================

def call_gemma(prompt):

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
            "num_predict": 300,
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
            "Gemma returned empty content."
        )

    return content


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found:\n" + text
        )

    return json.loads(
        text[start:end + 1]
    )


# ============================================================
# SEMANTIC RELEVANCE JUDGE
# ============================================================

def judge_relevance(question, passage):

    passage = str(passage or "")[
        :MAX_PASSAGE_LENGTH
    ]

    prompt = f"""
You are evaluating a scientific literature retrieval system.

Judge whether the RETRIEVED PASSAGE is relevant to answering
the QUESTION.

Use ONLY the question and passage shown below.

Do not use outside knowledge.
Do not assume information that is absent from the passage.

Assign exactly one relevance grade:

2 = DIRECTLY RELEVANT
The passage contains information that directly answers the
question, or contains the specific scientific evidence,
method, observation, parameter, mechanism, or result needed
to answer it.

1 = PARTIALLY RELEVANT
The passage discusses the same scientific topic and provides
useful supporting information, but does not contain enough
information to directly answer the question.

0 = IRRELEVANT
The passage does not meaningfully help answer the question,
even if it contains some overlapping scientific words.

Important:
A passage does NOT need to come from the same paper that was
used to create the question. Judge only scientific relevance.

Return ONLY valid JSON:

{{
  "grade": 0,
  "reason": "brief explanation based only on the passage"
}}

QUESTION:
{question}

RETRIEVED PASSAGE:
{passage}
""".strip()

    response = call_gemma(prompt)

    result = extract_json(response)

    grade = result.get("grade")

    try:
        grade = int(grade)
    except Exception:
        raise ValueError(
            f"Invalid relevance grade: {grade}"
        )

    if grade not in (0, 1, 2):
        raise ValueError(
            f"Grade must be 0, 1, or 2. Got {grade}"
        )

    reason = str(
        result.get("reason", "")
    ).strip()

    return grade, reason


# ============================================================
# SAVE PROGRESS
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

def main():

    data = load_data()

    evaluated = []

    total_passages = sum(
        len(q.get("retrieved", []))
        for q in data
    )

    current = 0

    grade_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    errors = 0

    for q_index, q in enumerate(data, start=1):

        qid = q.get(
            "id",
            f"Q{q_index:02d}",
        )

        question = q.get(
            "question",
            "",
        )

        print(
            f"\n======================================"
        )

        print(
            f"[{q_index}/{len(data)}] "
            f"{qid}: {question}"
        )

        evaluated_question = dict(q)
        evaluated_retrieved = []

        for result in q.get(
            "retrieved",
            []
        ):

            current += 1

            rank = result.get("rank")

            source = result.get(
                "source",
                "Unknown",
            )

            print(
                f"  [{current}/{total_passages}] "
                f"Rank {rank}: {source}"
            )

            evaluated_result = dict(result)

            try:

                grade, reason = (
                    judge_relevance(
                        question,
                        result.get(
                            "text",
                            "",
                        ),
                    )
                )

                evaluated_result[
                    "semantic_relevance_grade"
                ] = grade

                evaluated_result[
                    "semantic_relevance_reason"
                ] = reason

                grade_counts[grade] += 1

                print(
                    f"      Grade: {grade}"
                )

                print(
                    f"      Reason: {reason}"
                )

            except Exception as exc:

                errors += 1

                evaluated_result[
                    "semantic_relevance_grade"
                ] = None

                evaluated_result[
                    "semantic_relevance_reason"
                ] = (
                    f"EVALUATION ERROR: {exc}"
                )

                print(
                    f"      ERROR: {exc}"
                )

            evaluated_retrieved.append(
                evaluated_result
            )

        evaluated_question[
            "retrieved"
        ] = evaluated_retrieved

        evaluated.append(
            evaluated_question
        )

        # Save after every question so progress is preserved.
        save_results(evaluated)

    print(
        "\n======================================"
    )

    print(
        "SEMANTIC RELEVANCE EVALUATION COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Questions: {len(evaluated)}"
    )

    print(
        f"Passages evaluated: {current}"
    )

    print(
        f"Grade 2 (direct): {grade_counts[2]}"
    )

    print(
        f"Grade 1 (partial): {grade_counts[1]}"
    )

    print(
        f"Grade 0 (irrelevant): {grade_counts[0]}"
    )

    print(
        f"Errors: {errors}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
