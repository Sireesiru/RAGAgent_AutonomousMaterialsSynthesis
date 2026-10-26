#!/usr/bin/env python3

import json
import random
import time
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK_FILE = Path("rag_benchmark_50.json")
OUTPUT_FILE = Path("rag_ablation_results.json")

GEMMA_URL = "http://localhost:11434/api/chat"
RAG_URL = "http://localhost:8003/retrieve"

GEMMA_MODEL = "gemma4:12b"

TOP_K = 3
TEMPERATURE = 0.0
RANDOM_SEED = 42


# ============================================================
# LOAD BENCHMARK
# ============================================================

def load_benchmark():

    with open(
        BENCHMARK_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    print(
        f"Loaded {len(data)} benchmark questions."
    )

    return data


# ============================================================
# GEMMA CALL
# ============================================================

def call_gemma(messages):

    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 600,
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
# RAG RETRIEVAL
# ============================================================

def retrieve_literature(question):

    payload = {
        "question": question,
        "data_context": "",
        "top_k": TOP_K,
    }

    response = requests.post(
        RAG_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()

    # Handle either:
    # {"results": [...]}
    # or direct [...]
    if isinstance(result, dict):

        retrieved = result.get(
            "results",
            result.get(
                "retrieved",
                [],
            ),
        )

    elif isinstance(result, list):

        retrieved = result

    else:

        retrieved = []

    return retrieved


# ============================================================
# FORMAT RAG CONTEXT
# ============================================================

def build_rag_context(retrieved):

    blocks = []

    for i, item in enumerate(
        retrieved,
        start=1,
    ):

        source = item.get(
            "source",
            "Unknown source",
        )

        text = item.get(
            "text",
            "",
        )

        blocks.append(
            f"[Passage {i}]\n"
            f"Source: {source}\n"
            f"{text}"
        )

    return "\n\n".join(blocks)


# ============================================================
# COMMON INSTRUCTIONS
# ============================================================

COMMON_SYSTEM_PROMPT = """
You are answering a scientific question.

Give a concise scientific answer.

Do not invent information.

If the available information is insufficient to answer the
question confidently, explicitly state that the answer is
uncertain or cannot be determined from the available
information.

Do not discuss the evaluation procedure.
""".strip()


# ============================================================
# CONDITION A — GEMMA ONLY
# ============================================================

def answer_gemma_only(question):

    messages = [
        {
            "role": "system",
            "content": COMMON_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    return call_gemma(messages)


# ============================================================
# CONDITION B — RAG + GEMMA
# ============================================================

def answer_rag_gemma(
    question,
    retrieved,
):

    context = build_rag_context(
        retrieved
    )

    user_prompt = f"""
QUESTION:

{question}


RETRIEVED SCIENTIFIC LITERATURE:

{context}


Answer the QUESTION using the retrieved scientific
literature when it provides relevant evidence.

Do not introduce specific scientific claims that are not
supported by the retrieved passages.

If the passages do not contain enough information to answer
the question, explicitly say so.
""".strip()

    messages = [
        {
            "role": "system",
            "content": COMMON_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return call_gemma(messages)


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

def main():

    random.seed(RANDOM_SEED)

    benchmark = load_benchmark()

    results = []

    gemma_errors = 0
    rag_errors = 0
    retrieval_errors = 0

    for index, item in enumerate(
        benchmark,
        start=1,
    ):

        qid = item.get(
            "id",
            f"Q{index:02d}",
        )

        question = item["question"]

        print(
            "\n======================================"
        )

        print(
            f"[{index}/{len(benchmark)}] "
            f"{qid}"
        )

        print(
            f"Question: {question}"
        )

        record = {
            "id": qid,
            "question": question,

            # Keep gold answer only for later scoring.
            # It is NEVER supplied to either condition.
            "gold_answer":
                item.get(
                    "gold_answer",
                    "",
                ),

            "gold_source":
                item.get(
                    "gold_source",
                    "",
                ),

            "gold_chunk_id":
                item.get(
                    "gold_chunk_id",
                ),
        }

        # ----------------------------------------------------
        # CONDITION A
        # GEMMA ONLY
        # ----------------------------------------------------

        print(
            "\n  Condition A: Gemma only"
        )

        try:

            gemma_answer = (
                answer_gemma_only(
                    question
                )
            )

            record[
                "gemma_only_answer"
            ] = gemma_answer

            record[
                "gemma_only_error"
            ] = None

            print(
                "    Completed."
            )

        except Exception as exc:

            gemma_errors += 1

            record[
                "gemma_only_answer"
            ] = ""

            record[
                "gemma_only_error"
            ] = str(exc)

            print(
                f"    ERROR: {exc}"
            )

        # ----------------------------------------------------
        # RETRIEVE LITERATURE
        # ----------------------------------------------------

        print(
            "  Retrieving Top-3 literature..."
        )

        try:

            retrieved = (
                retrieve_literature(
                    question
                )
            )

            record[
                "retrieved"
            ] = retrieved

            record[
                "retrieval_error"
            ] = None

            print(
                f"    Retrieved "
                f"{len(retrieved)} passages."
            )

        except Exception as exc:

            retrieval_errors += 1

            retrieved = []

            record[
                "retrieved"
            ] = []

            record[
                "retrieval_error"
            ] = str(exc)

            print(
                f"    ERROR: {exc}"
            )

        # ----------------------------------------------------
        # CONDITION B
        # RAG + GEMMA
        # ----------------------------------------------------

        print(
            "  Condition B: RAG + Gemma"
        )

        try:

            rag_answer = (
                answer_rag_gemma(
                    question,
                    retrieved,
                )
            )

            record[
                "rag_gemma_answer"
            ] = rag_answer

            record[
                "rag_gemma_error"
            ] = None

            print(
                "    Completed."
            )

        except Exception as exc:

            rag_errors += 1

            record[
                "rag_gemma_answer"
            ] = ""

            record[
                "rag_gemma_error"
            ] = str(exc)

            print(
                f"    ERROR: {exc}"
            )

        results.append(record)

        # Save after every question.
        save_results(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "ABLATION GENERATION COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Questions: {len(results)}"
    )

    print(
        f"Gemma-only errors: {gemma_errors}"
    )

    print(
        f"Retrieval errors: {retrieval_errors}"
    )

    print(
        f"RAG+Gemma errors: {rag_errors}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
