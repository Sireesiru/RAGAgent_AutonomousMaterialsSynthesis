#!/usr/bin/env python3

import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

KB_DIR = Path("/mnt/cnms_lab_assist/sirisha/scientific_rag/scientific_kb_sfr")
METADATA_PATH = KB_DIR / "scientific.pkl"
OUTPUT_FILE = Path("rag_benchmark_50.json")
CORPUS_SUMMARY_FILE = Path(
    "rag_benchmark_corpus_summary.json"
)

GEMMA_URL = "http://localhost:11434/api/chat"
GEMMA_MODEL = "gemma4:12b"

N_QUESTIONS = 50
RANDOM_SEED = 42

# A chunk only needs enough scientific content to form a question.
MIN_TEXT_LENGTH = 400

# Only this much text is sent to Gemma.
MAX_TEXT_LENGTH = 2500

# Prefer broad source diversity.
MAX_QUESTIONS_PER_SOURCE = 1

# Keep several possible chunks from each paper.
CANDIDATES_PER_SOURCE = 5


# ============================================================
# LOAD CORPUS
# ============================================================

def load_chunks():

    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


# ============================================================
# CLEANING
# ============================================================

def clean_text(text):

    text = str(text or "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_source(source):

    return str(source or "").strip()


# ============================================================
# FILTER OBVIOUSLY POOR CHUNKS
# ============================================================

def looks_like_bad_chunk(text):
    text = clean_text(text)
    if len(text) < MIN_TEXT_LENGTH:
        return True
    lower = text.lower()
    # Reject chunks that are overwhelmingly references.
    if lower.count("doi.org/") >= 5:
        return True
    # Reject obvious reference-list beginnings.
    if (
        lower.startswith("references")
        or lower.startswith("bibliography")
    ):
        return True
    return False

# ============================================================
# GROUP CHUNKS BY SOURCE PAPER
# ============================================================

def group_chunks_by_source(chunks):
    grouped = defaultdict(list)
    for item in chunks:
        source = normalize_source(
            item.get("source")
        )
        text = clean_text(
            item.get("text")
        )
        if not source or not text:
            continue
        if looks_like_bad_chunk(text):
            continue
        grouped[source].append(
            {
                "source": source,
                "chunk_id": item.get("chunk_id"),
                "text": text,
                "embedding_model":
                    item.get("embedding_model"),
            }
        )

    return grouped

# ============================================================
# SIMPLE PASSAGE QUALITY SCORE
# ============================================================

def passage_quality(text):

    """
    Generic quality score.

    IMPORTANT:
    No AFM-specific or microscopy-specific keywords are used
    here because we want the benchmark to reflect the actual
    diversity of the 166-paper corpus.
    """

    text = clean_text(text)

    score = 0

    # Prefer reasonably substantive passages.
    score += min(len(text) / 500.0, 5)

    # Scientific prose often contains sentences.
    sentence_count = (
        text.count(".")
        + text.count("?")
        + text.count(";")
    )

    score += min(sentence_count / 5.0, 4)

    # Penalize chunks that look reference-heavy.
    lower = text.lower()

    score -= lower.count("doi.org/") * 2

    return score


# ============================================================
# CHOOSE SEVERAL CANDIDATES PER PAPER
# ============================================================

def choose_candidates_by_source(grouped):

    source_candidates = {}

    for source, items in grouped.items():

        ranked = sorted(
            items,
            key=lambda x: passage_quality(
                x["text"]
            ),
            reverse=True,
        )

        source_candidates[source] = (
            ranked[:CANDIDATES_PER_SOURCE]
        )

    return source_candidates


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

        # Gemma was spending tokens in its thinking field.
        # We do not need that for benchmark generation.
        "think": False,

        "options": {
            "temperature": 0.1,
            "num_predict": 500,
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
            "Gemma returned empty content. "
            f"done_reason="
            f"{result.get('done_reason')}"
        )

    return content


# ============================================================
# EXTRACT JSON
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
            "No JSON object found in model response:\n"
            + text
        )

    return json.loads(
        text[start:end + 1]
    )


# ============================================================
# GENERATE QUESTION FROM ONE PASSAGE
# ============================================================

def generate_question(candidate):

    text = candidate["text"][
        :MAX_TEXT_LENGTH
    ]

    prompt = f"""
Create ONE scientific question using ONLY the passage below.

This question will test whether a scientific literature
retrieval system can recover evidence relevant to a
scientific data-analysis agent.

Requirements:

1. The answer must be clearly supported by the passage.

2. Ask about the scientific content, such as:
   experimental observations,
   measurement,
   instrumentation,
   microscopy or imaging,
   biological structure or function,
   morphology,
   sample preparation,
   artifacts,
   quantitative analysis,
   computational analysis,
   data interpretation,
   materials characterization,
   or another scientific topic actually present in the passage.

3. Do NOT mention:
   authors,
   paper title,
   journal,
   publication year,
   DOI,
   citations.

4. Do NOT use knowledge absent from the passage.

5. Avoid yes/no questions.

6. Avoid bibliographic trivia.

7. Create a short category describing the scientific topic.

8. Provide a concise gold answer supported directly by the
   passage.

9. If the passage does not contain enough scientific
   information for a meaningful question, return
   "suitable": false.

Return ONLY valid JSON:

{{
  "suitable": true,
  "category": "short scientific category",
  "question": "one scientific question",
  "gold_answer": "short passage-supported answer",
  "support_reason": "brief explanation of why the passage supports the answer"
}}

PASSAGE:

{text}
""".strip()

    response_text = call_gemma(prompt)

    return extract_json(
        response_text
    )


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question):

    question = str(question).lower()

    question = re.sub(
        r"[^a-z0-9\s]",
        " ",
        question,
    )

    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    chunks = load_chunks()

    grouped = group_chunks_by_source(
        chunks
    )

    print(
        "Usable papers after filtering:",
        len(grouped),
    )

    if len(grouped) == 0:

        raise RuntimeError(
            "No usable papers remained after filtering. "
            "Check chunk-length/filter settings."
        )

    # --------------------------------------------------------
    # BUILD CANDIDATES PER PAPER
    # --------------------------------------------------------

    source_candidates = (
        choose_candidates_by_source(
            grouped
        )
    )

    sources = list(
        source_candidates.keys()
    )

    # Randomize paper order with fixed seed.
    random.shuffle(sources)

    total_candidates = sum(
        len(v)
        for v in source_candidates.values()
    )

    print(
        "Candidate source passages:",
        total_candidates,
    )

    print(
        "Candidate papers:",
        len(sources),
    )

    # --------------------------------------------------------
    # GENERATE QUESTIONS
    # --------------------------------------------------------

    benchmark = []

    source_counts = Counter()
    category_counts = Counter()

    existing_questions = set()

    attempted_passages = 0
    attempted_sources = set()

    # Multiple rounds allow trying another chunk from a paper
    # if its first candidate is unsuitable.
    for candidate_number in range(
        CANDIDATES_PER_SOURCE
    ):

        if len(benchmark) >= N_QUESTIONS:
            break

        for source in sources:

            if len(benchmark) >= N_QUESTIONS:
                break

            if (
                source_counts[source]
                >= MAX_QUESTIONS_PER_SOURCE
            ):
                continue

            candidates = (
                source_candidates[source]
            )

            if candidate_number >= len(
                candidates
            ):
                continue

            candidate = candidates[
                candidate_number
            ]

            attempted_passages += 1
            attempted_sources.add(source)

            print(
                f"\n"
                f"[Need question "
                f"{len(benchmark)+1}/{N_QUESTIONS}]"
            )

            print(
                f"Trying paper: {source}"
            )

            print(
                "Chunk:",
                candidate["chunk_id"],
            )

            try:

                generated = (
                    generate_question(
                        candidate
                    )
                )

            except Exception as exc:

                print(
                    "  Generation failed:",
                    exc,
                )

                continue

            if not generated.get(
                "suitable",
                False,
            ):

                print(
                    "  Passage marked unsuitable."
                )

                continue

            question = clean_text(
                generated.get(
                    "question",
                    "",
                )
            )

            gold_answer = clean_text(
                generated.get(
                    "gold_answer",
                    "",
                )
            )

            category = clean_text(
                generated.get(
                    "category",
                    "Other",
                )
            )

            support_reason = clean_text(
                generated.get(
                    "support_reason",
                    "",
                )
            )

            if (
                not question
                or not gold_answer
            ):

                print(
                    "  Missing question or "
                    "gold answer."
                )

                continue

            normalized = normalize_question(
                question
            )

            if normalized in existing_questions:

                print(
                    "  Duplicate question skipped."
                )

                continue

            # --------------------------------------------
            # ACCEPT
            # --------------------------------------------

            qid = (
                f"Q{len(benchmark)+1:02d}"
            )

            record = {

                "id": qid,

                "category":
                    category,

                "question":
                    question,

                # GOLD DATA — NEVER SENT TO RETRIEVER
                "gold_source":
                    candidate["source"],

                "gold_chunk_id":
                    candidate["chunk_id"],

                "gold_text":
                    candidate["text"],

                "gold_answer":
                    gold_answer,

                "support_reason":
                    support_reason,

                # Literature-only benchmark.
                "data_context": "",
            }

            benchmark.append(
                record
            )

            existing_questions.add(
                normalized
            )

            source_counts[source] += 1

            category_counts[
                category
            ] += 1

            print(
                "  ACCEPTED:",
                question,
            )

            print(
                "  Category:",
                category,
            )

            print(
                "  Gold answer:",
                gold_answer,
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            benchmark,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {

        "total_corpus_chunks":
            len(chunks),

        "usable_sources":
            len(grouped),

        "candidate_passages":
            total_candidates,

        "attempted_passages":
            attempted_passages,

        "attempted_sources":
            len(attempted_sources),

        "benchmark_questions":
            len(benchmark),

        "unique_gold_sources":
            len(source_counts),

        "random_seed":
            RANDOM_SEED,

        "max_questions_per_source":
            MAX_QUESTIONS_PER_SOURCE,

        "category_counts":
            dict(category_counts),

        "source_counts":
            dict(source_counts),
    }

    with open(
        CORPUS_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        "\n"
        "======================================"
    )

    print(
        "BENCHMARK BUILD COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "Questions:",
        len(benchmark),
    )

    print(
        "Unique source papers:",
        len(source_counts),
    )

    print(
        "Categories:",
        len(category_counts),
    )

    print(
        "Attempted passages:",
        attempted_passages,
    )

    print(
        "Saved:",
        OUTPUT_FILE,
    )

    print(
        "Saved:",
        CORPUS_SUMMARY_FILE,
    )

    print(
        "\nCATEGORY DISTRIBUTION"
    )

    for category, count in (
        category_counts.most_common()
    ):

        print(
            f"{count:3d}  {category}"
        )

    if len(benchmark) < N_QUESTIONS:

        print(
            "\nWARNING:"
        )

        print(
            f"Only {len(benchmark)} of "
            f"{N_QUESTIONS} questions "
            "were generated."
        )

        print(
            "Do NOT change the benchmark yet. "
            "Inspect the failures first."
        )


if __name__ == "__main__":
    main()