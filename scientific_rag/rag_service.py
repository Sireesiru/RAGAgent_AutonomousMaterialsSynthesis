import pickle
from pathlib import Path

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

KB_DIR = Path(
    "/mnt/cnms_lab_assist/sirisha/scientific_rag/scientific_kb_sfr"
)

INDEX_PATH = KB_DIR / "scientific.index"
METADATA_PATH = KB_DIR / "scientific.pkl"

SFR_BASE_URL = "http://10.123.23.27:8001/v1"
SFR_MODEL = "sfr-embedding-mistral"

TOP_K = 3
FETCH_K = 1000
MMR_LAMBDA = 0.7


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="BRaVE Scientific Literature RAG",
    description=(
        "Literature retrieval service for the BRaVE DataFed "
        "scientific agent."
    ),
    version="1.0.0",
)


# ============================================================
# LOAD KNOWLEDGE BASE ONCE
# ============================================================

print("Loading scientific RAG knowledge base...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )

index = faiss.read_index(str(INDEX_PATH))

with open(METADATA_PATH, "rb") as f:
    chunks = pickle.load(f)

if index.ntotal != len(chunks):
    raise RuntimeError(
        f"Index/chunk mismatch: "
        f"{index.ntotal} vectors vs {len(chunks)} chunks"
    )

print(f"Loaded {index.ntotal} literature chunks.")


# ============================================================
# REQUEST SCHEMA
# ============================================================

class RAGRequest(BaseModel):
    question: str
    data_context: str = ""
    top_k: int = TOP_K


# ============================================================
# EMBEDDING
# ============================================================

def embed_text(text: str) -> np.ndarray:

    url = f"{SFR_BASE_URL}/embeddings"

    payload = {
        "model": SFR_MODEL,
        "input": [text],
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    vector = np.asarray(
        data["data"][0]["embedding"],
        dtype=np.float32,
    )

    return vector


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b) / (a_norm * b_norm)
    )


# ============================================================
# MMR
# ============================================================

def mmr_select(
    query_vector,
    candidate_vectors,
    candidate_indices,
    top_k,
    lambda_mult,
):

    if len(candidate_indices) == 0:
        return []

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    candidate_vectors = np.asarray(
        candidate_vectors,
        dtype=np.float32,
    )

    selected_positions = []
    remaining_positions = list(
        range(len(candidate_indices))
    )

    # First result = most similar to query
    query_scores = [
        cosine_similarity(query_vector, vec)
        for vec in candidate_vectors
    ]

    first_position = int(
        np.argmax(query_scores)
    )

    selected_positions.append(first_position)
    remaining_positions.remove(first_position)

    while (
        len(selected_positions) < top_k
        and remaining_positions
    ):

        best_position = None
        best_score = -float("inf")

        for position in remaining_positions:

            relevance = cosine_similarity(
                query_vector,
                candidate_vectors[position],
            )

            redundancy = max(
                cosine_similarity(
                    candidate_vectors[position],
                    candidate_vectors[selected],
                )
                for selected in selected_positions
            )

            score = (
                lambda_mult * relevance
                - (1.0 - lambda_mult) * redundancy
            )

            if score > best_score:
                best_score = score
                best_position = position

        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    return [
        candidate_indices[position]
        for position in selected_positions
    ]


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_literature(
    question: str,
    data_context: str,
    top_k: int,
):

    # Dataset context is deliberately included.
    # This makes retrieval specific to the experimental situation
    # discovered by the DataFed agent.

    retrieval_query = f"""
Experimental dataset context:
{data_context}

Scientific question:
{question}
""".strip()

    query_vector = embed_text(
        retrieval_query
    )

    query_vector_2d = query_vector.reshape(
        1, -1
    )

    fetch_k = min(
        FETCH_K,
        index.ntotal,
    )

    distances, indices = index.search(
        query_vector_2d,
        fetch_k,
    )

    candidate_indices = [
        int(i)
        for i in indices[0]
        if i >= 0
    ]

    if not candidate_indices:
        return []

    # IndexFlatL2 stores the original vectors.
    candidate_vectors = np.vstack(
        [
            index.reconstruct(i)
            for i in candidate_indices
        ]
    ).astype(np.float32)

    selected_indices = mmr_select(
        query_vector=query_vector,
        candidate_vectors=candidate_vectors,
        candidate_indices=candidate_indices,
        top_k=min(top_k, len(candidate_indices)),
        lambda_mult=MMR_LAMBDA,
    )

    results = []

    for rank, chunk_index in enumerate(
        selected_indices,
        start=1,
    ):

        item = chunks[chunk_index]

        vector = index.reconstruct(
            chunk_index
        ).astype(np.float32)

        cosine = cosine_similarity(
            query_vector,
            vector,
        )

        results.append(
            {
                "rank": rank,
                "source": item.get(
                    "source",
                    item.get(
                        "filename",
                        "Unknown source",
                    ),
                ),
                "chunk_id": item.get(
                    "chunk_id",
                    chunk_index,
                    ),
                "text": item.get(
                    "text",
                    "",
                ),
                "cosine_similarity": cosine,
            }
        )

    return results


# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "chunks": index.ntotal,
        "embedding_model": SFR_MODEL,
    }


@app.post("/retrieve")
def retrieve(request: RAGRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    top_k = max(
        1,
        min(request.top_k, 10),
    )

    try:

        results = retrieve_literature(
            question=request.question,
            data_context=request.data_context,
            top_k=top_k,
        )

        return {
            "question": request.question,
            "num_results": len(results),
            "results": results,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
