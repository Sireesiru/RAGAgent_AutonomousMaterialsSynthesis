#!/usr/bin/env python3
"""
Build collaborator-compatible FAISS .index + list-of-dicts .pkl
using SFR-Embedding-Mistral through an OpenAI-compatible vLLM endpoint.

Preserves the existing RAG pipeline:
- PDF extraction
- text cleaning
- cl100k chunking: 300 tokens, 30-word overlap
- FAISS IndexFlatL2
- metadata structure
- recursive PDF discovery
- per-PDF resumable cache

Embedding backend:
    model: sfr-embedding-mistral
    endpoint: http://10.123.23.27:8001/v1
"""

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import fitz
import numpy as np
import pycountry
import requests
import tiktoken


# ============================================================
# CHUNKING
# ============================================================

TOKENS_PER_CHUNK = 300
WORDS_PER_CHUNK_OVERLAP = 30
ENC = tiktoken.get_encoding("cl100k_base")


# ============================================================
# PDF PROCESSING
# ============================================================

def extract_text_from_pdf(path):
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def remove_junk_sections(text, section_markers=None):
    if section_markers is None:
        section_markers = [
            "references",
            "acknowledgment",
            "acknowledgement",
            "acknowledgments",
            "acknowledgements",
            "author information",
            "author contribution",
            "author contributions",
            "associated content",
        ]

    pattern = re.compile(
        rf"^\s*[\.\-\u25A0\u2022]*\s*({'|'.join(section_markers)})",
        re.IGNORECASE | re.MULTILINE,
    )

    match = pattern.search(text)

    return text[:match.start()] if match else text


def remove_junk_lines(text, junk_patterns=None):

    if junk_patterns is None:
        junk_patterns = [
            "doi:",
            "et al.",
            "https://",
            "http://",
            ".org",
            ".com",
            "conflict of interest",
            "bio:",
            "funding",
            "journal",
            "citation",
            "cc-by",
            "preprint",
            "arxiv",
            "license",
            "open access",
            "submitted to",
            "peer review",
            "double-blind",
            "published",
            "copyright",
            "all rights reserved",
            "correspondence should be addressed to",
            "authors contributed equally",
            "this manuscript has been authored by",
            "contract no",
            "supporting information",
            "read online",
            "received:",
            "revised:",
            "accepted:",
            "cite this:",
            "©",
            "download",
            "public access plan",
            "department of",
            "university",
            "national laboratory",
            "laboratory",
            "government",
        ]

    countries = [country.name.lower() for country in pycountry.countries]

    cleaned = []

    for line in text.splitlines():

        lower = line.lower()

        if any(pat in lower for pat in junk_patterns):
            continue

        if "*" in line or "†" in line or "∇" in line:
            continue

        if any(
            re.search(rf"\b{re.escape(c)}\b", lower)
            for c in countries
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================================
# CHUNKING
# ============================================================

def chunk_text2(
    text,
    max_tokens=TOKENS_PER_CHUNK,
    tokenizer=ENC,
    overlap=WORDS_PER_CHUNK_OVERLAP,
):

    words = text.split()

    chunks = []

    current = []

    i = 0

    while i < len(words):

        word = words[i]

        current.append(word)

        test_chunk = " ".join(current)

        if len(tokenizer.encode(test_chunk)) > max_tokens:

            current.pop()

            if current:
                chunks.append(" ".join(current))

            overlap_start = max(len(current) - overlap, 0)

            current = current[overlap_start:] + [word]

        i += 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ============================================================
# SFR EMBEDDING
# OpenAI-compatible /v1/embeddings endpoint
# ============================================================

def embed(session, base_url, model, inputs, attempts=5):

    url = base_url.rstrip("/") + "/embeddings"

    for attempt in range(attempts):

        try:

            response = session.post(
                url,
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": inputs,
                },
                timeout=300,
            )

            response.raise_for_status()

            payload = response.json()

            records = payload["data"]

            # OpenAI-compatible responses include an index field.
            # Sort explicitly so vectors always correspond to input order.
            records = sorted(
                records,
                key=lambda item: item.get("index", 0),
            )

            data = [
                item["embedding"]
                for item in records
            ]

            if len(data) != len(inputs):
                raise RuntimeError(
                    "Embedding response count mismatch"
                )

            return np.asarray(
                data,
                dtype=np.float32,
            )

        except (
            requests.RequestException,
            KeyError,
            ValueError,
        ) as exc:

            if attempt == attempts - 1:

                raise RuntimeError(
                    f"SFR embedding failed: {exc}"
                ) from exc

            wait = min(2 ** attempt, 20)

            print(
                f"Embedding request failed. "
                f"Retrying in {wait}s...",
                file=sys.stderr,
                flush=True,
            )

            time.sleep(wait)


# ============================================================
# PDF SELECTION
# ============================================================

def select_pdfs(root, manifest):

    pdfs = (
        sorted(root.rglob("*.pdf"))
        + sorted(root.rglob("*.PDF"))
    )

    pdfs = sorted(set(pdfs))

    if not manifest:
        return pdfs

    with manifest.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:

        rows = list(csv.DictReader(f))

    yes = [
        r
        for r in rows
        if r.get(
            "include_in_rag",
            "",
        ).strip().upper()
        == "YES"
    ]

    if not yes:
        raise RuntimeError(
            "Manifest has no include_in_rag=YES rows. "
            "Check CSV columns."
        )

    matched = []

    for pdf in pdfs:

        for row in yes:

            fields = [
                str(v).replace("\\", "/")
                for v in row.values()
                if v
            ]

            if any(
                v.endswith(
                    "/" + pdf.parent.name + "/" + pdf.name
                )
                or v.endswith("/" + pdf.name)
                or v == pdf.name
                for v in fields
            ):

                matched.append(pdf)

                break

    if not matched:
        raise RuntimeError(
            "Manifest selected zero PDFs; "
            "check file paths/column format."
        )

    return matched


# ============================================================
# MAIN
# ============================================================

def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--papers",
        default="/mnt/cnms_lab_assist/sirisha/scientific_rag/papers",
    )

    p.add_argument(
        "--manifest",
        default="/mnt/cnms_lab_assist/sirisha/scientific_rag/master_rag_manifest.csv",
    )

    p.add_argument(
        "--output",
        default="/mnt/cnms_lab_assist/sirisha/scientific_rag/scientific_kb_sfr",
    )

    p.add_argument(
        "--base-url",
        default="http://10.123.23.27:8001/v1",
    )

    p.add_argument(
        "--model",
        default="sfr-embedding-mistral",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Index only N PDFs for smoke test",
    )

    args = p.parse_args()

    root = Path(args.papers)

    output = Path(args.output)

    if not root.is_dir():

        p.error(
            f"PDF folder not found: {root}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = Path(args.manifest)

    if not manifest.exists():

        print(
            "WARNING: manifest not found; "
            "indexing all PDFs recursively.",
            file=sys.stderr,
        )

        manifest = None


    # ========================================================
    # CONNECT TO SFR SERVER
    # ========================================================

    session = requests.Session()

    try:

        response = session.get(
            args.base_url.rstrip("/") + "/models",
            timeout=10,
        )

        response.raise_for_status()

        models = [
            item["id"]
            for item in response.json()["data"]
        ]

        if args.model not in models:

            raise RuntimeError(
                f"{args.model} not available. "
                f"Available models: {models}"
            )

    except requests.RequestException as exc:

        raise SystemExit(
            f"SFR endpoint unreachable at "
            f"{args.base_url}: {exc}"
        )


    # ========================================================
    # TEST EMBEDDING + DETERMINE DIMENSION
    # ========================================================

    test = embed(
        session,
        args.base_url,
        args.model,
        ["scientific microscopy"],
    )[0]

    dim = len(test)

    print(
        f"SFR endpoint OK | "
        f"model={args.model} | "
        f"dimension={dim}",
        flush=True,
    )


    # ========================================================
    # FIND PDFs
    # ========================================================

    pdfs = select_pdfs(
        root,
        manifest,
    )


    # ========================================================
    # EXACT PDF DEDUPLICATION
    # ========================================================

    unique = []

    hashes = set()

    for pdf in pdfs:

        digest = hashlib.sha256(
            pdf.read_bytes()
        ).hexdigest()

        if digest not in hashes:

            unique.append(
                (pdf, digest)
            )

            hashes.add(digest)


    if args.limit:

        unique = unique[
            :args.limit
        ]


    print(
        f"PDFs selected: {len(unique)}",
        flush=True,
    )


    # ========================================================
    # CREATE FAISS INDEX
    # ========================================================

    index = faiss.IndexFlatL2(dim)

    metadata = []

    cache = output / "cache"

    cache.mkdir(
        exist_ok=True
    )

    errors = []


    # ========================================================
    # PROCESS EACH PAPER
    # ========================================================

    for count, (
        pdf,
        digest,
    ) in enumerate(
        unique,
        1,
    ):

        safe_model = (
            args.model
            .replace(":", "_")
            .replace("/", "_")
        )

        cache_file = (
            cache
            / (
                digest
                + "_"
                + safe_model
                + ".pkl"
            )
        )

        try:

            # ------------------------------------------------
            # USE CACHE IF AVAILABLE
            # ------------------------------------------------

            if cache_file.exists():

                with cache_file.open(
                    "rb"
                ) as f:

                    saved = pickle.load(f)

                vectors = saved[
                    "vectors"
                ]

                entries = saved[
                    "metadata"
                ]

                if vectors.shape[1] != dim:

                    raise RuntimeError(
                        "Cached vector dimension mismatch"
                    )


            # ------------------------------------------------
            # OTHERWISE PROCESS PDF
            # ------------------------------------------------

            else:

                text = extract_text_from_pdf(
                    pdf
                )

                cleaned = remove_junk_lines(
                    remove_junk_sections(
                        text
                    )
                )

                chunks = [
                    c
                    for c in chunk_text2(
                        cleaned
                    )
                    if c.strip()
                ]

                if not chunks:

                    print(
                        f"[{count}/{len(unique)}] "
                        f"SKIP empty: {pdf.name}",
                        flush=True,
                    )

                    continue


                # --------------------------------------------
                # EMBED IN BATCHES
                # --------------------------------------------

                vector_batches = []

                for i in range(
                    0,
                    len(chunks),
                    args.batch_size,
                ):

                    batch = chunks[
                        i:i + args.batch_size
                    ]

                    vectors_batch = embed(
                        session,
                        args.base_url,
                        args.model,
                        batch,
                    )

                    vector_batches.append(
                        vectors_batch
                    )


                vectors = np.concatenate(
                    vector_batches,
                    axis=0,
                )


                entries = [
                    {
                        "source": pdf.name,
                        "chunk_id": i,
                        "text": chunk,
                        "embedding_model": args.model,
                    }
                    for i, chunk
                    in enumerate(chunks)
                ]


                # --------------------------------------------
                # SAVE PER-PDF CACHE
                # --------------------------------------------

                temp = cache_file.with_suffix(
                    ".tmp"
                )

                with temp.open(
                    "wb"
                ) as f:

                    pickle.dump(
                        {
                            "vectors": vectors,
                            "metadata": entries,
                        },
                        f,
                    )

                temp.replace(
                    cache_file
                )


            # ------------------------------------------------
            # ADD TO FAISS
            # ------------------------------------------------

            index.add(
                np.ascontiguousarray(
                    vectors,
                    dtype=np.float32,
                )
            )

            metadata.extend(
                entries
            )

            print(
                f"[{count}/{len(unique)}] "
                f"{pdf.name}: "
                f"{len(entries)} chunks | "
                f"total {index.ntotal}",
                flush=True,
            )


        except Exception as exc:

            errors.append(
                {
                    "file": str(pdf),
                    "error": str(exc),
                }
            )

            print(
                f"[{count}/{len(unique)}] "
                f"ERROR {pdf.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )


    # ========================================================
    # SAVE FINAL KB
    # ========================================================

    if index.ntotal == 0:

        raise SystemExit(
            "No chunks indexed. "
            "Check PDFs and errors."
        )


    stem = (
        "scientific_test"
        if args.limit
        else "scientific"
    )


    index_path = (
        output
        / (stem + ".index")
    )

    meta_path = (
        output
        / (stem + ".pkl")
    )


    faiss.write_index(
        index,
        str(index_path),
    )


    with meta_path.open(
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f,
        )


    # ========================================================
    # BUILD INFORMATION
    # ========================================================

    build_info = {

        "embedding_model":
            args.model,

        "embedding_endpoint":
            args.base_url,

        "embedding_dimension":
            dim,

        "pdfs_selected":
            len(unique),

        "chunks":
            index.ntotal,

        "errors":
            errors,

        "index":
            str(index_path),

        "metadata":
            str(meta_path),
    }


    (
        output
        / (
            stem
            + "_build.json"
        )
    ).write_text(
        json.dumps(
            build_info,
            indent=2,
        )
    )


    print(
        f"\nDONE | "
        f"chunks={index.ntotal} | "
        f"errors={len(errors)}"
        f"\n{index_path}"
        f"\n{meta_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()