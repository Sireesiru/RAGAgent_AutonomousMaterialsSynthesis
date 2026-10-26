# Scientific Literature RAG

This directory contains the literature retrieval and evaluation component used by the Agentic Scientific Data Assistant.

## Architecture

Scientific Data Agent (Microflows)
        |
        | MCP: search_scientific_literature
        v
Scientific RAG Service (Jawad)
        |
        +-- SFR embedding model
        +-- FAISS vector index
        +-- MMR reranking
        +-- Scientific literature corpus

The RAG service provides literature evidence to the scientific agent. Experimental data access and scientific analysis are handled separately by the DataFed/MCP agent.

## Environment

Create a Python environment and install the frozen dependencies:

python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements_scientific_rag.txt

## Knowledge Base Construction

`build_scientific_kb_sfr.py` builds the scientific literature knowledge base using SFR embeddings and FAISS indexing.

The literature corpus, generated FAISS indexes, serialized knowledge-base files, and PDFs are not stored in this repository.

## RAG Service

`rag_service.py` provides the retrieval service used by the scientific agent.

The production system uses:

- SFR scientific text embeddings
- FAISS vector search
- Maximum Marginal Relevance (MMR) reranking
- Top-K literature retrieval
- FastAPI service interface

## Evaluation

The `evaluation/` directory contains scripts used for formal validation of the retrieval and literature-grounded question-answering system.

The evaluation includes:

- 50-question retrieval benchmark
- source-level retrieval evaluation
- chunk-level retrieval evaluation
- semantic relevance evaluation
- Gemma-only vs RAG+Gemma ablation
- blinded answer scoring
- bootstrap confidence intervals
- paired statistical testing
- manual audit

## Results

The `results/` directory contains numerical outputs from the validation experiments used to evaluate the scientific RAG system.

Large generated artifacts and source literature are intentionally excluded from version control.
