# DataPulse — RAG Knowledge Base (LLM)

> Project developed as part of the Master in AI Solution Architecture
> (ProfessionAI) — LLM module. Educational case study.

## Description
A Retrieval-Augmented Generation (RAG) backend that lets employees of
DataPulse S.p.A., a fictional fintech software company, query internal
documentation in natural language. Documents such as compliance policies,
operational guides and consent registers are chunked by tokens, indexed in a
vector store, retrieved with hybrid search, and used to generate an answer
with source citations.

## Stack
- Python 3
- sentence-transformers (`all-MiniLM-L6-v2` embeddings, local)
- ChromaDB (vector store)
- BM25 (`rank-bm25`) for lexical search, combined with semantic search
- LangChain (text splitters, Google Generative AI integration)
- Google Gemini (optional, for embeddings/generation)
- transformers (`google/flan-t5-base`), RAGAS (evaluation)

## Demonstrated skills
- Document ingestion with metadata (source, date, validity, type)
- Token-based chunking
- Hybrid retrieval (semantic + lexical) and filtering on document validity
- Answer generation with source attribution and a simple confidence score
- RAG evaluation with RAGAS

## How to run
The retrieval and generation pipeline runs locally. The Gemini key is optional.

```bash
pip install chromadb sentence-transformers transformers torch
python Datapulse_Rag.py
```

Notebook: install the dependencies listed in its first cells and run them in order.
To enable the optional Gemini check, set the environment variable before starting Jupyter:

```bash
export GEMINI_API_KEY="your-key-here"
jupyter notebook "DataPulse S.p.A.ipynb"
```

## Note
- Educational case study. All data, companies and contacts mentioned in the
  code are entirely fictional.
- `GEMINI_API_KEY` must be set as an environment variable. It is never stored in the code.
- Models are downloaded from Hugging Face on first run.
