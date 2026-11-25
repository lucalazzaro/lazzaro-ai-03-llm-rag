# DataPulse — RAG Knowledge Base (LLM)

> Progetto sviluppato nell'ambito del Master in AI Solution Architecture
> (ProfessionAI) — modulo LLM. Caso di studio a fini didattici.

## Descrizione
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

## Skill dimostrate
- Document ingestion with metadata (source, date, validity, type)
- Token-based chunking
- Hybrid retrieval (semantic + lexical) and filtering on document validity
- Answer generation with source attribution and a simple confidence score
- RAG evaluation with RAGAS

## Come eseguire
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
- Caso di studio a fini didattici. I dati, le aziende e i contatti citati nel
  codice sono interamente fittizi.
- `GEMINI_API_KEY` must be set as an environment variable. It is never stored in the code.
- Models are downloaded from Hugging Face on first run.
