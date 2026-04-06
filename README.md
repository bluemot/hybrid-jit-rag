# Hybrid JIT RAG Orchestrator

A lightweight, lazy-loading RAG (Retrieval-Augmented Generation) system that combines Qdrant local vector database with ripgrep fallback for on-the-fly document search.

## Features

- **Just-In-Time Ingestion**: Only indexes what you actually search for
- **Cache & Grep Hybrid**: Vector search with smart grep fallback
- **Semantic Query Expansion**: Auto-expands queries with synonyms and related terms
- **Background Processing**: Non-blocking JIT ingestion via threading
- **No Docker Required**: Uses Qdrant embedded mode and local file storage

## Quick Start

```bash
pip install -r requirements.txt

# Upload documents
python -c "from jit_rag_skill import jit_rag_upload; jit_rag_upload('/path/to/doc.txt')"

# Query
python -c "from jit_rag_skill import jit_rag_query; print(jit_rag_query('your question'))"
```

## Architecture

```
User Query → Query Expansion → Cache Check → (Miss) → Smart Grep → JIT Index → Answer
                          ↓
                    (Hit) → Return Cached
```

## Files

- `jit_rag_orchestrator.py` - Core RAG engine
- `jit_rag_skill.py` - OpenClaw AgentSkill wrapper
- `JIT_SKILL.md` - Documentation

## License

MIT
