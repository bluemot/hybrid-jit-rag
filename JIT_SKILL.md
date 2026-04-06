# JIT RAG Skill

Hybrid Just-In-Time RAG Orchestrator for OpenClaw.

## Overview

This skill implements a **lazy-loading RAG system** that:
1. **Caches** search results in LanceDB (embedded vector DB)
2. **Falls back** to ripgrep (`rg`) on cache miss
3. **JIT ingests** new findings in background threads
4. **Learns** from your queries over time

## How It Works

```
Upload File → Save to Workspace
                 ↓
User Query → Check Cache (LanceDB) 
                 ↓
    Cache Hit? → Yes → Return immediately
        ↓ No
    Smart Grep (rg)
        ↓
    Background JIT Ingest
        ↓
    Return Context
```

## Tools

### `jit_rag_upload(file_path: str) -> str`

Upload a document to the RAG workspace.

**Example:**
```python
jit_rag_upload("/path/to/notes.md")
jit_rag_upload("/tmp/specifications.txt")
```

### `jit_rag_query(query: str) -> str`

Ask a question about your uploaded documents.

**Example:**
```python
jit_rag_query("What are the UART register addresses?")
jit_rag_query("Explain the interrupt handling flow")
```

### `jit_rag_status() -> str`

Check system status and statistics.

**Example:**
```python
jit_rag_status()
```

### `jit_rag_clear() -> str`

Clear all data and start fresh.

**Example:**
```python
jit_rag_clear()
```

## Requirements

- `lancedb` - Embedded vector database
- `ripgrep` - System package (rg command)
- `httpx` - For GPU server embeddings
- `numpy` - Vector operations

## Configuration

Edit `jit_rag_orchestrator.py` Config class:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_url` | `http://192.168.122.1:11434` | Your GPU server |
| `embedding_model` | `bge-m3` | Embedding model |
| `cache_threshold` | `0.85` | Cache hit threshold |
| `top_k` | `5` | Number of results |

## Workflow

1. **Upload** your documents
2. **Query** naturally
3. System automatically:
   - Checks cache first
   - Falls back to grep if needed
   - Caches new findings for next time

## File Support

- `.md` - Markdown files (with YAML frontmatter support)
- `.txt` - Plain text files
- `.markdown` - Alternative markdown extension
