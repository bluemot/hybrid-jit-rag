"""
OpenClaw AgentSkill for Hybrid JIT RAG Orchestrator.
Allows direct file upload and RAG-based Q&A.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from jit_rag_orchestrator import HybridJITRAGOrchestrator, Config, CONFIG

# Global orchestrator instance
_orchestrator: Optional[HybridJITRAGOrchestrator] = None
_workspace_dir: str = os.path.join(os.path.dirname(__file__), "jit_workspace")


def _get_orchestrator() -> HybridJITRAGOrchestrator:
    """Get or create JIT RAG orchestrator with persistent workspace."""
    global _orchestrator
    if _orchestrator is None:
        os.makedirs(_workspace_dir, exist_ok=True)
        print(f"[JIT RAG] Using workspace: {_workspace_dir}")
        _orchestrator = HybridJITRAGOrchestrator(_workspace_dir)
    return _orchestrator


def jit_rag_upload(file_path: str) -> str:
    """
    Upload a document to the JIT RAG workspace for later querying.
    
    Supports: .md, .txt, .pdf files
    PDF files will be automatically converted to text.
    
    Args:
        file_path: Absolute path to the uploaded file
    
    Returns:
        Confirmation with file details and workspace stats.
    """
    orchestrator = _get_orchestrator()
    
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(filename)[1].lower()
    dest_path = os.path.join(_workspace_dir, filename)
    
    try:
        # Handle PDF conversion
        if file_ext == '.pdf':
            print(f"[JIT RAG] Converting PDF to text: {filename}")
            try:
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(file_path)
                text_content = result.text_content
                
                # Save as .txt instead of .pdf
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(_workspace_dir, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                file_size = os.path.getsize(txt_path)
                filename = txt_filename  # Update filename for stats
                
            except Exception as e:
                return f"Error converting PDF: {str(e)}\nPlease ensure 'markitdown' is installed."
        else:
            # Copy text files directly
            shutil.copy2(file_path, dest_path)
            file_size = os.path.getsize(dest_path)
        
        stats = orchestrator.get_db_stats()
        
        return (
            f"File uploaded successfully!\n"
            f"  Name: {filename}\n"
            f"  Size: {file_size:,} bytes\n"
            f"  Workspace: {_workspace_dir}\n"
            f"  Total files: {len([f for f in os.listdir(_workspace_dir) if f.endswith(('.md', '.txt'))])}\n"
            f"  Cached records: {stats.get('total_records', 0)}\n\n"
            f"You can now query this file using: jit_rag_query(\"your question\")"
        )
    
    except Exception as e:
        return f"Error uploading file: {str(e)}"


def jit_rag_query(query: str) -> str:
    """
    Query the JIT RAG system with a natural language question.
    
    This implements the Hybrid JIT pipeline:
    1. Check cache (LanceDB) for similar vectors
    2. If high confidence: Cache Hit - return immediately
    3. If low confidence: Cache Miss - execute smart grep on workspace files
    4. Background thread embeds new findings for future queries
    
    Args:
        query: Natural language question about your documents
    
    Returns:
        Answer with relevant context from your documents.
    
    Example:
        jit_rag_query("What are the UART register addresses?")
        jit_rag_query("Summarize the main points about interrupts")
    """
    orchestrator = _get_orchestrator()
    
    # Check if workspace has files
    md_files = list(Path(_workspace_dir).glob("*.md"))
    txt_files = list(Path(_workspace_dir).glob("*.txt"))
    total_files = len(md_files) + len(txt_files)
    
    if total_files == 0:
        return (
            "No documents found in workspace.\n"
            "Please upload files first using: jit_rag_upload(\"path/to/file.md\")"
        )
    
    # Execute query
    result = orchestrator.query(query)
    
    # Format response
    source = result.get("source", "unknown")
    confidence = result.get("confidence", 0.0)
    context = result.get("context", "No results found.")
    sources = result.get("sources", [])
    stats = result.get("stats", {})
    
    lines = [
        f"Query: {query}",
        f"",
        f"Source: {source.upper()} | Confidence: {confidence:.1%}",
        f"Stats: {stats.get('cache_hits', 0)} hits / {stats.get('cache_misses', 0)} misses / {stats.get('grep_searches', 0)} greps",
    ]
    
    if result.get("grep_matches"):
        lines.append(f"Grep matches: {result['grep_matches']}")
    
    lines.extend([
        f"",
        f"Answer:",
        f"{'-'*60}",
        context[:2000] if len(context) > 2000 else context,
    ])
    
    if len(context) > 2000:
        lines.append("... (truncated)")
    
    if sources:
        lines.extend([
            f"",
            f"Sources:",
            f"{'-'*60}",
        ])
        for src in sources[:5]:
            lines.append(f"  - {src}")
    
    return "\n".join(lines)


def jit_rag_clear() -> str:
    """
    Clear all uploaded files and cached data from the workspace.
    
    Use this to start fresh with a clean slate.
    
    Returns:
        Confirmation message.
    
    Example:
        jit_rag_clear()
    """
    global _orchestrator, _workspace_dir
    
    if _workspace_dir and os.path.exists(_workspace_dir):
        try:
            shutil.rmtree(_workspace_dir)
            _workspace_dir = None
            _orchestrator = None
            return (
                "Workspace cleared successfully!\n"
                "All uploaded files and cached data have been removed.\n"
                "Ready for new uploads."
            )
        except Exception as e:
            return f"Error clearing workspace: {str(e)}"
    
    return "No workspace to clear. Already clean!"


def jit_rag_status() -> str:
    """
    Check the current status of the JIT RAG system.
    
    Returns:
        Statistics including cached records, workspace files, and query stats.
    
    Example:
        jit_rag_status()
    """
    orchestrator = _get_orchestrator()
    stats = orchestrator.get_db_stats()
    
    # Count files in workspace
    if _workspace_dir and os.path.exists(_workspace_dir):
        md_files = list(Path(_workspace_dir).glob("*.md"))
        txt_files = list(Path(_workspace_dir).glob("*.txt"))
        total_files = len(md_files) + len(txt_files)
    else:
        total_files = 0
    
    lines = [
        "JIT RAG System Status",
        f"{'='*60}",
        f"",
        f"Configuration:",
        f"  Vector DB: Qdrant (local embedded mode, no Docker)",
        f"  Cache threshold: {CONFIG.cache_threshold}",
        f"  Top K results: {CONFIG.top_k}",
        f"  Embedding model: {CONFIG.embedding_model}",
        f"  Embedding server: {CONFIG.embedding_url}",
        f"",
        f"Workspace:",
        f"  Directory: {_workspace_dir or 'Not initialized'}",
        f"  Total files: {total_files}",
        f"",
        f"Database:",
        f"  Path: {stats.get('db_path', 'N/A')}",
        f"  Collection: {stats.get('collection_name', 'N/A')}",
        f"  Cached records: {stats.get('total_records', 0)}",
        f"",
        f"Query Statistics:",
        f"  Cache hits: {stats.get('cache_hits', 0)}",
        f"  Cache misses: {stats.get('cache_misses', 0)}",
        f"  Grep searches: {stats.get('grep_searches', 0)}",
        f"",
        f"Available Commands:",
        f"  - jit_rag_upload(file_path): Upload a document",
        f"  - jit_rag_query(query): Ask a question",
        f"  - jit_rag_status(): Show this status",
        f"  - jit_rag_clear(): Clear all data",
    ]
    
    return "\n".join(lines)


# Export for OpenClaw
__all__ = [
    "jit_rag_upload",
    "jit_rag_query", 
    "jit_rag_status",
    "jit_rag_clear",
]
