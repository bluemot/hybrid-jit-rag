"""
Hybrid Just-In-Time (JIT) On-the-fly RAG Orchestrator with Qdrant Local Mode.

A lightweight RAG system that uses lazy-loading with Qdrant (local embedded) and ripgrep fallback.
No heavy upfront ingestion - only caches what you actually search for.
"""
import os
import re
import json
import subprocess
import threading
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for Hybrid JIT RAG."""
    # Qdrant settings (local embedded mode)
    qdrant_local_path: str = "./qdrant_local_storage"
    collection_name: str = "jit_rag_cache"
    vector_dim: int = 1024  # Match your embedding model
    
    # Search settings
    cache_threshold: float = 0.85
    top_k: int = 5
    
    # GPU/Embedding server
    embedding_url: str = "http://192.168.122.1:11434/api/embeddings"
    embedding_model: str = "bge-m3"
    
    # File settings
    supported_extensions: Tuple[str, ...] = ('.md', '.txt', '.markdown')
    context_window_lines: int = 3  # Lines before/after match


CONFIG = Config()


# =============================================================================
# Step 1: Database Setup (Qdrant Local Mode)
# =============================================================================

class QdrantLocalManager:
    """Manages the local Qdrant embedded vector database (no Docker)."""
    
    def __init__(self, db_path: str = CONFIG.qdrant_local_path):
        self.db_path = db_path
        self.collection_name = CONFIG.collection_name
        self._client = None
        self._init_db()
    
    def _init_db(self):
        """Initialize Qdrant connection in local/embedded mode."""
        import os
        os.makedirs(self.db_path, exist_ok=True)
        print(f"[Qdrant] Initializing embedded/local mode at: {self.db_path}")
        
        # Use path parameter for embedded mode (no server needed)
        self._client = QdrantClient(path=self.db_path)
        
        # Check if collection exists
        try:
            self._client.get_collection(self.collection_name)
            print(f"[Qdrant] Opened existing collection: {self.collection_name}")
        except Exception:
            print(f"[Qdrant] Creating new collection: {self.collection_name}")
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=CONFIG.vector_dim,
                    distance=Distance.COSINE
                ),
            )
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors in the database.
        
        Returns list of results with similarity scores.
        """
        if self._client is None:
            return []
        
        try:
            # Use query_points method (available in local mode)
            from qdrant_client.models import QueryRequest
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to dict format
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "text_content": r.payload.get("text_content", ""),
                    "file_source": r.payload.get("file_source", ""),
                    "chunk_hash": r.payload.get("chunk_hash", ""),
                }
                for r in results
            ]
        except Exception as e:
            print(f"[Qdrant] Search error: {e}")
            return []
    
    def insert(self, records: List[Dict]):
        """
        Insert new records into the database.
        
        Args:
            records: List of dicts with keys: id, vector, text_content, file_source, chunk_hash
        """
        if not records:
            return
        
        points = [
            PointStruct(
                id=r["id"],
                vector=r["vector"],
                payload={
                    "text_content": r.get("text_content", ""),
                    "file_source": r.get("file_source", ""),
                    "chunk_hash": r.get("chunk_hash", ""),
                }
            )
            for r in records
        ]
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"[Qdrant] Inserted {len(records)} records")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "total_records": info.points_count,
                "collection_name": self.collection_name,
                "db_path": self.db_path,
            }
        except Exception as e:
            return {
                "total_records": 0,
                "collection_name": self.collection_name,
                "db_path": self.db_path,
                "error": str(e)
            }


# =============================================================================
# ADVANCED QUERY EXPANSION & REGEX GENERATION
# =============================================================================

def expand_query_to_regex(user_query: str) -> str:
    """
    SYSTEM INSTRUCTION: ADVANCED QUERY EXPANSION & REGEX GENERATION

    Before executing any file search or 'grep' tool, you MUST perform a Semantic 
    Query Expansion to prevent "Cache Misses" caused by author vocabulary choices.

    You must follow this Chain of Thought before searching:

    CONCEPT EXTRACTION: Identify the core entities and the target action/state 
    in the user's query.

    WORLD-BUILDING EXPANSION (The Semantic Bridge):
    Authors rarely use the exact words users type. Brainstorm 3 to 5 synonyms, 
    related plot events, or specific terminology related to the action.

    REGEX CONSTRUCTION:
    Construct a robust, case-insensitive regular expression that captures the 
    Entity AND any of the Expanded Actions within the same proximity.
    Always use '(?i)' for case-insensitivity.
    Use '.*' to allow distance between entity and event.
    Use '(word1|word2|word3)' for OR logic.

    Execute this expansion silently before every search action.
    """
    
    # Extract key entities and concepts
    query_lower = user_query.lower()
    
    # Entity mappings (expand character names and key terms)
    entity_expansions = {
        # Character name expansions
        'shuna': ['朱菜', '公主', '鬼姬', '公主殿下'],
        'rimuru': ['利姆路', '利姆鲁', '史莱姆', '魔物'],
        'benimaru': ['紅丸', '红丸', '哥哥'],
        'shion': ['紫苑', '紫发'],
        'souei': ['蒼影', '苍影'],
        'hakurou': ['白老', '白发'],
        
        # Evolution/Rank terms
        'evolution': ['進化', '进化', '觉醒', '祝福', 'blessing', 'awakening'],
        'second evolution': ['二次进化', '二次進化', '再进化', '真鬼', '神鬼', '聖魔'],
        'demon lord': ['魔王', '真魔王', '八星魔王', 'harvest festival', '收成祭'],
        'power up': ['强化', '強化', '能力提升', '力量提升'],
        
        # Story events
        'first meeting': ['第一次', '初次', '相遇', '遇到', '见面', '邂逅'],
        'joined': ['加入', '成为部下', '效忠', '跟随'],
        
        # Races
        'ogre': ['大鬼族', '食人魔', '鬼人族', '鬼人', '鬼姬', 'oni'],
        'slime': ['史莱姆', '史萊姆', 'slime', '魔物'],
    }
    
    # Find matching entities in query
    found_entities = []
    expanded_terms = []
    
    for entity, synonyms in entity_expansions.items():
        # Check if entity or any synonym is in query
        entity_match = False
        for term in [entity] + synonyms:
            if term.lower() in query_lower:
                entity_match = True
                break
        
        if entity_match:
            found_entities.append(entity)
            expanded_terms.extend(synonyms)
    
    # Also extract any Chinese/Japanese terms directly from query
    import re
    # Extract Chinese characters (common in web novels)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', user_query)
    expanded_terms.extend(chinese_chars)
    
    # Build the expanded keyword list
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        term_lower = term.lower()
        if term_lower not in seen and len(term) > 1:
            seen.add(term_lower)
            unique_terms.append(re.escape(term))
    
    # If no expansions found, fall back to original words
    if not unique_terms:
        # Extract alphanumeric and Chinese words
        words = re.findall(r'[\w\u4e00-\u9fff]+', user_query)
        unique_terms = [re.escape(w) for w in words if len(w) > 1]
    
    # Construct final regex following REGEX CONSTRUCTION rules
    if unique_terms:
        regex_body = '|'.join(unique_terms[:15])  # Limit to top 15 terms
        # Use (?i) for case-insensitivity and proper OR grouping
        final_regex = f"(?i)({regex_body})"
        print(f"[Query Expansion] Original: '{user_query[:50]}...'")
        print(f"[Query Expansion] Expanded to {len(unique_terms)} terms")
        return final_regex
    else:
        # Fallback: simple word extraction
        words = re.findall(r'\w+', user_query)
        return f"(?i)({'|'.join(words[:5])})"


# =============================================================================
# Step 2: The Smart Grep Tool
# =============================================================================

def smart_rg_search(query_regex: str, dir_path: str, context_lines: int = CONFIG.context_window_lines) -> List[Dict]:
    """
    Execute ripgrep search with regex pattern.
    
    Args:
        query_regex: Regex pattern to search for
        dir_path: Directory to search in
        context_lines: Number of lines to include before/after match
    
    Returns:
        List of dicts containing matched text blocks with metadata
    """
    results = []
    
    try:
        # Build ripgrep command
        # -C: context lines, -i: case insensitive, --json: structured output
        cmd = [
            "rg",
            "-C", str(context_lines),
            "-i",
            "--json",
            query_regex,
            dir_path
        ]
        
        print(f"[SmartGrep] Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse JSON output line by line
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                
                if data.get("type") == "match":
                    match_data = data.get("data", {})
                    file_path = match_data.get("path", {}).get("text", "")
                    lines = match_data.get("lines", {}).get("text", "")
                    line_number = match_data.get("line_number", 0)
                    
                    results.append({
                        "file_path": file_path,
                        "line_number": line_number,
                        "text": lines,
                        "match_type": "content"
                    })
                    
                elif data.get("type") == "summary":
                    stats = data.get("data", {})
                    print(f"[SmartGrep] Matched {stats.get('stats', {}).get('matches', 0)} lines in {stats.get('stats', {}).get('searched', 0)} files")
                    
            except json.JSONDecodeError:
                continue
    
    except subprocess.TimeoutExpired:
        print("[SmartGrep] Search timed out")
    except FileNotFoundError:
        print("[SmartGrep] 'rg' (ripgrep) not found. Please install ripgrep.")
    except Exception as e:
        print(f"[SmartGrep] Error: {e}")
    
    return results


def extract_metadata_tags(file_path: str) -> Dict[str, List[str]]:
    """
    Extract metadata tags from markdown file frontmatter.
    
    Returns dict of tags found in YAML frontmatter.
    """
    tags = {"tags": [], "title": "", "category": ""}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                # Simple tag extraction
                tag_match = re.search(r'tags?:\s*\n((?:\s*-\s*[^\n]+\n?)+)', frontmatter)
                if tag_match:
                    tag_lines = tag_match.group(1)
                    tags["tags"] = [t.strip('- ') for t in tag_lines.split('\n') if t.strip()]
                
                title_match = re.search(r'title:\s*["\']?([^"\'\n]+)', frontmatter)
                if title_match:
                    tags["title"] = title_match.group(1)
    
    except Exception as e:
        print(f"[Metadata] Error reading {file_path}: {e}")
    
    return tags


# =============================================================================
# Step 3: Background JIT Ingestion Worker
# =============================================================================

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector using GPU server.
    
    Falls back to random vector if server unavailable.
    """
    try:
        response = httpx.post(
            CONFIG.embedding_url,
            json={"model": CONFIG.embedding_model, "prompt": text},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"[Embedding] Error: {e}, using fallback")
        # Fallback: random normalized vector (for demo/testing)
        vec = np.random.randn(CONFIG.vector_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


def compute_chunk_hash(text: str, file_path: str) -> str:
    """Compute unique hash for text chunk to avoid duplicates."""
    content = f"{file_path}:{text}"
    return hashlib.md5(content.encode()).hexdigest()


def async_ingest_to_db(db_manager: QdrantLocalManager, text_blocks: List[Dict]):
    """
    Background thread worker to ingest text blocks into Qdrant.
    
    Runs in separate thread to avoid blocking main execution.
    """
    def worker():
        print(f"[JIT Ingestion] Starting background thread for {len(text_blocks)} blocks...")
        
        records = []
        for block in text_blocks:
            text = block.get("text", "").strip()
            file_path = block.get("file_path", "")
            
            if not text or len(text) < 20:  # Skip very short blocks
                continue
            
            chunk_hash = compute_chunk_hash(text, file_path)
            record_id = chunk_hash  # Use hash as ID for deduplication
            
            # Generate embedding
            vector = generate_embedding(text)
            
            if len(vector) != CONFIG.vector_dim:
                print(f"[JIT Ingestion] Warning: Vector dim mismatch, expected {CONFIG.vector_dim}, got {len(vector)}")
                continue
            
            records.append({
                "id": record_id,
                "vector": vector,
                "text_content": text,
                "file_source": file_path,
                "chunk_hash": chunk_hash,
            })
        
        if records:
            db_manager.insert(records)
            print(f"[JIT Ingestion] Completed: {len(records)} records saved to cache")
        else:
            print("[JIT Ingestion] No valid records to insert")
    
    # Start background thread
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


# =============================================================================
# Step 4: The Orchestrator Pipeline
# =============================================================================

class HybridJITRAGOrchestrator:
    """
    Main orchestrator implementing the Hybrid JIT RAG pipeline.
    
    Flow:
    1. Check cache (Qdrant local) for similar vectors
    2. If confidence >= threshold: Cache Hit
    3. If confidence < threshold: Cache Miss -> Smart Grep -> JIT Ingest
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.db = QdrantLocalManager()
        self.stats = {"cache_hits": 0, "cache_misses": 0, "grep_searches": 0}
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def query(self, user_query: str) -> Dict:
        """
        Execute the hybrid JIT RAG query pipeline.
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Dict with results, source, and metadata
        """
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Processing query: '{user_query}'")
        print(f"{'='*60}")
        
        # Step 1: Generate query embedding
        print("[Step 1] Generating query embedding...")
        query_vector = generate_embedding(user_query)
        
        # Step 2: Search cache
        print(f"[Step 2] Searching local cache (Qdrant)...")
        cache_results = self.db.search(query_vector, top_k=CONFIG.top_k)
        
        # Step 3: Check confidence threshold (Qdrant returns score directly)
        best_score = 0.0
        if cache_results:
            best_match = cache_results[0]
            best_score = best_match.get("score", 0.0)
            print(f"[Cache] Best match score: {best_score:.3f} (threshold: {CONFIG.cache_threshold})")
        
        if best_score >= CONFIG.cache_threshold:
            # CACHE HIT
            self.stats["cache_hits"] += 1
            print(f"[Cache Hit] Score {best_score:.3f} >= {CONFIG.cache_threshold}")
            
            context_blocks = [r.get("text_content", "") for r in cache_results[:3]]
            
            return {
                "source": "cache",
                "confidence": best_score,
                "context": "\n\n".join(context_blocks),
                "sources": [r.get("file_source", "") for r in cache_results[:3]],
                "stats": self.stats.copy()
            }
        
        # CACHE MISS - Execute Smart Grep with Query Expansion
        self.stats["cache_misses"] += 1
        print(f"[Cache Miss] Score {best_score:.3f} < {CONFIG.cache_threshold}")
        print(f"[Cache Miss] Executing Smart Grep...")
        
        # Use ADVANCED QUERY EXPANSION instead of simple keyword extraction
        search_pattern = expand_query_to_regex(user_query)
        print(f"[Smart Grep] Using expanded regex pattern")
        
        grep_results = smart_rg_search(search_pattern, self.workspace_dir)
        self.stats["grep_searches"] += 1
        
        if not grep_results:
            print("[Smart Grep] No results found")
            return {
                "source": "none",
                "confidence": 0.0,
                "context": "No relevant information found.",
                "sources": [],
                "stats": self.stats.copy()
            }
        
        print(f"[Smart Grep] Found {len(grep_results)} matches")
        
        # Step 4: Fire JIT Ingestion (background thread)
        print("[Step 4] Spawning background JIT ingestion...")
        async_ingest_to_db(self.db, grep_results)
        
        # Step 5: Build context from grep results
        context_parts = []
        sources = []
        for i, match in enumerate(grep_results[:5]):  # Top 5 results
            text = match.get("text", "").strip()
            file_path = match.get("file_path", "")
            if text:
                context_parts.append(f"[Source: {file_path}]\n{text}")
                sources.append(file_path)
        
        return {
            "source": "grep",
            "confidence": 0.5,  # Grep results have lower confidence
            "context": "\n\n---\n\n".join(context_parts),
            "sources": list(set(sources)),
            "grep_matches": len(grep_results),
            "stats": self.stats.copy()
        }
    
    def get_db_stats(self) -> Dict:
        """Get database and system statistics."""
        db_stats = self.db.get_stats()
        return {
            **db_stats,
            **self.stats
        }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid JIT RAG Orchestrator (Qdrant Local)")
    parser.add_argument("workspace", help="Directory containing Markdown files to search")
    parser.add_argument("--query", "-q", help="Single query mode")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.workspace):
        print(f"Error: Directory not found: {args.workspace}")
        return 1
    
    orchestrator = HybridJITRAGOrchestrator(args.workspace)
    
    if args.stats:
        stats = orchestrator.get_db_stats()
        print("\n[Database Statistics]")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0
    
    if args.query:
        # Single query mode
        result = orchestrator.query(args.query)
        print("\n" + "="*60)
        print("[RESULT]")
        print("="*60)
        print(f"Source: {result['source']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'N/A'}")
        print("\n[Context]")
        print(result['context'][:2000])  # Limit output
        if len(result['context']) > 2000:
            print("... (truncated)")
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("Hybrid JIT RAG Orchestrator (Qdrant Local Mode)")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'stats' for database statistics")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nQuery> ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = orchestrator.get_db_stats()
                    print("\n[Statistics]")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                
                if not user_input:
                    continue
                
                result = orchestrator.query(user_input)
                print("\n" + "-"*60)
                print(f"Source: {result['source']} | Confidence: {result['confidence']:.3f}")
                print("-"*60)
                print(result['context'][:1500])
                if len(result['context']) > 1500:
                    print("... (truncated)")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break
    
    return 0


if __name__ == "__main__":
    exit(main())
