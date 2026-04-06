# Working Rules for Hybrid JIT RAG

## Core Principles

### 1. Subagent-Only Execution
- **NEVER** execute queries directly from the main agent
- ALL document queries must be delegated to subagents
- Main agent acts ONLY as a "forwarder" (轉發者)

### 2. No Hardcoded Domain Entities
- **ABSOLUTELY NO** hardcoded character names, story elements, or domain-specific terms
- Entity expansions must be loaded from external configuration files
- Default behavior should work with ANY text corpus (novels, technical docs, etc.)

### 3. Generic Implementation
- Code must be domain-agnostic
- No assumptions about content type (novel vs technical spec vs code)
- Configuration-driven, not code-driven

## File Structure

```
hybrid-jit-rag/
├── jit_rag_orchestrator.py    # Core engine (domain-agnostic)
├── jit_rag_skill.py            # OpenClaw skill wrapper
├── run_jit_query.py            # CLI entry point for subagents
├── subagent_tool.py            # Subagent interface
├── config/
│   └── entity_expansions.json  # Optional: domain-specific synonyms
├── working-rules.md            # This file
└── README.md
```

## External Configuration (Optional)

If domain-specific query expansion is needed:

1. Create `config/entity_expansions.json`
2. Load it dynamically at runtime
3. Never hardcode in Python source

Example:
```json
{
  "character_names": {
    "rimuru": ["利姆路", "史莱姆"],
    "shuna": ["朱菜", "公主"]
  }
}
```

## Query Flow

```
User Query
    ↓
Main Agent (forward only)
    ↓
Subagent
    ↓
run_jit_query.py
    ↓
jit_rag_orchestrator.py
    ↓
Results (JSON)
    ↓
Subagent processes
    ↓
Final answer to user
```

## Testing

All tests must be run through subagents:
```python
# CORRECT: Via subagent
sessions_spawn({
    "task": "python run_jit_query.py 'test query'",
    ...
})

# WRONG: Direct execution
exec("python -c 'from jit_rag import ...'")
```

## Reminder

> This is a **generic RAG framework**, not a "That Time I Got Reincarnated as a Slime" query tool.
> 
> Every feature must be applicable to:
> - Technical documentation
> - Legal contracts
> - Scientific papers
> - Any text corpus

## Violations

If you see:
- ❌ Hardcoded character names
- ❌ Story-specific logic
- ❌ Direct execution bypassing subagent

**Fix immediately.**
