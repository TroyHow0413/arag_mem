# Re-MEMR1 Memory Integration Guide

This directory contains the integration of Re-MEMR1's memory mechanism into ARAG.

## Overview

The memory module implements Re-MEMR1's callback-based memory mechanism:
- **Progressive reading**: Processes document chunks sequentially
- **Memory accumulation**: LLM extracts and updates memory at each step
- **Memory recall**: TF-IDF retrieval from history memory
- **Final answer**: Generated based on accumulated memory

## File Structure

```
src/arag/core/memory/
├── __init__.py              # Module exports
├── memory_config.py         # Configuration and templates
├── memory_processor.py      # Core memory logic
└── tf_idf_retriever.py      # TF-IDF based retrieval

src/arag/agent/
└── memory_agent.py          # MemoryAgent wrapper (compatible with BaseAgent)

src/arag/utils/
└── __init__.py              # Data adaptation utilities
```

## Usage

### 1. Install Dependencies

The memory module requires scikit-learn:

```bash
pip install scikit-learn
```

### 2. Run with Base Agent (Original A-RAG)

```bash
python scripts/batch_runner.py \
    --config configs/example.yaml \
    --questions data/musique/questions.json \
    --output results/musique_base \
    --agent-type base \
    --limit 100
```

### 3. Run with Memory Agent (Re-MEMR1)

```bash
python scripts/batch_runner.py \
    --config configs/memory_example.yaml \
    --questions data/musique/questions.json \
    --output results/musique_memory \
    --agent-type memory \
    --limit 100
```

### 4. Compare Results

Both agents output identical format, so you can use the same evaluation:

```bash
# Evaluate base agent
python scripts/eval.py \
    --predictions results/musique_base/predictions.jsonl \
    --workers 5

# Evaluate memory agent
python scripts/eval.py \
    --predictions results/musique_memory/predictions.jsonl \
    --workers 5
```

## Configuration

Add memory-specific settings to your config YAML:

```yaml
memory:
  chunk_size: 512                    # Tokens per chunk
  max_memorization_length: 512       # Max tokens for memory update
  max_final_response_length: 1024    # Max tokens for final answer
```

## Key Differences

| Aspect | Base Agent | Memory Agent |
|--------|-----------|--------------|
| **Retrieval** | Selective (tool-based) | Sequential (all chunks) |
| **Strategy** | Agent decides what to read | Pre-defined order |
| **Memory** | Implicit in conversation | Explicit with recall |
| **Best for** | Multi-hop questions | Long documents |

## Output Format

Both agents produce identical output structure:

```json
{
    "answer": "...",
    "trajectory": [...],
    "total_cost": 0.123,
    "loops": 5,
    "total_retrieved_tokens": 1234,
    "retrieval_logs": [...],
    "chunks_read_count": 5,
    "chunks_read_ids": ["0", "1", "2", "3", "4"]
}
```

For Memory Agent, additional field:
```json
{
    "memory_state": {
        "final_memory": "...",
        "history_size": 5
    }
}
```

## Research Value

This integration enables controlled comparison experiments:

1. **Direct Comparison**: Same input → different mechanisms → compare accuracy
2. **Ablation Study**: Test with/without memory recall
3. **Cost Analysis**: Compare token usage and API costs
4. **Use Case Analysis**: Which mechanism works better for what type of questions

## Implementation Notes

- **Zero modification** to original A-RAG BaseAgent
- **Compatible output** ensures fair comparison
- **Re-MEMR1 algorithm preserved** from original implementation
- **Modular design** allows easy switching between agents
