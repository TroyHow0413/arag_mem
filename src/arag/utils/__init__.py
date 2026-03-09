"""Utility functions for data adaptation between ARAG and Memory systems."""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_chunks_from_file(chunks_file: str) -> List[Dict[str, Any]]:
    """
    Load chunks from JSON file.
    
    Args:
        chunks_file: Path to chunks.json file
        
    Returns:
        List of chunk dictionaries with 'id' and 'text' keys
    """
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if data and isinstance(data[0], dict) and 'id' in data[0] and 'text' in data[0]:
        return data
    
    # Convert old format if needed
    chunks = []
    for item in data:
        if isinstance(item, str):
            parts = item.split(':', 1)
            if len(parts) == 2:
                chunks.append({'id': parts[0], 'text': parts[1]})
        elif isinstance(item, dict):
            chunks.append(item)
    
    return chunks


def assemble_chunks_to_context(chunks_file: str, sort_by_id: bool = True) -> str:
    """
    Assemble all chunks into a single context string.
    
    This converts ARAG's chunk-based format to Memory's continuous context format.
    
    Args:
        chunks_file: Path to chunks.json file
        sort_by_id: Whether to sort chunks by ID before assembling
        
    Returns:
        Assembled context string
    """
    chunks = load_chunks_from_file(chunks_file)
    
    if sort_by_id:
        # Sort by numeric ID
        chunks = sorted(chunks, key=lambda x: int(x['id']))
    
    # Join all chunk texts
    context = '\n\n'.join([chunk['text'] for chunk in chunks])
    
    return context


def get_context_for_dataset(questions_file: str, chunks_file: str = None) -> str:
    """
    Get context string for a dataset.
    
    Args:
        questions_file: Path to questions.json
        chunks_file: Path to chunks.json (auto-detected if None)
        
    Returns:
        Assembled context string
    """
    questions_path = Path(questions_file)
    
    # Auto-detect chunks file in same directory
    if chunks_file is None:
        data_dir = questions_path.parent
        chunks_file = data_dir / "chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {chunks_file}. "
                f"Please specify chunks_file explicitly."
            )
    
    return assemble_chunks_to_context(str(chunks_file))
