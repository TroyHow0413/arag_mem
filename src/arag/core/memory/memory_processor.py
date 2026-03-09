"""Core memory processing logic extracted from Re-MEMR1.

This module implements the callback-based memory mechanism from Re-MEMR1:
- Progressive chunk-by-chunk reading
- Memory update and accumulation
- TF-IDF based memory recall
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import tiktoken

from arag.core.memory.memory_config import MemoryConfig
from arag.core.memory.tf_idf_retriever import TfidfRetriever


class MemoryProcessor:
    """
    Memory processor implementing Re-MEMR1's callback mechanism.
    
    Key features:
    - Chunk-by-chunk document processing
    - Memory update with <update>...</update> tags
    - Memory recall with <recall>...</recall> tags
    - TF-IDF based retrieval from history
    """
    
    def __init__(self, config: MemoryConfig = None, tokenizer_name: str = "gpt-4o"):
        """
        Args:
            config: Memory configuration
            tokenizer_name: Name of tokenizer for token counting
        """
        self.config = config or MemoryConfig.default()
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        self.retriever = TfidfRetriever(tokenizer=None)  # Use simple tokenizer
        
        # Memory state
        self.history_memory: List[set] = []  # List of sets, one per sample
        self.current_memory: List[Optional[str]] = []  # Current memory for each sample
        self.recalled_memory: List[Optional[str]] = []  # Recalled memory for each sample
        
        # Constants
        self.NO_MEMORY_STRING = "No previous memory"
        self.NO_MEMORY_RECALLED_STRING = "No memory was recalled."
    
    def initialize(self, num_samples: int):
        """Initialize memory state for a batch of samples."""
        self.history_memory = [set() for _ in range(num_samples)]
        self.current_memory = [None for _ in range(num_samples)]
        self.recalled_memory = [None for _ in range(num_samples)]
    
    def split_context_into_chunks(self, context: str) -> List[str]:
        """
        Split context into fixed-size chunks based on token count.
        
        Args:
            context: Full document text
            
        Returns:
            List of chunk strings
        """
        tokens = self.tokenizer.encode(context)
        chunks = []
        
        chunk_size = self.config.chunk_size
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def format_prompt(
        self,
        problem: str,
        chunk: str,
        memory: Optional[str] = None,
        recalled_memory: Optional[str] = None,
        is_final: bool = False
    ) -> str:
        """
        Format prompt using Re-MEMR1 templates.
        
        Args:
            problem: The question to answer
            chunk: Current chunk being processed
            memory: Current memory state
            recalled_memory: Memory recalled from history
            is_final: Whether this is the final answer generation
            
        Returns:
            Formatted prompt string
        """
        template = self.config.TEMPLATE_FINAL if is_final else self.config.TEMPLATE
        
        return template.format(
            prompt=problem,
            chunk=chunk if not is_final else "",
            memory=memory or self.NO_MEMORY_STRING,
            recalled_memory=recalled_memory or self.NO_MEMORY_RECALLED_STRING
        )
    
    def parse_recall_query(self, text_response: str) -> Optional[str]:
        """
        Extract recall query from LLM response.
        
        Looks for pattern: <recall>query text</recall>
        
        Args:
            text_response: LLM's text output
            
        Returns:
            Extracted query string or None
        """
        try:
            match = re.search(r'<recall>(.+?)</recall>', text_response, re.DOTALL)
            if match:
                query = match.group(1).strip()
                return query
        except (ValueError, TypeError):
            pass
        return None
    
    def parse_update_memory(self, text_response: str) -> Optional[str]:
        """
        Extract updated memory from LLM response.
        
        Removes <recall>...</recall> tags and returns the cleaned text.
        
        Args:
            text_response: LLM's text output
            
        Returns:
            Updated memory string or None
        """
        try:
            # Remove recall tags
            cleaned = re.sub(r'<recall>.*?</recall>', '', text_response, flags=re.DOTALL)
            # Remove update tags if present
            cleaned = re.sub(r'<update>|</update>', '', cleaned, flags=re.DOTALL)
            return cleaned.strip()
        except (ValueError, TypeError):
            return None
    
    def retrieve_from_history(
        self,
        query: Optional[str],
        sample_idx: int
    ) -> Optional[str]:
        """
        Retrieve relevant memory from history using TF-IDF.
        
        Args:
            query: Query string for retrieval
            sample_idx: Index of the sample
            
        Returns:
            Retrieved memory string or None
        """
        if query is None or not self.history_memory[sample_idx]:
            return None
        
        corpus = list(self.history_memory[sample_idx])
        retrieved = self.retriever.top1_retrieve(query, corpus)
        return retrieved
    
    def update_memory_state(
        self,
        sample_idx: int,
        llm_response: str
    ) -> Tuple[str, Optional[str]]:
        """
        Update memory state based on LLM response.
        
        This implements the core callback mechanism:
        1. Parse recall query from response
        2. Retrieve relevant memory from history
        3. Parse and update current memory
        4. Add to history
        
        Args:
            sample_idx: Index of the sample
            llm_response: LLM's response text
            
        Returns:
            Tuple of (updated_memory, recalled_memory)
        """
        # Parse recall query
        recall_query = self.parse_recall_query(llm_response)
        
        # Retrieve from history
        recalled = self.retrieve_from_history(recall_query, sample_idx)
        self.recalled_memory[sample_idx] = recalled
        
        # Parse updated memory
        updated_memory = self.parse_update_memory(llm_response)
        
        # Update current memory
        if updated_memory:
            self.current_memory[sample_idx] = updated_memory
            # Add to history
            self.history_memory[sample_idx].add(updated_memory)
        
        return updated_memory, recalled
    
    def get_memory_state(self, sample_idx: int) -> Dict[str, Any]:
        """Get current memory state for a sample."""
        return {
            'current_memory': self.current_memory[sample_idx],
            'recalled_memory': self.recalled_memory[sample_idx],
            'history_size': len(self.history_memory[sample_idx])
        }
    
    def reset(self):
        """Reset all memory state."""
        self.history_memory = []
        self.current_memory = []
        self.recalled_memory = []
