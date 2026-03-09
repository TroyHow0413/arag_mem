"""Memory-based Agent implementing Re-MEMR1 mechanism.

This agent processes documents using the Re-MEMR1 memory mechanism:
- Progressive chunk-by-chunk reading
- Memory accumulation with callback-based recall
- Compatible output format with BaseAgent for comparison experiments
"""

import json
from typing import Dict, Any, List
import tiktoken

from arag.core.llm import LLMClient
from arag.core.memory import MemoryConfig, MemoryProcessor


class MemoryAgent:
    """
    Memory-driven agent using Re-MEMR1's callback mechanism.
    
    Key differences from BaseAgent:
    - Processes ALL chunks sequentially (not selective retrieval)
    - Accumulates memory across chunks
    - Uses TF-IDF for memory recall
    - Final answer based on accumulated memory
    
    Output format is identical to BaseAgent for comparison experiments.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        context: str,
        memory_config: MemoryConfig = None,
        verbose: bool = False,
    ):
        """
        Args:
            llm_client: LLM client for generation
            context: Full document context to process
            memory_config: Memory configuration
            verbose: Whether to print processing details
        """
        self.llm = llm_client
        self.context = context
        self.config = memory_config or MemoryConfig.default()
        self.verbose = verbose
        
        # Initialize memory processor
        self.processor = MemoryProcessor(config=self.config)
        
        # Tokenizer for cost tracking
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    def _calculate_tokens(self, text: str) -> int:
        """Calculate number of tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run memory-based agent on a query.
        
        Args:
            query: Question to answer
            
        Returns:
            Result dictionary with same format as BaseAgent:
            {
                "answer": str,
                "trajectory": List[Dict],
                "total_cost": float,
                "loops": int,
                "total_retrieved_tokens": int,
                "retrieval_logs": List[Dict],
                "chunks_read_count": int,
                "chunks_read_ids": List[str]
            }
        """
        # Split context into chunks
        chunks = self.processor.split_context_into_chunks(self.context)
        
        # Initialize memory for single sample
        self.processor.initialize(num_samples=1)
        sample_idx = 0
        
        trajectory = []
        total_cost = 0.0
        total_tokens = 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {query}")
            print(f"Total chunks: {len(chunks)}")
            print(f"{'='*60}\n")
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            if self.verbose:
                print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
            
            # Get current memory state
            memory_state = self.processor.get_memory_state(sample_idx)
            current_memory = memory_state['current_memory']
            recalled_memory = memory_state['recalled_memory']
            
            # Format prompt using Re-MEMR1 template
            prompt = self.processor.format_prompt(
                problem=query,
                chunk=chunk,
                memory=current_memory,
                recalled_memory=recalled_memory,
                is_final=False
            )
            
            # Calculate input tokens
            input_tokens = self._calculate_tokens(prompt)
            total_tokens += input_tokens
            
            # Call LLM
            try:
                response = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    temperature=0.0
                )
                
                total_cost += response["cost"]
                llm_response = response["message"].get("content", "")
                
                if self.verbose:
                    print(f"  Response preview: {llm_response[:200]}...")
                
                # Update memory state (implements callback mechanism)
                updated_memory, recalled = self.processor.update_memory_state(
                    sample_idx, llm_response
                )
                
                if self.verbose and recalled:
                    print(f"  Recalled memory: {recalled[:100]}...")
                
                # Record trajectory
                traj_entry = {
                    "loop": chunk_idx + 1,
                    "tool_name": "memory_update",
                    "arguments": {
                        "chunk_index": chunk_idx,
                        "chunk_preview": chunk[:100] + "..."
                    },
                    "tool_result": f"Memory updated. Recall query: {self.processor.parse_recall_query(llm_response) or 'None'}",
                    "retrieved_tokens": input_tokens,
                    "memory_updated": updated_memory is not None,
                    "memory_recalled": recalled is not None
                }
                trajectory.append(traj_entry)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error processing chunk: {e}")
                
                traj_entry = {
                    "loop": chunk_idx + 1,
                    "tool_name": "memory_update",
                    "arguments": {"chunk_index": chunk_idx},
                    "tool_result": f"Error: {str(e)}",
                    "retrieved_tokens": 0,
                    "error": str(e)
                }
                trajectory.append(traj_entry)
        
        # Generate final answer
        if self.verbose:
            print(f"\nGenerating final answer...")
        
        memory_state = self.processor.get_memory_state(sample_idx)
        final_prompt = self.processor.format_prompt(
            problem=query,
            chunk="",
            memory=memory_state['current_memory'],
            recalled_memory=memory_state['recalled_memory'],
            is_final=True
        )
        
        final_input_tokens = self._calculate_tokens(final_prompt)
        total_tokens += final_input_tokens
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": final_prompt}],
                tools=None,
                temperature=0.0
            )
            
            total_cost += response["cost"]
            final_answer = response["message"].get("content", "")
            
            if self.verbose:
                print(f"Final answer: {final_answer[:200]}...")
                print(f"Total cost: ${total_cost:.6f}")
                print(f"Total chunks processed: {len(chunks)}")
        
        except Exception as e:
            if self.verbose:
                print(f"Error generating final answer: {e}")
            final_answer = f"Error: {str(e)}"
        
        # Add final answer to trajectory
        trajectory.append({
            "loop": len(chunks) + 1,
            "tool_name": "final_answer",
            "arguments": {"memory_history_size": memory_state['history_size']},
            "tool_result": final_answer,
            "retrieved_tokens": final_input_tokens
        })
        
        # Format output to match BaseAgent
        return {
            "answer": final_answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": len(chunks) + 1,
            "total_retrieved_tokens": total_tokens,
            "retrieval_logs": [
                {
                    "tool_name": "memory_chunk_processing",
                    "tokens": total_tokens,
                    "metadata": {
                        "total_chunks": len(chunks),
                        "memory_history_size": memory_state['history_size']
                    }
                }
            ],
            "chunks_read_count": len(chunks),
            "chunks_read_ids": [str(i) for i in range(len(chunks))],
            "memory_state": {
                "final_memory": memory_state['current_memory'],
                "history_size": memory_state['history_size']
            }
        }
