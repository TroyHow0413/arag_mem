"""Memory configuration and templates for Re-MEMR1.

Extracted from Re-MEMR1 memory_revisit.py
"""

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for Memory-based processing."""
    
    chunk_size: int = 512  # size of each context chunk in number of tokens
    max_memorization_length: int = 512  # max number of tokens to memorize per step
    max_final_response_length: int = 1024  # max tokens for final answer
    
    # Templates from Re-MEMR1
    TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. You should generate response in the following format:
- Output your thinking process in <thinking>your_thinking_process</thinking>.
- Read the provided section carefully and update the memory with the new information that helps to answer the problem in only one <update>the_updated_memory</update> action. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
- If you notice partial key evidence that is not enough to answer the problem, also output only one `<recall>query</recall>` (e.g. `<recall>who's the president of the United States?</recall>`) to retrieve information in previous memories.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""
    
    TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and provide a clear, direct answer.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Your answer:
"""
    
    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            chunk_size=512,
            max_memorization_length=512,
            max_final_response_length=1024
        )
