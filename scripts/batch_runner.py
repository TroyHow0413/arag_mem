#!/usr/bin/env python3
"""
Batch Runner for ARAG - Supports concurrent execution and checkpoint resume.

Usage:
    python scripts/batch_runner.py \
        --config configs/example.yaml \
        --questions data/questions.json \
        --output results/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from tqdm import tqdm
from arag import LLMClient, BaseAgent, MemoryAgent, ToolRegistry, Config, MemoryConfig
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.utils import get_context_for_dataset

logging.basicConfig(level=logging.ERROR)


class BatchRunner:
    """Batch runner with concurrent execution and checkpoint resume."""
    
    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int = None,
        num_workers: int = 10,
        verbose: bool = False,
        agent_type: str = "base"
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose
        self.agent_type = agent_type
        
        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()
        
        self.questions = self._load_questions()
        
        # Pre-initialize shared tools (load embedding model only once)
        self._shared_tools = self._init_shared_tools()
        
        # Load system prompt once
        prompt_file = Path(__file__).parent.parent / "src/arag/agent/prompts/default.txt"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
        
        # For memory agent, pre-load context
        self._context = None
        if self.agent_type == "memory":
            print(f"Loading context for Memory Agent...")
            self._context = get_context_for_dataset(str(self.questions_file))
            print(f"Context loaded: {len(self._context)} characters")
        else:
            self._system_prompt = "You are a helpful assistant."
    
    def _init_shared_tools(self) -> ToolRegistry:
        """Initialize shared tools (load embedding model only once)."""
        data_config = self.config.get('data', {})
        
        # Infer chunks_file and index_dir from questions_file path
        questions_path = Path(self.questions_file)
        data_dir = questions_path.parent
        chunks_file = data_dir / "chunks.json"
        index_dir = data_dir / "index" if (data_dir / "index").exists() else Path(data_config.get('index_dir', 'data/index'))
        
        # Fallback to config if inferred paths don't exists
        if not chunks_file.exists():
            chunks_file = Path(data_config.get('chunks_file', 'data/chunks.json'))
        
        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=str(chunks_file)))
        tools.register(ReadChunkTool(chunks_file=str(chunks_file)))
        
        # Add semantic search if index exists
        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            embedding_config = self.config.get('embedding', {})
            print(f"Loading embedding model: {embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')}")
            tools.register(SemanticSearchTool(
                chunks_file=str(chunks_file),
                index_dir=str(index_dir),
                model_name=embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                device=embedding_config.get('device')
            ))
            print("Embedding model loaded successfully!")
        else:
            print(f"Warning: Index not found at {index_file}, semantic search disabled")
        
        return tools
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from file."""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if self.limit:
            questions = questions[:self.limit]
        
        return questions
    
    def _load_completed_qids(self) -> set:
        """Load completed question IDs for checkpoint resume."""
        completed_qids = set()
        
        if not self.predictions_file.exists():
            return completed_qids
        
        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'question' in data and 'pred_answer' in data:
                            qid = data.get('qid') or data.get('id')
                            if qid is not None:
                                completed_qids.add(qid)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error loading completed data: {e}")
        
        return completed_qids
    :
        """Create agent instance based on agent_type."""
        if self.agent_type == "memory":
            return self._create_memory_agent()
        else:
            return self._create_base_agent()
    
    def _create_base_agent(self) -> BaseAgent:
        """Create base agent instance with shared tools."""
        llm_config = self.config.get('llm', {})
        
        client = LLMClient(
            model=llm_config.get('model') or os.getenv('ARAG_MODEL', 'gpt-4o-mini'),
            api_key=llm_config.get('api_key') or os.getenv('ARAG_API_KEY'),
            base_url=llm_config.get('base_url') or os.getenv('ARAG_BASE_URL', 'https://api.openai.com/v1'),
            reasoning_effort=llm_config.get('reasoning_effort')
        )
        
        agent_config = self.config.get('agent', {})
        
        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,  # Use shared tools
            system_prompt=self._system_prompt,
            max_loops=agent_config.get('max_loops', 10),
            max_token_budget=agent_config.get('max_token_budget', 128000),
            verbose=self.verbose
        )
    
    def _create_memory_agent(self) -> MemoryAgent:
        """Create memory agent instance."""
        llm_config = self.config.get('llm', {})
        
        client = LLMClient(
            model=llm_config.get('model') or os.getenv('ARAG_MODEL', 'gpt-4o-mini'),
            api_key=llm_config.get('api_key') or os.getenv('ARAG_API_KEY'),
            base_url=llm_config.get('base_url') or os.getenv('ARAG_BASE_URL', 'https://api.openai.com/v1'),
            reasoning_effort=llm_config.get('reasoning_effort')
        )
        
        # Get memory config from config file or use defaults
        memory_config_dict = self.config.get('memory', {})
        memory_config = MemoryConfig(
            chunk_size=memory_config_dict.get('chunk_size', 512),
            max_memorization_length=memory_config_dict.get('max_memorization_length', 512),
            max_final_response_length=memory_config_dict.get('max_final_response_length', 1024)
        )
        
        return MemoryAgent(
            llm_client=client,
            context=self._context,
            memory_config=memory_config
        
        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,  # Use shared tools
            system_prompt=self._system_prompt,
            max_loops=agent_config.get('max_loops', 10),
            max_token_budget=agent_config.get('max_token_budget', 128000),
            verbose=self.verbose
        )
    
    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        """Process one question."""
        qid = item.get('qid') or item.get('id')
        question = item.get('question', '')
        gold_answer = item.get('answer', item.get('gold_answer', ''))
        
        llm_config = self.config.get('llm', {})
        model_name = llm_config.get('model') or os.getenv('ARAG_MODEL', 'gpt-4o-mini')
        
        try:
            result = agent.run(question)
            
            return {
                'qid': qid,
                'question': question,
                'trajectory': result['trajectory'],
                'gold_answer': gold_answer,
                'pred_answer': result['answer'],
                'total_cost': result['total_cost'],
                'loops': result['loops'],
                'total_retrieved_tokens': result.get('total_retrieved_tokens', 0),
                'retrieval_logs': result.get('retrieval_logs', []),
                'chunks_read_count': result.get('chunks_read_count', 0),
                'chunks_read_ids': result.get('chunks_read_ids', []),
                'model': model_name
            }
        except Exception as e:
            return {
                'qid': qid,
                'question': question,
                'trajectory': [],
                'gold_answer': gold_answer,
                'pred_answer': f"Error: {str(e)}",
                'total_cost': 0,
                'loops': 0,
                'total_retrieved_tokens': 0,
                'retrieval_logs': [],
                'chunks_read_count': 0,
                'chunks_read_ids': [],
                'error': str(e),
                'model': model_name
            }
    
    def run(self):
        """Run batch processing."""
        completed_qids = self._load_completed_qids()
        
        # Filter pending questions
        pending = [q for q in self.questions 
                   if (q.get('qid') or q.get('id')) not in completed_qids]
    parser.add_argument("--agent-type", choices=["base", "memory"], default="base", 
                        help="Agent type: 'base' for tool-based A-RAG, 'memory' for Re-MEMR1")
    
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    runner = BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
        agent_type=args.agent_typutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent)
                futures[future] = item.get('qid') or item.get('id')
            
            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        print(f"Error processing {qid}: {e}")
                    pbar.update(1)
        
        print(f"\nResults saved to: {self.predictions_file}")


def main():
    parser = argparse.ArgumentParser(description="ARAG Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--questions", "-q", required=True, help="Questions file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    runner = BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose
    )
    
    runner.run()


if __name__ == "__main__":
    main()
