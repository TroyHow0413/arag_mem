"""LLM client for ARAG - unified interface for OpenAI-compatible APIs."""

import json
import os
from typing import Any, Dict, List, Optional

import requests
import tiktoken


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs."""
    
    # Model name mappings for different providers
    OPENROUTER_MODEL_MAP = {
        # Qwen models
        "qwen3.5-flash-02-23": "qwen/qwen3.5-flash-02-23",
        "qwen-flash": "qwen/qwen-2.5-72b-instruct",
        "qwen2.5": "qwen/qwen-2.5-72b-instruct",
        # Claude models
        "claude-4.5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-haiku": "anthropic/claude-3-haiku",
        # GPT models
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4": "openai/gpt-4-turbo",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        # Gemini models
        "gemini-flash": "google/gemini-flash-1.5",
        "gemini-pro": "google/gemini-pro-1.5",
    }
    
    # Official pricing (USD per 1M tokens): (input, cached_input, output)
    PRICING = {
        # OpenAI GPT-5 series
        "gpt-5.2-pro": (21.0, 2.1, 168.0),
        "gpt-5.2": (1.75, 0.175, 14.0),
        "gpt-5.1": (1.5, 0.15, 12.0),
        "gpt-5-mini": (0.25, 0.025, 2.0),
        "gpt-5": (1.5, 0.15, 12.0),
        # OpenAI GPT-4 series
        "gpt-4.1-mini": (0.4, 0.1, 1.6),
        "gpt-4.1-nano": (0.1, 0.025, 0.4),
        "gpt-4.1": (2.0, 0.5, 8.0),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "gpt-4o": (2.5, 1.25, 10.0),
        "gpt-4-turbo": (10.0, 5.0, 30.0),
        # OpenAI o-series
        "o4-mini": (1.1, 0.275, 4.4),
        "o3-mini": (1.1, 0.275, 4.4),
        "o3": (10.0, 2.5, 40.0),
        "o1-mini": (1.1, 0.275, 4.4),
        "o1": (15.0, 3.75, 60.0),
        # Anthropic Claude 4 series
        "claude-4.5-opus": (5.0, 0.5, 25.0),
        "claude-4.5-sonnet": (3.0, 0.3, 15.0),
        "claude-4.5-haiku": (1.0, 0.1, 5.0),
        "claude-4-opus": (15.0, 1.5, 75.0),
        "claude-4-sonnet": (3.0, 0.3, 15.0),
        "claude-sonnet": (3.0, 0.3, 15.0),
        "claude-opus": (5.0, 0.5, 25.0),
        "claude-haiku": (1.0, 0.1, 5.0),
        # Google Gemini 3 series
        "gemini-3-pro": (2.0, 0.2, 12.0),
        "gemini-3-flash": (0.5, 0.05, 3.0),
        # Google Gemini 2.5 series
        "gemini-2.5-pro": (1.25, 0.125, 10.0),
        "gemini-2.5-flash": (0.3, 0.075, 2.5),
        # Google Gemini 2.0 series
        "gemini-2.0-flash": (0.1, 0.025, 0.4),
        "gemini-pro": (1.25, 0.125, 5.0),
        "gemini-flash": (0.075, 0.02, 0.3),
        # Default fallback
        "default": (1.0, 0.1, 5.0),
    }
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: str = None,
    ):
        self.model = model or os.getenv("ARAG_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("ARAG_API_KEY")
        # Clean up base_url: remove trailing slash and /chat/completions if present
        url = (base_url or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1")).rstrip('/')
        if url.endswith('/chat/completions'):
            url = url[:-len('/chat/completions')]
        self.base_url = url
        
        # Auto-convert model names for OpenRouter
        if 'openrouter.ai' in self.base_url.lower():
            # If model name doesn't contain '/', it needs provider prefix
            if '/' not in self.model:
                # Try to find in mapping table first
                if self.model in self.OPENROUTER_MODEL_MAP:
                    original = self.model
                    self.model = self.OPENROUTER_MODEL_MAP[self.model]
                    print(f"[LLM] Converted model '{original}' -> '{self.model}' for OpenRouter")
                else:
                    # Auto-detect provider from model name
                    original = self.model
                    if 'qwen' in self.model.lower():
                        self.model = f"qwen/{self.model}"
                    elif 'claude' in self.model.lower():
                        self.model = f"anthropic/{self.model}"
                    elif 'gpt' in self.model.lower() or self.model.startswith('o'):
                        self.model = f"openai/{self.model}"
                    elif 'gemini' in self.model.lower():
                        self.model = f"google/{self.model}"
                    print(f"[LLM] Auto-prefixed model '{original}' -> '{self.model}' for OpenRouter")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        
        if not self.api_key:
            raise ValueError("API key required. Set ARAG_API_KEY environment variable or pass api_key parameter.")
        
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total
    
    def calculate_cost(self, usage: dict) -> float:
        """Calculate API cost from usage info."""
        model_lower = self.model.lower()
        
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        prompt_details = usage.get('prompt_tokens_details', {}) or {}
        cached_tokens = prompt_details.get('cached_tokens', 0)
        input_tokens = max(prompt_tokens - cached_tokens, 0)
        
        # Find matching pricing
        for key in self.PRICING:
            if key in model_lower:
                input_price, cached_price, output_price = self.PRICING[key]
                break
        else:
            input_price, cached_price, output_price = self.PRICING["default"]
        
        usd_cost = (
            (input_tokens / 1_000_000) * input_price +
            (cached_tokens / 1_000_000) * cached_price +
            (completion_tokens / 1_000_000) * output_price
        )
        
        return round(usd_cost, 6)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # Only add reasoning_effort if it's set and not None
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        if not response.ok:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = json.dumps(error_json, indent=2)
            except:
                pass
            raise requests.exceptions.HTTPError(
                f"{response.status_code} Error for {url}\nResponse: {error_detail}",
                response=response
            )
        result = response.json()
        
        usage = result.get("usage", {})
        
        return {
            "message": result["choices"][0]["message"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost": self.calculate_cost(usage),
            "raw_response": result,
        }
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        **kwargs
    ) -> tuple:
        """Generate response (compatibility method for eval script)."""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        cost = result["cost"]
        return content, cost
