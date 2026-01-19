"""
Token Counter Module - Utility for counting tokens in messages.
Uses tiktoken for accurate token counting compatible with OpenAI models.
"""

import tiktoken
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from dataclasses import dataclass


@dataclass
class TokenStats:
    """Statistics for token usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # Tokens saved by using cache

    def add(self, other: 'TokenStats') -> 'TokenStats':
        """Add another TokenStats to this one."""
        return TokenStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens
        )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens
        }


class TokenCounter:
    """
    Token counter for tracking token usage across conversations.
    """

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
    }

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the token counter.

        Args:
            model_name: The model name to use for encoding
        """
        self.model_name = model_name
        self._encoding = None
        self._load_encoding()

    def _load_encoding(self) -> None:
        """Load the appropriate encoding for the model."""
        try:
            # Try to get encoding for the specific model
            encoding_name = self.MODEL_ENCODINGS.get(self.model_name, "cl100k_base")
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to cl100k_base if model not found
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def count_message(self, message: BaseMessage) -> int:
        """
        Count tokens in a LangChain message.

        Args:
            message: A LangChain message object

        Returns:
            Number of tokens
        """
        tokens = 0

        # Count content tokens
        if message.content:
            tokens += self.count_text(str(message.content))

        # Count tool call tokens if present (for AIMessage)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                # Count tool name
                tokens += self.count_text(tool_call.get('name', ''))
                # Count arguments
                args = tool_call.get('args', {})
                if args:
                    import json
                    tokens += self.count_text(json.dumps(args))

        # Add overhead for message structure (role, etc.)
        tokens += 4  # Approximate overhead per message

        return tokens

    def count_messages(self, messages: List[BaseMessage]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of LangChain messages

        Returns:
            Total number of tokens
        """
        total = 0
        for msg in messages:
            total += self.count_message(msg)

        # Add base overhead for the conversation
        total += 3

        return total

    def count_dict_message(self, message: Dict[str, Any]) -> int:
        """
        Count tokens in a dictionary-format message.

        Args:
            message: A message in dict format with 'role' and 'content'

        Returns:
            Number of tokens
        """
        tokens = 0

        # Count role
        if 'role' in message:
            tokens += self.count_text(message['role'])

        # Count content
        if 'content' in message:
            tokens += self.count_text(str(message['content']))

        # Add overhead
        tokens += 4

        return tokens

    def estimate_cost(self, stats: TokenStats, model: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate the cost based on token usage.

        Args:
            stats: Token statistics
            model: Model name (uses self.model_name if not provided)

        Returns:
            Dictionary with cost breakdown
        """
        model = model or self.model_name

        # Pricing per 1M tokens (as of 2024)
        PRICING = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }

        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])

        input_cost = (stats.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (stats.completion_tokens / 1_000_000) * pricing["output"]
        saved_cost = (stats.cached_tokens / 1_000_000) * pricing["input"]

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(input_cost + output_cost, 6),
            "saved_cost": round(saved_cost, 6)
        }

    def format_token_display(self, stats: TokenStats) -> str:
        """
        Format token statistics for display.

        Args:
            stats: Token statistics

        Returns:
            Formatted string for display
        """
        parts = []
        parts.append(f"Total: {stats.total_tokens:,}")
        parts.append(f"Prompt: {stats.prompt_tokens:,}")
        parts.append(f"Completion: {stats.completion_tokens:,}")
        if stats.cached_tokens > 0:
            parts.append(f"Cached: {stats.cached_tokens:,}")
        return " | ".join(parts)


# Global instance for convenience
_default_counter: Optional[TokenCounter] = None


def get_token_counter(model_name: str = "gpt-4o-mini") -> TokenCounter:
    """Get or create a token counter instance."""
    global _default_counter
    if _default_counter is None or _default_counter.model_name != model_name:
        _default_counter = TokenCounter(model_name)
    return _default_counter
