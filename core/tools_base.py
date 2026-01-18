"""
Tools Module - The "Hands" of the Agent.
Provides the base interface and registry for all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
import json


@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        if self.success:
            return str(self.output)
        return f"Error: {self.error}"


class ToolRegistry:
    """
    Central registry for all available tools.
    Allows dynamic registration and discovery of tools.
    """
    _instance = None
    _tools: Dict[str, BaseTool] = {}
    _categories: Dict[str, List[str]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._categories = {}
        return cls._instance
    
    def register(self, tool: BaseTool, category: str = "general") -> None:
        """Register a tool with the registry."""
        self._tools[tool.name] = tool
        if category not in self._categories:
            self._categories[category] = []
        if tool.name not in self._categories[category]:
            self._categories[category].append(tool.name)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_categories(self) -> List[str]:
        """Get all tool categories."""
        return list(self._categories.keys())
    
    def list_tools(self) -> Dict[str, str]:
        """List all tools with their descriptions."""
        return {name: tool.description for name, tool in self._tools.items()}
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()


# Global registry instance
tool_registry = ToolRegistry()


def register_tool(category: str = "general"):
    """Decorator to register a tool with the registry."""
    def decorator(tool_instance: BaseTool) -> BaseTool:
        tool_registry.register(tool_instance, category)
        return tool_instance
    return decorator
