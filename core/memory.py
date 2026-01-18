"""
Memory Module - State Management for the Agent.
Handles short-term (conversation) and long-term memory.
"""

from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import json


class AgentState(TypedDict, total=False):
    """
    The state that flows through the agent graph.
    This represents the agent's working memory during task execution.
    """
    # Core message history
    messages: List[BaseMessage]
    
    # Current mission/goal
    mission: str
    
    # Planning state
    current_plan: List[str]
    current_step: int
    
    # Reasoning trace (for ReAct)
    thoughts: List[str]
    observations: List[str]
    actions: List[str]
    
    # Execution state
    iteration_count: int
    is_complete: bool
    final_answer: Optional[str]
    
    # Error handling
    error: Optional[str]
    
    # Metadata
    start_time: str
    tool_calls_made: int


@dataclass
class ThoughtStep:
    """Represents a single step in the ReAct reasoning process."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationMemory:
    """
    Short-term memory for the current conversation.
    Tracks the immediate context of the interaction.
    """
    messages: List[BaseMessage] = field(default_factory=list)
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def add_thought_step(self, step: ThoughtStep) -> None:
        """Add a reasoning step."""
        self.thought_steps.append(step)
    
    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        """Get the n most recent messages."""
        return self.messages[-n:]
    
    def get_reasoning_trace(self) -> str:
        """Get a formatted trace of all reasoning steps."""
        trace = []
        for i, step in enumerate(self.thought_steps, 1):
            trace.append(f"Step {i}:")
            trace.append(f"  Thought: {step.thought}")
            if step.action:
                trace.append(f"  Action: {step.action}")
                if step.action_input:
                    trace.append(f"  Input: {json.dumps(step.action_input)}")
            if step.observation:
                trace.append(f"  Observation: {step.observation}")
        return "\n".join(trace)
    
    def clear(self) -> None:
        """Clear all memory."""
        self.messages.clear()
        self.thought_steps.clear()


@dataclass
class LongTermMemory:
    """
    Long-term memory for persistent information across conversations.
    In a production system, this would be backed by a database or vector store.
    """
    facts: Dict[str, str] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    past_interactions: List[Dict[str, Any]] = field(default_factory=list)
    
    def store_fact(self, key: str, value: str) -> None:
        """Store a fact for later retrieval."""
        self.facts[key] = value
    
    def retrieve_fact(self, key: str) -> Optional[str]:
        """Retrieve a stored fact."""
        return self.facts.get(key)
    
    def store_preference(self, key: str, value: Any) -> None:
        """Store a user preference."""
        self.user_preferences[key] = value
    
    def log_interaction(self, summary: Dict[str, Any]) -> None:
        """Log an interaction summary."""
        summary["timestamp"] = datetime.now().isoformat()
        self.past_interactions.append(summary)
    
    def search_facts(self, query: str) -> List[str]:
        """Simple keyword search through facts."""
        query_lower = query.lower()
        results = []
        for key, value in self.facts.items():
            if query_lower in key.lower() or query_lower in value.lower():
                results.append(f"{key}: {value}")
        return results


def create_initial_state(mission: str) -> AgentState:
    """Create the initial state for an agent run."""
    return AgentState(
        messages=[HumanMessage(content=mission)],
        mission=mission,
        current_plan=[],
        current_step=0,
        thoughts=[],
        observations=[],
        actions=[],
        iteration_count=0,
        is_complete=False,
        final_answer=None,
        error=None,
        start_time=datetime.now().isoformat(),
        tool_calls_made=0
    )
