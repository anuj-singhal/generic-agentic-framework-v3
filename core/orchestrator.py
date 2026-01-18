"""
Orchestration Layer - The "Nervous System" of the Agent.
Manages the ReAct loop using LangGraph for state management.
"""

from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict, Any
from datetime import datetime
import json
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from core.memory import AgentState
from core.config import FrameworkConfig, get_config


# ReAct System Prompt
REACT_SYSTEM_PROMPT = """You are a helpful AI assistant operating in ReAct (Reasoning and Acting) mode.

For each user request, you follow this process:
1. **Think**: Analyze the request and reason about what needs to be done
2. **Act**: Choose and execute the appropriate tool(s)
3. **Observe**: Review the results of your actions
4. **Repeat**: Continue until the task is complete

Available Tools:
{tool_descriptions}

IMPORTANT GUIDELINES:
- Always think step-by-step before taking action
- Use tools when you need external information or to perform actions
- Be concise but thorough in your reasoning
- If you have enough information to answer, provide the final response
- If a tool call fails, try an alternative approach

When you have completed the task and have the final answer, respond with your complete answer to the user.
"""


# Define GraphState at module level so it can be referenced by all methods
class GraphState(TypedDict):
    """State that flows through the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    mission: str
    thoughts: List[str]
    observations: List[str]
    actions: List[str]
    iteration_count: int
    is_complete: bool
    final_answer: Optional[str]
    error: Optional[str]
    tool_calls_made: int


class ReActOrchestrator:
    """
    The orchestration layer that manages the ReAct loop.
    Uses LangGraph for state management and tool execution.
    """

    def __init__(
        self,
        tools: List,
        config: Optional[FrameworkConfig] = None,
        callbacks: Optional[List] = None,
        system_prompt: Optional[str] = None
    ):
        self.config = config or get_config()
        self.tools = tools
        self.callbacks = callbacks or []
        self.custom_system_prompt = system_prompt

        # Initialize the LLM (The Brain)
        self.llm = ChatOpenAI(
            model=self.config.model.model_name,
            temperature=self.config.model.temperature,
            api_key=self.config.model.api_key
        )

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

        # Build the graph
        self.graph = self._build_graph()
    
    def _get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(descriptions)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for ReAct."""
        
        # Create the graph using module-level GraphState
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Set entry point
        workflow.set_entry_point("reason")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "tools": "tools",
                "synthesize": "synthesize",
                "end": END
            }
        )
        
        # After tool execution, go back to reasoning
        workflow.add_edge("tools", "reason")
        
        # After synthesis, end
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _reason_node(self, state: GraphState) -> Dict[str, Any]:
        """
        The reasoning node - where the agent thinks.
        This is the core "Think" step of ReAct.
        """
        messages = state["messages"]
        iteration = state.get("iteration_count", 0)

        # Add system prompt if this is the first iteration
        if iteration == 0:
            # Use custom system prompt if provided, otherwise use default
            if self.custom_system_prompt:
                # Combine custom prompt with tool descriptions
                tool_info = f"\n\nAvailable Tools:\n{self._get_tool_descriptions()}"
                system_prompt = self.custom_system_prompt + tool_info
            else:
                system_prompt = REACT_SYSTEM_PROMPT.format(
                    tool_descriptions=self._get_tool_descriptions()
                )
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        # Call the LLM
        response = self.llm_with_tools.invoke(messages)

        # Update state
        new_thoughts = list(state.get("thoughts", []))
        content_preview = response.content[:200] if response.content else "[Tool Call]"
        new_thoughts.append(f"Iteration {iteration + 1}: {content_preview}...")

        return {
            "messages": [response],
            "thoughts": new_thoughts,
            "iteration_count": iteration + 1
        }
    
    def _should_continue(self, state: GraphState) -> Literal["tools", "synthesize", "end"]:
        """
        Determine the next step based on the current state.
        """
        messages = state["messages"]
        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)
        
        # Check iteration limit
        if iteration >= self.config.max_iterations:
            return "synthesize"
        
        # Check if the LLM wants to use tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # If no tool calls, we have a final answer
        return "end"
    
    def _synthesize_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Synthesize a final answer when max iterations reached.
        """
        messages = state["messages"]
        
        synthesis_prompt = """Based on all the work done so far, provide a final comprehensive answer to the user's original request. 
Summarize the key findings and conclusions."""
        
        messages_with_synthesis = list(messages) + [HumanMessage(content=synthesis_prompt)]
        response = self.llm.invoke(messages_with_synthesis)
        
        return {
            "messages": [response],
            "is_complete": True,
            "final_answer": response.content
        }
    
    def run(self, mission: str) -> Dict[str, Any]:
        """
        Execute the agent with a given mission.
        
        Args:
            mission: The user's request/goal
            
        Returns:
            The final state including the answer and execution trace
        """
        initial_state: GraphState = {
            "messages": [HumanMessage(content=mission)],
            "mission": mission,
            "thoughts": [],
            "observations": [],
            "actions": [],
            "iteration_count": 0,
            "is_complete": False,
            "final_answer": None,
            "error": None,
            "tool_calls_made": 0
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            return {
                **initial_state,
                "error": str(e),
                "is_complete": True
            }
    
    def stream(self, mission: str):
        """
        Stream the agent execution for real-time updates.
        
        Args:
            mission: The user's request/goal
            
        Yields:
            State updates as they occur
        """
        initial_state: GraphState = {
            "messages": [HumanMessage(content=mission)],
            "mission": mission,
            "thoughts": [],
            "observations": [],
            "actions": [],
            "iteration_count": 0,
            "is_complete": False,
            "final_answer": None,
            "error": None,
            "tool_calls_made": 0
        }
        
        try:
            for state in self.graph.stream(initial_state):
                yield state
        except Exception as e:
            yield {"error": str(e)}
