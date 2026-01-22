"""
Multi-Agent Orchestrator for Data Analysis
==========================================

Implements a sophisticated 6-agent workflow using LangGraph for complex SQL query
generation with multi-level validation and retry logic.

Agent Workflow:
1. Agent1 (Pre-validation): Validates if query is data-related
2. Agent2 (Schema Extraction): Extracts relevant schema (tool-based, no LLM)
3. Agent3 (Query Orchestrator): Determines simple vs complex query, creates subtasks
4. Agent4 (SQL Generator): Generates SQL (4.1 simple, 4.2 complex with CTEs)
5. Agent5 (Validator): Multi-level validation with retry logic
6. Agent6 (Executor): Executes SQL and presents results
"""

from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict, Any
from datetime import datetime
import json
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from core.config import FrameworkConfig, get_config
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


# =============================================================================
# QUERY CACHE FOR FOLLOW-UP DETECTION
# =============================================================================

@dataclass
class CachedQuery:
    """Represents a cached query with its SQL and results."""
    nl_query: str
    generated_sql: str
    query_results: str
    timestamp: str
    tables_used: List[str] = field(default_factory=list)


class QueryCache:
    """
    Cache for storing the last N queries for follow-up detection.
    Uses a deque for efficient FIFO operations.
    """
    _instance = None
    _max_size: int = 3

    def __new__(cls):
        """Singleton pattern to ensure one cache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: Deque[CachedQuery] = deque(maxlen=cls._max_size)
        return cls._instance

    def add_query(self, nl_query: str, generated_sql: str, query_results: str, tables_used: List[str] = None):
        """Add a new query to the cache."""
        cached = CachedQuery(
            nl_query=nl_query,
            generated_sql=generated_sql,
            query_results=query_results,
            timestamp=datetime.now().isoformat(),
            tables_used=tables_used or []
        )
        self._cache.append(cached)

    def get_recent_queries(self) -> List[CachedQuery]:
        """Get all cached queries, most recent last."""
        return list(self._cache)

    def get_context_for_followup(self) -> str:
        """Get formatted context string for follow-up detection."""
        if not self._cache:
            return "No previous queries in this session."

        context_parts = []
        for i, cached in enumerate(self._cache, 1):
            # Truncate results if too long
            results_preview = cached.query_results[:500] + "..." if len(cached.query_results) > 500 else cached.query_results
            context_parts.append(f"""
--- Previous Query {i} ---
Question: {cached.nl_query}
SQL Generated: {cached.generated_sql}
Results Preview: {results_preview}
Tables Used: {', '.join(cached.tables_used) if cached.tables_used else 'Unknown'}
""")
        return "\n".join(context_parts)

    def get_last_query(self) -> Optional[CachedQuery]:
        """Get the most recent cached query."""
        return self._cache[-1] if self._cache else None

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# Global cache instance
query_cache = QueryCache()


def format_sql(sql: str) -> str:
    """
    Format SQL query for better readability.
    Adds proper indentation and line breaks.
    """
    if not sql:
        return sql

    # Clean up the SQL first
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace

    # Keywords that should start on a new line (with proper spacing)
    newline_before = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
        'LIMIT', 'OFFSET', 'UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT',
        'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN', 'FULL JOIN',
        'CROSS JOIN', 'JOIN', 'ON', 'AND', 'OR'
    ]

    # Sort by length (longest first) to avoid partial matches
    newline_before.sort(key=len, reverse=True)

    # First, handle CTEs specially
    # Pattern: WITH name AS ( ... ), name2 AS ( ... ) SELECT
    if re.search(r'\bWITH\b', sql, re.IGNORECASE):
        # Split into CTE part and main query
        # Find the final SELECT that's not inside a CTE
        parts = re.split(r'\)\s*(?=SELECT\s+(?!.*\bAS\s*\())', sql, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 2:
            cte_part = parts[0] + ')'
            main_query = parts[1]

            # Format CTE part
            # Handle WITH keyword
            cte_part = re.sub(r'\bWITH\b', '\nWITH', cte_part, flags=re.IGNORECASE)

            # Handle each CTE definition: name AS (
            cte_part = re.sub(r'(\w+)\s+AS\s*\(', r'\n    \1 AS (\n        ', cte_part, flags=re.IGNORECASE)

            # Handle CTE separators: ), name AS (
            cte_part = re.sub(r'\)\s*,\s*(\w+)\s+AS\s*\(', r'\n    ),\n    \1 AS (\n        ', cte_part, flags=re.IGNORECASE)

            # Add newlines for keywords inside CTEs
            for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'JOIN', 'LEFT JOIN', 'AND']:
                cte_part = re.sub(r'(?<!\w)(' + kw + r')(?!\w)', r'\n        \1', cte_part, flags=re.IGNORECASE)

            # Close final CTE
            cte_part = re.sub(r'\)\s*$', '\n    )', cte_part)

            # Format main query
            for kw in newline_before:
                main_query = re.sub(r'(?<!\w)(' + kw + r')(?!\w)', r'\n\1', main_query, flags=re.IGNORECASE)

            sql = cte_part + '\n' + main_query
        else:
            # Fallback: just format normally
            for kw in newline_before:
                sql = re.sub(r'(?<!\w)(' + kw + r')(?!\w)', r'\n\1', sql, flags=re.IGNORECASE)
    else:
        # No CTEs, just format with newlines before keywords
        for kw in newline_before:
            sql = re.sub(r'(?<!\w)(' + kw + r')(?!\w)', r'\n\1', sql, flags=re.IGNORECASE)

    # Clean up
    lines = [line.rstrip() for line in sql.split('\n')]
    lines = [line for line in lines if line.strip()]  # Remove empty lines

    # Remove leading newline if present
    result = '\n'.join(lines).strip()

    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


# Load schema document path
SCHEMA_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "sample_files", "wealth_tables.json")


class AgentTrace(TypedDict):
    """Trace information for a single agent step."""
    agent_id: str
    agent_name: str
    status: str  # "running", "completed", "skipped"
    input_summary: str
    output_summary: str
    details: Dict[str, Any]
    timestamp: str


class MultiAgentState(TypedDict):
    """State that flows through the multi-agent graph."""
    # Core message history
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Original user query
    nl_query: str

    # Agent1 outputs
    is_data_query: bool
    general_answer: Optional[str]
    related_tables: List[str]
    relationships: List[str]

    # Query Intent Detection
    is_modified_query: bool  # User wants to modify/filter previous query (needs new SQL)
    is_followup_question: bool  # User asks question about previous results (answer from cache)
    followup_context: Optional[Dict[str, Any]]  # Contains previous query info
    previous_sql: Optional[str]  # SQL from previous query
    previous_results: Optional[str]  # Results from previous query
    followup_answer: Optional[str]  # Direct answer from cache for followup questions

    # Agent2 outputs
    extracted_schema: Dict[str, Any]

    # Agent3 outputs
    query_complexity: str  # "SIMPLE" or "COMPLEX"
    subtasks: List[str]

    # Agent4 outputs
    generated_sql: str
    sub_queries: List[Dict[str, str]]  # For complex queries

    # Agent5 outputs
    validation_results: Dict[str, Any]
    overall_confidence: float
    validation_issues: List[str]
    validation_suggestions: List[str]

    # Retry logic
    retry_count: int
    max_retries: int

    # Agent6 outputs
    query_results: str
    final_answer: str

    # Execution state
    current_agent: str
    is_complete: bool
    error: Optional[str]

    # Trace information for UI display
    agent_traces: List[AgentTrace]


class MultiAgentDataOrchestrator:
    """
    Multi-Agent Data Orchestrator implementing a 6-agent workflow
    for sophisticated SQL query generation and validation.
    """

    # Agent definitions for trace display
    AGENT_INFO = {
        "agent1": {"name": "Intent Classification Agent", "icon": "🔍"},
        "agent2": {"name": "Schema Extraction Agent", "icon": "📋"},
        "agent3": {"name": "Query Orchestrator Agent", "icon": "🎯"},
        "agent4": {"name": "SQL Generator Agent", "icon": "⚙️"},
        "agent5": {"name": "Validation Agent", "icon": "✅"},
        "agent5.1": {"name": "Syntax Validator", "icon": "📝"},
        "agent5.2": {"name": "Schema Validator", "icon": "🗄️"},
        "agent5.3": {"name": "Semantic Validator", "icon": "🎯"},
        "retry": {"name": "Retry Handler", "icon": "🔄"},
        "agent6": {"name": "Executor Agent", "icon": "🚀"},
    }

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        callbacks: Optional[List] = None
    ):
        self.config = config or get_config()
        self.callbacks = callbacks or []

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=self.config.model.model_name,
            temperature=self.config.model.temperature,
            api_key=self.config.model.api_key
        )

        # Load schema document
        self.schema_document = self._load_schema_document()

        # Build the graph
        self.graph = self._build_graph()

    def _create_trace(
        self,
        agent_id: str,
        status: str,
        input_summary: str,
        output_summary: str,
        details: Dict[str, Any] = None
    ) -> AgentTrace:
        """Create a trace entry for an agent step."""
        agent_info = self.AGENT_INFO.get(agent_id, {"name": agent_id, "icon": "🔷"})
        return AgentTrace(
            agent_id=agent_id,
            agent_name=f"{agent_info['icon']} {agent_info['name']}",
            status=status,
            input_summary=input_summary,
            output_summary=output_summary,
            details=details or {},
            timestamp=datetime.now().isoformat()
        )

    def _add_trace(self, state: MultiAgentState, trace: AgentTrace) -> List[AgentTrace]:
        """Add a trace to the state's trace list."""
        traces = list(state.get("agent_traces", []))
        traces.append(trace)
        return traces

    def _load_schema_document(self) -> Dict[str, Any]:
        """Load the schema definition document."""
        try:
            if os.path.exists(SCHEMA_DOCUMENT_PATH):
                with open(SCHEMA_DOCUMENT_PATH, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Warning: Could not load schema document: {e}")
            return {}

    def _get_schema_summary(self) -> str:
        """Get a summary of all tables and their descriptions."""
        if not self.schema_document or "tables" not in self.schema_document:
            return "No schema document available."

        summary = []
        for table in self.schema_document["tables"]:
            summary.append(f"- {table['table_name']}: {table['table_description']}")
            columns = [f"  - {col['column_name']}: {col['description']}"
                      for col in table['columns'][:5]]  # First 5 columns
            summary.extend(columns)
            if len(table['columns']) > 5:
                summary.append(f"  ... and {len(table['columns']) - 5} more columns")

        return "\n".join(summary)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for multi-agent workflow."""
        workflow = StateGraph(MultiAgentState)

        # Add nodes for each agent
        workflow.add_node("agent1_prevalidation", self._agent1_prevalidation)
        workflow.add_node("agent2_schema_extraction", self._agent2_schema_extraction)
        workflow.add_node("agent3_orchestrator", self._agent3_orchestrator)
        workflow.add_node("agent4_sql_generator", self._agent4_sql_generator)
        workflow.add_node("agent5_validator", self._agent5_validator)
        workflow.add_node("handle_retry", self._handle_retry)  # Retry handler node
        workflow.add_node("agent6_executor", self._agent6_executor)
        workflow.add_node("final_response", self._final_response_node)

        # Set entry point
        workflow.set_entry_point("agent1_prevalidation")

        # Add conditional edges from agent1
        workflow.add_conditional_edges(
            "agent1_prevalidation",
            self._route_after_prevalidation,
            {
                "schema_extraction": "agent2_schema_extraction",
                "end": "final_response"
            }
        )

        # Linear flow from agent2 to agent3
        workflow.add_edge("agent2_schema_extraction", "agent3_orchestrator")

        # Agent3 always goes to agent4
        workflow.add_edge("agent3_orchestrator", "agent4_sql_generator")

        # Agent4 goes to agent5
        workflow.add_edge("agent4_sql_generator", "agent5_validator")

        # Conditional edges from agent5 (validation)
        workflow.add_conditional_edges(
            "agent5_validator",
            self._route_after_validation,
            {
                "execute": "agent6_executor",
                "regenerate": "handle_retry",  # Route through retry handler first
                "fail": "final_response"
            }
        )

        # Retry handler goes back to agent4
        workflow.add_edge("handle_retry", "agent4_sql_generator")

        # Agent6 goes to final response
        workflow.add_edge("agent6_executor", "final_response")

        # Final response ends
        workflow.add_edge("final_response", END)

        return workflow.compile()

    # =========================================================================
    # AGENT 1: Intent Classification Agent
    # =========================================================================
    def _agent1_prevalidation(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent1: Intent Classification Agent
        Classifies user query into one of four intents:
        1. NEW_DATA_QUERY: Fresh query requiring SQL generation
        2. MODIFIED_QUERY: Modify/filter previous query (needs new SQL with context)
        3. FOLLOWUP_QUESTION: Question about previous results (answer from cache)
        4. GENERAL_QUESTION: Non-data question (answer directly)
        """
        nl_query = state["nl_query"]
        schema_summary = self._get_schema_summary()

        # Get previous query context from cache
        previous_context = query_cache.get_context_for_followup()
        last_query = query_cache.get_last_query()
        has_previous_queries = len(query_cache) > 0

        # Build the prompt with context
        cache_section = ""
        if has_previous_queries:
            cache_section = f"""
PREVIOUS QUERIES IN THIS SESSION (for context):
{previous_context}
"""

        prompt = f"""You are an intent classification agent. Analyze the user's query and classify it into ONE of these intents:

1. **NEW_DATA_QUERY**: A fresh data query that needs SQL generation
   - No reference to previous queries
   - Asking for specific data from the database
   - Examples: "Show me all clients", "List portfolios with value > 1M"

2. **MODIFIED_QUERY**: User wants to modify/filter/extend the previous query (needs NEW SQL)
   - References previous query but needs different/additional data
   - Wants to filter, sort, group, or expand previous results
   - Examples: "Show only the ones from UAE", "Add their email addresses", "Sort by value descending"

3. **FOLLOWUP_QUESTION**: User asks a question ABOUT the previous results (answer from CACHE, no new SQL)
   - Asking for analysis, summary, or explanation of existing results
   - Can be answered by looking at the cached query results
   - Examples: "What's the total?", "Which one has the highest value?", "How many are there?", "Summarize this"

4. **GENERAL_QUESTION**: Non-data question that doesn't need database access
   - General knowledge questions
   - Questions about SQL syntax or concepts
   - Examples: "What is SQL?", "How do JOINs work?"

AVAILABLE DATABASE SCHEMA:
{schema_summary}
{cache_section}
CURRENT USER QUERY: {nl_query}

Respond in JSON format:
{{
    "intent": "NEW_DATA_QUERY" | "MODIFIED_QUERY" | "FOLLOWUP_QUESTION" | "GENERAL_QUESTION",
    "reasoning": "Why you classified it this way",
    "related_tables": ["TABLE1", "TABLE2"],
    "relationships": ["TABLE1.COL -> TABLE2.COL"],
    "general_answer": "If GENERAL_QUESTION, provide the answer here. Otherwise null.",
    "followup_answer": "If FOLLOWUP_QUESTION, analyze the cached results and provide answer here. Otherwise null.",
    "modification_type": "filter/aggregate/sort/expand/new"
}}

IMPORTANT GUIDELINES:
- FOLLOWUP_QUESTION: The answer must come from analyzing the CACHED RESULTS shown above. Do NOT suggest running new SQL.
- MODIFIED_QUERY: Requires generating NEW SQL that builds upon the previous query context.
- For MODIFIED_QUERY, include ALL relevant tables (previous + new).
- If no previous queries exist, cannot be MODIFIED_QUERY or FOLLOWUP_QUESTION.

Respond ONLY with the JSON, no additional text."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content.strip())
            intent = result.get("intent", "NEW_DATA_QUERY")
            reasoning = result.get("reasoning", "")
            related_tables = result.get("related_tables", [])
            relationships = result.get("relationships", [])
            general_answer = result.get("general_answer")
            followup_answer = result.get("followup_answer")
            modification_type = result.get("modification_type", "new")

        except json.JSONDecodeError:
            intent = "NEW_DATA_QUERY"
            reasoning = "Failed to parse response, defaulting to new data query"
            related_tables = []
            relationships = []
            general_answer = None
            followup_answer = None
            modification_type = "new"

        # Determine intent flags
        is_data_query = intent in ["NEW_DATA_QUERY", "MODIFIED_QUERY"]
        is_modified_query = intent == "MODIFIED_QUERY"
        is_followup_question = intent == "FOLLOWUP_QUESTION"

        # Build context for modified queries
        followup_context = None
        previous_sql = None
        previous_results = None

        if (is_modified_query or is_followup_question) and last_query:
            followup_context = {
                "previous_nl_query": last_query.nl_query,
                "previous_sql": last_query.generated_sql,
                "previous_results": last_query.query_results,
                "tables_used": last_query.tables_used,
                "modification_type": modification_type
            }
            previous_sql = last_query.generated_sql
            previous_results = last_query.query_results

        # Create output summary based on intent
        if intent == "NEW_DATA_QUERY":
            output_summary = f"New data query. Related tables: {', '.join(related_tables) if related_tables else 'analyzing...'}"
        elif intent == "MODIFIED_QUERY":
            output_summary = f"Modified query ({modification_type}). Related tables: {', '.join(related_tables) if related_tables else 'analyzing...'}"
        elif intent == "FOLLOWUP_QUESTION":
            output_summary = "Follow-up question - answering from cached results"
        else:
            output_summary = "General question - answering directly"

        trace = self._create_trace(
            agent_id="agent1",
            status="completed",
            input_summary=f"Query: {nl_query[:100]}{'...' if len(nl_query) > 100 else ''}",
            output_summary=output_summary,
            details={
                "intent": intent,
                "is_data_query": is_data_query,
                "is_modified_query": is_modified_query,
                "is_followup_question": is_followup_question,
                "modification_type": modification_type if is_modified_query else None,
                "related_tables": related_tables,
                "reasoning": reasoning,
                "cached_queries_count": len(query_cache)
            }
        )

        # Set final answer for non-SQL intents
        final_general_answer = None
        if intent == "GENERAL_QUESTION":
            final_general_answer = general_answer
        elif intent == "FOLLOWUP_QUESTION":
            final_general_answer = followup_answer

        return {
            "messages": [response],
            "is_data_query": is_data_query,
            "is_modified_query": is_modified_query,
            "is_followup_question": is_followup_question,
            "followup_context": followup_context,
            "previous_sql": previous_sql,
            "previous_results": previous_results,
            "followup_answer": followup_answer,
            "general_answer": final_general_answer,
            "related_tables": related_tables,
            "relationships": relationships,
            "current_agent": "agent1_prevalidation",
            "agent_traces": self._add_trace(state, trace)
        }

    def _route_after_prevalidation(self, state: MultiAgentState) -> Literal["schema_extraction", "end"]:
        """Route after pre-validation based on query intent."""
        # Follow-up questions are answered from cache - skip SQL generation
        if state.get("is_followup_question", False):
            return "end"
        # Data queries (new or modified) need SQL generation
        if state.get("is_data_query", True):
            return "schema_extraction"
        # General questions don't need SQL
        return "end"

    # =========================================================================
    # AGENT 2: Schema Extraction Agent (Tool-based, no LLM)
    # =========================================================================
    def _agent2_schema_extraction(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent2: Schema Extraction Agent
        - Extracts relevant table descriptions, column descriptions, and sample data
        - Tool-based extraction, no LLM needed
        """
        related_tables = state.get("related_tables", [])

        # If no tables identified, get all tables
        if not related_tables:
            related_tables = [t["table_name"] for t in self.schema_document.get("tables", [])]

        extracted_schema = {
            "tables": {},
            "relationships": []
        }

        # Extract schema for each related table
        total_columns = 0
        for table_info in self.schema_document.get("tables", []):
            table_name = table_info["table_name"]
            if table_name in related_tables or not related_tables:
                extracted_schema["tables"][table_name] = {
                    "description": table_info.get("table_description", ""),
                    "columns": []
                }

                for col in table_info.get("columns", []):
                    extracted_schema["tables"][table_name]["columns"].append({
                        "name": col.get("column_name", ""),
                        "data_type": col.get("data_type", ""),
                        "description": col.get("description", ""),
                        "is_primary_key": col.get("is_primary_key", False),
                        "nullable": col.get("nullable", True),
                        "sample_values": col.get("sample_values", [])
                    })
                    total_columns += 1

        # Add relationships from state
        extracted_schema["relationships"] = state.get("relationships", [])

        # Add standard relationships if not provided
        if not extracted_schema["relationships"]:
            extracted_schema["relationships"] = [
                "PORTFOLIOS.CLIENT_ID -> CLIENTS.CLIENT_ID",
                "TRANSACTIONS.PORTFOLIO_ID -> PORTFOLIOS.PORTFOLIO_ID",
                "TRANSACTIONS.ASSET_ID -> ASSETS.ASSET_ID",
                "HOLDINGS.PORTFOLIO_ID -> PORTFOLIOS.PORTFOLIO_ID",
                "HOLDINGS.ASSET_ID -> ASSETS.ASSET_ID"
            ]

        # Create trace for this agent
        tables_extracted = list(extracted_schema["tables"].keys())
        trace = self._create_trace(
            agent_id="agent2",
            status="completed",
            input_summary=f"Tables to extract: {', '.join(related_tables) if related_tables else 'all tables'}",
            output_summary=f"Extracted {len(tables_extracted)} tables with {total_columns} columns",
            details={
                "tables_extracted": tables_extracted,
                "total_columns": total_columns,
                "relationships_count": len(extracted_schema["relationships"]),
                "note": "No LLM used - direct schema extraction from document"
            }
        )

        return {
            "extracted_schema": extracted_schema,
            "current_agent": "agent2_schema_extraction",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 3: Query Orchestrator Agent
    # =========================================================================
    def _agent3_orchestrator(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent3: Query Orchestrator Agent
        - Determines if query is SIMPLE or COMPLEX
        - For complex queries, breaks down into subtasks
        """
        nl_query = state["nl_query"]
        extracted_schema = state.get("extracted_schema", {})

        schema_str = json.dumps(extracted_schema, indent=2)

        prompt = f"""You are a SQL query orchestrator agent. Your job is to analyze the user query and determine its complexity.

USER QUERY: {nl_query}

AVAILABLE SCHEMA:
{schema_str}

COMPLEXITY DEFINITIONS:
- SIMPLE: Query can be answered with a single task using:
  - Single table queries
  - Simple JOINs (1-2 tables)
  - Basic aggregations (COUNT, SUM, AVG) with GROUP BY
  - Simple filtering with WHERE clauses

- COMPLEX: Query requires multiple tasks or advanced SQL:
  - Multiple aggregations with different groupings
  - Subqueries or CTEs (Common Table Expressions)
  - Multiple JOINs (3+ tables)
  - Window functions
  - Calculations involving multiple steps
  - Conditional logic (CASE statements)
  - Multiple questions in one query

Analyze the query and respond in JSON format:
{{
    "complexity": "SIMPLE" or "COMPLEX",
    "reasoning": "Why you classified it this way",
    "subtasks": [
        "If COMPLEX: First subtask description",
        "Second subtask description",
        "etc."
    ],
    "final_synthesis": "If COMPLEX: How to combine subtasks into final CTE query"
}}

For SIMPLE queries, subtasks should be empty or contain just the main task.
For COMPLEX queries, break down into logical subtasks that can be combined with CTEs.

Respond ONLY with the JSON, no additional text."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content.strip())
            complexity = result.get("complexity", "SIMPLE")
            subtasks = result.get("subtasks", [])
            reasoning = result.get("reasoning", "")
        except json.JSONDecodeError:
            complexity = "SIMPLE"
            subtasks = []
            reasoning = "Failed to parse response"

        # Create trace for this agent
        if complexity == "SIMPLE":
            output_summary = "Query classified as SIMPLE - single SQL statement will be generated"
        else:
            output_summary = f"Query classified as COMPLEX - {len(subtasks)} subtasks identified for CTE generation"

        trace = self._create_trace(
            agent_id="agent3",
            status="completed",
            input_summary=f"Analyzing complexity for: {nl_query[:80]}{'...' if len(nl_query) > 80 else ''}",
            output_summary=output_summary,
            details={
                "complexity": complexity,
                "subtasks": subtasks,
                "reasoning": reasoning
            }
        )

        return {
            "messages": [response],
            "query_complexity": complexity,
            "subtasks": subtasks,
            "current_agent": "agent3_orchestrator",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 4: SQL Generator Agent
    # =========================================================================
    def _agent4_sql_generator(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent4: SQL Generator Agent
        - Agent4.1: Simple query generation
        - Agent4.2: Complex query generation with CTE synthesis
        """
        nl_query = state["nl_query"]
        extracted_schema = state.get("extracted_schema", {})
        complexity = state.get("query_complexity", "SIMPLE")
        subtasks = state.get("subtasks", [])
        retry_count = state.get("retry_count", 0)

        # Include validation feedback if this is a retry
        validation_feedback = ""
        if retry_count > 0:
            issues = state.get("validation_issues", [])
            suggestions = state.get("validation_suggestions", [])
            if issues or suggestions:
                validation_feedback = f"""

IMPORTANT - PREVIOUS QUERY FAILED VALIDATION. Please fix these issues:
ISSUES:
{chr(10).join(f'- {issue}' for issue in issues)}

SUGGESTIONS:
{chr(10).join(f'- {suggestion}' for suggestion in suggestions)}

Generate a CORRECTED query that addresses all the above issues."""

        schema_str = json.dumps(extracted_schema, indent=2)

        if complexity == "SIMPLE":
            result = self._generate_simple_query(nl_query, schema_str, validation_feedback, state, retry_count)
        else:
            result = self._generate_complex_query(nl_query, schema_str, subtasks, validation_feedback, state, retry_count)

        return result

    def _generate_simple_query(self, nl_query: str, schema_str: str,
                                validation_feedback: str, state: MultiAgentState,
                                retry_count: int) -> Dict[str, Any]:
        """Agent 4.1: Generate simple SQL query."""

        # Build follow-up context section if this is a follow-up query
        followup_section = ""
        followup_context = state.get("followup_context")
        if followup_context and state.get("is_modified_query", False):
            previous_sql = followup_context.get("previous_sql", "")
            previous_results = followup_context.get("previous_results_preview", "")
            modification_type = followup_context.get("modification_type", "")
            previous_nl = followup_context.get("previous_nl_query", "")

            followup_section = f"""

FOLLOW-UP QUERY CONTEXT:
This is a follow-up to a previous query. Use this context to generate a more accurate query.

Previous Question: {previous_nl}
Previous SQL:
{previous_sql}

Previous Results Preview:
{previous_results[:800]}{'...' if len(previous_results) > 800 else ''}

Modification Type: {modification_type}
- If "filter": Add additional WHERE clauses to filter the previous results
- If "aggregate": Add aggregations (SUM, COUNT, AVG) to the query
- If "detail": Expand the query to include more columns or related data

IMPORTANT: Build upon or modify the previous query based on the user's new request.
"""

        prompt = f"""You are an expert SQL generator. Generate a SQL query for the following request.

USER QUERY: {nl_query}

AVAILABLE SCHEMA:
{schema_str}
{followup_section}{validation_feedback}

Generate a SQL query that:
1. Uses the correct table and column names from the schema
2. Includes proper JOINs if multiple tables are needed
3. Uses appropriate WHERE clauses for filtering
4. Includes GROUP BY with aggregate functions if needed
5. Has meaningful column aliases
6. Is syntactically correct for DuckDB

Respond in JSON format:
{{
    "sql": "Your SQL query here",
    "explanation": "Brief explanation of what the query does"
}}

Respond ONLY with the JSON, no additional text."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content.strip())
            sql = result.get("sql", "")
            explanation = result.get("explanation", "")
        except json.JSONDecodeError:
            # Try to extract SQL from response
            sql = response.content.strip()
            explanation = ""
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()

        # Format SQL for readability
        sql = format_sql(sql)

        # Create trace for this agent
        agent_type = "4.1 (Simple)" if retry_count == 0 else f"4.1 (Simple - Retry {retry_count})"
        trace = self._create_trace(
            agent_id="agent4",
            status="completed",
            input_summary=f"Generating simple SQL for: {nl_query[:60]}{'...' if len(nl_query) > 60 else ''}",
            output_summary=f"Generated SQL ({len(sql)} chars){' - RETRY' if retry_count > 0 else ''}",
            details={
                "agent_type": agent_type,
                "sql_preview": sql,  # Full formatted SQL
                "explanation": explanation,
                "retry_count": retry_count,
                "has_validation_feedback": bool(validation_feedback)
            }
        )

        return {
            "messages": [response],
            "generated_sql": sql,
            "sub_queries": [],
            "current_agent": "agent4_sql_generator",
            "agent_traces": self._add_trace(state, trace)
        }

    def _generate_complex_query(self, nl_query: str, schema_str: str,
                                 subtasks: List[str],
                                 validation_feedback: str, state: MultiAgentState,
                                 retry_count: int) -> Dict[str, Any]:
        """Agent 4.2: Generate complex SQL query with CTEs."""
        subtasks_str = "\n".join(f"{i+1}. {task}" for i, task in enumerate(subtasks))

        # Build follow-up context section if this is a follow-up query
        followup_section = ""
        followup_context = state.get("followup_context")
        if followup_context and state.get("is_modified_query", False):
            previous_sql = followup_context.get("previous_sql", "")
            previous_results = followup_context.get("previous_results_preview", "")
            modification_type = followup_context.get("modification_type", "")
            previous_nl = followup_context.get("previous_nl_query", "")

            followup_section = f"""

FOLLOW-UP QUERY CONTEXT:
This is a follow-up to a previous query. Use this context to generate a more accurate query.

Previous Question: {previous_nl}
Previous SQL:
{previous_sql}

Previous Results Preview:
{previous_results[:800]}{'...' if len(previous_results) > 800 else ''}

Modification Type: {modification_type}
IMPORTANT: Build upon or modify the previous query based on the user's new request. Consider incorporating the previous SQL as a CTE if useful.
"""

        prompt = f"""You are an expert SQL generator specializing in complex queries with CTEs.

USER QUERY: {nl_query}

SUBTASKS TO ADDRESS:
{subtasks_str}

AVAILABLE SCHEMA:
{schema_str}
{followup_section}{validation_feedback}

Generate a complex SQL query using CTEs (Common Table Expressions) that:
1. Creates separate CTEs for each logical subtask
2. Uses proper JOINs between CTEs
3. Synthesizes all subtasks into a final result
4. Uses meaningful CTE and column names
5. Is syntactically correct for DuckDB

STRUCTURE EXAMPLE:
WITH
    subtask_1 AS (
        -- Query for subtask 1
    ),
    subtask_2 AS (
        -- Query for subtask 2
    )
SELECT ...
FROM subtask_1
JOIN subtask_2 ON ...

Respond in JSON format:
{{
    "sql": "Your complete CTE query here",
    "sub_queries": [
        {{"name": "cte_name_1", "purpose": "What this CTE does", "sql": "SELECT ..."}},
        {{"name": "cte_name_2", "purpose": "What this CTE does", "sql": "SELECT ..."}}
    ],
    "explanation": "How the CTEs combine to answer the query"
}}

Respond ONLY with the JSON, no additional text."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content.strip())
            sql = result.get("sql", "")
            sub_queries = result.get("sub_queries", [])
            explanation = result.get("explanation", "")
        except json.JSONDecodeError:
            sql = response.content.strip()
            explanation = ""
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            sub_queries = []

        # Format SQL for readability
        sql = format_sql(sql)

        # Create trace for this agent
        agent_type = "4.2 (Complex/CTE)" if retry_count == 0 else f"4.2 (Complex/CTE - Retry {retry_count})"
        trace = self._create_trace(
            agent_id="agent4",
            status="completed",
            input_summary=f"Generating complex CTE query with {len(subtasks)} subtasks",
            output_summary=f"Generated CTE query with {len(sub_queries)} CTEs{' - RETRY' if retry_count > 0 else ''}",
            details={
                "agent_type": agent_type,
                "subtasks": subtasks,
                "cte_count": len(sub_queries),
                "sql_preview": sql,  # Full formatted SQL
                "explanation": explanation,
                "retry_count": retry_count,
                "has_validation_feedback": bool(validation_feedback)
            }
        )

        return {
            "messages": [response],
            "generated_sql": sql,
            "sub_queries": sub_queries,
            "current_agent": "agent4_sql_generator",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 5: Validation Agent
    # =========================================================================
    def _agent5_validator(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent5: Multi-level Validation Agent
        - Agent5.1: Syntax validation
        - Agent5.2: Schema validation
        - Agent5.3: Semantic validation
        """
        generated_sql = state.get("generated_sql", "")
        nl_query = state["nl_query"]
        extracted_schema = state.get("extracted_schema", {})
        retry_count = state.get("retry_count", 0)

        schema_str = json.dumps(extracted_schema, indent=2)

        # Get current traces
        traces = list(state.get("agent_traces", []))

        # Run all three validations with individual traces
        syntax_result = self._validate_syntax(generated_sql, schema_str)
        syntax_trace = self._create_trace(
            agent_id="agent5.1",
            status="completed",
            input_summary="Validating SQL syntax",
            output_summary=f"Syntax: {'VALID' if syntax_result.get('is_valid') else 'INVALID'} (conf: {syntax_result.get('confidence', 0):.0%})",
            details=syntax_result
        )
        traces.append(syntax_trace)

        schema_result = self._validate_schema(generated_sql, schema_str)
        schema_trace = self._create_trace(
            agent_id="agent5.2",
            status="completed",
            input_summary="Validating schema references",
            output_summary=f"Schema: {'VALID' if schema_result.get('is_valid') else 'INVALID'} (conf: {schema_result.get('confidence', 0):.0%})",
            details=schema_result
        )
        traces.append(schema_trace)

        semantic_result = self._validate_semantic(generated_sql, nl_query, schema_str)
        semantic_trace = self._create_trace(
            agent_id="agent5.3",
            status="completed",
            input_summary="Validating semantic correctness",
            output_summary=f"Semantic: {'VALID' if semantic_result.get('is_valid') else 'INVALID'} (conf: {semantic_result.get('confidence', 0):.0%})",
            details=semantic_result
        )
        traces.append(semantic_trace)

        # Combine results
        all_issues = []
        all_suggestions = []
        confidence_scores = []

        for result in [syntax_result, schema_result, semantic_result]:
            all_issues.extend(result.get("issues", []))
            all_suggestions.extend(result.get("suggestions", []))
            confidence_scores.append(result.get("confidence", 0))

        # Calculate overall confidence (weighted average)
        overall_confidence = (
            confidence_scores[0] * 0.3 +  # Syntax: 30%
            confidence_scores[1] * 0.35 + # Schema: 35%
            confidence_scores[2] * 0.35   # Semantic: 35%
        )

        validation_results = {
            "syntax": syntax_result,
            "schema": schema_result,
            "semantic": semantic_result,
            "overall_confidence": overall_confidence
        }

        # Create main Agent5 trace
        if overall_confidence >= 0.75:
            status_msg = f"PASSED - Overall confidence: {overall_confidence:.0%} (>= 75%)"
            next_action = "Proceeding to execution"
        else:
            status_msg = f"FAILED - Overall confidence: {overall_confidence:.0%} (< 75%)"
            if retry_count < state.get("max_retries", 3):
                next_action = f"Will retry SQL generation (attempt {retry_count + 1}/{state.get('max_retries', 3)})"
            else:
                next_action = "Max retries reached - will return error"

        main_trace = self._create_trace(
            agent_id="agent5",
            status="completed",
            input_summary=f"Validating SQL query ({len(generated_sql)} chars)",
            output_summary=f"{status_msg}. {next_action}",
            details={
                "syntax_confidence": confidence_scores[0],
                "schema_confidence": confidence_scores[1],
                "semantic_confidence": confidence_scores[2],
                "overall_confidence": overall_confidence,
                "issues_count": len(all_issues),
                "threshold": 0.75,
                "passed": overall_confidence >= 0.75
            }
        )
        traces.append(main_trace)

        return {
            "validation_results": validation_results,
            "overall_confidence": overall_confidence,
            "validation_issues": all_issues,
            "validation_suggestions": all_suggestions,
            "current_agent": "agent5_validator",
            "agent_traces": traces
        }

    def _validate_syntax(self, sql: str, schema_str: str) -> Dict[str, Any]:
        """Agent 5.1: Syntax Validation"""
        prompt = f"""You are a SQL syntax validation expert. Analyze the following SQL query for syntax correctness.

SQL QUERY:
{sql}

SCHEMA FOR REFERENCE:
{schema_str}

Check for:
1. Proper SQL syntax (SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY)
2. Correct keyword usage
3. Proper use of quotes and string literals
4. Balanced parentheses
5. Correct function syntax

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0 to 1.0,
    "issues": ["List of syntax issues found"],
    "suggestions": ["How to fix each issue"]
}}

Respond ONLY with the JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            return json.loads(response.content.strip())
        except json.JSONDecodeError:
            return {"is_valid": True, "confidence": 0.7, "issues": [], "suggestions": []}

    def _validate_schema(self, sql: str, schema_str: str) -> Dict[str, Any]:
        """Agent 5.2: Schema Validation"""
        prompt = f"""You are a database schema validation expert. Validate that the SQL query correctly uses the schema.

SQL QUERY:
{sql}

AVAILABLE SCHEMA:
{schema_str}

Check for:
1. All table names exist in the schema
2. All column names exist in their respective tables
3. JOINs use correct foreign key relationships
4. Data types are used correctly (e.g., comparing strings to strings)
5. Primary keys are used correctly

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0 to 1.0,
    "issues": ["List of schema issues found"],
    "suggestions": ["How to fix each issue"]
}}

Respond ONLY with the JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            return json.loads(response.content.strip())
        except json.JSONDecodeError:
            return {"is_valid": True, "confidence": 0.7, "issues": [], "suggestions": []}

    def _validate_semantic(self, sql: str, nl_query: str, schema_str: str) -> Dict[str, Any]:
        """Agent 5.3: Semantic Validation"""
        prompt = f"""You are a semantic validation expert. Validate that the SQL query correctly captures the user's intent.

USER'S NATURAL LANGUAGE QUERY:
{nl_query}

SQL QUERY:
{sql}

AVAILABLE SCHEMA:
{schema_str}

Check for:
1. Does the query answer what the user asked?
2. Are all requirements from the NL query covered?
3. Is the query returning the correct data?
4. Are the aggregations/calculations correct for the intent?
5. Is any part of the user's request missing?

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0 to 1.0,
    "issues": ["List of semantic issues - what's wrong or missing"],
    "suggestions": ["How to better capture user intent"]
}}

Respond ONLY with the JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            return json.loads(response.content.strip())
        except json.JSONDecodeError:
            return {"is_valid": True, "confidence": 0.7, "issues": [], "suggestions": []}

    def _route_after_validation(self, state: MultiAgentState) -> Literal["execute", "regenerate", "fail"]:
        """Route after validation based on confidence score."""
        overall_confidence = state.get("overall_confidence", 0)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        # If confidence >= 75%, execute (lowered from 90% for better success rate)
        if overall_confidence >= 0.75:
            return "execute"

        # If confidence < 75% and retries remaining, regenerate
        if retry_count < max_retries:
            return "regenerate"

        # Max retries exceeded, fail
        return "fail"

    def _handle_retry(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Handle retry by incrementing retry count and creating a trace.
        This node is called when validation fails and we need to regenerate SQL.
        """
        current_retry = state.get("retry_count", 0)
        new_retry_count = current_retry + 1
        max_retries = state.get("max_retries", 3)
        overall_confidence = state.get("overall_confidence", 0)
        issues = state.get("validation_issues", [])

        # Create trace for retry
        trace = self._create_trace(
            agent_id="retry",
            status="completed",
            input_summary=f"Validation failed with {overall_confidence:.0%} confidence",
            output_summary=f"Retry {new_retry_count}/{max_retries} - Regenerating SQL with feedback",
            details={
                "retry_count": new_retry_count,
                "max_retries": max_retries,
                "confidence_was": overall_confidence,
                "issues_count": len(issues),
                "issues_summary": issues[:3] if issues else []
            }
        )

        return {
            "retry_count": new_retry_count,
            "current_agent": "handle_retry",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 6: Executor Agent
    # =========================================================================
    def _agent6_executor(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent6: SQL Executor Agent
        - Executes the validated SQL query
        - Formats results based on user intent
        """
        from tools.duckdb_tools import run_sql_query

        generated_sql = state.get("generated_sql", "")
        nl_query = state["nl_query"]

        # Execute the query
        try:
            results = run_sql_query.invoke({"sql_query": generated_sql})

            # Check if results contain an error message (run_sql_query returns errors as strings)
            error_indicators = ["Error:", "SQL Syntax Error:", "Database Error:", "error executing"]
            if any(indicator.lower() in results.lower() for indicator in error_indicators):
                execution_success = False
                error_msg = results
            else:
                execution_success = True
                error_msg = None
        except Exception as e:
            results = f"Error executing query: {str(e)}"
            execution_success = False
            error_msg = str(e)

        # Generate final answer based on results
        prompt = f"""You are a data analyst presenting query results to a user.

USER'S ORIGINAL QUESTION:
{nl_query}

SQL QUERY EXECUTED:
{generated_sql}

QUERY RESULTS:
{results}

Please provide a clear, user-friendly response that:
1. Directly answers the user's question
2. Presents the data in an easy-to-understand format
3. Highlights key findings or insights
4. Uses tables or bullet points where appropriate

Your response should be conversational and helpful."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Count result rows (approximate)
        result_lines = results.strip().split('\n') if results else []
        result_rows = max(0, len(result_lines) - 1)  # Subtract header row

        # Create trace for this agent
        if execution_success:
            output_summary = f"Query executed successfully. {result_rows} rows returned."
        else:
            output_summary = f"Query execution failed: {error_msg}"

        trace = self._create_trace(
            agent_id="agent6",
            status="completed" if execution_success else "failed",
            input_summary=f"Executing SQL: {generated_sql[:80]}{'...' if len(generated_sql) > 80 else ''}",
            output_summary=output_summary,
            details={
                "execution_success": execution_success,
                "rows_returned": result_rows,
                "sql_executed": generated_sql,
                "error": error_msg
            }
        )

        return {
            "messages": [response],
            "query_results": results,
            "final_answer": response.content,
            "is_complete": True,
            "current_agent": "agent6_executor",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # Final Response Node
    # =========================================================================
    def _final_response_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Generate final response based on the workflow outcome."""

        # Preserve important state fields in all cases
        preserved_fields = {
            "agent_traces": state.get("agent_traces", []),
            "generated_sql": state.get("generated_sql", ""),
            "overall_confidence": state.get("overall_confidence", 0),
            "validation_results": state.get("validation_results", {}),
            "query_results": state.get("query_results", ""),
            "final_answer": state.get("final_answer", ""),  # Always preserve final_answer
            "is_data_query": state.get("is_data_query", False),  # For cache update
            "related_tables": state.get("related_tables", []),   # For cache update
            "is_followup_question": state.get("is_followup_question", False),
            "is_modified_query": state.get("is_modified_query", False),
        }

        # Case 1: Follow-up question - answer from cache (no SQL needed)
        if state.get("is_followup_question", False):
            followup_answer = state.get("followup_answer") or state.get("general_answer")
            return {
                **preserved_fields,
                "final_answer": followup_answer or "I couldn't determine an answer from the previous results.",
                "is_complete": True
            }

        # Case 2: General question answered by LLM
        if not state.get("is_data_query", True):
            return {
                **preserved_fields,
                "final_answer": state.get("general_answer", "I couldn't determine an answer."),
                "is_complete": True
            }

        # Case 2: Query execution completed (final_answer already set by agent6)
        if state.get("final_answer"):
            return {
                **preserved_fields,
                "is_complete": True
            }

        # Case 3: Validation failed after max retries
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if retry_count >= max_retries:
            issues = state.get("validation_issues", [])
            suggestions = state.get("validation_suggestions", [])

            error_message = f"""I apologize, but I was unable to generate a valid SQL query after {max_retries} attempts.

**Issues encountered:**
{chr(10).join(f"- {issue}" for issue in issues[:5])}

**Suggestions:**
{chr(10).join(f"- {suggestion}" for suggestion in suggestions[:5])}

Please try rephrasing your question or providing more specific details about what data you need."""

            return {
                **preserved_fields,
                "final_answer": error_message,
                "is_complete": True,
                "error": "Max retries exceeded"
            }

        return {
            **preserved_fields,
            "is_complete": True
        }

    # =========================================================================
    # Public Methods
    # =========================================================================
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the multi-agent workflow with a given query.

        Args:
            query: The user's natural language query

        Returns:
            The final state including the answer and execution trace
        """
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=query)],
            "nl_query": query,
            "is_data_query": True,
            "general_answer": None,
            "related_tables": [],
            "relationships": [],
            # Query intent fields
            "is_modified_query": False,
            "is_followup_question": False,
            "followup_context": None,
            "previous_sql": None,
            "previous_results": None,
            "followup_answer": None,
            # Other fields
            "extracted_schema": {},
            "query_complexity": "SIMPLE",
            "subtasks": [],
            "generated_sql": "",
            "sub_queries": [],
            "validation_results": {},
            "overall_confidence": 0.0,
            "validation_issues": [],
            "validation_suggestions": [],
            "retry_count": 0,
            "max_retries": 3,
            "query_results": "",
            "final_answer": "",
            "current_agent": "start",
            "is_complete": False,
            "error": None,
            "agent_traces": []
        }

        try:
            # Create a custom graph with retry logic
            final_state = self._run_with_retry(initial_state)

            # Cache the successful query for follow-up detection
            if final_state.get("is_data_query", False) and final_state.get("generated_sql"):
                query_cache.add_query(
                    nl_query=query,
                    generated_sql=final_state.get("generated_sql", ""),
                    query_results=final_state.get("query_results", ""),
                    tables_used=final_state.get("related_tables", [])
                )

            return final_state
        except Exception as e:
            return {
                **initial_state,
                "error": str(e),
                "is_complete": True,
                "final_answer": f"An error occurred: {str(e)}"
            }

    def _run_with_retry(self, state: MultiAgentState) -> MultiAgentState:
        """Run the graph with retry logic for validation failures."""
        current_state = state.copy()

        while True:
            # Run the graph
            result = self.graph.invoke(current_state)

            # Check if we need to retry (handled by conditional edges)
            if result.get("is_complete", False):
                return result

            # Update retry count for next iteration
            current_state = result.copy()
            current_state["retry_count"] = current_state.get("retry_count", 0) + 1

            # Safety check to prevent infinite loops
            if current_state["retry_count"] > current_state.get("max_retries", 3):
                break

        return result

    def stream(self, query: str):
        """
        Stream the multi-agent execution for real-time updates.

        Args:
            query: The user's natural language query

        Yields:
            State updates as they occur
        """
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=query)],
            "nl_query": query,
            "is_data_query": True,
            "general_answer": None,
            "related_tables": [],
            "relationships": [],
            # Query intent fields
            "is_modified_query": False,
            "is_followup_question": False,
            "followup_context": None,
            "previous_sql": None,
            "previous_results": None,
            "followup_answer": None,
            # Other fields
            "extracted_schema": {},
            "query_complexity": "SIMPLE",
            "subtasks": [],
            "generated_sql": "",
            "sub_queries": [],
            "validation_results": {},
            "overall_confidence": 0.0,
            "validation_issues": [],
            "validation_suggestions": [],
            "retry_count": 0,
            "max_retries": 3,
            "query_results": "",
            "final_answer": "",
            "current_agent": "start",
            "is_complete": False,
            "error": None,
            "agent_traces": []
        }

        try:
            for state in self.graph.stream(initial_state):
                yield state
            # Note: Cache update is handled by the caller (app.py) after streaming completes
            # This ensures proper access to the final state
        except Exception as e:
            yield {"error": str(e)}

    def format_trace_for_display(self, traces: List[AgentTrace]) -> str:
        """Format agent traces for display in UI observation box."""
        if not traces:
            return "No trace information available."

        lines = []
        lines.append("=" * 60)
        lines.append("MULTI-AGENT WORKFLOW TRACE")
        lines.append("=" * 60)

        for i, trace in enumerate(traces, 1):
            lines.append("")
            lines.append(f"[{i}] {trace['agent_name']}")
            lines.append("-" * 40)
            lines.append(f"   Status: {trace['status'].upper()}")
            lines.append(f"   Input: {trace['input_summary']}")
            lines.append(f"   Output: {trace['output_summary']}")

            # Add key details
            details = trace.get('details', {})
            if details:
                key_details = []
                for key, value in details.items():
                    if key not in ['sql_preview', 'explanation', 'reasoning'] and value is not None:
                        if isinstance(value, (list, dict)):
                            if isinstance(value, list) and len(value) > 0:
                                key_details.append(f"{key}: {len(value)} items")
                        else:
                            key_details.append(f"{key}: {value}")

                if key_details:
                    lines.append(f"   Details: {', '.join(key_details[:5])}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the multi-agent workflow."""
        return {
            "name": "Multi-Agent Data Orchestrator",
            "description": "6-agent workflow for SQL generation with intent classification and caching",
            "agents": [
                {"id": "agent1", "name": "Intent Classification", "description": "Classifies: NEW_DATA_QUERY, MODIFIED_QUERY, FOLLOWUP_QUESTION, GENERAL_QUESTION"},
                {"id": "agent2", "name": "Schema Extraction", "description": "Extracts relevant schema (no LLM)"},
                {"id": "agent3", "name": "Orchestrator", "description": "Determines query complexity (SIMPLE/COMPLEX)"},
                {"id": "agent4", "name": "SQL Generator", "description": "Generates SQL queries (with modified query context)"},
                {"id": "agent5", "name": "Validator", "description": "Multi-level SQL validation"},
                {"id": "agent6", "name": "Executor", "description": "Executes SQL and presents results"}
            ],
            "intents": [
                {"name": "NEW_DATA_QUERY", "description": "Fresh query requiring SQL generation"},
                {"name": "MODIFIED_QUERY", "description": "Modify previous query (needs new SQL)"},
                {"name": "FOLLOWUP_QUESTION", "description": "Question about results (answer from cache)"},
                {"name": "GENERAL_QUESTION", "description": "Non-data question"}
            ],
            "max_retries": 3,
            "validation_threshold": 0.75,
            "cache_size": 3,
            "cached_queries": len(query_cache)
        }

    # =========================================================================
    # Cache Management Methods
    # =========================================================================
    def get_cached_queries(self) -> List[Dict[str, Any]]:
        """Get all cached queries for display/debugging."""
        return [
            {
                "nl_query": q.nl_query,
                "generated_sql": q.generated_sql,
                "results_preview": q.query_results[:200] + "..." if len(q.query_results) > 200 else q.query_results,
                "tables_used": q.tables_used,
                "timestamp": q.timestamp
            }
            for q in query_cache.get_recent_queries()
        ]

    def clear_query_cache(self):
        """Clear the query cache."""
        query_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached queries."""
        return len(query_cache)
