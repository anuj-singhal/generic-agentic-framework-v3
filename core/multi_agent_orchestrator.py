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
    Cache for storing the LAST query result for follow-up detection.
    Only stores 1 result - cleared only on NEW_QUERY intent.
    """
    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure one cache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cached_query: Optional[CachedQuery] = None
        return cls._instance

    def set_query(self, nl_query: str, generated_sql: str, query_results: str, tables_used: List[str] = None):
        """Set the cached query (replaces any existing)."""
        self._cached_query = CachedQuery(
            nl_query=nl_query,
            generated_sql=generated_sql,
            query_results=query_results,
            timestamp=datetime.now().isoformat(),
            tables_used=tables_used or []
        )

    def get_cached_query(self) -> Optional[CachedQuery]:
        """Get the cached query."""
        return self._cached_query

    def has_cache(self) -> bool:
        """Check if there's a cached query."""
        return self._cached_query is not None

    def get_cache_context(self) -> str:
        """Get formatted cache context for LLM."""
        if not self._cached_query:
            return ""

        cached = self._cached_query
        # Include full results for analysis (up to 3000 chars to ensure good analysis)
        results = cached.query_results[:3000] + "..." if len(cached.query_results) > 3000 else cached.query_results

        return f"""
=== PREVIOUS QUERY AND RESULTS (from cache) ===
PREVIOUS QUESTION: {cached.nl_query}

PREVIOUS SQL:
{cached.generated_sql}

PREVIOUS RESULTS DATA:
{results}

TABLES USED: {', '.join(cached.tables_used) if cached.tables_used else 'Unknown'}
=== END OF CACHED DATA ===
"""

    def clear(self):
        """Clear the cache."""
        self._cached_query = None

    # Backward compatibility methods
    def add_query(self, nl_query: str, generated_sql: str, query_results: str, tables_used: List[str] = None):
        """Alias for set_query for backward compatibility."""
        self.set_query(nl_query, generated_sql, query_results, tables_used)

    def get_last_query(self) -> Optional[CachedQuery]:
        """Alias for get_cached_query for backward compatibility."""
        return self.get_cached_query()

    def get_recent_queries(self) -> List[CachedQuery]:
        """Get cached query as list for backward compatibility."""
        return [self._cached_query] if self._cached_query else []

    def __len__(self):
        return 1 if self._cached_query else 0


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

    # Memory context (passed from app.py)
    short_term_memory: Optional[str]  # Recent conversations
    long_term_memory: Optional[str]   # Historical summaries

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

    def _get_strict_schema_reference(self) -> str:
        """
        Get a STRICT schema reference with explicit column listings.
        This is used in SQL generation to prevent hallucinated columns.
        """
        if not self.schema_document or "tables" not in self.schema_document:
            return "No schema document available."

        lines = []
        lines.append("=" * 70)
        lines.append("STRICT SCHEMA REFERENCE - USE ONLY THESE COLUMNS")
        lines.append("=" * 70)
        lines.append("")

        for table in self.schema_document["tables"]:
            table_name = table["table_name"]
            lines.append(f"TABLE: {table_name}")
            lines.append("-" * 50)
            lines.append("COLUMNS (ONLY THESE EXIST):")

            for col in table["columns"]:
                col_name = col["column_name"]
                data_type = col["data_type"]
                desc = col["description"]
                pk = " [PK]" if col.get("is_primary_key") else ""
                lines.append(f"  • {col_name} ({data_type}){pk} - {desc}")

            lines.append("")

        lines.append("=" * 70)
        lines.append("VALID RELATIONSHIPS (JOIN CONDITIONS):")
        lines.append("-" * 50)
        lines.append("• PORTFOLIOS.CLIENT_ID = CLIENTS.CLIENT_ID")
        lines.append("• TRANSACTIONS.PORTFOLIO_ID = PORTFOLIOS.PORTFOLIO_ID")
        lines.append("• TRANSACTIONS.ASSET_ID = ASSETS.ASSET_ID")
        lines.append("• HOLDINGS.PORTFOLIO_ID = PORTFOLIOS.PORTFOLIO_ID")
        lines.append("• HOLDINGS.ASSET_ID = ASSETS.ASSET_ID")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _get_all_valid_columns(self) -> dict:
        """
        Get a dictionary of all valid columns per table.
        Used for programmatic validation.
        """
        if not self.schema_document or "tables" not in self.schema_document:
            return {}

        valid_columns = {}
        for table in self.schema_document["tables"]:
            table_name = table["table_name"]
            valid_columns[table_name] = [col["column_name"] for col in table["columns"]]

        return valid_columns

    def _extract_columns_from_sql(self, sql: str) -> dict:
        """
        Extract table.column references from SQL query.
        Returns dict of {table_name: [columns used]}
        """
        import re

        # Normalize SQL
        sql_upper = sql.upper()

        # Extract table aliases
        # Pattern: FROM/JOIN table_name alias or FROM/JOIN table_name AS alias
        alias_pattern = r'(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)(?:\s|$|,|ON)'
        aliases = {}
        for match in re.finditer(alias_pattern, sql_upper, re.IGNORECASE):
            table_name = match.group(1)
            alias = match.group(2)
            if alias.upper() not in ('ON', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'AS', 'AND', 'OR'):
                aliases[alias] = table_name

        # Also handle simple FROM table_name (no alias)
        simple_from_pattern = r'(?:FROM|JOIN)\s+(\w+)(?:\s+(?:WHERE|ON|JOIN|LEFT|RIGHT|INNER|GROUP|ORDER|HAVING|LIMIT|$))'
        for match in re.finditer(simple_from_pattern, sql_upper, re.IGNORECASE):
            table_name = match.group(1)
            if table_name not in aliases.values():
                aliases[table_name] = table_name

        # Extract column references with table/alias prefix (e.g., c.FULL_NAME, CLIENTS.FULL_NAME)
        column_pattern = r'(\w+)\.(\w+)'
        columns_used = {}

        for match in re.finditer(column_pattern, sql_upper):
            prefix = match.group(1)
            column = match.group(2)

            # Skip if it looks like a function or keyword
            if column in ('COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'COALESCE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'AND', 'OR', 'NULL'):
                continue

            # Resolve alias to table name
            table_name = aliases.get(prefix, prefix)

            if table_name not in columns_used:
                columns_used[table_name] = set()
            columns_used[table_name].add(column)

        # Convert sets to lists
        return {k: list(v) for k, v in columns_used.items()}

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
    def _build_memory_context_for_prompt(self, state: MultiAgentState) -> str:
        """Build memory context string for LLM prompt."""
        memory_parts = []

        # Add long-term memory if available
        long_term = state.get("long_term_memory", "")
        if long_term and long_term.strip():
            memory_parts.append("=== LONG-TERM MEMORY (Historical Context) ===")
            memory_parts.append(long_term)
            memory_parts.append("=== END LONG-TERM MEMORY ===\n")

        # Add short-term memory if available
        short_term = state.get("short_term_memory", "")
        if short_term and short_term.strip():
            memory_parts.append("=== SHORT-TERM MEMORY (Recent Conversations) ===")
            memory_parts.append(short_term)
            memory_parts.append("=== END SHORT-TERM MEMORY ===\n")

        return "\n".join(memory_parts) if memory_parts else ""

    def _has_memory(self, state: MultiAgentState) -> bool:
        """Check if state has any memory context."""
        long_term = state.get("long_term_memory", "")
        short_term = state.get("short_term_memory", "")
        return bool((long_term and long_term.strip()) or (short_term and short_term.strip()))

    def _agent1_prevalidation(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Agent1: Intent Classification Agent

        Enhanced Flow with Memory Integration:

        Scenario 1 - No cache, No memory:
            → Check: GENERAL_QUESTION or NEW_DATA_QUERY

        Scenario 2 - No cache, Has memory:
            → Pass: NL query + memory context
            → Check: GENERAL_QUESTION, NEW_DATA_QUERY, or MODIFIED_QUERY
            → Memory allows detecting modifications to past queries even without cache

        Scenario 3 - Has cache, Has memory:
            → Pass: NL query + cache data + memory context
            → Check: GENERAL_QUESTION, FOLLOWUP_QUESTION, NEW_DATA_QUERY, or MODIFIED_QUERY
            → Full context for best decision

        Cache is cleared ONLY on NEW_DATA_QUERY intent.
        """
        nl_query = state["nl_query"]
        schema_summary = self._get_schema_summary()
        cached_query = query_cache.get_cached_query()
        has_cache = query_cache.has_cache()
        has_memory = self._has_memory(state)
        memory_context = self._build_memory_context_for_prompt(state) if has_memory else ""

        # Determine scenario and build appropriate prompt
        if has_cache:
            # SCENARIO 3: Has cache (and possibly memory)
            cache_context = query_cache.get_cache_context()

            prompt = f"""You are an intelligent query classifier. A user has asked a question, and we have CACHED DATA from a previous query.

YOUR TASK: Analyze the user's question and determine the best way to handle it.

{cache_context}

{memory_context if memory_context else ""}

CURRENT USER QUESTION: {nl_query}

AVAILABLE DATABASE SCHEMA (for reference):
{schema_summary}

DECISION PROCESS (evaluate in this order):

1. **GENERAL_QUESTION**: Is this a general knowledge question NOT about database data?
   - Examples: "What is SQL?", "How do JOINs work?", "What does portfolio mean?", "Explain risk profile"
   - These can be answered WITHOUT querying the database
   - ALSO check if this relates to a previous general question in memory and provide a contextual answer

2. **CAN_ANSWER_FROM_CACHE**: Can you answer the user's question by analyzing the CACHED RESULTS above?
   - If YES: Analyze the data and provide the SPECIFIC ANSWER (with actual numbers/values)
   - Examples: "How many?", "What's the total?", "Which has the highest?", "Summarize this"
   - The answer must come DIRECTLY from the cached data

3. **MODIFIED_QUERY**: Does the user want to MODIFY the previous query (filter, sort, add columns, etc.)?
   - This requires generating NEW SQL based on the previous context
   - Examples: "Show only from USA", "Add email column", "Sort by value", "Group by country", "Filter by high risk"
   - Check memory for context - user might be modifying a query from conversation history

4. **NEW_QUERY**: Is this a completely NEW question unrelated to the previous query or conversation?
   - This clears the cache and starts fresh
   - Examples: "Show me all assets", "What transactions happened today?"

Respond in JSON format:
{{
    "is_general_question": true/false,
    "general_answer": "If is_general_question=true, provide comprehensive answer using memory context if relevant. Otherwise null.",
    "can_answer_from_cache": true/false,
    "cache_answer": "If can_answer_from_cache=true, provide the COMPLETE ANSWER here with specific numbers. Otherwise null.",
    "is_modified_query": true/false,
    "modification_type": "filter/sort/aggregate/add_columns/other (only if is_modified_query=true)",
    "is_new_query": true/false,
    "reasoning": "Brief explanation of your decision, mentioning if memory context influenced it",
    "related_tables": ["TABLE1", "TABLE2"],
    "memory_influenced": true/false
}}

CRITICAL RULES:
- ALWAYS check for GENERAL_QUESTION first - these should NOT trigger SQL generation
- For CACHE_ANSWER: Provide ACTUAL NUMBERS from the data (e.g., "There are 4 high-risk clients" NOT "count the rows")
- Only ONE of is_general_question, can_answer_from_cache, is_modified_query, is_new_query should be true
- If the question references "the results", "those", "them", "this data" → likely CACHE or MODIFIED
- If asking for totally different data → NEW_QUERY
- For general questions related to previous general questions in memory, provide contextual answers

Respond ONLY with JSON."""

        elif has_memory:
            # SCENARIO 2: No cache but has memory - can still detect modifications from conversation history
            prompt = f"""You are an intelligent query classifier. A user has asked a question. We have CONVERSATION MEMORY but no cached query results.

YOUR TASK: Analyze the user's question along with the conversation memory to determine the best way to handle it.

{memory_context}

CURRENT USER QUESTION: {nl_query}

AVAILABLE DATABASE SCHEMA (for reference):
{schema_summary}

IMPORTANT: The memory above contains previous User Queries, Generated SQL queries, and Assistant Responses. Use this context to understand:
- What queries the user has previously asked
- What SQL was generated for those queries
- What data/answers they received

DECISION PROCESS (evaluate in this order):

1. **GENERAL_QUESTION**: Is this a general knowledge question NOT about database data?
   - Examples: "What is SQL?", "How do JOINs work?", "What does portfolio mean?"
   - Check if this relates to a previous general question in memory - if so, provide contextual answer
   - Examples of follow-up general questions: "Tell me more about that", "Can you explain further?"

2. **MODIFIED_QUERY**: Does the user want to MODIFY a query mentioned in the conversation memory?
   - Look at SHORT-TERM MEMORY for recent "User Query" and "Generated SQL" entries
   - If user says "now filter by...", "add X to that", "go back to..." → this references a past query
   - Examples: "Now show only high-risk clients", "Add the total value", "Filter the previous results"
   - If MODIFIED_QUERY, extract the previous_sql and previous_nl_query from memory

3. **NEW_DATA_QUERY**: Is this a new database query unrelated to conversation history?
   - This is a fresh query that needs SQL generation from scratch
   - Examples: "Show me all clients", "List all portfolios"

Respond in JSON format:
{{
    "is_general_question": true/false,
    "general_answer": "If is_general_question=true, provide comprehensive answer using memory context if relevant. Otherwise null.",
    "is_modified_query": true/false,
    "modification_context": "If is_modified_query=true, describe what query from memory is being modified and include the previous SQL if found. Otherwise null.",
    "previous_nl_query": "If is_modified_query=true, the previous user query being modified. Otherwise null.",
    "previous_sql": "If is_modified_query=true, the previous SQL from memory being modified. Otherwise null.",
    "modification_type": "filter/sort/aggregate/add_columns/other (only if is_modified_query=true)",
    "is_data_query": true/false,
    "reasoning": "Brief explanation of your decision, mentioning if memory context influenced it",
    "related_tables": ["TABLE1", "TABLE2"],
    "memory_influenced": true/false
}}

CRITICAL RULES:
- ALWAYS check for GENERAL_QUESTION first
- For general questions that follow up on previous general questions in memory, provide contextual answers
- is_modified_query can be true even without cache if memory shows a relevant previous query
- If modified_query is true, EXTRACT the previous_sql and previous_nl_query from memory

Respond ONLY with JSON."""

        else:
            # SCENARIO 1: No cache, No memory - simplest classification
            prompt = f"""You are a query classifier. Determine if this is a DATABASE query or a GENERAL question.

USER QUESTION: {nl_query}

AVAILABLE DATABASE SCHEMA:
{schema_summary}

DECISION PROCESS:

1. **GENERAL_QUESTION**: Is this a general knowledge question NOT about database data?
   - Examples: "What is SQL?", "How do JOINs work?", "What does portfolio mean?", "Explain risk profiles"
   - These questions can be answered WITHOUT querying the database

2. **DATA_QUERY**: Does this require querying the database to answer?
   - Examples: "Show me all clients", "What is the total portfolio value?", "List high-risk clients"

Respond in JSON format:
{{
    "is_general_question": true/false,
    "general_answer": "If is_general_question=true, provide comprehensive answer. Otherwise null.",
    "is_data_query": true/false,
    "reasoning": "Brief explanation",
    "related_tables": ["TABLE1", "TABLE2"]
}}

CRITICAL RULES:
- ALWAYS check for GENERAL_QUESTION first
- General questions should NOT trigger SQL generation

Respond ONLY with JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Parse response and determine intent
        try:
            # Handle potential markdown code block wrapping
            content = response.content.strip()
            if content.startswith("```"):
                # Remove markdown code block
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {}

        # Determine intent based on response and scenario
        intent = "NEW_DATA_QUERY"
        is_data_query = True
        is_modified_query = False
        is_followup_question = False
        general_answer = None
        followup_answer = None
        related_tables = result.get("related_tables", [])
        reasoning = result.get("reasoning", "")
        memory_influenced = result.get("memory_influenced", False)
        modification_context = result.get("modification_context")

        if has_cache:
            # SCENARIO 3: Parse cache-aware response
            # Check general question FIRST (fix for general_question not working with cache)
            if result.get("is_general_question"):
                intent = "GENERAL_QUESTION"
                is_data_query = False
                general_answer = result.get("general_answer")
            elif result.get("can_answer_from_cache"):
                intent = "FOLLOWUP_QUESTION"
                is_data_query = False
                is_followup_question = True
                followup_answer = result.get("cache_answer")
            elif result.get("is_modified_query"):
                intent = "MODIFIED_QUERY"
                is_modified_query = True
            else:
                # NEW_QUERY - clear the cache
                intent = "NEW_DATA_QUERY"
                query_cache.clear()

        elif has_memory:
            # SCENARIO 2: Parse memory-aware response (no cache)
            if result.get("is_general_question"):
                intent = "GENERAL_QUESTION"
                is_data_query = False
                general_answer = result.get("general_answer")
            elif result.get("is_modified_query"):
                # Modified query detected from memory context
                intent = "MODIFIED_QUERY"
                is_modified_query = True
                is_data_query = True
            else:
                intent = "NEW_DATA_QUERY"
                is_data_query = True
        else:
            # SCENARIO 1: No cache, no memory - simple classification
            if result.get("is_general_question"):
                intent = "GENERAL_QUESTION"
                is_data_query = False
                general_answer = result.get("general_answer")
            else:
                intent = "NEW_DATA_QUERY"
                is_data_query = True

        # Build context for modified queries
        followup_context = None
        previous_sql = None
        previous_results = None

        if is_modified_query:
            if cached_query:
                # Use cached query context
                followup_context = {
                    "previous_nl_query": cached_query.nl_query,
                    "previous_sql": cached_query.generated_sql,
                    "previous_results": cached_query.query_results,
                    "tables_used": cached_query.tables_used,
                    "modification_type": result.get("modification_type", "unknown"),
                    "source": "cache"
                }
                previous_sql = cached_query.generated_sql
                previous_results = cached_query.query_results
            elif has_memory:
                # Use memory context for modification (when no cache)
                # LLM should have extracted previous_sql and previous_nl_query from memory
                memory_previous_sql = result.get("previous_sql")
                memory_previous_nl_query = result.get("previous_nl_query")

                followup_context = {
                    "previous_nl_query": memory_previous_nl_query,
                    "previous_sql": memory_previous_sql,
                    "modification_context": modification_context,
                    "modification_type": result.get("modification_type", "unknown"),
                    "source": "memory",
                    "memory_context": memory_context[:2000] if memory_context else ""  # Truncate for context
                }
                previous_sql = memory_previous_sql
                # previous_results not available from memory, only from cache

        # Create output summary
        memory_note = " (memory-influenced)" if memory_influenced else ""
        if intent == "NEW_DATA_QUERY":
            output_summary = f"New query{' (cache cleared)' if has_cache else ''}{memory_note}. Tables: {', '.join(related_tables) if related_tables else 'analyzing...'}"
        elif intent == "MODIFIED_QUERY":
            source = "cache" if cached_query else "memory"
            output_summary = f"Modified query (from {source}){memory_note}. Tables: {', '.join(related_tables) if related_tables else 'analyzing...'}"
        elif intent == "FOLLOWUP_QUESTION":
            output_summary = f"Follow-up question - answering from cached results{memory_note}"
        else:
            output_summary = f"General question - answering directly{memory_note}"

        # Calculate memory lengths for trace
        st_memory_len = len(state.get("short_term_memory") or "")
        lt_memory_len = len(state.get("long_term_memory") or "")

        trace = self._create_trace(
            agent_id="agent1",
            status="completed",
            input_summary=f"Query: {nl_query[:100]}{'...' if len(nl_query) > 100 else ''}",
            output_summary=output_summary,
            details={
                "intent": intent,
                "scenario": "cache+memory" if has_cache else ("memory_only" if has_memory else "no_context"),
                "has_cache": has_cache,
                "has_memory": has_memory,
                "short_term_memory_chars": st_memory_len,
                "long_term_memory_chars": lt_memory_len,
                "memory_influenced": memory_influenced,
                "is_data_query": is_data_query,
                "is_modified_query": is_modified_query,
                "is_followup_question": is_followup_question,
                "related_tables": related_tables,
                "reasoning": reasoning
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
            "relationships": [],  # Relationships determined by Agent2
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
        - For complex queries, breaks down into subtasks with explicit column references
        """
        nl_query = state["nl_query"]
        extracted_schema = state.get("extracted_schema", {})

        schema_str = json.dumps(extracted_schema, indent=2)

        prompt = f"""You are a SQL query orchestrator agent. Your job is to analyze the user query and determine its complexity.

USER QUERY: {nl_query}

AVAILABLE SCHEMA:
{schema_str}

═══════════════════════════════════════════════════════════════════════════
VALID COLUMNS REFERENCE (USE ONLY THESE IN SUBTASK DEFINITIONS):
═══════════════════════════════════════════════════════════════════════════
CLIENTS: CLIENT_ID, FULL_NAME, COUNTRY, RISK_PROFILE, ONBOARDING_DATE, KYC_STATUS
PORTFOLIOS: PORTFOLIO_ID, CLIENT_ID, PORTFOLIO_NAME, BASE_CURRENCY, INCEPTION_DATE, STATUS
ASSETS: ASSET_ID, SYMBOL, ASSET_NAME, ASSET_TYPE, CURRENCY, EXCHANGE
TRANSACTIONS: TRANSACTION_ID, PORTFOLIO_ID, ASSET_ID, TRADE_DATE, TRANSACTION_TYPE, QUANTITY, PRICE, FEES, CURRENCY, CREATED_AT
HOLDINGS: PORTFOLIO_ID, ASSET_ID, QUANTITY, AVG_COST, LAST_UPDATED
═══════════════════════════════════════════════════════════════════════════

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
        "If COMPLEX: Subtask description with SPECIFIC COLUMNS to use (e.g., 'Calculate portfolio value using SUM(HOLDINGS.QUANTITY * HOLDINGS.AVG_COST)')",
        "Second subtask with columns (e.g., 'Get client info using CLIENTS.FULL_NAME, CLIENTS.COUNTRY')",
        "etc."
    ],
    "tables_needed": ["List of tables needed for this query"],
    "key_calculations": [
        "For portfolio value: SUM(HOLDINGS.QUANTITY * HOLDINGS.AVG_COST)",
        "For transaction value: TRANSACTIONS.QUANTITY * TRANSACTIONS.PRICE",
        "etc."
    ],
    "final_synthesis": "If COMPLEX: How to combine subtasks into final CTE query"
}}

IMPORTANT FOR SUBTASK DEFINITIONS:
- Be SPECIFIC about which columns to use
- Reference ONLY columns that exist in the schema above
- For "total value" calculations, specify: HOLDINGS.QUANTITY * HOLDINGS.AVG_COST
- For "client name", specify: CLIENTS.FULL_NAME (NOT "client name" or "CLIENT_NAME")
- For "most recent transaction", specify: MAX(TRANSACTIONS.TRADE_DATE)

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
            validation_results = state.get("validation_results", {})

            # Extract programmatic check results if available
            schema_validation = validation_results.get("schema", {})
            programmatic_check = schema_validation.get("programmatic_check", {})
            missing_columns = programmatic_check.get("missing_columns", [])

            validation_feedback = f"""

═══════════════════════════════════════════════════════════════════════════
⚠️  CRITICAL: PREVIOUS QUERY FAILED VALIDATION (Attempt {retry_count})
═══════════════════════════════════════════════════════════════════════════

ISSUES FOUND:
{chr(10).join(f'  ✗ {issue}' for issue in issues)}

{"INVALID COLUMNS DETECTED: " + ", ".join(missing_columns) if missing_columns else ""}

SUGGESTIONS FOR FIX:
{chr(10).join(f'  → {suggestion}' for suggestion in suggestions)}

COLUMN REFERENCE GUIDE:
- For client name: Use CLIENTS.FULL_NAME (not CLIENT_NAME)
- For portfolio value: Calculate SUM(HOLDINGS.QUANTITY * HOLDINGS.AVG_COST)
- For transaction value: Calculate TRANSACTIONS.QUANTITY * TRANSACTIONS.PRICE
- For asset count: Use COUNT(DISTINCT HOLDINGS.ASSET_ID)
- For recent date: Use MAX(TRANSACTIONS.TRADE_DATE) or MAX(HOLDINGS.LAST_UPDATED)

═══════════════════════════════════════════════════════════════════════════
GENERATE A CORRECTED QUERY USING ONLY VALID COLUMNS FROM THE SCHEMA
═══════════════════════════════════════════════════════════════════════════
"""

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

═══════════════════════════════════════════════════════════════════════════
VALID COLUMNS PER TABLE (USE ONLY THESE):
═══════════════════════════════════════════════════════════════════════════
CLIENTS: CLIENT_ID, FULL_NAME, COUNTRY, RISK_PROFILE, ONBOARDING_DATE, KYC_STATUS
PORTFOLIOS: PORTFOLIO_ID, CLIENT_ID, PORTFOLIO_NAME, BASE_CURRENCY, INCEPTION_DATE, STATUS
ASSETS: ASSET_ID, SYMBOL, ASSET_NAME, ASSET_TYPE, CURRENCY, EXCHANGE
TRANSACTIONS: TRANSACTION_ID, PORTFOLIO_ID, ASSET_ID, TRADE_DATE, TRANSACTION_TYPE, QUANTITY, PRICE, FEES, CURRENCY, CREATED_AT
HOLDINGS: PORTFOLIO_ID, ASSET_ID, QUANTITY, AVG_COST, LAST_UPDATED
═══════════════════════════════════════════════════════════════════════════

CRITICAL: Only use columns listed above. Do NOT use:
- CLIENT_NAME (use FULL_NAME)
- PORTFOLIO_VALUE (calculate: QUANTITY * AVG_COST)
- TOTAL_VALUE (must be calculated)
- Any column not explicitly listed

Generate a SQL query that:
1. Uses ONLY the correct table and column names from the schema above
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
            # Handle potential markdown code block wrapping
            content = response.content.strip()
            if content.startswith("```"):
                # Remove markdown code block (```json or ```)
                lines = content.split("\n")
                # Remove first line (```json) and last line (```)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            result = json.loads(content)
            sql = result.get("sql", "")
            explanation = result.get("explanation", "")
        except json.JSONDecodeError:
            # Try to extract SQL from response
            sql = response.content.strip()
            explanation = ""
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```json" in sql:
                # Try to parse JSON from markdown block
                try:
                    json_content = sql.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_content)
                    sql = result.get("sql", "")
                    explanation = result.get("explanation", "")
                except:
                    pass

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

        # Get strict schema reference for complex queries
        strict_schema = self._get_strict_schema_reference()

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

{strict_schema}

ADDITIONAL SCHEMA CONTEXT:
{schema_str}
{followup_section}{validation_feedback}

═══════════════════════════════════════════════════════════════════════════
CRITICAL RULES - READ CAREFULLY BEFORE GENERATING SQL
═══════════════════════════════════════════════════════════════════════════

1. **ONLY USE COLUMNS THAT EXIST** - Every column in your SQL MUST exist in the schema above.
   DO NOT invent columns like:
   - PORTFOLIO_VALUE (does not exist - calculate from HOLDINGS.QUANTITY * HOLDINGS.AVG_COST)
   - TOTAL_VALUE (does not exist - must be calculated)
   - CLIENT_NAME (use FULL_NAME from CLIENTS)
   - ASSET_VALUE (does not exist - calculate from QUANTITY * AVG_COST or QUANTITY * PRICE)

2. **For portfolio value calculations:**
   - Use: SUM(h.QUANTITY * h.AVG_COST) AS portfolio_value
   - From: HOLDINGS table joined with PORTFOLIOS

3. **For transaction value calculations:**
   - Use: (t.QUANTITY * t.PRICE) AS transaction_value
   - From: TRANSACTIONS table

4. **For most frequent/common aggregations:**
   - Use: COUNT(*) with GROUP BY and ORDER BY DESC LIMIT 1
   - Or use window functions with RANK()

5. **CTE column aliases:**
   - When you create a calculated column in a CTE (e.g., SUM(...) AS total_value)
   - That alias becomes available in subsequent CTEs or final SELECT
   - BUT base table columns remain unchanged

6. **Always prefix columns with table alias** (e.g., c.FULL_NAME, p.PORTFOLIO_ID)

═══════════════════════════════════════════════════════════════════════════

Generate a complex SQL query using CTEs (Common Table Expressions) that:
1. Creates separate CTEs for each logical subtask
2. Uses proper JOINs between CTEs
3. Synthesizes all subtasks into a final result
4. Uses meaningful CTE and column names
5. Is syntactically correct for DuckDB
6. ONLY uses columns that exist in the schema above

STRUCTURE EXAMPLE:
WITH
    subtask_1 AS (
        -- Query for subtask 1 using ONLY real columns
        SELECT p.PORTFOLIO_ID, SUM(h.QUANTITY * h.AVG_COST) AS calculated_value
        FROM PORTFOLIOS p
        JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID
        GROUP BY p.PORTFOLIO_ID
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
    "columns_used": ["List every column from base tables you used (e.g., CLIENTS.FULL_NAME, HOLDINGS.QUANTITY)"],
    "explanation": "How the CTEs combine to answer the query"
}}

Respond ONLY with the JSON, no additional text."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            # Handle potential markdown code block wrapping
            content = response.content.strip()
            if content.startswith("```"):
                # Remove markdown code block (```json or ```)
                lines = content.split("\n")
                # Remove first line (```json) and last line (```)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            result = json.loads(content)
            sql = result.get("sql", "")
            sub_queries = result.get("sub_queries", [])
            explanation = result.get("explanation", "")
        except json.JSONDecodeError:
            sql = response.content.strip()
            explanation = ""
            sub_queries = []
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```json" in sql:
                # Try to parse JSON from markdown block
                try:
                    json_content = sql.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_content)
                    sql = result.get("sql", "")
                    sub_queries = result.get("sub_queries", [])
                    explanation = result.get("explanation", "")
                except:
                    pass

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
        """Agent 5.2: Schema Validation - Hybrid programmatic + LLM validation"""

        # Step 1: Programmatic validation - extract columns used and check against schema
        valid_columns = self._get_all_valid_columns()
        columns_used = self._extract_columns_from_sql(sql)

        programmatic_issues = []
        programmatic_missing = []

        for table, cols in columns_used.items():
            if table not in valid_columns:
                # Check if it might be a CTE name (not a base table)
                # CTEs are defined in the WITH clause
                if table not in self._extract_cte_names(sql):
                    programmatic_issues.append(f"Unknown table referenced: {table}")
            else:
                for col in cols:
                    if col not in valid_columns[table]:
                        programmatic_missing.append(f"{table}.{col}")
                        programmatic_issues.append(f"Column '{col}' does not exist in table '{table}'. Valid columns are: {', '.join(valid_columns[table])}")

        # Build validation context with programmatic findings
        programmatic_context = ""
        if programmatic_issues:
            programmatic_context = f"""

PROGRAMMATIC VALIDATION FOUND THESE ISSUES:
{chr(10).join(f'- {issue}' for issue in programmatic_issues)}

Please verify these findings and check for any additional issues."""

        # Step 2: LLM validation for semantic/logical issues
        prompt = f"""You are a STRICT database schema validation expert. Verify that EVERY column and table in the SQL exists in the schema.

SQL QUERY:
{sql}

AVAILABLE SCHEMA (ONLY these tables and columns exist):
{schema_str}
{programmatic_context}

STRICT VALIDATION RULES:
1. ONLY columns explicitly listed in the schema exist
2. Columns like PORTFOLIO_VALUE, TOTAL_VALUE, CLIENT_NAME do NOT exist unless in schema
3. Calculated columns in CTEs (e.g., SUM(...) AS total) are valid aliases, but base columns must exist
4. Verify every table.column reference against the schema

VALID COLUMNS PER TABLE:
- CLIENTS: CLIENT_ID, FULL_NAME, COUNTRY, RISK_PROFILE, ONBOARDING_DATE, KYC_STATUS
- PORTFOLIOS: PORTFOLIO_ID, CLIENT_ID, PORTFOLIO_NAME, BASE_CURRENCY, INCEPTION_DATE, STATUS
- ASSETS: ASSET_ID, SYMBOL, ASSET_NAME, ASSET_TYPE, CURRENCY, EXCHANGE
- TRANSACTIONS: TRANSACTION_ID, PORTFOLIO_ID, ASSET_ID, TRADE_DATE, TRANSACTION_TYPE, QUANTITY, PRICE, FEES, CURRENCY, CREATED_AT
- HOLDINGS: PORTFOLIO_ID, ASSET_ID, QUANTITY, AVG_COST, LAST_UPDATED

Check for:
1. Invalid column names (columns that don't exist in base tables)
2. Wrong table references
3. Incorrect JOIN conditions
4. Column name typos

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0 to 1.0,
    "tables_found": ["list of tables used"],
    "columns_verified": ["list of columns that exist"],
    "columns_missing": ["list of columns that DO NOT exist in schema"],
    "issues": ["List of schema issues found"],
    "suggestions": ["Specific fixes - e.g., 'Use FULL_NAME instead of CLIENT_NAME'"]
}}

Respond ONLY with the JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            # Handle potential markdown wrapping
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            result = json.loads(content)

            # Merge programmatic findings with LLM findings
            llm_issues = result.get("issues", [])
            llm_missing = result.get("columns_missing", [])

            # Add programmatic issues if not already present
            for issue in programmatic_issues:
                if issue not in llm_issues:
                    llm_issues.append(issue)

            for col in programmatic_missing:
                if col not in llm_missing:
                    llm_missing.append(col)

            # Adjust confidence based on programmatic findings
            confidence = result.get("confidence", 0.7)
            if programmatic_missing:
                # Significant penalty for missing columns found programmatically
                confidence = min(confidence, 0.3)
                result["is_valid"] = False

            result["issues"] = llm_issues
            result["columns_missing"] = llm_missing
            result["confidence"] = confidence
            result["programmatic_check"] = {
                "issues_found": len(programmatic_issues),
                "missing_columns": programmatic_missing
            }

            return result
        except json.JSONDecodeError:
            # If LLM response parsing fails, return programmatic results
            is_valid = len(programmatic_issues) == 0
            return {
                "is_valid": is_valid,
                "confidence": 0.9 if is_valid else 0.2,
                "issues": programmatic_issues,
                "columns_missing": programmatic_missing,
                "suggestions": ["Fix the column references listed in issues"]
            }

    def _extract_cte_names(self, sql: str) -> set:
        """Extract CTE names from a SQL query."""
        import re
        # Pattern: name AS ( in WITH clause
        cte_pattern = r'(\w+)\s+AS\s*\('
        sql_upper = sql.upper()

        # Only look at the WITH clause part
        if 'WITH' in sql_upper:
            # Find everything between WITH and the final SELECT
            with_match = re.search(r'WITH\s+(.*?)\s+SELECT', sql_upper, re.DOTALL)
            if with_match:
                with_clause = with_match.group(1)
                cte_names = set()
                for match in re.finditer(cte_pattern, with_clause):
                    cte_names.add(match.group(1))
                return cte_names
        return set()

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
    def run(
        self,
        query: str,
        short_term_memory: Optional[str] = None,
        long_term_memory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the multi-agent workflow with a given query.

        Args:
            query: The user's natural language query
            short_term_memory: Recent conversation history (optional)
            long_term_memory: Historical summaries (optional)

        Returns:
            The final state including the answer and execution trace
        """
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=query)],
            "nl_query": query,
            # Memory context
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory,
            # Agent1 outputs
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

    def stream(
        self,
        query: str,
        short_term_memory: Optional[str] = None,
        long_term_memory: Optional[str] = None
    ):
        """
        Stream the multi-agent execution for real-time updates.

        Args:
            query: The user's natural language query
            short_term_memory: Recent conversation history (optional)
            long_term_memory: Historical summaries (optional)

        Yields:
            State updates as they occur
        """
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=query)],
            "nl_query": query,
            # Memory context
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory,
            # Agent1 outputs
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
