"""
Multi-EDA Orchestrator
======================

12-agent LangGraph orchestrator for comprehensive Exploratory Data Analysis.
Agents run sequentially through a state graph, each performing a specific EDA phase.

Agent Workflow:
1. Intent Classification - Determine EDA vs general question
2. Schema Extraction - Get table schemas from DB or JSON
3. Data Loading - Load tables into pandas with row limits
4. Join & Target Detection - Detect joins and target variable
5. Structure Inspection - Shape, dtypes, head, column classification
6. Descriptive Statistics - Numerical and categorical stats
7. Distribution Analysis - Histograms (7.1), Individual histograms (7.2), Countplots (7.3)
8. Segmentation Analysis - Boxplots (8.1), Violin plots (8.2), LM plots (8.3)
9. Outlier Detection - IQR-based outlier detection with plots
10. Correlation Analysis - Correlation matrix and heatmap
11. Deep Analysis - LLM-synthesized insights and suggestions
12. Dashboard Generation - Self-contained HTML dashboard with embedded plots
"""

from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict, Any
from datetime import datetime
import json
import os
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from core.config import FrameworkConfig, get_config

# Load schema document path
SCHEMA_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "sample_files", "wealth_tables.json")


# =============================================================================
# TRACE TYPE
# =============================================================================

class AgentTrace(TypedDict):
    """Trace information for a single agent step."""
    agent_id: str
    agent_name: str
    status: str  # "running", "completed", "skipped"
    input_summary: str
    output_summary: str
    details: Dict[str, Any]
    timestamp: str


# =============================================================================
# STATE DEFINITION
# =============================================================================

class MultiEDAState(TypedDict):
    """State that flows through the multi-EDA agent graph."""
    # Core
    messages: Annotated[Sequence[BaseMessage], add_messages]
    nl_query: str
    short_term_memory: Optional[str]
    long_term_memory: Optional[str]

    # Agent1: Intent Classification
    intent: str  # NEW_EDA, MODIFY_EDA, FOLLOWUP_EDA, GENERAL_QUESTION
    general_answer: Optional[str]
    related_tables: List[str]
    target_variable: Optional[str]

    # Agent2: Schema Extraction
    extracted_schema: Dict[str, Any]
    schema_json_path: Optional[str]

    # Agent3: Data Loading
    session_id: str
    loaded_tables: Dict[str, Any]
    row_limit: int
    total_rows_loaded: int

    # Agent4: Join & Target Variable
    has_target_variable: bool
    target_variable_confirmed: str
    join_strategy: Optional[str]
    joined_dataframe_info: Optional[Dict]
    stop_reason: Optional[str]

    # Agent5: Structure Inspection
    structure_summary: str
    shape_info: Dict[str, int]
    dtypes_info: Dict[str, str]
    head_sample: str
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]

    # Agent6: Descriptive Statistics
    descriptive_stats: Dict[str, Any]
    stats_tables_html: List[str]

    # Agent7: Distribution Analysis
    histogram_all_path: Optional[str]
    histogram_individual_paths: List[str]
    countplot_paths: List[str]
    distribution_summary: str
    sparse_classes: List[Dict]

    # Agent8: Segmentation
    boxplot_paths: List[str]
    violinplot_paths: List[str]
    lmplot_paths: List[str]
    segmentation_summary: str

    # Agent9: Outlier Detection
    outlier_plot_paths: List[str]
    outlier_summary: str
    outliers_detected: Dict[str, Any]

    # Agent10: Correlation
    heatmap_path: Optional[str]
    correlation_summary: str
    strong_correlations: List[Dict]

    # Agent11: Deep Analysis
    deep_analysis: str
    cleaning_suggestions: List[str]
    feature_engineering_suggestions: List[str]

    # Agent12: Dashboard
    dashboard_path: Optional[str]

    # Control
    current_agent: str
    is_complete: bool
    error: Optional[str]
    agent_traces: List[AgentTrace]


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class MultiEDAOrchestrator:
    """
    Multi-EDA Orchestrator implementing a 12-agent workflow using LangGraph
    for comprehensive Exploratory Data Analysis with visualization.
    """

    AGENT_INFO = {
        "eda_agent1":   {"name": "Intent Classification Agent",   "icon": "🔍"},
        "eda_agent2":   {"name": "Schema Extraction Agent",       "icon": "📋"},
        "eda_agent3":   {"name": "Data Loading Agent",            "icon": "📥"},
        "eda_agent4":   {"name": "Join & Target Detection Agent", "icon": "🔗"},
        "eda_agent5":   {"name": "Structure Inspection Agent",    "icon": "🏗️"},
        "eda_agent6":   {"name": "Descriptive Statistics Agent",  "icon": "📊"},
        "eda_agent7":   {"name": "Distribution Analysis Agent",   "icon": "📈"},
        "eda_agent7.1": {"name": "Histogram (All Numerical)",     "icon": "📊"},
        "eda_agent7.2": {"name": "Individual Histograms",         "icon": "📉"},
        "eda_agent7.3": {"name": "Categorical Countplots",        "icon": "📶"},
        "eda_agent8":   {"name": "Segmentation Analysis Agent",   "icon": "🎯"},
        "eda_agent8.1": {"name": "Boxplot Analysis",              "icon": "📦"},
        "eda_agent8.2": {"name": "Violin Plot Analysis",          "icon": "🎻"},
        "eda_agent8.3": {"name": "LM Plot Analysis",              "icon": "📐"},
        "eda_agent9":   {"name": "Outlier Detection Agent",       "icon": "🔎"},
        "eda_agent10":  {"name": "Correlation Analysis Agent",    "icon": "🔥"},
        "eda_agent11":  {"name": "Deep Analysis Agent",           "icon": "🧠"},
        "eda_agent12":  {"name": "Dashboard Generation Agent",    "icon": "🎨"},
    }

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or get_config()
        self.llm = ChatOpenAI(
            model=self.config.model.model_name,
            temperature=self.config.model.temperature,
            api_key=self.config.model.api_key
        )
        self.schema_document = self._load_schema_document()
        self.graph = self._build_graph()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _load_schema_document(self) -> Dict[str, Any]:
        """Load the schema definition document."""
        try:
            if os.path.exists(SCHEMA_DOCUMENT_PATH):
                with open(SCHEMA_DOCUMENT_PATH, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def _get_schema_summary(self) -> str:
        """Get a summary of all tables and their descriptions."""
        if not self.schema_document or "tables" not in self.schema_document:
            return "No schema document available."
        summary = []
        for table in self.schema_document["tables"]:
            summary.append(f"- {table['table_name']}: {table['table_description']}")
            columns = [f"  - {col['column_name']}: {col['description']}"
                       for col in table['columns'][:5]]
            summary.extend(columns)
            if len(table['columns']) > 5:
                summary.append(f"  ... and {len(table['columns']) - 5} more columns")
        return "\n".join(summary)

    def _create_trace(self, agent_id: str, status: str, input_summary: str,
                      output_summary: str, details: Dict[str, Any] = None) -> AgentTrace:
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

    def _add_trace(self, state: MultiEDAState, trace: AgentTrace) -> List[AgentTrace]:
        """Add a trace to the state's trace list."""
        traces = list(state.get("agent_traces", []))
        traces.append(trace)
        return traces

    def _parse_llm_json(self, content: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def _extract_row_limit(self, query: str) -> int:
        """Extract row limit from user query, default 50000."""
        patterns = [
            r'(\d+)\s*rows?',
            r'limit\s*(\d+)',
            r'top\s*(\d+)',
            r'first\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                limit = int(match.group(1))
                if limit > 0:
                    return min(limit, 100000)
        return 50000

    def _invoke_tool(self, tool_func, **kwargs) -> str:
        """Invoke a tool function and return its result string."""
        try:
            return tool_func.invoke(kwargs)
        except Exception as e:
            return f"Error: {str(e)}"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for the 12-agent EDA workflow."""
        workflow = StateGraph(MultiEDAState)

        # Add nodes
        workflow.add_node("agent1_intent", self._agent1_intent_classification)
        workflow.add_node("agent2_schema", self._agent2_schema_extraction)
        workflow.add_node("agent3_loading", self._agent3_data_loading)
        workflow.add_node("agent4_join_target", self._agent4_join_target)
        workflow.add_node("agent5_structure", self._agent5_structure_inspection)
        workflow.add_node("agent6_stats", self._agent6_descriptive_statistics)
        workflow.add_node("agent7_distribution", self._agent7_distribution)
        workflow.add_node("agent8_segmentation", self._agent8_segmentation)
        workflow.add_node("agent9_outliers", self._agent9_outlier_detection)
        workflow.add_node("agent10_correlation", self._agent10_correlation)
        workflow.add_node("agent11_deep", self._agent11_deep_analysis)
        workflow.add_node("agent12_dashboard", self._agent12_dashboard)
        workflow.add_node("final_response", self._final_response_node)

        # Entry point
        workflow.set_entry_point("agent1_intent")

        # Conditional after agent1
        workflow.add_conditional_edges(
            "agent1_intent",
            self._route_after_intent,
            {
                "schema_extraction": "agent2_schema",
                "end": "final_response"
            }
        )

        # Linear: agent2 -> agent3 -> agent4
        workflow.add_edge("agent2_schema", "agent3_loading")
        workflow.add_edge("agent3_loading", "agent4_join_target")

        # Conditional after agent4
        workflow.add_conditional_edges(
            "agent4_join_target",
            self._route_after_join_target,
            {
                "proceed": "agent5_structure",
                "stop": "final_response"
            }
        )

        # Linear: agent5 -> agent6 -> ... -> agent12 -> final
        workflow.add_edge("agent5_structure", "agent6_stats")
        workflow.add_edge("agent6_stats", "agent7_distribution")
        workflow.add_edge("agent7_distribution", "agent8_segmentation")
        workflow.add_edge("agent8_segmentation", "agent9_outliers")
        workflow.add_edge("agent9_outliers", "agent10_correlation")
        workflow.add_edge("agent10_correlation", "agent11_deep")
        workflow.add_edge("agent11_deep", "agent12_dashboard")
        workflow.add_edge("agent12_dashboard", "final_response")

        # Final -> END
        workflow.add_edge("final_response", END)

        return workflow.compile()

    # =========================================================================
    # ROUTING
    # =========================================================================

    def _route_after_intent(self, state: MultiEDAState) -> Literal["schema_extraction", "end"]:
        """Route after intent classification."""
        intent = state.get("intent", "NEW_EDA")
        if intent in ("NEW_EDA", "MODIFY_EDA"):
            return "schema_extraction"
        return "end"

    def _route_after_join_target(self, state: MultiEDAState) -> Literal["proceed", "stop"]:
        """Route after join & target detection."""
        if state.get("stop_reason"):
            return "stop"
        return "proceed"

    # =========================================================================
    # AGENT 1: Intent Classification
    # =========================================================================

    def _agent1_intent_classification(self, state: MultiEDAState) -> Dict[str, Any]:
        """Classify user intent as EDA request or general question."""
        nl_query = state["nl_query"]
        schema_summary = self._get_schema_summary()
        short_term_memory = state.get("short_term_memory") or ""
        long_term_memory = state.get("long_term_memory") or ""

        # Build memory context
        memory_context = ""
        if short_term_memory:
            memory_context += f"\nRECENT CONVERSATION CONTEXT:\n{short_term_memory[:1500]}\n"
        if long_term_memory:
            memory_context += f"\nLONG-TERM MEMORY:\n{long_term_memory[:1000]}\n"

        prompt = f"""You are an EDA (Exploratory Data Analysis) intent classifier for a multi-agent system.

USER QUESTION: {nl_query}

AVAILABLE DATABASE TABLES:
{schema_summary}
{memory_context}

Your task is to classify the user's intent into one of these categories:

1. **NEW_EDA**: The user wants to perform Exploratory Data Analysis on one or more tables.
   Examples: "Do EDA on TRANSACTIONS", "Analyze the clients table", "Run a complete analysis on HOLDINGS",
   "Show me distributions for the assets data", "Check for outliers in transactions"

2. **GENERAL_QUESTION**: A question that does NOT require performing EDA. This includes:
   - Questions about the database: "What tables are available?", "What columns does CLIENTS have?"
   - Questions about EDA concepts: "What is EDA?", "What is a correlation heatmap?"
   - Questions about the system: "What can you do?", "How does this agent work?"
   - Follow-up questions about previous analysis
   - General knowledge questions

CLASSIFICATION RULES:
- If the user mentions analyzing, exploring, profiling, or doing EDA on specific tables → NEW_EDA
- If the user asks about table contents, schema, or metadata without requesting analysis → GENERAL_QUESTION
- If the user asks conceptual or informational questions → GENERAL_QUESTION
- If uncertain, default to NEW_EDA only if table names are explicitly mentioned with analysis intent

Also extract:
- **related_tables**: Which specific tables the user wants to analyze (empty list if GENERAL_QUESTION)
- **target_variable**: Any column explicitly mentioned as the target/dependent variable (null if not mentioned)
- **row_limit**: Any explicit row limit mentioned (null if not mentioned)

Respond in JSON format:
{{
    "intent": "NEW_EDA" or "GENERAL_QUESTION",
    "related_tables": ["TABLE1", "TABLE2"],
    "target_variable": "column_name or null",
    "row_limit": null,
    "reasoning": "Brief explanation of why this intent was chosen"
}}

Respond ONLY with valid JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._parse_llm_json(response.content)

        intent = result.get("intent", "NEW_EDA")
        related_tables = result.get("related_tables", [])
        target_variable = result.get("target_variable")
        general_answer = None

        # For GENERAL_QUESTION: make a dedicated LLM call to answer the question directly
        if intent == "GENERAL_QUESTION":
            # Build richer memory context for general questions (more budget since no EDA pipeline)
            gq_memory = ""
            if short_term_memory:
                gq_memory += f"\nRECENT CONVERSATION HISTORY (use this to answer follow-up questions):\n{short_term_memory[:3000]}\n"
            if long_term_memory:
                gq_memory += f"\nLONG-TERM MEMORY (past sessions - use if the user refers to earlier analysis):\n{long_term_memory[:2000]}\n"

            answer_prompt = f"""You are a helpful assistant. Answer the user's question directly and concisely.

USER QUESTION: {nl_query}
{gq_memory}

BACKGROUND KNOWLEDGE (use ONLY if directly relevant to the question - do NOT list these unprompted):
{schema_summary}

RULES:
- Answer ONLY what the user asked. Do not volunteer extra information they did not request.
- If the user is asking a follow-up question (e.g., "what was the target variable?", "how many rows?", "tell me more about that"), use the CONVERSATION HISTORY and LONG-TERM MEMORY above to provide the answer.
- Do NOT list database tables, relationships, or system capabilities unless the user explicitly asked about them.
- Do NOT describe what the system can do unless the user explicitly asked "what can you do" or similar.
- Keep the answer focused, clear, and concise.
- Use markdown formatting for readability."""

            answer_response = self.llm.invoke([HumanMessage(content=answer_prompt)])
            # Append a polite one-liner about EDA capability
            general_answer = answer_response.content.strip() + \
                "\n\n---\n*Tip: You can also ask me to perform a complete EDA on any table " \
                "(e.g., \"Do EDA on TRANSACTIONS\") and I will generate a full analysis with visualizations and a dashboard.*"

        trace = self._create_trace(
            agent_id="eda_agent1",
            status="completed",
            input_summary=f"Query: {nl_query[:100]}{'...' if len(nl_query) > 100 else ''}",
            output_summary=f"Intent: {intent}. Tables: {', '.join(related_tables) if related_tables else 'none'}"
                           + (f". Answer generated." if intent == "GENERAL_QUESTION" else ""),
            details={"intent": intent, "tables": related_tables, "target": target_variable}
        )

        return {
            "messages": [response],
            "intent": intent,
            "general_answer": general_answer,
            "related_tables": related_tables,
            "target_variable": target_variable,
            "current_agent": "agent1_intent",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 2: Schema Extraction
    # =========================================================================

    def _agent2_schema_extraction(self, state: MultiEDAState) -> Dict[str, Any]:
        """Extract schema for the related tables."""
        from tools.multi_eda_tools import eda_get_table_schema, eda_get_all_tables

        related_tables = state.get("related_tables", [])
        schema_json_path = state.get("schema_json_path")

        extracted_schema = {"tables": {}}

        if schema_json_path:
            from tools.multi_eda_tools import eda_get_schema_from_json
            schema_result = self._invoke_tool(eda_get_schema_from_json, json_path=schema_json_path)
            extracted_schema["raw"] = schema_result
        elif related_tables:
            for table in related_tables:
                result = self._invoke_tool(eda_get_table_schema, table_name=table)
                extracted_schema["tables"][table] = result
        else:
            all_tables = self._invoke_tool(eda_get_all_tables)
            extracted_schema["all_tables"] = all_tables

        trace = self._create_trace(
            agent_id="eda_agent2",
            status="completed",
            input_summary=f"Tables: {', '.join(related_tables) if related_tables else 'all'}",
            output_summary=f"Schema extracted for {len(extracted_schema.get('tables', {}))} tables",
            details={"tables_extracted": related_tables or list(extracted_schema.get("tables", {}).keys())}
        )

        return {
            "extracted_schema": extracted_schema,
            "current_agent": "agent2_schema",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 3: Data Loading
    # =========================================================================

    def _agent3_data_loading(self, state: MultiEDAState) -> Dict[str, Any]:
        """Load tables into pandas DataFrames."""
        from tools.multi_eda_tools import eda_load_table, eda_load_multiple_tables

        related_tables = state.get("related_tables", [])
        row_limit = self._extract_row_limit(state.get("nl_query", ""))

        if len(related_tables) > 1:
            result = self._invoke_tool(
                eda_load_multiple_tables,
                table_names=",".join(related_tables),
                row_limit=row_limit
            )
        elif related_tables:
            result = self._invoke_tool(
                eda_load_table,
                table_name=related_tables[0],
                row_limit=row_limit
            )
        else:
            result = "Error: No tables specified for loading."

        # Extract session_id from result
        session_id = ""
        total_rows = 0
        loaded_tables = {}
        for line in result.split("\n"):
            if line.startswith("SESSION_ID:"):
                session_id = line.split(":")[1].strip()
            if "Rows loaded:" in line:
                try:
                    total_rows = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "Total rows" in line:
                try:
                    total_rows = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        trace = self._create_trace(
            agent_id="eda_agent3",
            status="completed",
            input_summary=f"Loading {', '.join(related_tables)} (limit: {row_limit})",
            output_summary=f"Session: {session_id}. {result.split(chr(10))[1] if chr(10) in result else result[:100]}",
            details={"session_id": session_id, "row_limit": row_limit, "tables": related_tables}
        )

        return {
            "session_id": session_id,
            "row_limit": row_limit,
            "total_rows_loaded": total_rows,
            "loaded_tables": loaded_tables,
            "current_agent": "agent3_loading",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 4: Join & Target Variable Detection
    # =========================================================================

    def _agent4_join_target(self, state: MultiEDAState) -> Dict[str, Any]:
        """Detect target variable and handle multi-table joins."""
        from tools.multi_eda_tools import (
            eda_detect_target_variable, eda_detect_joins,
            eda_join_tables, eda_validate_target
        )

        session_id = state.get("session_id", "")
        related_tables = state.get("related_tables", [])
        user_target = state.get("target_variable")

        # Detect target variable
        if user_target:
            target_result = self._invoke_tool(eda_validate_target, session_id=session_id, target_col=user_target)
            has_target = "VALIDATED" in target_result
            target_confirmed = user_target if has_target else ""
        else:
            target_result = self._invoke_tool(eda_detect_target_variable, session_id=session_id)
            has_target = "NO_TARGET_FOUND" not in target_result
            # Extract top suggestion
            target_confirmed = ""
            if has_target:
                for line in target_result.split("\n"):
                    line = line.strip()
                    if line and line[0].isdigit() and "." in line:
                        # Extract column name from "1. COLUMN_NAME (score=X)"
                        parts = line.split(".")
                        if len(parts) > 1:
                            target_confirmed = parts[1].strip().split("(")[0].strip()
                            break

        # Handle multi-table joins
        join_strategy = None
        joined_info = None
        stop_reason = None

        if len(related_tables) > 1:
            join_result = self._invoke_tool(
                eda_detect_joins,
                session_id=session_id,
                table_names=",".join(related_tables)
            )

            if "NO_JOINS_FOUND" in join_result:
                stop_reason = "No valid joins found between tables. Cannot proceed with multi-table EDA."
            else:
                # Use LLM to construct the JOIN SQL
                prompt = f"""Given these detected joins between tables:
{join_result}

Tables: {', '.join(related_tables)}

Generate a single SQL SELECT * JOIN query that combines all tables.
Return ONLY the SQL query, nothing else."""

                response = self.llm.invoke([HumanMessage(content=prompt)])
                join_sql = response.content.strip()
                # Clean markdown
                if join_sql.startswith("```"):
                    join_sql = join_sql.split("```")[1]
                    if join_sql.startswith("sql"):
                        join_sql = join_sql[3:]
                    join_sql = join_sql.strip()

                join_result2 = self._invoke_tool(eda_join_tables, session_id=session_id, join_sql=join_sql)
                join_strategy = join_sql
                if "Error" in join_result2:
                    stop_reason = f"Join failed: {join_result2}"
                else:
                    joined_info = {"result": join_result2}

        output_parts = []
        if has_target:
            output_parts.append(f"Target: {target_confirmed}")
        else:
            output_parts.append("No target variable detected (unsupervised)")
        if join_strategy:
            output_parts.append("Tables joined successfully")
        if stop_reason:
            output_parts.append(f"STOPPED: {stop_reason[:80]}")

        trace = self._create_trace(
            agent_id="eda_agent4",
            status="completed" if not stop_reason else "failed",
            input_summary=f"Session: {session_id}, Tables: {', '.join(related_tables)}",
            output_summary="; ".join(output_parts),
            details={"has_target": has_target, "target": target_confirmed,
                     "stop_reason": stop_reason, "join_strategy": join_strategy is not None}
        )

        return {
            "has_target_variable": has_target,
            "target_variable_confirmed": target_confirmed,
            "join_strategy": join_strategy,
            "joined_dataframe_info": joined_info,
            "stop_reason": stop_reason,
            "current_agent": "agent4_join_target",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 5: Structure Inspection
    # =========================================================================

    def _agent5_structure_inspection(self, state: MultiEDAState) -> Dict[str, Any]:
        """Inspect DataFrame structure: shape, dtypes, head, column classification."""
        from tools.multi_eda_tools import eda_get_shape, eda_get_dtypes, eda_get_head, eda_classify_columns

        session_id = state.get("session_id", "")

        shape_result = self._invoke_tool(eda_get_shape, session_id=session_id)
        dtypes_result = self._invoke_tool(eda_get_dtypes, session_id=session_id)
        head_result = self._invoke_tool(eda_get_head, session_id=session_id, n=10)
        classify_result = self._invoke_tool(eda_classify_columns, session_id=session_id)

        # Parse shape
        shape_info = {"rows": 0, "columns": 0}
        if "rows" in shape_result and "columns" in shape_result:
            parts = shape_result.replace("Shape:", "").strip().split("x")
            if len(parts) == 2:
                try:
                    shape_info["rows"] = int(parts[0].strip().split()[0])
                    shape_info["columns"] = int(parts[1].strip().split()[0])
                except ValueError:
                    pass

        # Parse column types from classify result
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        for line in classify_result.split("\n"):
            if "Numeric" in line and ":" in line:
                cols_str = line.split(":", 1)[1].strip()
                if cols_str != "None":
                    numeric_cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            elif "Categorical" in line and ":" in line:
                cols_str = line.split(":", 1)[1].strip()
                if cols_str != "None":
                    categorical_cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            elif "Datetime" in line and ":" in line:
                cols_str = line.split(":", 1)[1].strip()
                if cols_str != "None":
                    datetime_cols = [c.strip() for c in cols_str.split(",") if c.strip()]

        summary = f"{shape_result}\n{classify_result}"

        trace = self._create_trace(
            agent_id="eda_agent5",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"{shape_result}. Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Datetime: {len(datetime_cols)}",
            details={"shape": shape_info, "numeric_count": len(numeric_cols),
                     "categorical_count": len(categorical_cols), "datetime_count": len(datetime_cols)}
        )

        return {
            "structure_summary": summary,
            "shape_info": shape_info,
            "dtypes_info": {},
            "head_sample": head_result,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "current_agent": "agent5_structure",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 6: Descriptive Statistics
    # =========================================================================

    def _agent6_descriptive_statistics(self, state: MultiEDAState) -> Dict[str, Any]:
        """Generate descriptive statistics for numerical and categorical columns."""
        from tools.multi_eda_tools import eda_describe_numerical, eda_describe_categorical, eda_generate_stats_table_html

        session_id = state.get("session_id", "")

        num_result = self._invoke_tool(eda_describe_numerical, session_id=session_id)
        cat_result = self._invoke_tool(eda_describe_categorical, session_id=session_id)
        html_result = self._invoke_tool(eda_generate_stats_table_html, session_id=session_id)

        trace = self._create_trace(
            agent_id="eda_agent6",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"Numerical stats computed. Categorical stats computed. HTML tables generated.",
            details={"has_numerical": "No numerical" not in num_result,
                     "has_categorical": "No categorical" not in cat_result}
        )

        return {
            "descriptive_stats": {"numerical": num_result, "categorical": cat_result},
            "stats_tables_html": [html_result],
            "current_agent": "agent6_stats",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 7: Distribution Analysis (with sub-agents 7.1, 7.2, 7.3)
    # =========================================================================

    def _agent7_distribution(self, state: MultiEDAState) -> Dict[str, Any]:
        """Distribution analysis with sub-agents for histograms and countplots."""
        from tools.multi_eda_tools import (
            eda_plot_all_histograms, eda_plot_all_individual_histograms, eda_plot_countplots
        )

        session_id = state.get("session_id", "")
        traces = list(state.get("agent_traces", []))

        # 7.1: All histograms grid
        hist_all_result = self._invoke_tool(eda_plot_all_histograms, session_id=session_id)
        hist_all_path = None
        if "SAVED:" in hist_all_result:
            hist_all_path = hist_all_result.split("SAVED:")[1].strip().split("\n")[0]

        traces.append(self._create_trace(
            agent_id="eda_agent7.1",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=hist_all_result[:120],
            details={"path": hist_all_path}
        ))

        # 7.2: Individual histograms
        hist_indiv_result = self._invoke_tool(eda_plot_all_individual_histograms, session_id=session_id)
        hist_indiv_paths = []
        for line in hist_indiv_result.split("\n"):
            line = line.strip()
            if line.startswith("sample_files/") or line.startswith("SAVED"):
                if "sample_files/" in line:
                    path = line.split("sample_files/")[0] + "sample_files/" + line.split("sample_files/")[1] if "sample_files/" in line else line
                    # Clean path
                    for part in line.split("\n"):
                        part = part.strip()
                        if part.endswith(".png"):
                            hist_indiv_paths.append(part)

        traces.append(self._create_trace(
            agent_id="eda_agent7.2",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"Generated {len(hist_indiv_paths)} individual histograms" if hist_indiv_paths else hist_indiv_result[:120],
            details={"count": len(hist_indiv_paths)}
        ))

        # 7.3: Countplots
        countplot_result = self._invoke_tool(eda_plot_countplots, session_id=session_id)
        countplot_paths = []
        sparse_classes = []
        for line in countplot_result.split("\n"):
            line = line.strip()
            if line.endswith(".png"):
                countplot_paths.append(line)
        if "SPARSE CLASSES" in countplot_result:
            sparse_part = countplot_result.split("SPARSE CLASSES")[1]
            try:
                count = int(sparse_part.split(":")[1].strip().split()[0])
                sparse_classes = [{"count": count}]
            except (ValueError, IndexError):
                pass

        traces.append(self._create_trace(
            agent_id="eda_agent7.3",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"Generated {len(countplot_paths)} countplots" if countplot_paths else countplot_result[:120],
            details={"count": len(countplot_paths), "sparse_classes": len(sparse_classes)}
        ))

        # Synthesize distribution summary via LLM
        prompt = f"""Summarize the distribution analysis results:

Histograms (All): {hist_all_result[:300]}
Individual Histograms: {hist_indiv_result[:300]}
Countplots: {countplot_result[:300]}

Provide a brief summary (3-5 bullet points) of key distribution findings."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        distribution_summary = response.content

        # Main agent 7 trace
        traces.append(self._create_trace(
            agent_id="eda_agent7",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"Distribution analysis complete. Histograms + countplots generated.",
            details={"hist_all": hist_all_path is not None,
                     "hist_individual": len(hist_indiv_paths),
                     "countplots": len(countplot_paths)}
        ))

        return {
            "histogram_all_path": hist_all_path,
            "histogram_individual_paths": hist_indiv_paths,
            "countplot_paths": countplot_paths,
            "distribution_summary": distribution_summary,
            "sparse_classes": sparse_classes,
            "current_agent": "agent7_distribution",
            "agent_traces": traces
        }

    # =========================================================================
    # AGENT 8: Segmentation Analysis (with sub-agents 8.1, 8.2, 8.3)
    # =========================================================================

    def _agent8_segmentation(self, state: MultiEDAState) -> Dict[str, Any]:
        """Segmentation analysis with boxplots, violin plots, and lmplots."""
        from tools.multi_eda_tools import eda_plot_boxplots, eda_plot_violinplots, eda_plot_lmplots

        session_id = state.get("session_id", "")
        target = state.get("target_variable_confirmed", "")
        traces = list(state.get("agent_traces", []))

        boxplot_paths = []
        violin_paths = []
        lmplot_paths = []

        if not target:
            # No target variable - skip segmentation
            traces.append(self._create_trace(
                agent_id="eda_agent8",
                status="skipped",
                input_summary=f"Session: {session_id}",
                output_summary="Skipped: No target variable for segmentation.",
                details={"reason": "no_target"}
            ))
            return {
                "boxplot_paths": [],
                "violinplot_paths": [],
                "lmplot_paths": [],
                "segmentation_summary": "Segmentation skipped - no target variable detected.",
                "current_agent": "agent8_segmentation",
                "agent_traces": traces
            }

        # 8.1: Boxplots
        box_result = self._invoke_tool(eda_plot_boxplots, session_id=session_id, target=target)
        for line in box_result.split("\n"):
            if line.strip().endswith(".png"):
                boxplot_paths.append(line.strip())

        traces.append(self._create_trace(
            agent_id="eda_agent8.1",
            status="completed",
            input_summary=f"Boxplots by {target}",
            output_summary=f"Generated {len(boxplot_paths)} boxplots",
            details={"count": len(boxplot_paths)}
        ))

        # 8.2: Violin plots
        violin_result = self._invoke_tool(eda_plot_violinplots, session_id=session_id, target=target)
        for line in violin_result.split("\n"):
            if line.strip().endswith(".png"):
                violin_paths.append(line.strip())

        traces.append(self._create_trace(
            agent_id="eda_agent8.2",
            status="completed",
            input_summary=f"Violin plots by {target}",
            output_summary=f"Generated {len(violin_paths)} violin plots",
            details={"count": len(violin_paths)}
        ))

        # 8.3: LM plots
        lm_result = self._invoke_tool(eda_plot_lmplots, session_id=session_id, target=target)
        for line in lm_result.split("\n"):
            if line.strip().endswith(".png"):
                lmplot_paths.append(line.strip())

        traces.append(self._create_trace(
            agent_id="eda_agent8.3",
            status="completed",
            input_summary=f"LM plots by {target}",
            output_summary=f"Generated {len(lmplot_paths)} lm plots",
            details={"count": len(lmplot_paths)}
        ))

        # Synthesize
        prompt = f"""Summarize the segmentation analysis:

Target variable: {target}
Boxplots: {box_result[:200]}
Violin plots: {violin_result[:200]}
LM plots: {lm_result[:200]}

Provide 3-5 bullet points about segmentation findings."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        seg_summary = response.content

        traces.append(self._create_trace(
            agent_id="eda_agent8",
            status="completed",
            input_summary=f"Segmentation by {target}",
            output_summary=f"Complete. Boxplots: {len(boxplot_paths)}, Violins: {len(violin_paths)}, LM: {len(lmplot_paths)}",
            details={"target": target}
        ))

        return {
            "boxplot_paths": boxplot_paths,
            "violinplot_paths": violin_paths,
            "lmplot_paths": lmplot_paths,
            "segmentation_summary": seg_summary,
            "current_agent": "agent8_segmentation",
            "agent_traces": traces
        }

    # =========================================================================
    # AGENT 9: Outlier Detection
    # =========================================================================

    def _agent9_outlier_detection(self, state: MultiEDAState) -> Dict[str, Any]:
        """Detect and visualize outliers using IQR method."""
        from tools.multi_eda_tools import eda_detect_outliers_iqr, eda_plot_outlier_boxplots, eda_plot_outlier_scatter

        session_id = state.get("session_id", "")

        # Detect outliers
        iqr_result = self._invoke_tool(eda_detect_outliers_iqr, session_id=session_id)

        # Plot outlier boxplots
        box_result = self._invoke_tool(eda_plot_outlier_boxplots, session_id=session_id)

        # Plot outlier scatter
        scatter_result = self._invoke_tool(eda_plot_outlier_scatter, session_id=session_id)

        plot_paths = []
        for result in [box_result, scatter_result]:
            if "SAVED:" in result:
                path = result.split("SAVED:")[1].strip().split("\n")[0]
                plot_paths.append(path)

        trace = self._create_trace(
            agent_id="eda_agent9",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"Outlier detection complete. {len(plot_paths)} plots generated.",
            details={"plots": len(plot_paths), "method": "IQR"}
        )

        return {
            "outlier_plot_paths": plot_paths,
            "outlier_summary": iqr_result,
            "outliers_detected": {},
            "current_agent": "agent9_outliers",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 10: Correlation Analysis
    # =========================================================================

    def _agent10_correlation(self, state: MultiEDAState) -> Dict[str, Any]:
        """Compute correlations and generate heatmap."""
        from tools.multi_eda_tools import eda_compute_correlations, eda_plot_heatmap

        session_id = state.get("session_id", "")

        corr_result = self._invoke_tool(eda_compute_correlations, session_id=session_id)
        heatmap_result = self._invoke_tool(eda_plot_heatmap, session_id=session_id)

        heatmap_path = None
        if "SAVED:" in heatmap_result:
            heatmap_path = heatmap_result.split("SAVED:")[1].strip().split("\n")[0]

        # Extract strong correlations
        strong_corrs = []
        if "STRONG CORRELATIONS" in corr_result:
            for line in corr_result.split("STRONG CORRELATIONS")[1].split("\n"):
                line = line.strip()
                if "<->" in line:
                    parts = line.split("<->")
                    if len(parts) == 2:
                        col1 = parts[0].strip()
                        rest = parts[1].strip().split(":")
                        col2 = rest[0].strip() if rest else ""
                        val = float(rest[1].strip()) if len(rest) > 1 else 0
                        strong_corrs.append({"col1": col1, "col2": col2, "correlation": val})

        trace = self._create_trace(
            agent_id="eda_agent10",
            status="completed",
            input_summary=f"Session: {session_id}",
            output_summary=f"Correlation analysis complete. Strong correlations: {len(strong_corrs)}. Heatmap: {'generated' if heatmap_path else 'N/A'}",
            details={"strong_correlations": len(strong_corrs), "heatmap": heatmap_path is not None}
        )

        return {
            "heatmap_path": heatmap_path,
            "correlation_summary": corr_result,
            "strong_correlations": strong_corrs,
            "current_agent": "agent10_correlation",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 11: Deep Analysis
    # =========================================================================

    def _agent11_deep_analysis(self, state: MultiEDAState) -> Dict[str, Any]:
        """LLM-powered deep analysis synthesizing all previous results."""
        from tools.multi_eda_tools import eda_get_agent_summaries

        session_id = state.get("session_id", "")
        all_summaries = self._invoke_tool(eda_get_agent_summaries, session_id=session_id)

        target = state.get("target_variable_confirmed", "None")
        distribution_summary = state.get("distribution_summary", "N/A")
        segmentation_summary = state.get("segmentation_summary", "N/A")
        outlier_summary = state.get("outlier_summary", "N/A")
        correlation_summary = state.get("correlation_summary", "N/A")

        prompt = f"""You are a senior data scientist performing deep EDA analysis.

DATASET SUMMARIES:
{all_summaries[:3000]}

DISTRIBUTION ANALYSIS:
{distribution_summary[:500]}

SEGMENTATION ANALYSIS:
{segmentation_summary[:500]}

OUTLIER DETECTION:
{outlier_summary[:500]}

CORRELATION ANALYSIS:
{correlation_summary[:500]}

TARGET VARIABLE: {target}

Based on ALL the above analysis, provide:

1. **DATA CLEANING SUGGESTIONS** (5-8 bullet points):
   - Missing data treatment
   - Outlier handling
   - Data type corrections
   - Structural issues

2. **FEATURE ENGINEERING SUGGESTIONS** (5-8 bullet points):
   - New features from existing columns
   - Indicator/dummy variables for sparse classes
   - Interaction features based on correlations
   - Datetime features if applicable
   - Grouping/binning suggestions

3. **KEY INSIGHTS** (3-5 bullet points):
   - Most important patterns found
   - Relationships between variables
   - Data quality concerns

Respond in JSON format:
{{
    "cleaning_suggestions": ["suggestion1", "suggestion2", ...],
    "feature_engineering_suggestions": ["suggestion1", "suggestion2", ...],
    "key_insights": ["insight1", "insight2", ...],
    "overall_assessment": "Brief overall assessment of the dataset"
}}

Respond ONLY with JSON."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._parse_llm_json(response.content)

        cleaning = result.get("cleaning_suggestions", [])
        feature_eng = result.get("feature_engineering_suggestions", [])
        key_insights = result.get("key_insights", [])
        overall = result.get("overall_assessment", "Analysis complete.")

        deep_text = f"""DEEP ANALYSIS RESULTS:

KEY INSIGHTS:
{chr(10).join(f'- {i}' for i in key_insights)}

DATA CLEANING SUGGESTIONS:
{chr(10).join(f'- {s}' for s in cleaning)}

FEATURE ENGINEERING SUGGESTIONS:
{chr(10).join(f'- {s}' for s in feature_eng)}

OVERALL: {overall}"""

        trace = self._create_trace(
            agent_id="eda_agent11",
            status="completed",
            input_summary=f"Synthesizing all agent results for session {session_id}",
            output_summary=f"Generated {len(cleaning)} cleaning + {len(feature_eng)} feature suggestions.",
            details={"cleaning_count": len(cleaning), "feature_eng_count": len(feature_eng),
                     "insights_count": len(key_insights)}
        )

        return {
            "deep_analysis": deep_text,
            "cleaning_suggestions": cleaning,
            "feature_engineering_suggestions": feature_eng,
            "current_agent": "agent11_deep",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # AGENT 12: Dashboard Generation
    # =========================================================================

    def _agent12_dashboard(self, state: MultiEDAState) -> Dict[str, Any]:
        """Generate a comprehensive HTML dashboard with all plots and analysis."""
        from tools.multi_eda_tools import eda_generate_dashboard, eda_get_agent_summaries

        session_id = state.get("session_id", "")
        table_name = ", ".join(state.get("related_tables", ["Dataset"]))

        # Gather agent summaries from tools
        agent_summaries = self._invoke_tool(eda_get_agent_summaries, session_id=session_id)

        # Prepend the deep analysis text (contains KEY INSIGHTS and OVERALL)
        # so the dashboard builder can parse structured sections
        deep_analysis = state.get("deep_analysis", "")
        all_summaries = deep_analysis
        if agent_summaries:
            all_summaries += "\n\nAGENT SUMMARIES:\n" + agent_summaries

        # Build suggestions text
        cleaning = state.get("cleaning_suggestions", [])
        feature_eng = state.get("feature_engineering_suggestions", [])
        suggestions = "DATA CLEANING:\n"
        suggestions += "\n".join(f"- {s}" for s in cleaning)
        suggestions += "\n\nFEATURE ENGINEERING:\n"
        suggestions += "\n".join(f"- {s}" for s in feature_eng)

        title = f"EDA Report - {table_name}"

        dashboard_result = self._invoke_tool(
            eda_generate_dashboard,
            session_id=session_id,
            title=title,
            all_summaries=all_summaries[:5000],
            suggestions=suggestions
        )

        dashboard_path = None
        if "DASHBOARD SAVED:" in dashboard_result:
            dashboard_path = dashboard_result.split("DASHBOARD SAVED:")[1].strip()

        trace = self._create_trace(
            agent_id="eda_agent12",
            status="completed",
            input_summary=f"Generating dashboard for {table_name}",
            output_summary=f"Dashboard generated: {dashboard_path}" if dashboard_path else dashboard_result[:120],
            details={"dashboard_path": dashboard_path}
        )

        return {
            "dashboard_path": dashboard_path,
            "current_agent": "agent12_dashboard",
            "agent_traces": self._add_trace(state, trace)
        }

    # =========================================================================
    # FINAL RESPONSE
    # =========================================================================

    def _final_response_node(self, state: MultiEDAState) -> Dict[str, Any]:
        """Generate the final response based on workflow outcome."""
        intent = state.get("intent", "NEW_EDA")

        # General question - pass through the comprehensive answer from Agent1
        if intent == "GENERAL_QUESTION":
            return {
                "general_answer": state.get("general_answer", "I couldn't determine an answer to your question."),
                "is_complete": True,
                "agent_traces": state.get("agent_traces", [])
            }

        # Stopped early (no target or no joins)
        if state.get("stop_reason"):
            stop_msg = state.get("stop_reason", "EDA could not proceed.")
            return {
                "general_answer": f"**EDA Stopped Early**\n\n{stop_msg}\n\nPlease try rephrasing your query or specify different tables.",
                "is_complete": True,
                "agent_traces": state.get("agent_traces", [])
            }

        # Full EDA completed - build final answer
        parts = []
        parts.append("**EDA Analysis Complete!**\n")

        tables = state.get("related_tables", [])
        if tables:
            parts.append(f"**Tables analyzed:** {', '.join(tables)}")

        shape = state.get("shape_info", {})
        if shape:
            parts.append(f"**Dataset shape:** {shape.get('rows', '?')} rows x {shape.get('columns', '?')} columns")

        target = state.get("target_variable_confirmed", "")
        if target:
            parts.append(f"**Target variable:** {target}")

        # Deep analysis summary
        deep = state.get("deep_analysis", "")
        if deep:
            parts.append(f"\n{deep}")

        # Dashboard link
        dashboard = state.get("dashboard_path", "")
        if dashboard:
            parts.append(f"\n**Dashboard:** {dashboard}")

        return {
            "general_answer": "\n".join(parts),
            "is_complete": True,
            "agent_traces": state.get("agent_traces", [])
        }

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def stream(self, query: str, short_term_memory: Optional[str] = None,
               long_term_memory: Optional[str] = None):
        """Stream the multi-EDA execution for real-time UI updates."""
        initial_state: MultiEDAState = {
            "messages": [HumanMessage(content=query)],
            "nl_query": query,
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory,
            # Agent1
            "intent": "NEW_EDA",
            "general_answer": None,
            "related_tables": [],
            "target_variable": None,
            # Agent2
            "extracted_schema": {},
            "schema_json_path": None,
            # Agent3
            "session_id": "",
            "loaded_tables": {},
            "row_limit": 50000,
            "total_rows_loaded": 0,
            # Agent4
            "has_target_variable": False,
            "target_variable_confirmed": "",
            "join_strategy": None,
            "joined_dataframe_info": None,
            "stop_reason": None,
            # Agent5
            "structure_summary": "",
            "shape_info": {},
            "dtypes_info": {},
            "head_sample": "",
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            # Agent6
            "descriptive_stats": {},
            "stats_tables_html": [],
            # Agent7
            "histogram_all_path": None,
            "histogram_individual_paths": [],
            "countplot_paths": [],
            "distribution_summary": "",
            "sparse_classes": [],
            # Agent8
            "boxplot_paths": [],
            "violinplot_paths": [],
            "lmplot_paths": [],
            "segmentation_summary": "",
            # Agent9
            "outlier_plot_paths": [],
            "outlier_summary": "",
            "outliers_detected": {},
            # Agent10
            "heatmap_path": None,
            "correlation_summary": "",
            "strong_correlations": [],
            # Agent11
            "deep_analysis": "",
            "cleaning_suggestions": [],
            "feature_engineering_suggestions": [],
            # Agent12
            "dashboard_path": None,
            # Control
            "current_agent": "start",
            "is_complete": False,
            "error": None,
            "agent_traces": []
        }

        try:
            for state_update in self.graph.stream(initial_state):
                yield state_update
        except Exception as e:
            yield {"error": str(e)}

    def run(self, query: str, short_term_memory: Optional[str] = None,
            long_term_memory: Optional[str] = None) -> Dict[str, Any]:
        """Execute the multi-EDA workflow synchronously."""
        initial_state: MultiEDAState = {
            "messages": [HumanMessage(content=query)],
            "nl_query": query,
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory,
            "intent": "NEW_EDA",
            "general_answer": None,
            "related_tables": [],
            "target_variable": None,
            "extracted_schema": {},
            "schema_json_path": None,
            "session_id": "",
            "loaded_tables": {},
            "row_limit": 50000,
            "total_rows_loaded": 0,
            "has_target_variable": False,
            "target_variable_confirmed": "",
            "join_strategy": None,
            "joined_dataframe_info": None,
            "stop_reason": None,
            "structure_summary": "",
            "shape_info": {},
            "dtypes_info": {},
            "head_sample": "",
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "descriptive_stats": {},
            "stats_tables_html": [],
            "histogram_all_path": None,
            "histogram_individual_paths": [],
            "countplot_paths": [],
            "distribution_summary": "",
            "sparse_classes": [],
            "boxplot_paths": [],
            "violinplot_paths": [],
            "lmplot_paths": [],
            "segmentation_summary": "",
            "outlier_plot_paths": [],
            "outlier_summary": "",
            "outliers_detected": {},
            "heatmap_path": None,
            "correlation_summary": "",
            "strong_correlations": [],
            "deep_analysis": "",
            "cleaning_suggestions": [],
            "feature_engineering_suggestions": [],
            "dashboard_path": None,
            "current_agent": "start",
            "is_complete": False,
            "error": None,
            "agent_traces": []
        }

        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            return {
                **initial_state,
                "error": str(e),
                "is_complete": True,
                "general_answer": f"An error occurred: {str(e)}"
            }

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the multi-EDA workflow."""
        return {
            "name": "Multi-EDA Orchestrator",
            "description": "12-agent workflow for comprehensive Exploratory Data Analysis",
            "agents": [
                {"id": f"eda_agent{i}", "name": info["name"]}
                for i, (key, info) in enumerate(self.AGENT_INFO.items(), 0)
                if not "." in key  # Only main agents
            ]
        }
