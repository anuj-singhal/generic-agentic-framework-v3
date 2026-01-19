"""
Streamlit UI for the Agentic AI Framework.
Provides an interactive interface for running agents in ReAct mode.
"""

import streamlit as st
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_config, FrameworkConfig, ModelConfig
from core.tools_base import tool_registry
from tools.example_tools import get_all_tools
from agents.agent_definitions import get_available_agents, create_agent, AgentFactory
from core.token_counter import get_token_counter

# DuckDB imports for database explorer
import duckdb
import pandas as pd

DB_PATH = "agent_ddb.db"


# Page configuration
st.set_page_config(
    page_title="Agentic AI Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    /* Sidebar styling - Light Blue Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #1565c0;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #1565c0 !important;
        font-size: 0.85rem !important;
    }
    /* Compact sidebar elements with consistent font */
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput {
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox input,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton button p {
        font-size: 0.75rem !important;
    }
    [data-testid="stSidebar"] .stSlider {
        padding-top: 0 !important;
        margin-bottom: 0.3rem !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.75rem !important;
    }
    /* Primary button - Green color with white text */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: #1b5e20 !important;
        border-color: #1b5e20 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] p {
        color: #ffffff !important;
    }
    /* Section headers */
    .sidebar-header {
        color: #0d47a1;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.8rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #1976d2;
    }
    /* Database card */
    .db-card {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        border: 1px solid #1565c0;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .db-card-title {
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .db-card-info {
        color: #e3f2fd;
        font-size: 0.7rem;
    }
    /* Agent info box */
    .agent-info {
        background: rgba(25, 118, 210, 0.1);
        border-left: 3px solid #1976d2;
        padding: 8px 10px;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #1565c0;
        margin: 5px 0;
    }
    /* Tool expander styling - Compact */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #90caf9;
        border-radius: 4px;
        margin: 2px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #1565c0 !important;
        font-weight: 500;
        font-size: 0.75rem !important;
        padding: 4px 8px !important;
        min-height: unset !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        font-size: 0.75rem !important;
    }
    /* Expander content - white background with black text */
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        background: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stText"],
    [data-testid="stSidebar"] [data-testid="stExpander"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] span,
    [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    /* Tools container - scrollable */
    .tools-container {
        max-height: 200px;
        overflow-y: auto;
        padding-right: 5px;
    }
    .tools-container::-webkit-scrollbar {
        width: 4px;
    }
    .tools-container::-webkit-scrollbar-track {
        background: #e3f2fd;
        border-radius: 2px;
    }
    .tools-container::-webkit-scrollbar-thumb {
        background: #90caf9;
        border-radius: 2px;
    }
    /* Compact tool description */
    .tool-desc {
        background-color: #ffffff;
        color: #000000;
        padding: 4px 6px;
        font-size: 0.7rem;
        line-height: 1.3;
    }
    /* Main content boxes */
    .thought-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .action-box {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .observation-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .final-answer {
        background-color: #e3f2fd;
        border: 2px solid #1E88E5;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
    }
    /* Session Tokens Card */
    .token-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #90caf9;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .token-card-header {
        color: #1565c0;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .token-stats-row {
        display: flex;
        justify-content: space-between;
        text-align: center;
    }
    .token-stat {
        flex: 1;
        padding: 4px;
    }
    .token-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .token-value-total {
        color: #1565c0;
    }
    .token-value-prompt {
        color: #2e7d32;
    }
    .token-value-output {
        color: #7b1fa2;
    }
    .token-label {
        font-size: 0.65rem;
        color: #546e7a;
        text-transform: capitalize;
    }
    /* Message token badge */
    .msg-token-badge {
        display: inline-block;
        background: rgba(25, 118, 210, 0.1);
        color: #1565c0;
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 10px;
        margin-left: 8px;
    }
    /* Memory badge */
    .memory-badge {
        display: inline-block;
        background: rgba(255, 152, 0, 0.15);
        color: #e65100;
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 12px;
        margin-bottom: 8px;
    }
    /* Memory answer box */
    .memory-answer {
        background-color: #fff8e1;
        border: 2px solid #ff9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def get_duckdb_tables():
    """Get list of tables from DuckDB database."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute("SHOW TABLES;").fetchall()
        conn.close()
        return [row[0] for row in result]
    except Exception as e:
        return []


def get_table_schema(table_name: str):
    """Get schema for a specific table."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute(f"DESCRIBE {table_name};").fetchall()
        columns = conn.execute(f"DESCRIBE {table_name};").description
        conn.close()
        return {
            "columns": [col[0] for col in columns],
            "data": result
        }
    except Exception as e:
        return None


def get_sample_data(table_name: str, limit: int = 10):
    """Get sample data from a table."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        columns = [col[0] for col in result.description]
        data = result.fetchall()
        conn.close()
        return {
            "columns": columns,
            "data": data
        }
    except Exception as e:
        return None


def get_table_row_count(table_name: str):
    """Get row count for a table."""
    if not os.path.exists(DB_PATH):
        return 0
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception as e:
        return 0


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = "general_assistant"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4o-mini"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = 10
    if "show_schema_modal" not in st.session_state:
        st.session_state.show_schema_modal = False
    if "selected_table" not in st.session_state:
        st.session_state.selected_table = None
    # Token tracking
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0  # Cumulative total
    if "last_input_tokens" not in st.session_state:
        st.session_state.last_input_tokens = 0
    if "last_output_tokens" not in st.session_state:
        st.session_state.last_output_tokens = 0
    if "last_total_tokens" not in st.session_state:
        st.session_state.last_total_tokens = 0
    if "needs_rerun" not in st.session_state:
        st.session_state.needs_rerun = False
    # Short-term memory (last 5 conversations)
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []  # List of {query, response, agent, tokens, timestamp}
    if "memory_tokens" not in st.session_state:
        st.session_state.memory_tokens = 0


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        # App title - compact
        st.markdown("### 🤖 AI Agent Console")

        # ═══════════════════════════════════════════════════════════════
        # SESSION TOKENS (Top Priority Display)
        # ═══════════════════════════════════════════════════════════════
        memory_count = len(st.session_state.conversation_memory)
        st.markdown(f"""
        <div class="token-card">
            <div class="token-card-header">SESSION TOKENS</div>
            <div class="token-stats-row" style="margin-bottom: 8px;">
                <div class="token-stat" style="flex: 1;">
                    <div class="token-value token-value-total" style="font-size: 1.4rem;">{st.session_state.total_tokens:,}</div>
                    <div class="token-label">Total</div>
                </div>
                <div class="token-stat" style="flex: 1;">
                    <div class="token-value" style="color: #ff9800; font-size: 1.4rem;">{st.session_state.memory_tokens:,}</div>
                    <div class="token-label">Memory ({memory_count}/5)</div>
                </div>
            </div>
            <div style="border-top: 1px solid #90caf9; padding-top: 8px; margin-top: 4px;">
                <div style="font-size: 0.65rem; color: #546e7a; margin-bottom: 6px; text-transform: uppercase;">Last Conversation</div>
                <div class="token-stats-row">
                    <div class="token-stat">
                        <div class="token-value token-value-prompt">{st.session_state.last_input_tokens:,}</div>
                        <div class="token-label">Input</div>
                    </div>
                    <div class="token-stat">
                        <div class="token-value token-value-output">{st.session_state.last_output_tokens:,}</div>
                        <div class="token-label">Output</div>
                    </div>
                    <div class="token-stat">
                        <div class="token-value" style="color: #1565c0;">{st.session_state.last_total_tokens:,}</div>
                        <div class="token-label">Total</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # DATABASE EXPLORER (Top Priority)
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<p class="sidebar-header">🗄️ Database Explorer</p>', unsafe_allow_html=True)

        if os.path.exists(DB_PATH):
            tables = get_duckdb_tables()
            if tables:
                # Compact database info - inline layout
                total_tables = len(tables)
                st.markdown(f"<div class='db-card'><span class='db-card-title'>📁 {DB_PATH}</span><span class='db-card-info'>{total_tables} tables</span></div>", unsafe_allow_html=True)

                # Table selection - compact
                selected_table = st.selectbox(
                    "Table",
                    tables,
                    key="table_selector",
                    format_func=lambda x: f"{x} ({get_table_row_count(x):,} rows)"
                )

                # Show Schema button
                if st.button("📋 View Schema & Data", use_container_width=True, type="primary"):
                    st.session_state.show_schema_modal = True
                    st.session_state.selected_table = selected_table
                    st.rerun()
            else:
                st.warning("No tables found.")
        else:
            st.error("Database not found")
            st.caption("Run: `python init_duckdb.py`")

        # ═══════════════════════════════════════════════════════════════
        # SETTINGS (API Key)
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<p class="sidebar-header">🔑 Settings</p>', unsafe_allow_html=True)

        # API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Enter your OpenAI API key...",
            help="Required to run agents"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        # ═══════════════════════════════════════════════════════════════
        # MODEL CONFIG (Expanded)
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<p class="sidebar-header">🧠 Model Config</p>', unsafe_allow_html=True)

        model_name = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
        st.session_state.model_name = model_name

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
            st.session_state.temperature = temperature
        with col2:
            max_iterations = st.slider("Max Iters", 1, 20, st.session_state.max_iterations)
            st.session_state.max_iterations = max_iterations

        # ═══════════════════════════════════════════════════════════════
        # AGENT SELECTION
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<p class="sidebar-header">🤖 Agent Selection</p>', unsafe_allow_html=True)

        agents = get_available_agents()
        agent_names = list(agents.keys())

        selected_agent = st.selectbox(
            "Select Agent",
            agent_names,
            index=agent_names.index(st.session_state.current_agent) if st.session_state.current_agent in agent_names else 0,
            format_func=lambda x: f"{x.replace('_', ' ').title()}"
        )
        st.session_state.current_agent = selected_agent

        # Agent description
        if selected_agent in agents:
            st.markdown(f"<div class='agent-info'>{agents[selected_agent]}</div>", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # AGENT TOOLS (Compact & Scrollable)
        # ═══════════════════════════════════════════════════════════════
        agent_def = AgentFactory.get_agent_definition(selected_agent)
        if agent_def:
            tools = agent_def.get_tools()
            st.markdown(f'<p class="sidebar-header">🔧 Tools ({len(tools)})</p>', unsafe_allow_html=True)

            # Create scrollable container
            with st.container(height=300):
                for tool in tools:
                    with st.expander(f"📌 {tool.name}", expanded=False):
                        st.markdown(f"<div class='tool-desc'>{tool.description}</div>", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # ACTIONS
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.execution_history = []
                # Reset token counts
                st.session_state.total_tokens = 0
                st.session_state.last_input_tokens = 0
                st.session_state.last_output_tokens = 0
                st.session_state.last_total_tokens = 0
                st.rerun()
        with col2:
            if st.button("🧠 Clear Memory", use_container_width=True):
                st.session_state.conversation_memory = []
                st.session_state.memory_tokens = 0
                st.rerun()


def render_react_trace(execution_data: Dict[str, Any]):
    """Render the ReAct execution trace."""
    messages = execution_data.get("messages", [])
    token_stats = execution_data.get("token_stats", {})
    message_tokens = token_stats.get("message_tokens", [])

    # Token counter for messages without pre-calculated tokens
    token_counter = get_token_counter(st.session_state.model_name)

    # Show execution token summary
    total_tokens = token_stats.get("total_tokens", 0)
    input_tokens = token_stats.get("prompt_tokens", 0)
    output_tokens = token_stats.get("completion_tokens", 0)

    trace_header = "🔍 View ReAct Execution Trace"
    if total_tokens > 0:
        trace_header += f" ({total_tokens:,} tokens)"

    with st.expander(trace_header, expanded=False):
        # Show token summary at top if available
        if total_tokens > 0:
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; font-size: 0.8rem;">
                <strong>Tokens:</strong> {total_tokens:,} total
                (<span style="color: #2e7d32;">{input_tokens:,} input</span> |
                <span style="color: #7b1fa2;">{output_tokens:,} output</span>)
            </div>
            """, unsafe_allow_html=True)

        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__

            # Get token count from pre-calculated or calculate now
            if i < len(message_tokens):
                token_count = message_tokens[i].get("tokens", 0)
            else:
                token_count = token_counter.count_message(msg)

            token_badge = f'<span class="msg-token-badge">{token_count} tokens</span>'

            if msg_type == "HumanMessage":
                st.markdown(f"""
                <div class="thought-box">
                    <strong>📝 Mission:</strong>{token_badge}<br>
                    {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)

            elif msg_type == "AIMessage":
                content = msg.content if msg.content else "[Tool Call]"

                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        st.markdown(f"""
                        <div class="action-box">
                            <strong>🛠️ Action:</strong> {tool_call['name']}{token_badge}<br>
                            <strong>Input:</strong> <code>{json.dumps(tool_call['args'], indent=2)}</code>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="thought-box">
                        <strong>💭 Thought/Response:</strong>{token_badge}<br>
                        {content[:500]}{'...' if len(content) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)

            elif msg_type == "ToolMessage":
                st.markdown(f"""
                <div class="observation-box">
                    <strong>👁️ Observation:</strong>{token_badge}<br>
                    {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)


def render_schema_modal():
    """Render the schema and sample data modal as an overlay."""
    if not st.session_state.show_schema_modal or not st.session_state.selected_table:
        return False

    table_name = st.session_state.selected_table

    # Get data
    schema = get_table_schema(table_name)
    sample = get_sample_data(table_name, 10)
    row_count = get_table_row_count(table_name)

    # Create modal container
    modal = st.container()

    with modal:
        # Modal header with close button
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"## 📊 Table: {table_name}")
        with col2:
            if st.button("✕ Close", key="close_modal"):
                st.session_state.show_schema_modal = False
                st.rerun()

        st.markdown(f"**Total Rows:** {row_count:,}")
        st.markdown("---")

        # Schema Section
        st.markdown("### 📋 Schema")

        if schema:
            schema_df_data = []
            for row in schema['data']:
                schema_df_data.append({
                    "Column": row[0],
                    "Type": row[1],
                    "Null": row[2] if len(row) > 2 else "YES",
                    "Key": row[3] if len(row) > 3 else "",
                    "Default": row[4] if len(row) > 4 else ""
                })

            schema_df = pd.DataFrame(schema_df_data)
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        else:
            st.error("Could not retrieve schema.")

        # Sample Data Section
        st.markdown("---")
        st.markdown("### 📝 Sample Data (10 rows)")

        if sample and sample['data']:
            sample_df = pd.DataFrame(sample['data'], columns=sample['columns'])
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No sample data available.")

        st.markdown("---")

        # Close button at bottom
        if st.button("Close", type="primary", use_container_width=True, key="close_modal_bottom"):
            st.session_state.show_schema_modal = False
            st.rerun()

    return True


def create_agent_orchestrator():
    """Create and return the agent orchestrator."""
    # Create config
    config = FrameworkConfig(
        model=ModelConfig(
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
            api_key=st.session_state.api_key
        ),
        max_iterations=st.session_state.max_iterations
    )

    # Create the agent
    orchestrator = create_agent(st.session_state.current_agent, config)
    return orchestrator


def render_streaming_trace_item(msg, container, token_count: int = None):
    """Render a single trace item during streaming."""
    msg_type = type(msg).__name__

    # Get token count if not provided
    if token_count is None:
        token_counter = get_token_counter(st.session_state.model_name)
        token_count = token_counter.count_message(msg)

    token_badge = f'<span class="msg-token-badge">{token_count} tokens</span>'

    if msg_type == "HumanMessage":
        container.markdown(f"""
        <div class="thought-box">
            <strong>📝 Mission:</strong>{token_badge}<br>
            {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}
        </div>
        """, unsafe_allow_html=True)

    elif msg_type == "AIMessage":
        content = msg.content if msg.content else ""

        # Check for tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                container.markdown(f"""
                <div class="action-box">
                    <strong>🛠️ Action:</strong> {tool_call['name']}{token_badge}<br>
                    <strong>Input:</strong> <code>{json.dumps(tool_call['args'], indent=2)}</code>
                </div>
                """, unsafe_allow_html=True)
        elif content:
            container.markdown(f"""
            <div class="thought-box">
                <strong>💭 Thought/Response:</strong>{token_badge}<br>
                {content[:500]}{'...' if len(content) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)

    elif msg_type == "ToolMessage":
        container.markdown(f"""
        <div class="observation-box">
            <strong>👁️ Observation:</strong>{token_badge}<br>
            {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}
        </div>
        """, unsafe_allow_html=True)


def calculate_memory_tokens():
    """Calculate total tokens used by conversation memory."""
    token_counter = get_token_counter(st.session_state.model_name)
    total = 0
    for conv in st.session_state.conversation_memory:
        total += token_counter.count_text(conv.get("query", ""))
        total += token_counter.count_text(conv.get("response", ""))
    return total


def build_memory_context() -> str:
    """Build a context string from conversation memory for the agent."""
    if not st.session_state.conversation_memory:
        return ""

    memory_parts = ["[CONVERSATION MEMORY - Use this context for follow-up questions]\n"]
    for i, conv in enumerate(st.session_state.conversation_memory, 1):
        memory_parts.append(f"--- Conversation {i} (Agent: {conv.get('agent', 'unknown')}) ---")
        memory_parts.append(f"User: {conv.get('query', '')}")
        # Truncate long responses to save tokens
        response = conv.get('response', '')
        if len(response) > 500:
            response = response[:500] + "..."
        memory_parts.append(f"Assistant: {response}")
        memory_parts.append("")

    memory_parts.append("[END OF MEMORY]\n")
    return "\n".join(memory_parts)


def check_duplicate_question(query: str) -> dict | None:
    """Check if a similar question exists in memory and return the cached response."""
    if not st.session_state.conversation_memory:
        return None

    # Normalize the query for comparison
    query_lower = query.lower().strip()

    for conv in st.session_state.conversation_memory:
        stored_query = conv.get("query", "").lower().strip()
        # Check for exact or very similar match
        if query_lower == stored_query:
            return conv
        # Check for high similarity (simple substring check)
        if len(query_lower) > 10 and len(stored_query) > 10:
            if query_lower in stored_query or stored_query in query_lower:
                return conv

    return None


def store_in_memory(query: str, response: str, agent: str, tokens: int):
    """Store a conversation in short-term memory (sliding window of last 5 conversations)."""
    conversation = {
        "query": query,
        "response": response,
        "agent": agent,
        "tokens": tokens,
        "timestamp": datetime.now().isoformat()
    }

    # Remove oldest conversation if at max capacity (sliding window)
    while len(st.session_state.conversation_memory) >= 5:
        st.session_state.conversation_memory.pop(0)  # Remove first/oldest

    # Add new conversation to memory
    st.session_state.conversation_memory.append(conversation)

    # Update memory tokens count
    st.session_state.memory_tokens = calculate_memory_tokens()


def run_agent_with_streaming(mission: str, trace_container, answer_placeholder):
    """Run the agent with real-time streaming of trace."""
    orchestrator = create_agent_orchestrator()

    if orchestrator is None:
        return {"error": f"Agent '{st.session_state.current_agent}' not found"}

    # Initialize token counter for current model
    token_counter = get_token_counter(st.session_state.model_name)

    # Build memory context and inject into mission
    memory_context = build_memory_context()
    if memory_context:
        enhanced_mission = f"{memory_context}\n[CURRENT QUESTION]\n{mission}"
    else:
        enhanced_mission = mission

    # Track tokens for this execution
    execution_prompt_tokens = 0
    execution_completion_tokens = 0
    message_tokens = []  # Track tokens per message

    # Track messages we've already displayed
    displayed_message_count = 0
    final_state = None
    all_messages = []

    # Stream the execution
    for state_update in orchestrator.stream(enhanced_mission):
        # state_update is a dict with node name as key
        for node_name, node_state in state_update.items():
            if node_name == "__end__":
                continue

            # Get messages from this update
            messages = node_state.get("messages", [])

            # Display new messages and count tokens
            for msg in messages:
                all_messages.append(msg)

                # Count tokens for this message
                msg_tokens = token_counter.count_message(msg)
                msg_type = type(msg).__name__

                # Categorize: AI messages are completion, others are prompt
                if msg_type == "AIMessage":
                    execution_completion_tokens += msg_tokens
                else:
                    execution_prompt_tokens += msg_tokens

                message_tokens.append({
                    "type": msg_type,
                    "tokens": msg_tokens
                })

                render_streaming_trace_item(msg, trace_container)
                displayed_message_count += 1

            # Update the final state
            final_state = node_state

    # Update session token totals
    execution_total = execution_prompt_tokens + execution_completion_tokens
    st.session_state.total_tokens += execution_total
    st.session_state.last_input_tokens = execution_prompt_tokens
    st.session_state.last_output_tokens = execution_completion_tokens
    st.session_state.last_total_tokens = execution_total

    # Build the complete result
    result = {
        "messages": all_messages,
        "mission": mission,
        "thoughts": final_state.get("thoughts", []) if final_state else [],
        "observations": final_state.get("observations", []) if final_state else [],
        "actions": final_state.get("actions", []) if final_state else [],
        "iteration_count": final_state.get("iteration_count", 0) if final_state else 0,
        "is_complete": True,
        "final_answer": final_state.get("final_answer") if final_state else None,
        "error": final_state.get("error") if final_state else None,
        "tool_calls_made": final_state.get("tool_calls_made", 0) if final_state else 0,
        "token_stats": {
            "prompt_tokens": execution_prompt_tokens,
            "completion_tokens": execution_completion_tokens,
            "total_tokens": execution_prompt_tokens + execution_completion_tokens,
            "message_tokens": message_tokens
        }
    }

    return result


def get_final_answer(result: Dict[str, Any]) -> str:
    """Extract the final answer from the execution result."""
    if "error" in result and result["error"]:
        return f"❌ Error: {result['error']}"
    
    messages = result.get("messages", [])
    
    # Get the last AI message
    for msg in reversed(messages):
        if type(msg).__name__ == "AIMessage" and msg.content:
            return msg.content
    
    return "No response generated."


def main():
    """Main application entry point."""
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.title("🤖 Agentic AI Framework")
    st.markdown("*An intelligent assistant operating in ReAct mode*")

    # Render schema modal if triggered (this will take over the main area)
    if render_schema_modal():
        return  # Stop rendering rest of the page when modal is shown

    # Check for API key
    if not st.session_state.api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
        
        # Show framework overview
        with st.expander("📚 About This Framework", expanded=True):
            st.markdown("""
            ### Architecture Overview
            
            This framework implements an agentic AI system with three core components:
            
            **1. 🧠 The Model (Brain)**
            - GPT-4o-mini as the reasoning engine
            - Processes information and makes decisions
            
            **2. 🤚 Tools (Hands)**
            - Calculator, unit converter
            - Date/time operations
            - Text analysis and transformation
            - Task management
            - Knowledge base search
            
            **3. 🔗 Orchestration Layer (Nervous System)**
            - ReAct pattern implementation
            - LangGraph for state management
            - Think → Act → Observe loop
            
            ### Available Agents
            """)
            
            agents = get_available_agents()
            for name, description in agents.items():
                st.markdown(f"- **{name.replace('_', ' ').title()}**: {description}")
        
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("from_memory"):
                st.markdown('<span class="memory-badge">📝 From Memory</span>', unsafe_allow_html=True)
            st.markdown(message["content"])
            if "trace" in message:
                render_react_trace(message["trace"])
    
    # Chat input
    if prompt := st.chat_input("Enter your task or question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check for duplicate question in memory
        cached_response = check_duplicate_question(prompt)

        if cached_response:
            # Return cached response
            with st.chat_message("assistant"):
                final_answer = cached_response.get("response", "")
                st.markdown(f"""
                <div class="final-answer" style="border-color: #ff9800;">
                    <strong>📝 From Memory</strong> <span style="font-size: 0.8rem; color: #666;">(previously answered by {cached_response.get('agent', 'agent')})</span><br><br>
                    {final_answer}
                </div>
                """, unsafe_allow_html=True)

                # Store in history (no trace for cached)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"[From Memory] {final_answer}",
                    "from_memory": True
                })

                # Rerun to update display
                st.rerun()
        else:
            # Run agent with real-time streaming
            with st.chat_message("assistant"):
                # Header showing agent is working
                status_placeholder = st.empty()
                status_placeholder.markdown(f"**🔄 {st.session_state.current_agent.replace('_', ' ').title()} is working...**")

                # Create expander for real-time trace - starts expanded
                with st.expander("🔍 ReAct Execution Trace (Live)", expanded=True):
                    trace_container = st.container()

                # Placeholder for the final answer
                answer_placeholder = st.empty()

                # Run with streaming
                result = run_agent_with_streaming(prompt, trace_container, answer_placeholder)
                final_answer = get_final_answer(result)

                # Clear the status and show final answer
                status_placeholder.empty()
                answer_placeholder.markdown(f"""
                <div class="final-answer">
                    <strong>✅ Final Answer:</strong><br><br>
                    {final_answer}
                </div>
                """, unsafe_allow_html=True)

                # Store in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "trace": result
                })

                # Store execution history
                st.session_state.execution_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": st.session_state.current_agent,
                    "mission": prompt,
                    "result": result
                })

                # Store in memory for future reference
                token_stats = result.get("token_stats", {})
                total_tokens = token_stats.get("total_tokens", 0)
                store_in_memory(prompt, final_answer, st.session_state.current_agent, total_tokens)

                # Rerun to update sidebar token display
                st.rerun()


if __name__ == "__main__":
    main()
