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
from openai import OpenAI

# DuckDB imports for database explorer
import duckdb
import pandas as pd

# RAG imports
from tools.rag_tools import (
    get_relevant_context,
    has_knowledge_base_content,
    get_rag_store_state
)
from pages.knowledge_base import render_knowledge_base

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
    /* Multi-Agent Trace Styles */
    .multi-agent-trace {
        background: #fafafa;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .multi-agent-header {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .agent-trace-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        margin: 6px 0;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    .agent-trace-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .agent-trace-header {
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
    }
    .agent-icon {
        font-size: 1.2rem;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }
    .agent-name {
        font-weight: 600;
        font-size: 0.85rem;
        flex-grow: 1;
    }
    .agent-status {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 500;
    }
    .status-completed {
        background: #e8f5e9;
        color: #2e7d32;
    }
    .status-running {
        background: #fff3e0;
        color: #ef6c00;
    }
    .status-failed {
        background: #ffebee;
        color: #c62828;
    }
    .status-skipped {
        background: #f5f5f5;
        color: #757575;
    }
    .agent-trace-content {
        padding: 8px 12px;
        border-top: 1px solid #f0f0f0;
        font-size: 0.8rem;
    }
    .trace-row {
        display: flex;
        margin: 4px 0;
    }
    .trace-label {
        color: #666;
        width: 70px;
        flex-shrink: 0;
        font-weight: 500;
    }
    .trace-value {
        color: #333;
        word-break: break-word;
    }
    .confidence-bar {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 4px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    .confidence-high {
        background: linear-gradient(90deg, #4caf50, #8bc34a);
    }
    .confidence-medium {
        background: linear-gradient(90deg, #ff9800, #ffc107);
    }
    .confidence-low {
        background: linear-gradient(90deg, #f44336, #ff5722);
    }
    /* Agent type colors */
    .agent-prevalidation { background: #e3f2fd; border-left: 3px solid #1976d2; }
    .agent-prevalidation .agent-icon { background: #1976d2; color: white; }
    .agent-schema { background: #f3e5f5; border-left: 3px solid #7b1fa2; }
    .agent-schema .agent-icon { background: #7b1fa2; color: white; }
    .agent-orchestrator { background: #fff3e0; border-left: 3px solid #ef6c00; }
    .agent-orchestrator .agent-icon { background: #ef6c00; color: white; }
    .agent-generator { background: #e8f5e9; border-left: 3px solid #388e3c; }
    .agent-generator .agent-icon { background: #388e3c; color: white; }
    .agent-validator { background: #fce4ec; border-left: 3px solid #c2185b; }
    .agent-validator .agent-icon { background: #c2185b; color: white; }
    .agent-retry { background: #fff8e1; border-left: 3px solid #ffa000; }
    .agent-retry .agent-icon { background: #ffa000; color: white; }
    .agent-executor { background: #e0f7fa; border-left: 3px solid #0097a7; }
    .agent-executor .agent-icon { background: #0097a7; color: white; }
    .sql-preview {
        background: #263238;
        color: #aed581;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.75rem;
        overflow-x: auto;
        margin-top: 6px;
    }
    .validation-summary {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 6px;
    }
    .validation-item {
        background: #f5f5f5;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
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
    # Long-term memory (summaries after every 5 conversations)
    if "long_term_memory" not in st.session_state:
        st.session_state.long_term_memory = []  # List of {summary, conversation_range, tokens, timestamp}
    if "long_term_memory_tokens" not in st.session_state:
        st.session_state.long_term_memory_tokens = 0
    # Total conversation counter (for tracking 5/10/15 thresholds)
    if "total_conversation_count" not in st.session_state:
        st.session_state.total_conversation_count = 0
    # Current view/page
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Chat"


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        # App title - compact
        st.markdown("### 🤖 AI Agent Console")

        # ═══════════════════════════════════════════════════════════════
        # NAVIGATION
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<p class="sidebar-header">📍 Navigation</p>', unsafe_allow_html=True)

        # Get RAG document count for badge
        rag_state = get_rag_store_state()
        rag_doc_count = len(rag_state["documents"])
        kb_label = f"📚 Knowledge Base ({rag_doc_count})" if rag_doc_count > 0 else "📚 Knowledge Base"

        current_view = st.radio(
            "Select View",
            ["💬 Chat", "📝 Example Queries", kb_label],
            index=["💬 Chat", "📝 Example Queries", kb_label].index(
                st.session_state.current_view if st.session_state.current_view in ["💬 Chat", "📝 Example Queries", kb_label] else "💬 Chat"
            ),
            label_visibility="collapsed",
            key="nav_radio"
        )

        # Update session state (handle KB label with count)
        if current_view.startswith("📚 Knowledge Base"):
            st.session_state.current_view = "📚 Knowledge Base"
        else:
            st.session_state.current_view = current_view

        # ═══════════════════════════════════════════════════════════════
        # SESSION TOKENS (Top Priority Display)
        # ═══════════════════════════════════════════════════════════════
        # Calculate memory counters
        total_conv_count = st.session_state.total_conversation_count
        next_threshold = get_next_summary_threshold()
        short_term_count = len(st.session_state.conversation_memory)
        long_term_count = len(st.session_state.long_term_memory)

        st.markdown(f"""
        <div class="token-card">
            <div class="token-card-header">SESSION TOKENS</div>
            <div class="token-stats-row" style="margin-bottom: 8px;">
                <div class="token-stat" style="flex: 1;">
                    <div class="token-value token-value-total" style="font-size: 1.4rem;">{st.session_state.total_tokens:,}</div>
                    <div class="token-label">Total</div>
                </div>
            </div>
            <div style="border-top: 1px solid #90caf9; padding-top: 8px; margin-top: 4px;">
                <div style="font-size: 0.65rem; color: #546e7a; margin-bottom: 6px; text-transform: uppercase;">Memory</div>
                <div class="token-stats-row">
                    <div class="token-stat">
                        <div class="token-value" style="color: #ff9800;">{st.session_state.memory_tokens:,}</div>
                        <div class="token-label">Short ({total_conv_count}/{next_threshold})</div>
                    </div>
                    <div class="token-stat">
                        <div class="token-value" style="color: #9c27b0;">{st.session_state.long_term_memory_tokens:,}</div>
                        <div class="token-label">Long ({long_term_count})</div>
                    </div>
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
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.execution_history = []
            # Reset token counts
            st.session_state.total_tokens = 0
            st.session_state.last_input_tokens = 0
            st.session_state.last_output_tokens = 0
            st.session_state.last_total_tokens = 0
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Clear Short", use_container_width=True, help="Clear short-term memory"):
                st.session_state.conversation_memory = []
                st.session_state.memory_tokens = 0
                st.rerun()
        with col2:
            if st.button("📚 Clear Long", use_container_width=True, help="Clear long-term memory"):
                st.session_state.long_term_memory = []
                st.session_state.long_term_memory_tokens = 0
                st.rerun()

        if st.button("🔄 Clear All Memory", use_container_width=True):
            st.session_state.conversation_memory = []
            st.session_state.memory_tokens = 0
            st.session_state.long_term_memory = []
            st.session_state.long_term_memory_tokens = 0
            st.session_state.total_conversation_count = 0
            st.rerun()


def render_react_trace(execution_data: Dict[str, Any]):
    """Render the ReAct execution trace or multi-agent trace."""
    # Check if this is a multi-agent trace
    agent_traces = execution_data.get("agent_traces", [])

    if agent_traces:
        # Render multi-agent trace
        token_stats = execution_data.get("token_stats", {})
        total_tokens = token_stats.get("total_tokens", 0)
        confidence = execution_data.get("overall_confidence", 0)

        trace_header = "🤖 View Multi-Agent Workflow Trace"
        if total_tokens > 0:
            trace_header += f" ({total_tokens:,} tokens)"

        with st.expander(trace_header, expanded=False):
            # Show summary at top
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 10px 15px; border-radius: 6px; margin-bottom: 12px;">
                <div style="font-size: 0.85rem; color: #1565c0;">
                    <strong>Agents Executed:</strong> {len(agent_traces)} |
                    <strong>Confidence:</strong> {confidence:.0%} |
                    <strong>Tokens:</strong> {total_tokens:,}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Render each agent trace card
            for trace in agent_traces:
                render_agent_trace_card(trace, st)

            # Show generated SQL if available
            sql = execution_data.get("generated_sql")
            if sql:
                st.markdown("**Generated SQL:**")
                st.code(sql, language="sql")

        return

    # Standard ReAct trace rendering
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


def get_agent_class(agent_id: str) -> str:
    """Get CSS class for agent type."""
    class_map = {
        "agent1": "agent-prevalidation",
        "agent2": "agent-schema",
        "agent3": "agent-orchestrator",
        "agent4": "agent-generator",
        "agent5": "agent-validator",
        "agent5.1": "agent-validator",
        "agent5.2": "agent-validator",
        "agent5.3": "agent-validator",
        "retry": "agent-retry",
        "agent6": "agent-executor",
    }
    return class_map.get(agent_id, "agent-prevalidation")


def get_agent_icon(agent_id: str) -> str:
    """Get icon for agent type."""
    icon_map = {
        "agent1": "🔍",
        "agent2": "📋",
        "agent3": "🎯",
        "agent4": "⚙️",
        "agent5": "✅",
        "agent5.1": "📝",
        "agent5.2": "🗄️",
        "agent5.3": "🎯",
        "retry": "🔄",
        "agent6": "🚀",
    }
    return icon_map.get(agent_id, "🔷")


def render_agent_trace_card(trace: dict, container, expanded: bool = False):
    """Render a single agent trace as a styled card."""
    agent_id = trace.get("agent_id", "unknown")
    agent_name = trace.get("agent_name", "Unknown Agent")
    status = trace.get("status", "unknown")
    input_summary = trace.get("input_summary", "")
    output_summary = trace.get("output_summary", "")
    details = trace.get("details", {})

    agent_class = get_agent_class(agent_id)
    icon = get_agent_icon(agent_id)

    # Clean agent name (remove emoji if present)
    clean_name = ''.join(c for c in agent_name if ord(c) < 128 or c.isalnum() or c.isspace()).strip()
    if not clean_name:
        clean_name = agent_id.replace("_", " ").title()

    status_class = f"status-{status}"

    # Build details HTML
    details_html = ""

    # Add confidence bar for validators
    confidence = details.get("confidence") or details.get("overall_confidence")
    if confidence is not None:
        conf_percent = float(confidence) * 100
        conf_class = "confidence-high" if conf_percent >= 90 else "confidence-medium" if conf_percent >= 70 else "confidence-low"
        details_html += f"""
        <div class="trace-row">
            <span class="trace-label">Confidence:</span>
            <span class="trace-value">{conf_percent:.0f}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill {conf_class}" style="width: {conf_percent}%"></div>
        </div>
        """

    # Add SQL preview for generator
    sql_preview = details.get("sql_preview", "")
    if sql_preview:
        # Escape HTML in SQL
        sql_escaped = sql_preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        details_html += f'<div class="sql-preview">{sql_escaped}</div>'

    # Add validation summary for main validator
    if agent_id == "agent5" and "syntax_confidence" in details:
        syntax_conf = details.get("syntax_confidence", 0) * 100
        schema_conf = details.get("schema_confidence", 0) * 100
        semantic_conf = details.get("semantic_confidence", 0) * 100
        details_html += f"""
        <div class="validation-summary">
            <span class="validation-item">Syntax: {syntax_conf:.0f}%</span>
            <span class="validation-item">Schema: {schema_conf:.0f}%</span>
            <span class="validation-item">Semantic: {semantic_conf:.0f}%</span>
        </div>
        """

    # Add complexity for orchestrator
    if "complexity" in details:
        details_html += f"""
        <div class="trace-row">
            <span class="trace-label">Complexity:</span>
            <span class="trace-value"><strong>{details['complexity']}</strong></span>
        </div>
        """

    # Add tables for schema extraction
    if "tables_extracted" in details:
        tables = details["tables_extracted"]
        if isinstance(tables, list):
            details_html += f"""
            <div class="trace-row">
                <span class="trace-label">Tables:</span>
                <span class="trace-value">{', '.join(tables) if tables else 'None'}</span>
            </div>
            """

    # Add rows returned for executor
    if "rows_returned" in details:
        details_html += f"""
        <div class="trace-row">
            <span class="trace-label">Rows:</span>
            <span class="trace-value">{details['rows_returned']}</span>
        </div>
        """

    # Add retry info
    if details.get("retry_count", 0) > 0:
        details_html += f"""
        <div class="trace-row">
            <span class="trace-label">Retry:</span>
            <span class="trace-value">Attempt #{details['retry_count'] + 1}</span>
        </div>
        """

    html = f"""
    <div class="agent-trace-card {agent_class}">
        <div class="agent-trace-header">
            <span class="agent-icon">{icon}</span>
            <span class="agent-name">{clean_name}</span>
            <span class="agent-status {status_class}">{status.upper()}</span>
        </div>
        <div class="agent-trace-content">
            <div class="trace-row">
                <span class="trace-label">Input:</span>
                <span class="trace-value">{input_summary}</span>
            </div>
            <div class="trace-row">
                <span class="trace-label">Output:</span>
                <span class="trace-value">{output_summary}</span>
            </div>
            {details_html}
        </div>
    </div>
    """

    container.markdown(html, unsafe_allow_html=True)


def run_multi_agent_with_streaming(mission: str, trace_container, status_placeholder):
    """Run the multi-agent orchestrator with real-time trace updates."""
    from core.multi_agent_orchestrator import MultiAgentDataOrchestrator
    from core.config import FrameworkConfig, ModelConfig

    # Create config
    config = FrameworkConfig(
        model=ModelConfig(
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
            api_key=st.session_state.api_key
        ),
        max_iterations=st.session_state.max_iterations
    )

    orchestrator = MultiAgentDataOrchestrator(config=config)

    # Initialize token counter
    token_counter = get_token_counter(st.session_state.model_name)

    # Track displayed traces
    displayed_traces = set()
    final_state = None
    all_traces = []

    # Render header
    trace_container.markdown("""
    <div class="multi-agent-header">
        <span>🤖</span>
        <span>Multi-Agent Workflow Execution</span>
    </div>
    """, unsafe_allow_html=True)

    # Create a container for agent cards
    agents_container = trace_container.container()

    # Stream the execution
    try:
        for state_update in orchestrator.stream(mission):
            for node_name, node_state in state_update.items():
                if node_name == "__end__":
                    continue

                # Update status
                current_agent = node_state.get("current_agent", "processing")
                agent_display = current_agent.replace("_", " ").title()
                status_placeholder.markdown(f"**🔄 Processing: {agent_display}...**")

                # Get new traces and render them
                traces = node_state.get("agent_traces", [])
                for i, trace in enumerate(traces):
                    trace_id = f"{trace.get('agent_id')}_{trace.get('timestamp', i)}"
                    if trace_id not in displayed_traces:
                        displayed_traces.add(trace_id)
                        all_traces.append(trace)
                        render_agent_trace_card(trace, agents_container)

                final_state = node_state

    except Exception as e:
        trace_container.error(f"Error during execution: {str(e)}")
        return {
            "error": str(e),
            "is_complete": True,
            "final_answer": f"An error occurred: {str(e)}",
            "agent_traces": all_traces
        }

    # Calculate token stats (approximate)
    execution_tokens = 0
    for trace in all_traces:
        # Rough estimate based on trace content
        execution_tokens += len(str(trace)) // 4

    st.session_state.total_tokens += execution_tokens
    st.session_state.last_total_tokens = execution_tokens

    # Build result
    result = {
        "messages": final_state.get("messages", []) if final_state else [],
        "mission": mission,
        "agent_traces": all_traces,
        "is_complete": True,
        "final_answer": final_state.get("final_answer") if final_state else None,
        "general_answer": final_state.get("general_answer") if final_state else None,
        "generated_sql": final_state.get("generated_sql") if final_state else None,
        "overall_confidence": final_state.get("overall_confidence", 0) if final_state else 0,
        "query_results": final_state.get("query_results") if final_state else None,
        "error": final_state.get("error") if final_state else None,
        "token_stats": {
            "total_tokens": execution_tokens,
            "prompt_tokens": execution_tokens // 2,
            "completion_tokens": execution_tokens // 2
        }
    }

    return result


def render_multi_agent_trace(result: Dict[str, Any]):
    """Render the multi-agent execution trace for stored messages."""
    traces = result.get("agent_traces", [])

    if not traces:
        st.info("No agent trace available.")
        return

    # Render header
    st.markdown("""
    <div class="multi-agent-header">
        <span>🤖</span>
        <span>Multi-Agent Workflow Trace</span>
    </div>
    """, unsafe_allow_html=True)

    # Render each agent trace
    for trace in traces:
        render_agent_trace_card(trace, st)

    # Show SQL if available
    sql = result.get("generated_sql")
    if sql:
        st.markdown("**Generated SQL:**")
        st.code(sql, language="sql")

    # Show confidence
    confidence = result.get("overall_confidence", 0)
    if confidence > 0:
        st.markdown(f"**Validation Confidence:** {confidence:.0%}")


def calculate_memory_tokens():
    """Calculate total tokens used by short-term conversation memory."""
    token_counter = get_token_counter(st.session_state.model_name)
    total = 0
    for conv in st.session_state.conversation_memory:
        total += token_counter.count_text(conv.get("query", ""))
        total += token_counter.count_text(conv.get("response", ""))
    return total


def calculate_long_term_memory_tokens():
    """Calculate total tokens used by long-term memory summaries."""
    token_counter = get_token_counter(st.session_state.model_name)
    total = 0
    for mem in st.session_state.long_term_memory:
        total += token_counter.count_text(mem.get("summary", ""))
    return total


def get_next_summary_threshold():
    """Get the next conversation count threshold for summarization."""
    count = st.session_state.total_conversation_count
    # Thresholds are 5, 10, 15, 20, ...
    return ((count // 5) + 1) * 5


def summarize_conversations_for_long_term():
    """Summarize the current short-term memory conversations using LLM and store in long-term memory."""
    if not st.session_state.conversation_memory:
        return

    if not st.session_state.api_key:
        return

    try:
        # Build conversation text for summarization
        conversations_text = []
        for i, conv in enumerate(st.session_state.conversation_memory, 1):
            conversations_text.append(f"Conversation {i}:")
            conversations_text.append(f"User: {conv.get('query', '')}")
            conversations_text.append(f"Assistant: {conv.get('response', '')}")
            conversations_text.append("")

        conversations_str = "\n".join(conversations_text)

        # Create OpenAI client and summarize
        client = OpenAI(api_key=st.session_state.api_key)

        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes conversations. Create a concise summary that captures the key topics, questions asked, and important information from the conversations. The summary should be useful for providing context in future conversations."
                },
                {
                    "role": "user",
                    "content": f"Please summarize these {len(st.session_state.conversation_memory)} conversations into a brief, informative summary:\n\n{conversations_str}"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        summary = response.choices[0].message.content

        # Calculate tokens used for summarization
        token_counter = get_token_counter(st.session_state.model_name)
        summary_tokens = token_counter.count_text(summary)

        # Store in long-term memory
        start_conv = st.session_state.total_conversation_count - len(st.session_state.conversation_memory) + 1
        end_conv = st.session_state.total_conversation_count

        long_term_entry = {
            "summary": summary,
            "conversation_range": f"{start_conv}-{end_conv}",
            "tokens": summary_tokens,
            "timestamp": datetime.now().isoformat()
        }

        st.session_state.long_term_memory.append(long_term_entry)

        # Update long-term memory tokens count
        st.session_state.long_term_memory_tokens = calculate_long_term_memory_tokens()

        # Add tokens used for summarization to total
        input_tokens = token_counter.count_text(conversations_str) + 100  # ~100 for system prompt
        output_tokens = summary_tokens
        st.session_state.total_tokens += input_tokens + output_tokens

    except Exception as e:
        # Silently fail - don't break the main flow
        print(f"Error summarizing for long-term memory: {e}")


def build_memory_context() -> str:
    """Build a context string from both short-term and long-term memory for the agent."""
    memory_parts = []

    # Add long-term memory first (lower priority, provides historical context)
    if st.session_state.long_term_memory:
        memory_parts.append("[LONG-TERM MEMORY - Historical conversation summaries]\n")
        for i, mem in enumerate(st.session_state.long_term_memory, 1):
            memory_parts.append(f"--- Summary {i} (Conversations {mem.get('conversation_range', 'unknown')}) ---")
            memory_parts.append(mem.get("summary", ""))
            memory_parts.append("")
        memory_parts.append("[END OF LONG-TERM MEMORY]\n")

    # Add short-term memory (higher priority, recent conversations)
    if st.session_state.conversation_memory:
        memory_parts.append("[SHORT-TERM MEMORY - Recent conversations (use this for follow-up questions)]\n")
        for i, conv in enumerate(st.session_state.conversation_memory, 1):
            memory_parts.append(f"--- Conversation {i} (Agent: {conv.get('agent', 'unknown')}) ---")
            memory_parts.append(f"User: {conv.get('query', '')}")
            # Truncate long responses to save tokens
            response = conv.get('response', '')
            if len(response) > 500:
                response = response[:500] + "..."
            memory_parts.append(f"Assistant: {response}")
            memory_parts.append("")
        memory_parts.append("[END OF SHORT-TERM MEMORY]\n")

    if not memory_parts:
        return ""

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
    """Store a conversation in short-term memory and trigger long-term summarization when needed."""
    # Increment total conversation count
    st.session_state.total_conversation_count += 1

    conversation = {
        "query": query,
        "response": response,
        "agent": agent,
        "tokens": tokens,
        "timestamp": datetime.now().isoformat()
    }

    # Add new conversation to short-term memory
    st.session_state.conversation_memory.append(conversation)

    # Check if we need to summarize (every 5 conversations)
    if st.session_state.total_conversation_count > 0 and st.session_state.total_conversation_count % 5 == 0:
        # Summarize current short-term memory to long-term
        summarize_conversations_for_long_term()
        # Clear short-term memory after summarization
        st.session_state.conversation_memory = []

    # Remove oldest if short-term memory exceeds 5 (sliding window within the 5-conversation cycle)
    while len(st.session_state.conversation_memory) > 5:
        st.session_state.conversation_memory.pop(0)

    # Update memory tokens count
    st.session_state.memory_tokens = calculate_memory_tokens()


def run_agent_with_streaming(mission: str, trace_container, answer_placeholder):
    """Run the agent with real-time streaming of trace."""
    orchestrator = create_agent_orchestrator()

    if orchestrator is None:
        return {"error": f"Agent '{st.session_state.current_agent}' not found"}

    # Initialize token counter for current model
    token_counter = get_token_counter(st.session_state.model_name)

    # Build context from multiple sources
    context_parts = []

    # 1. RAG Knowledge Base Context (automatic retrieval)
    if has_knowledge_base_content():
        rag_context = get_relevant_context(mission, top_k=5, min_score=0.3)
        if rag_context:
            context_parts.append(rag_context)

    # 2. Memory Context (short-term and long-term)
    memory_context = build_memory_context()
    if memory_context:
        context_parts.append(memory_context)

    # Build enhanced mission with all context
    if context_parts:
        all_context = "\n".join(context_parts)
        enhanced_mission = f"{all_context}\n[CURRENT QUESTION]\n{mission}"
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
            - Knowledge base (automatic RAG)

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

    # Import pages
    from pages.example_queries import render_example_queries

    # Render content based on current view
    current_view = st.session_state.current_view

    if current_view == "📚 Knowledge Base":
        # Knowledge Base view
        render_knowledge_base()

    elif current_view == "📝 Example Queries":
        # Example Queries view
        render_example_queries()

    else:
        # Chat view (default)
        # Show RAG status indicator if knowledge base has content
        if has_knowledge_base_content():
            rag_state = get_rag_store_state()
            doc_count = len(rag_state["documents"])
            st.info(f"📚 Knowledge Base Active: {doc_count} document(s) loaded. Relevant context will be automatically included in responses.")

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
                # Check if multi_data_agent is selected - use special handler
                if st.session_state.current_agent == "multi_data_agent":
                    # Run multi-agent with live trace streaming
                    with st.chat_message("assistant"):
                        # Header showing agent is working
                        status_placeholder = st.empty()
                        status_placeholder.markdown("**🔄 Multi-Agent Data Orchestrator initializing...**")

                        # Create expander for real-time trace - starts expanded
                        with st.expander("🤖 Multi-Agent Workflow Trace (Live)", expanded=True):
                            trace_container = st.container()

                        # Placeholder for the final answer
                        answer_placeholder = st.empty()

                        # Run with streaming
                        result = run_multi_agent_with_streaming(prompt, trace_container, status_placeholder)

                        # Get final answer
                        final_answer = result.get("final_answer") or result.get("general_answer") or "No result generated."

                        # Clear the status and show final answer
                        status_placeholder.empty()

                        # Show SQL if generated
                        if result.get("generated_sql"):
                            answer_placeholder.markdown("**Generated SQL:**")
                            st.code(result["generated_sql"], language="sql")

                            # Show confidence
                            confidence = result.get("overall_confidence", 0)
                            if confidence > 0:
                                conf_color = "#4caf50" if confidence >= 0.9 else "#ff9800" if confidence >= 0.7 else "#f44336"
                                st.markdown(f'<span style="color: {conf_color}; font-weight: bold;">Validation Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)

                        # Show final answer
                        st.markdown(f"""
                        <div class="final-answer">
                            <strong>✅ Final Answer:</strong><br><br>
                            {final_answer}
                        </div>
                        """, unsafe_allow_html=True)

                        # Store in history with special multi-agent flag
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_answer,
                            "trace": result,
                            "is_multi_agent": True
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
                else:
                    # Run standard agent with real-time streaming
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
