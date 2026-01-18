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


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        # App title - compact
        st.markdown("### 🤖 AI Agent Console")

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
            st.rerun()


def render_react_trace(execution_data: Dict[str, Any]):
    """Render the ReAct execution trace."""
    messages = execution_data.get("messages", [])
    
    with st.expander("🔍 View ReAct Execution Trace", expanded=False):
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            
            if msg_type == "HumanMessage":
                st.markdown(f"""
                <div class="thought-box">
                    <strong>📝 Mission:</strong><br>
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
                            <strong>🛠️ Action:</strong> {tool_call['name']}<br>
                            <strong>Input:</strong> <code>{json.dumps(tool_call['args'], indent=2)}</code>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="thought-box">
                        <strong>💭 Thought/Response:</strong><br>
                        {content[:500]}{'...' if len(content) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            elif msg_type == "ToolMessage":
                st.markdown(f"""
                <div class="observation-box">
                    <strong>👁️ Observation:</strong><br>
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


def run_agent(mission: str) -> Dict[str, Any]:
    """Run the selected agent with the given mission."""
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
    
    if orchestrator is None:
        return {"error": f"Agent '{st.session_state.current_agent}' not found"}
    
    # Run the agent
    result = orchestrator.run(mission)
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
        
        # Run agent and display response
        with st.chat_message("assistant"):
            with st.spinner(f"🔄 {st.session_state.current_agent.replace('_', ' ').title()} is thinking..."):
                result = run_agent(prompt)
                final_answer = get_final_answer(result)
            
            # Display the final answer
            st.markdown(final_answer)
            
            # Show execution trace
            render_react_trace(result)
            
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


if __name__ == "__main__":
    main()
