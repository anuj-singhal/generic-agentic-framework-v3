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
    .tool-card {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
    }
    .agent-card {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


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


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key
        st.subheader("🔑 API Settings")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your OpenAI API key"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Model Settings
        st.subheader("🧠 Model Settings")
        model_name = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
        st.session_state.model_name = model_name
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random"
        )
        st.session_state.temperature = temperature
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=20,
            value=st.session_state.max_iterations,
            help="Maximum ReAct loop iterations"
        )
        st.session_state.max_iterations = max_iterations
        
        # Agent Selection
        st.subheader("🤖 Agent Selection")
        agents = get_available_agents()
        agent_names = list(agents.keys())
        
        selected_agent = st.selectbox(
            "Select Agent",
            agent_names,
            index=agent_names.index(st.session_state.current_agent) if st.session_state.current_agent in agent_names else 0,
            format_func=lambda x: f"{x.replace('_', ' ').title()}"
        )
        st.session_state.current_agent = selected_agent
        
        # Show agent description
        if selected_agent in agents:
            st.info(agents[selected_agent])
        
        # Show agent's available tools
        agent_def = AgentFactory.get_agent_definition(selected_agent)
        if agent_def:
            st.subheader("🔧 Agent Tools")
            tools = agent_def.get_tools()
            for tool in tools:
                with st.expander(f"📌 {tool.name}"):
                    st.write(tool.description)
        
        # Clear history button
        st.divider()
        if st.button("🗑️ Clear History", use_container_width=True):
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
    
    # # Footer with tabs for tools and examples
    # tab1, tab2 = st.tabs(["📋 Available Tools", "📝 Example Queries"])
    
    # with tab1:
    #     all_tools = get_all_tools()
    #     cols = st.columns(2)
    #     for i, tool in enumerate(all_tools):
    #         with cols[i % 2]:
    #             st.markdown(f"""
    #             <div class="tool-card">
    #                 <strong>🔧 {tool.name}</strong><br>
    #                 <small>{tool.description[:100]}...</small>
    #             </div>
    #             """, unsafe_allow_html=True)
    
    # with tab2:
    #     render_example_queries_inline()


def render_example_queries_inline():
    """Render example queries inline in the main app."""
    
    st.markdown("### 🟢 Simple (Single Tool)")
    simple = [
        "What is 15% of 850?",
        "Convert 98.6 Fahrenheit to Celsius",
        "What is today's date and time?",
        "Create a high priority task called 'Review report'",
        "Sort alphabetically: banana, apple, cherry, date",
    ]
    cols = st.columns(2)
    for i, q in enumerate(simple):
        with cols[i % 2]:
            st.code(q, language=None)
    
    st.markdown("### 🟡 Medium (Multiple Tools)")
    medium = [
        "I'm traveling 150 miles. Gas costs $3.50/gallon, car gets 30 mpg. Calculate cost and convert distance to km.",
        "Create tasks: 'Design' (high), 'Code' (medium), 'Test' (low). Then list all tasks.",
        "Split a $247.50 bill between 5 friends with 20% tip. How much each?",
    ]
    for q in medium:
        st.code(q, language=None)
    
    st.markdown("### 🔴 Complex (Multi-Step)")
    complex_ex = [
        """Plan my project:
1. Get today's date
2. Calculate end date (90 days from now)
3. Create tasks: Requirements (high), Development (high), Testing (medium)
4. List all tasks""",
        """Investment analysis:
1. Calculate $50,000 at 7% for 5 years: 50000 * (1.07)^5
2. Calculate the gain (result - 50000)
3. Calculate percentage gain
4. Convert to EUR (multiply by 0.92)""",
    ]
    for q in complex_ex:
        st.code(q, language=None)
    
    st.markdown("### 🗄️ SQL / Text-to-SQL")
    sql_examples = [
        "What databases are available?",
        "Show me the schema for the ecommerce database",
        "Write a SQL query to get all customers from USA in the ecommerce database",
        "How many orders are in pending status? Use the ecommerce database.",
        """Using the ecommerce database, write a query to find the top 5 customers by total order value. 
Show customer name, email, number of orders, and total spent.""",
        """Explain this SQL: SELECT c.first_name, COUNT(o.order_id) as order_count 
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id 
GROUP BY c.first_name ORDER BY order_count DESC""",
    ]
    for q in sql_examples:
        st.code(q, language=None)
    
    st.markdown("### 🏷️ Name Matching")
    name_examples = [
        '''Load these names for matching:
["Emirates NBD", "Emirates NBD PJSC", "Emirates NBD Bank", "ENBD",
"DEWA", "Dubai Electricity and Water", "Dubai Electricity and Water Authority"]''',
        "Find all names that match 'Emirates NBD Bank' from the loaded list",
        "Analyze how the name 'Dubai Electricity and Water Authority' will be processed",
        '''Batch match these canonical names: ["Emirates NBD", "DEWA"]''',
        "Create a canonical mapping for 'Emirates NBD' showing all variations",
    ]
    for q in name_examples:
        st.code(q, language=None)


if __name__ == "__main__":
    main()
