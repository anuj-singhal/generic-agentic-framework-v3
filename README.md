# 🤖 Agentic AI Framework

A modular, scalable agentic AI framework built with LangGraph and Streamlit, implementing the ReAct (Reasoning and Acting) pattern.

## 🏗️ Architecture

This framework implements the three essential elements of autonomous AI systems:

### 1. 🧠 The Model (Brain)
The core language model that serves as the agent's central reasoning engine.
- Uses GPT-4o-mini (configurable)
- Processes information, evaluates options, and makes decisions
- Manages the input context window

### 2. 🤚 Tools (Hands)
Mechanisms that connect the agent's reasoning to the outside world.
- **Math Tools**: Calculator, unit converter
- **DateTime Tools**: Current time, date calculations
- **Text Tools**: Text analysis, transformation
- **Data Tools**: List operations, JSON parsing
- **Task Tools**: Task management (create, list, update)
- **Knowledge Tools**: Knowledge base search

### 3. 🔗 Orchestration Layer (Nervous System)
The governing process that manages the agent's operational loop.
- Implements ReAct pattern with LangGraph
- Handles planning, memory, and reasoning strategy
- Manages the Think → Act → Observe cycle

## 🔄 The Agent Loop

The framework implements a five-step operational cycle:

1. **Get the Mission**: Receive the user's goal or request
2. **Scan the Scene**: Gather context from tools and memory
3. **Think It Through**: Analyze and devise a plan (ReAct reasoning)
4. **Take Action**: Execute tools and actions
5. **Observe and Iterate**: Review results and continue until complete

## 📁 Project Structure

```
agentic_framework/
├── app.py                    # Streamlit UI application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── core/                     # Core framework modules
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── memory.py            # State and memory management
│   ├── tools_base.py        # Tool registry and base classes
│   └── orchestrator.py      # LangGraph ReAct orchestration
│
├── tools/                    # Tool implementations
│   ├── __init__.py
│   └── example_tools.py     # Built-in example tools
│
└── agents/                   # Agent definitions
    ├── __init__.py
    └── agent_definitions.py # Pre-defined agent configurations
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file or export your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Access the UI

Open your browser to `http://localhost:8501`

## 🤖 Available Agents

| Agent | Description |
|-------|-------------|
| **General Assistant** | Versatile assistant for various tasks |
| **Math Specialist** | Focused on calculations and conversions |
| **Task Manager** | Specialized in task organization |
| **Researcher** | Information retrieval and analysis |
| **Data Analyst** | Data processing and parsing |

## 🔧 Available Tools

### Math Tools
- `calculator` - Evaluate mathematical expressions
- `unit_converter` - Convert between units

### DateTime Tools
- `get_current_datetime` - Get current date/time
- `calculate_date_difference` - Calculate days between dates
- `add_days_to_date` - Add/subtract days from a date

### Text Tools
- `text_analyzer` - Analyze text statistics
- `text_transformer` - Transform text (uppercase, reverse, etc.)

### Data Tools
- `list_operations` - Sort, filter, and manipulate lists
- `json_parser` - Parse and extract from JSON

### Task Tools
- `create_task` - Create a new task
- `list_tasks` - List all tasks
- `update_task_status` - Update task status

### Knowledge Tools
- `knowledge_base_search` - Search the knowledge base

## 🔌 Extending the Framework

### Adding New Tools

1. Create a new tool in `tools/example_tools.py` or a new file:

```python
from langchain_core.tools import tool
from core.tools_base import tool_registry

@tool
def my_new_tool(param: str) -> str:
    """Description of what the tool does."""
    # Implementation
    return result

# Register the tool
tool_registry.register(my_new_tool, "category_name")
```

### Adding New Agents

1. Add a new agent definition in `agents/agent_definitions.py`:

```python
new_agent = AgentDefinition(
    name="new_agent",
    description="Description of the agent's capabilities",
    system_prompt="Instructions for the agent...",
    tool_categories=["math", "text"],  # Tool categories to use
    specific_tools=["my_tool"],        # Specific tools by name
    max_iterations=10
)
AgentFactory.register_agent(new_agent)
```

## 🎯 Example Usage

### Simple Calculation
```
User: What is 25% of 480?
Agent: [Uses calculator tool]
Result: 120
```

### Multi-Step Task
```
User: Create a task to review the quarterly report, set it as high priority, then list all tasks.
Agent: 
1. [Thinks] I need to create a task and then list all tasks
2. [Acts] Uses create_task tool
3. [Observes] Task TASK-0001 created
4. [Acts] Uses list_tasks tool
5. [Observes] Shows all tasks including new one
```

### Date Calculation
```
User: How many days until December 25, 2025?
Agent: [Uses get_current_datetime and calculate_date_difference]
Result: X days until December 25, 2025
```

## ⚙️ Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `MODEL_NAME` | LLM model to use | gpt-4o-mini |
| `TEMPERATURE` | Response randomness | 0.7 |
| `MAX_ITERATIONS` | Max ReAct loops | 10 |
| `VERBOSE` | Enable detailed logging | true |

## 🔒 Security Notes

- API keys are stored in session state (not persisted)
- The calculator uses safe evaluation with limited functions
- No external network calls in example tools (extend carefully)

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

Built with ❤️ using LangGraph, LangChain, and Streamlit
