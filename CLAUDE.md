# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular, scalable agentic AI framework built with LangGraph and Streamlit that implements the ReAct (Reasoning and Acting) pattern. The framework creates autonomous AI agents that can use tools to accomplish tasks through iterative reasoning.

## Common Commands

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize DuckDB database (for data_agent)
python init_duckdb.py

# Run the Streamlit UI
streamlit run app.py
```

### Testing

```bash
# Run all tests
python run_tests.py

# Run specific test category
python run_tests.py --category simple
python run_tests.py --category sql
python run_tests.py --category name_matching

# Run tests for specific agent
python run_tests.py --agent sql_specialist
python run_tests.py --agent name_matcher
python run_tests.py --agent data_agent

# Test wealth portfolio queries (data_agent)
python run_tests.py --category wealth_simple
python run_tests.py --category wealth

# Quick test for data_agent
python test_data_agent.py

# Interactive test mode
python run_tests.py --interactive

# List all available test examples
python run_tests.py --list

# Export test results to JSON
python run_tests.py --export
```

### Environment Setup

```bash
# Set OpenAI API key (required)
export OPENAI_API_KEY="your-api-key-here"

# Optional configuration
export MODEL_NAME="gpt-4o-mini"
export TEMPERATURE="0.7"
export MAX_ITERATIONS="10"
```

## Architecture Overview

The framework implements three core components modeled after autonomous systems:

### 1. The Model (Brain) - `core/orchestrator.py`

The central reasoning engine powered by GPT-4o-mini (configurable). The `ReActOrchestrator` class manages the agent's operational loop:

- **LangGraph State Machine**: Uses `StateGraph` to manage the Think → Act → Observe cycle
- **Nodes**: `reason` (thinking), `tools` (acting), `synthesize` (finalizing)
- **State Flow**: Entry → Reason → [Tools → Reason]* → Synthesize/End
- **Iteration Control**: Configurable max iterations (default: 10) prevents infinite loops

The orchestrator binds tools to the LLM and manages message flow through the graph.

### 2. Tools (Hands) - `tools/` directory

Tools extend the agent's capabilities beyond pure reasoning:

- **Tool Registry Pattern**: `core/tools_base.py` provides centralized tool registration and discovery
- **Category-based Organization**: Tools grouped by function (math, datetime, text, data, tasks, knowledge, sql, name_matching)
- **LangChain Integration**: Tools decorated with `@tool` for automatic schema generation
- **Dynamic Tool Assignment**: Agents select tool subsets based on their specialization

Tool categories:
- `math`: calculator, unit_converter
- `datetime`: get_current_datetime, calculate_date_difference, add_days_to_date
- `text`: text_analyzer, text_transformer
- `data`: list_operations, json_parser
- `tasks`: create_task, list_tasks, update_task_status
- `knowledge`: knowledge_base_search
- `sql`: Text-to-SQL tools (generate_sql, validate_sql, execute_sql, explain_sql, get_schema, etc.)
- `name_matching`: Fuzzy name matching tools for entity resolution (load_names_for_matching, find_matching_names, batch_match_names, etc.)

### 3. Orchestration Layer (Nervous System) - `core/orchestrator.py`

Implements the ReAct loop using LangGraph:

- **Five-Step Cycle**:
  1. Get the Mission (user request)
  2. Scan the Scene (gather context)
  3. Think It Through (reason about approach)
  4. Take Action (execute tools)
  5. Observe and Iterate (review results, repeat if needed)

- **State Management**: `GraphState` TypedDict tracks messages, thoughts, observations, actions, iteration count, and completion status
- **Conditional Routing**: `_should_continue()` determines next step based on tool calls and iteration limits
- **Error Handling**: Graceful degradation with synthesis node when max iterations reached

## Agent System - `agents/agent_definitions.py`

### Agent Factory Pattern

The `AgentFactory` class manages agent definitions and instantiation:

- **Registration**: Agents defined as `AgentDefinition` dataclass instances
- **Tool Selection**: Each agent specifies `tool_categories` and `specific_tools` lists
- **Custom System Prompts**: Agents have specialized instructions for their domain
- **Orchestrator Creation**: Factory creates `ReActOrchestrator` instances with agent-specific tools

### Pre-defined Agents

- `general_assistant`: Versatile, uses math, datetime, text, tasks, knowledge tools
- `math_specialist`: Focused on calculations and conversions
- `task_manager`: Specialized in task organization
- `researcher`: Information retrieval and analysis
- `data_analyst`: Data processing and parsing
- `sql_specialist`: Text-to-SQL expert with schema understanding
- `sql_analyst`: Combines SQL with data analysis for business insights
- `name_matcher`: Entity name matching and canonicalization (handles 10-20k names)
- `name_data_processor`: Combines name matching with data processing

### Creating New Agents

1. Define an `AgentDefinition` with:
   - `name`: Unique identifier
   - `description`: Capabilities summary
   - `system_prompt`: Detailed instructions for the agent
   - `tool_categories`: List of tool categories to include
   - `specific_tools`: Optional list of specific tool names
   - `max_iterations`: Override default iteration limit

2. Register with `AgentFactory.register_agent(definition)`

3. Agent automatically available in UI and testing framework

## Configuration System - `core/config.py`

Two-level configuration structure:

- **ModelConfig**: LLM settings (model_name, temperature, max_tokens, api_key)
- **FrameworkConfig**: Agent settings (model, max_iterations, verbose, memory_enabled)

Configuration sources (priority order):
1. Programmatically passed config objects
2. Environment variables (MODEL_NAME, TEMPERATURE, MAX_ITERATIONS, OPENAI_API_KEY)
3. Default values

## Memory and State - `core/memory.py`

The `AgentState` class tracks:
- Conversation messages
- Agent thoughts and observations
- Action history
- Iteration count
- Completion status
- Final answer

State flows through the LangGraph using the `add_messages` reducer for message accumulation.

## Streamlit UI - `app.py`

Multi-component interface:

- **Sidebar**: Agent selection, API key input, model configuration, tool display
- **Main Chat**: Message history with ReAct trace expansion
- **Trace Visualization**: Color-coded boxes for thoughts (blue), actions (orange), observations (green)
- **Session State**: Persistent across reruns for messages, execution history, current agent

The UI automatically discovers agents from `AgentFactory` and displays their available tools.

## Testing Framework - `test_examples.py` and `run_tests.py`

### Test Organization

Tests categorized by complexity:
- `SIMPLE_EXAMPLES`: Single tool, straightforward tasks
- `MEDIUM_EXAMPLES`: Multiple tools, multi-step reasoning
- `COMPLEX_EXAMPLES`: Multi-agent capable, complex reasoning chains
- `EDGE_CASES`: Error handling and robustness
- `CONVERSATIONAL_EXAMPLES`: Natural language queries
- `SQL_*_EXAMPLES`: Text-to-SQL scenarios (simple, medium, complex)
- `NAME_MATCHING_*_EXAMPLES`: Entity name matching scenarios

### Test Structure

Each test is a dict with:
- `name`: Test identifier
- `agent`: Which agent to use
- `query`: User input
- `expected_tools`: Tools that should be used
- `description`: What the test validates

### TestRunner Features

- Run by category, agent, or tool
- Interactive mode for manual testing
- JSON export for CI/CD integration
- Detailed timing and success metrics
- Tool usage statistics

## SQL Tools - `tools/sql_tools.py`

Implements text-to-SQL capabilities with simulated database schemas:

- **Schema Management**: Pre-loaded example schemas (ecommerce, hr, analytics)
- **Tools**: list_databases, get_schema, get_table_info, generate_sql, validate_sql, execute_sql, explain_sql, sql_examples
- **In-Memory Store**: Simulated database (production would connect to actual DBs)
- **Schema Structure**: Tables with columns, types, constraints, relationships

Agents use multi-step workflow:
1. List/explore databases
2. Get schema for relevant database
3. Generate SQL from natural language
4. Validate syntax
5. Execute and return simulated results
6. Explain query in plain English

## Name Matching Tools - `tools/name_matching_tools.py`

Handles entity name matching and canonicalization for large datasets (10-20k names):

- **Batch Processing**: Manages token limits by processing in configurable batches
- **Normalization**: Uppercase, remove special chars, expand abbreviations, remove business suffixes
- **Multiple Similarity Algorithms**: SequenceMatcher, token-based Jaccard, containment checking
- **Session Management**: In-memory store for loaded names with session IDs
- **Confidence Levels**: HIGH (≥0.85), MEDIUM (0.70-0.85), LOW (0.55-0.70)

**Key Tools**:
- `load_names_for_matching`: Load names in batches with append mode
- `find_matching_names`: Find all matches for single canonical name
- `batch_match_names`: Match multiple canonical names efficiently
- `create_canonical_mapping`: Structured mapping output
- `bulk_create_mappings`: Create many mappings at once
- `analyze_name`: Debug how a name will be processed
- `get_session_info`: View session statistics

**Business Suffixes Handled**: LLC, Inc, Ltd, PJSC, Corp, Group, Holdings, Bank, Authority, plus UAE-specific (FZ, FZE, DMCC) and European (SA, AG, GmbH, BV)

**Common Abbreviations Expanded**: INTL→International, Dept→Department, Tech→Technology, Govt→Government, etc.

## DuckDB Data Agent - `tools/duckdb_tools.py`

The data_agent provides LLM-powered analysis of wealth management data using DuckDB. Unlike traditional text-to-SQL systems, this implementation uses **LLM intelligence at every step**.

### Database Schema (`agent_ddb.db`)

Created via `init_duckdb.py` with 5 tables and real wealth portfolio data:
- **CLIENTS** (10 records): Client profiles with risk assessment, KYC status
- **PORTFOLIOS** (15 records): Investment accounts (multiple per client)
- **ASSETS** (35 records): Tradable instruments (stocks, ETFs, crypto)
- **TRANSACTIONS** (1,200 records): Complete trade history (buy/sell)
- **HOLDINGS** (305 records): Current position snapshots

### LLM-Powered Tools Architecture

**Three Core LLM-Powered Tools:**

1. **`get_relevant_schema`** - LLM-based schema analysis
   - Provides complete schema WITH sample data (3 rows per table)
   - LLM analyzes which tables/columns are needed based on query semantics
   - Returns structured analysis: tables needed, key columns, relationships, query approach
   - NOT keyword-based - uses LLM intelligence to understand context

2. **`generate_sql`** - LLM-based SQL generation
   - Provides structured prompt with schema, relationships, complexity guidelines
   - LLM generates contextual SQL (not template-based)
   - Automatically suggests CTEs for complex queries
   - Includes sample patterns for SIMPLE/MEDIUM/COMPLEX queries

3. **`validate_sql`** - LLM-based multi-level validation
   - **Automated**: DuckDB EXPLAIN for syntax check
   - **LLM-Powered**: Schema validation (tables/columns exist?)
   - **LLM-Powered**: Semantic validation (JOINs correct? GROUP BY logic sound?)
   - **LLM-Powered**: Best practices review
   - **LLM-Powered**: Logical correctness assessment
   - Returns confidence score: HIGH/MEDIUM/LOW with specific recommendations

**Supporting Tools:**
- `show_tables`: List all tables
- `describe_table`: Get table schema
- `analyze_query_complexity`: Determine SIMPLE/MEDIUM/COMPLEX
- `execute_sql`: Run validated queries
- `explain_sql`: Plain English explanation
- `get_sample_data`: View sample rows
- `summarize_table`: Statistical summaries

### Agent Workflow with Retry Logic

The data_agent follows a 7-step process:

1. **Schema Analysis** (LLM): `get_relevant_schema` → LLM identifies needed tables/columns
2. **Complexity Analysis**: `analyze_query_complexity` → SIMPLE/MEDIUM/COMPLEX
3. **SQL Generation** (LLM): `generate_sql` → LLM creates SQL with CTEs if needed
4. **Validation** (LLM + automated): `validate_sql` → Comprehensive validation with confidence score
5. **Retry Logic** (up to 3 attempts):
   - HIGH confidence → Execute immediately
   - MEDIUM confidence → Proceed with warnings
   - LOW confidence → Fix and retry (3 attempts max)
6. **Execution**: `execute_sql` → Run validated query
7. **Explanation**: `explain_sql` → Plain English explanation

### Key Differentiators

**Traditional Text-to-SQL:**
- Template-based pattern matching
- Limited to predefined query types
- No semantic understanding
- Brittle with variations

**This Implementation:**
- LLM understands query semantics
- Generates contextual SQL based on schema AND sample data
- Multi-level validation (syntax + schema + semantics)
- Confidence scoring with retry logic
- Handles complex queries with CTEs
- Works with ANY query, not just templates

### Query Examples

**Simple**: "Show me all clients in the database"
**Medium**: "Show me all clients along with their portfolio names and currencies"
**Complex**: "Calculate the total value of each portfolio based on current holdings. Show client name, portfolio name, base currency, and total portfolio value. Order by total value descending."

See `DATA_AGENT_README.md` for comprehensive documentation.

## Development Patterns

### Adding New Tools

1. Create tool function in appropriate file under `tools/`
2. Decorate with `@tool` and include docstring (becomes tool description)
3. Register with `tool_registry.register(tool_instance, "category_name")`
4. Tool automatically available to agents that include that category

### Error Handling

- Tools should return descriptive error messages (not raise exceptions unnecessarily)
- Orchestrator catches exceptions and includes them in state
- Synthesis node provides fallback when max iterations reached
- Edge case tests validate error handling

### State Management Best Practices

- Use `add_messages` reducer for message accumulation in LangGraph
- Track iteration count to prevent infinite loops
- Include error field in state for debugging
- Use TypedDict for type safety in graph state

### Testing New Agents

1. Add examples to `test_examples.py` in appropriate category
2. Run with `python run_tests.py --agent <agent_name>`
3. Review trace output to validate reasoning chain
4. Check tool usage matches expectations

## Important Implementation Notes

- **DuckDB Database**: The data_agent uses a real DuckDB database (`agent_ddb.db`) with 1,565 records of wealth data - NOT in-memory
- **Simulated SQL Schemas**: The sql_specialist tools (not data_agent) use mock schemas for demonstration
- **In-Memory Storage**: Name matching and other tools use module-level dicts (session state in Streamlit)
- **Token Limits**: Name matching tools batch process to stay within context limits
- **LLM-Powered Tools**: data_agent tools (`get_relevant_schema`, `generate_sql`, `validate_sql`) provide structured prompts for LLM analysis - NOT black-box tools
- **LangGraph Compilation**: Graph must be compiled before use (`workflow.compile()`)
- **Tool Binding**: Tools must be bound to LLM before graph execution (`llm.bind_tools()`)
- **Message Types**: Use LangChain message types (HumanMessage, AIMessage, ToolMessage, SystemMessage)

## File Organization

```
agentic_framework/
├── app.py                          # Streamlit UI entry point
├── requirements.txt                # Python dependencies
├── test_examples.py                # Test case definitions (19 wealth examples added)
├── run_tests.py                    # Test runner with CLI
├── init_duckdb.py                  # DuckDB database initialization
├── test_data_agent.py              # Quick test script for data_agent
├── agent_ddb.db                    # DuckDB database (1,565 records)
├── CLAUDE.md                       # This file
├── DATA_AGENT_README.md            # Comprehensive data_agent documentation
├── core/                           # Core framework
│   ├── config.py                   # Configuration management
│   ├── memory.py                   # State and memory structures
│   ├── tools_base.py               # Tool registry and base classes
│   └── orchestrator.py             # ReAct orchestration with LangGraph
├── tools/                          # Tool implementations
│   ├── example_tools.py            # Built-in example tools
│   ├── sql_tools.py                # Text-to-SQL tools (simulated schemas)
│   ├── name_matching_tools.py      # Name matching and canonicalization
│   └── duckdb_tools.py             # LLM-powered DuckDB tools (10 tools)
├── agents/                         # Agent definitions
│   └── agent_definitions.py        # Pre-configured agents (data_agent added)
├── pages/                          # Streamlit pages
│   └── example_queries.py          # Example queries UI
└── sample_files/                   # Sample data and schemas
    ├── duckdb.py                   # Reference implementation
    ├── wealth_tables.json          # Database schema definition
    └── wealth_data/                # CSV files (clients, portfolios, etc.)
        ├── clients.csv
        ├── portfolios.csv
        ├── assets.csv
        ├── transactions.csv
        └── holdings.csv
```

## Key Dependencies

- **langgraph**: State machine orchestration for agents
- **langchain**: Agent framework and tool abstractions
- **langchain-openai**: OpenAI LLM integration
- **streamlit**: Web UI framework
- **pydantic**: Data validation and settings
- **python-dotenv**: Environment variable management
- **duckdb**: Embedded analytical database for data_agent (1.0.0+)
