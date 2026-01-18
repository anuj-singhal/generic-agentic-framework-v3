# Data Agent - LLM-Powered Wealth Management Analysis

## Overview

The `data_agent` is a specialized AI agent for analyzing wealth management data using DuckDB. Unlike traditional text-to-SQL systems that use templates or rule-based approaches, this agent leverages **LLM intelligence** at every step of the query process.

## Key Features

### 1. **LLM-Powered Schema Analysis**
The `get_relevant_schema` tool doesn't just use keyword matching - it provides the complete database schema with sample data to the LLM, which then intelligently analyzes:
- Which tables are needed based on semantic understanding of the query
- Which specific columns are relevant
- What relationships (JOINs) will be required
- The overall query approach (single table, simple JOIN, or complex CTEs)

### 2. **LLM-Powered SQL Generation**
The `generate_sql` tool provides structured instructions for the LLM to generate SQL, including:
- Complete database schema with all tables and columns
- Table relationships and foreign keys
- Query complexity guidelines
- Sample SQL patterns for different complexity levels
- The LLM generates contextual, accurate SQL (not template-based)

### 3. **LLM-Powered Multi-Level Validation**
The `validate_sql` tool performs comprehensive validation:
- **Automated**: Syntax check using DuckDB EXPLAIN
- **LLM-Powered**: Schema validation (do tables/columns exist?)
- **LLM-Powered**: Semantic validation (are JOINs correct? GROUP BY logic sound?)
- **LLM-Powered**: Best practices review
- **LLM-Powered**: Logical correctness assessment
- Returns confidence level: HIGH/MEDIUM/LOW

### 4. **Intelligent Retry Logic**
Built into the agent's workflow:
- If validation confidence is LOW: Up to 3 retry attempts
- Attempt 1: Fix identified issues manually
- Attempt 2: Regenerate SQL with corrections
- Attempt 3: Simplify query or break into smaller parts

## Database Schema

The wealth management database contains:

**CLIENTS** (10 records)
- Client profiles with risk assessment and KYC status
- Fields: CLIENT_ID, FULL_NAME, COUNTRY, RISK_PROFILE, ONBOARDING_DATE, KYC_STATUS

**PORTFOLIOS** (15 records)
- Investment accounts (clients can have multiple portfolios)
- Fields: PORTFOLIO_ID, CLIENT_ID, PORTFOLIO_NAME, BASE_CURRENCY, INCEPTION_DATE, STATUS

**ASSETS** (35 records)
- Tradable instruments (stocks, ETFs, crypto)
- Fields: ASSET_ID, SYMBOL, ASSET_NAME, ASSET_TYPE, CURRENCY, EXCHANGE

**TRANSACTIONS** (1,200 records)
- Complete trade history (buy/sell activity)
- Fields: TRANSACTION_ID, PORTFOLIO_ID, ASSET_ID, TRADE_DATE, TRANSACTION_TYPE, QUANTITY, PRICE, FEES, CURRENCY, CREATED_AT

**HOLDINGS** (305 records)
- Current position snapshots
- Fields: PORTFOLIO_ID, ASSET_ID, QUANTITY, AVG_COST, LAST_UPDATED

## Setup Instructions

### 1. Initialize the Database
```bash
python init_duckdb.py
```
This creates `agent_ddb.db` with all tables and sample data.

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Test the Agent
```bash
# Quick test
python test_data_agent.py

# Run test suite
python run_tests.py --agent data_agent

# Run by category
python run_tests.py --category wealth_simple
python run_tests.py --category wealth_medium
python run_tests.py --category wealth_complex

# Interactive mode
python run_tests.py --interactive
```

### 4. Use in Streamlit UI
```bash
streamlit run app.py
```
Then select **data_agent** from the sidebar.

## Example Queries

### Simple Queries
- "Show me all clients in the database"
- "List all active portfolios"
- "What are the different risk profiles of our clients?"
- "Show me sample data from the assets table"

### Medium Complexity
- "Show me all clients along with their portfolio names and currencies"
- "For each portfolio, show me the total number of different assets they hold"
- "Show me the 10 most recent transactions with client names, portfolio names, and asset symbols"
- "How many assets do we have of each type (Equity, ETF, Crypto)?"

### Complex Queries
- "Calculate the total value of each portfolio based on current holdings. Show client name, portfolio name, base currency, and total portfolio value (quantity * average cost). Order by total value descending."
- "For each client, show: Number of portfolios they own, Total number of unique assets across all their portfolios, Number of different asset types (Equity, ETF, Crypto) they hold. Group by client and order by number of assets descending."
- "Generate a trading activity report showing: Client name, Total number of buy transactions, Total number of sell transactions, Total transaction count, Most frequently traded asset symbol. Only include clients with more than 5 transactions."

## Agent Workflow

The data_agent follows a 7-step process:

1. **Understand the Query** (LLM-powered schema analysis)
   - Use `get_relevant_schema` to identify needed tables and columns
   - LLM analyzes query semantics and provides structured analysis

2. **Analyze Complexity**
   - Use `analyze_query_complexity` to determine SIMPLE/MEDIUM/COMPLEX
   - Guides the SQL generation approach

3. **Generate SQL** (LLM-powered)
   - Use `generate_sql` for structured SQL generation instructions
   - LLM generates contextual SQL based on schema and complexity
   - Complex queries use CTEs for readability

4. **Validate SQL** (LLM-powered multi-level)
   - Use `validate_sql` for comprehensive validation
   - Automated syntax check + LLM schema/semantic validation
   - Returns HIGH/MEDIUM/LOW confidence level

5. **Retry Logic** (Up to 3 attempts)
   - LOW confidence: Fix issues and retry
   - MEDIUM confidence: Proceed with warnings
   - HIGH confidence: Execute immediately

6. **Execute Query**
   - Use `execute_sql` to run validated query
   - Results limited to 100 rows by default

7. **Explain Results**
   - Use `explain_sql` for plain English explanation
   - Help user understand the analysis performed

## Available Tools

The data_agent has access to 10 specialized tools:

**Core LLM-Powered Tools:**
1. `get_relevant_schema` - LLM analyzes which tables/columns are needed
2. `generate_sql` - LLM generates SQL from structured instructions
3. `validate_sql` - LLM performs multi-level validation with confidence scoring

**Supporting Tools:**
4. `show_tables` - List all database tables
5. `describe_table` - Get table schema details
6. `analyze_query_complexity` - Determine query complexity level
7. `execute_sql` - Run validated SQL queries
8. `explain_sql` - Plain English explanation of queries
9. `get_sample_data` - View sample rows from tables
10. `summarize_table` - Statistical summaries with aggregations

## Test Examples

The framework includes **19 test examples** across 3 complexity levels:

**WEALTH_SIMPLE_EXAMPLES** (5 tests)
- Basic table queries
- Simple filters
- Schema inspection

**WEALTH_MEDIUM_EXAMPLES** (6 tests)
- Multi-table JOINs
- Aggregations with GROUP BY
- Recent data queries

**WEALTH_COMPLEX_EXAMPLES** (8 tests)
- CTEs with multiple JOINs
- Window functions and ranking
- Multi-level aggregations
- Time-based analysis
- Percentage calculations

## Architecture Highlights

### LLM-Powered vs Traditional Approaches

**Traditional Text-to-SQL:**
- Template-based pattern matching
- Limited to predefined query types
- Brittle with new query variations
- No semantic understanding

**This Implementation:**
- LLM understands query semantics
- Generates contextual SQL based on schema
- Validates comprehensively (syntax + schema + semantics)
- Handles complex queries with CTEs
- Provides confidence scoring
- Retry logic for low-confidence queries

### Integration with Framework

The data_agent follows all existing framework patterns:
- Uses `AgentDefinition` for configuration
- Registered with `AgentFactory`
- Tools registered with `tool_registry`
- Compatible with test framework
- Works in Streamlit UI
- Follows ReAct orchestration pattern

## Files Created/Modified

**New Files:**
- `init_duckdb.py` - Database initialization script
- `tools/duckdb_tools.py` - 10 DuckDB tools (LLM-powered)
- `agent_ddb.db` - DuckDB database with wealth data
- `test_data_agent.py` - Quick test script
- `DATA_AGENT_README.md` - This file

**Modified Files:**
- `requirements.txt` - Added duckdb>=1.0.0
- `tools/__init__.py` - Export DuckDB tools
- `agents/agent_definitions.py` - Registered data_agent
- `test_examples.py` - Added 19 wealth test examples
- `run_tests.py` - Added wealth test categories

## Performance Considerations

- **Database Size**: 1,565 total records across 5 tables
- **Query Execution**: Sub-second for most queries
- **LLM Calls**: 3-5 tool calls per query (schema → complexity → generate → validate → execute)
- **Retry Logic**: Max 3 attempts for low-confidence queries
- **Result Limits**: Default 100 rows to prevent overwhelming output

## Best Practices

1. **Always start with get_relevant_schema** - Provides context for accurate SQL generation
2. **Use analyze_query_complexity** - Helps determine appropriate query structure
3. **Validate before executing** - Catch errors early with multi-level validation
4. **Review validation confidence** - LOW confidence requires retry or manual review
5. **Use CTEs for complex queries** - Improves readability and maintainability
6. **Add meaningful aliases** - Makes results easier to understand
7. **Include LIMIT clauses** - Prevents overwhelming output on large result sets

## Troubleshooting

**Database not found:**
```bash
python init_duckdb.py
```

**API key not set:**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Low validation confidence:**
- Review the validation report for specific issues
- The agent will automatically retry up to 3 times
- Check if column names match schema exactly (case-sensitive)
- Verify JOIN conditions use correct foreign keys

**Query timeout:**
- Simplify the query
- Add more specific filters (WHERE clauses)
- Use LIMIT to restrict result set

## Future Enhancements

Potential improvements:
- Query result caching for common queries
- Query optimization suggestions
- Performance profiling and indexing recommendations
- Integration with real-time market data
- Support for additional database backends
- Query history and favorites
- Scheduled/automated reports
- Data visualization integration

## Support

For issues or questions:
1. Check the validation report for specific error details
2. Review test examples for similar query patterns
3. Use `get_sample_data` to understand table contents
4. Use `describe_table` to verify column names and types
5. Run test suite: `python run_tests.py --agent data_agent`

---

**Built with**: DuckDB, LangChain, LangGraph, OpenAI GPT-4o-mini, Streamlit

**License**: MIT
