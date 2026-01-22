"""
Multi Data Agent Tools Module
=============================

Provides tools for the multi-agent data workflow. These tools can be used
standalone or as part of the multi-agent orchestrator.
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any, List
import json
import os

from core.tools_base import tool_registry


# Path to schema document
SCHEMA_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "sample_files", "wealth_tables.json")

# Cache for schema document
_schema_cache: Dict[str, Any] = {}


def _load_schema_document() -> Dict[str, Any]:
    """Load and cache the schema document."""
    global _schema_cache

    if not _schema_cache:
        try:
            if os.path.exists(SCHEMA_DOCUMENT_PATH):
                with open(SCHEMA_DOCUMENT_PATH, 'r') as f:
                    _schema_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load schema document: {e}")
            _schema_cache = {}

    return _schema_cache


# =============================================================================
# AGENT 2 TOOLS: Schema Extraction (No LLM required)
# =============================================================================

@tool
def get_schema_document() -> str:
    """
    Get the complete schema document with all tables, columns, and descriptions.
    This is the foundational document for understanding the database structure.

    Returns:
        JSON string containing the complete schema definition
    """
    schema = _load_schema_document()
    if not schema:
        return "Error: Schema document not found or empty."

    return json.dumps(schema, indent=2)


@tool
def get_table_descriptions() -> str:
    """
    Get a summary of all tables with their descriptions.
    Useful for quick overview of available data.

    Returns:
        Formatted list of tables and their descriptions
    """
    schema = _load_schema_document()
    if not schema or "tables" not in schema:
        return "Error: Schema document not available."

    descriptions = []
    for table in schema["tables"]:
        descriptions.append(f"**{table['table_name']}**: {table['table_description']}")

    return "\n".join(descriptions)


@tool
def extract_table_schema(table_name: str) -> str:
    """
    Extract detailed schema for a specific table including column descriptions
    and sample values.

    Args:
        table_name: Name of the table (e.g., CLIENTS, PORTFOLIOS)

    Returns:
        JSON with table schema, column details, and sample values
    """
    schema = _load_schema_document()
    if not schema or "tables" not in schema:
        return "Error: Schema document not available."

    table_name_upper = table_name.upper()

    for table in schema["tables"]:
        if table["table_name"] == table_name_upper:
            result = {
                "table_name": table["table_name"],
                "description": table["table_description"],
                "columns": []
            }

            for col in table["columns"]:
                result["columns"].append({
                    "name": col["column_name"],
                    "data_type": col["data_type"],
                    "description": col["description"],
                    "is_primary_key": col.get("is_primary_key", False),
                    "nullable": col.get("nullable", True),
                    "sample_values": col.get("sample_values", [])
                })

            return json.dumps(result, indent=2)

    return f"Error: Table '{table_name}' not found in schema."


@tool
def extract_related_tables(table_names: str) -> str:
    """
    Extract schema for multiple related tables and their relationships.

    Args:
        table_names: Comma-separated list of table names (e.g., "CLIENTS,PORTFOLIOS")

    Returns:
        JSON with schemas for all specified tables and their relationships
    """
    schema = _load_schema_document()
    if not schema or "tables" not in schema:
        return "Error: Schema document not available."

    requested_tables = [t.strip().upper() for t in table_names.split(",")]

    result = {
        "tables": {},
        "relationships": []
    }

    # Standard relationships
    all_relationships = [
        {"from": "PORTFOLIOS.CLIENT_ID", "to": "CLIENTS.CLIENT_ID", "type": "many-to-one"},
        {"from": "TRANSACTIONS.PORTFOLIO_ID", "to": "PORTFOLIOS.PORTFOLIO_ID", "type": "many-to-one"},
        {"from": "TRANSACTIONS.ASSET_ID", "to": "ASSETS.ASSET_ID", "type": "many-to-one"},
        {"from": "HOLDINGS.PORTFOLIO_ID", "to": "PORTFOLIOS.PORTFOLIO_ID", "type": "many-to-one"},
        {"from": "HOLDINGS.ASSET_ID", "to": "ASSETS.ASSET_ID", "type": "many-to-one"}
    ]

    for table in schema["tables"]:
        if table["table_name"] in requested_tables:
            result["tables"][table["table_name"]] = {
                "description": table["table_description"],
                "columns": [
                    {
                        "name": col["column_name"],
                        "data_type": col["data_type"],
                        "description": col["description"],
                        "sample_values": col.get("sample_values", [])[:3]
                    }
                    for col in table["columns"]
                ]
            }

    # Add relevant relationships
    for rel in all_relationships:
        from_table = rel["from"].split(".")[0]
        to_table = rel["to"].split(".")[0]
        if from_table in requested_tables or to_table in requested_tables:
            result["relationships"].append(rel)

    return json.dumps(result, indent=2)


@tool
def get_column_sample_values(table_name: str, column_name: str) -> str:
    """
    Get sample values for a specific column.
    Useful for understanding what data looks like.

    Args:
        table_name: Name of the table
        column_name: Name of the column

    Returns:
        Sample values for the specified column
    """
    schema = _load_schema_document()
    if not schema or "tables" not in schema:
        return "Error: Schema document not available."

    table_name_upper = table_name.upper()
    column_name_upper = column_name.upper()

    for table in schema["tables"]:
        if table["table_name"] == table_name_upper:
            for col in table["columns"]:
                if col["column_name"] == column_name_upper:
                    sample_values = col.get("sample_values", [])
                    return json.dumps({
                        "table": table_name_upper,
                        "column": column_name_upper,
                        "data_type": col["data_type"],
                        "sample_values": sample_values
                    }, indent=2)

            return f"Error: Column '{column_name}' not found in table '{table_name}'."

    return f"Error: Table '{table_name}' not found."


@tool
def get_database_relationships() -> str:
    """
    Get all foreign key relationships between tables.
    Essential for understanding how to JOIN tables.

    Returns:
        List of all table relationships with explanations
    """
    relationships = [
        {
            "relationship": "PORTFOLIOS.CLIENT_ID -> CLIENTS.CLIENT_ID",
            "type": "Many-to-One",
            "description": "Each portfolio belongs to one client. A client can have multiple portfolios."
        },
        {
            "relationship": "TRANSACTIONS.PORTFOLIO_ID -> PORTFOLIOS.PORTFOLIO_ID",
            "type": "Many-to-One",
            "description": "Each transaction belongs to one portfolio. A portfolio can have many transactions."
        },
        {
            "relationship": "TRANSACTIONS.ASSET_ID -> ASSETS.ASSET_ID",
            "type": "Many-to-One",
            "description": "Each transaction involves one asset. An asset can be in many transactions."
        },
        {
            "relationship": "HOLDINGS.PORTFOLIO_ID -> PORTFOLIOS.PORTFOLIO_ID",
            "type": "Many-to-One (Composite PK)",
            "description": "Each holding record belongs to one portfolio."
        },
        {
            "relationship": "HOLDINGS.ASSET_ID -> ASSETS.ASSET_ID",
            "type": "Many-to-One (Composite PK)",
            "description": "Each holding record represents one asset in a portfolio."
        }
    ]

    return json.dumps(relationships, indent=2)


# =============================================================================
# QUERY COMPLEXITY ANALYSIS TOOLS
# =============================================================================

@tool
def analyze_query_requirements(nl_query: str) -> str:
    """
    Analyze a natural language query to determine data requirements.
    Identifies which tables might be needed and potential complexity.

    Args:
        nl_query: Natural language query from user

    Returns:
        Analysis of query requirements including potential tables and complexity indicators
    """
    nl_query_lower = nl_query.lower()

    # Table keywords mapping
    table_keywords = {
        "CLIENTS": ["client", "customer", "investor", "kyc", "onboarding", "country", "risk profile"],
        "PORTFOLIOS": ["portfolio", "investment account", "base currency", "inception"],
        "ASSETS": ["asset", "stock", "equity", "etf", "crypto", "symbol", "ticker", "instrument"],
        "TRANSACTIONS": ["transaction", "trade", "buy", "sell", "trade date", "fee", "price"],
        "HOLDINGS": ["holding", "position", "quantity", "avg cost", "current"]
    }

    # Complexity indicators
    complexity_keywords = {
        "aggregation": ["total", "sum", "count", "average", "max", "min", "top", "bottom"],
        "grouping": ["by", "each", "per", "grouped"],
        "comparison": ["more than", "less than", "greater", "highest", "lowest", "compare"],
        "multi_table": ["and their", "along with", "including", "with", "together"],
        "calculation": ["calculate", "compute", "value", "percentage", "ratio"]
    }

    # Identify potential tables
    potential_tables = []
    for table, keywords in table_keywords.items():
        if any(kw in nl_query_lower for kw in keywords):
            potential_tables.append(table)

    # Identify complexity
    complexity_indicators = []
    for indicator_type, keywords in complexity_keywords.items():
        if any(kw in nl_query_lower for kw in keywords):
            complexity_indicators.append(indicator_type)

    # Determine complexity level
    if len(potential_tables) <= 1 and len(complexity_indicators) <= 1:
        complexity = "SIMPLE"
    elif len(potential_tables) <= 2 and len(complexity_indicators) <= 2:
        complexity = "MEDIUM"
    else:
        complexity = "COMPLEX"

    result = {
        "original_query": nl_query,
        "potential_tables": potential_tables if potential_tables else ["Unable to determine - please specify"],
        "complexity_indicators": complexity_indicators,
        "estimated_complexity": complexity,
        "recommendations": []
    }

    if complexity == "SIMPLE":
        result["recommendations"].append("Use simple SELECT with optional WHERE clause")
    elif complexity == "MEDIUM":
        result["recommendations"].append("Use JOINs to combine tables")
        result["recommendations"].append("Consider GROUP BY for aggregations")
    else:
        result["recommendations"].append("Use CTEs (WITH clause) to break down logic")
        result["recommendations"].append("Consider multiple JOINs and subqueries")

    return json.dumps(result, indent=2)


# =============================================================================
# SQL VALIDATION HELPER TOOLS
# =============================================================================

@tool
def validate_table_references(sql_query: str) -> str:
    """
    Validate that all table references in SQL exist in the schema.

    Args:
        sql_query: SQL query to validate

    Returns:
        Validation result with any issues found
    """
    schema = _load_schema_document()
    if not schema or "tables" not in schema:
        return "Error: Schema document not available."

    valid_tables = {table["table_name"] for table in schema["tables"]}

    # Simple extraction of potential table names (not perfect, but helpful)
    sql_upper = sql_query.upper()
    words = sql_upper.replace(",", " ").replace("(", " ").replace(")", " ").split()

    # Find words after FROM and JOIN
    potential_references = []
    for i, word in enumerate(words):
        if word in ("FROM", "JOIN") and i + 1 < len(words):
            potential_references.append(words[i + 1])

    issues = []
    valid_refs = []
    for ref in potential_references:
        if ref in valid_tables:
            valid_refs.append(ref)
        elif ref not in ("SELECT", "WHERE", "ON", "AND", "OR", "AS", "LEFT", "RIGHT", "INNER", "OUTER"):
            issues.append(f"Unknown table reference: {ref}")

    result = {
        "valid_table_references": valid_refs,
        "issues": issues,
        "is_valid": len(issues) == 0
    }

    return json.dumps(result, indent=2)


@tool
def get_join_syntax_help(table1: str, table2: str) -> str:
    """
    Get the correct JOIN syntax for two tables based on relationships.

    Args:
        table1: First table name
        table2: Second table name

    Returns:
        JOIN syntax with explanation
    """
    relationships = {
        ("CLIENTS", "PORTFOLIOS"): "PORTFOLIOS.CLIENT_ID = CLIENTS.CLIENT_ID",
        ("PORTFOLIOS", "CLIENTS"): "PORTFOLIOS.CLIENT_ID = CLIENTS.CLIENT_ID",
        ("PORTFOLIOS", "TRANSACTIONS"): "TRANSACTIONS.PORTFOLIO_ID = PORTFOLIOS.PORTFOLIO_ID",
        ("TRANSACTIONS", "PORTFOLIOS"): "TRANSACTIONS.PORTFOLIO_ID = PORTFOLIOS.PORTFOLIO_ID",
        ("ASSETS", "TRANSACTIONS"): "TRANSACTIONS.ASSET_ID = ASSETS.ASSET_ID",
        ("TRANSACTIONS", "ASSETS"): "TRANSACTIONS.ASSET_ID = ASSETS.ASSET_ID",
        ("PORTFOLIOS", "HOLDINGS"): "HOLDINGS.PORTFOLIO_ID = PORTFOLIOS.PORTFOLIO_ID",
        ("HOLDINGS", "PORTFOLIOS"): "HOLDINGS.PORTFOLIO_ID = PORTFOLIOS.PORTFOLIO_ID",
        ("ASSETS", "HOLDINGS"): "HOLDINGS.ASSET_ID = ASSETS.ASSET_ID",
        ("HOLDINGS", "ASSETS"): "HOLDINGS.ASSET_ID = ASSETS.ASSET_ID"
    }

    table1_upper = table1.upper()
    table2_upper = table2.upper()

    key = (table1_upper, table2_upper)
    if key in relationships:
        condition = relationships[key]
        return json.dumps({
            "join_syntax": f"{table1_upper} JOIN {table2_upper} ON {condition}",
            "condition": condition,
            "example": f"SELECT * FROM {table1_upper} JOIN {table2_upper} ON {condition}"
        }, indent=2)
    else:
        return f"No direct relationship found between {table1} and {table2}. These tables may need to be joined through an intermediate table."


# =============================================================================
# MULTI-AGENT ORCHESTRATOR WRAPPER TOOL
# =============================================================================

@tool
def run_multi_agent_query(query: str) -> str:
    """
    Execute a natural language query using the multi-agent data workflow.
    This tool coordinates multiple specialized agents:
    1. Pre-validation: Determines if query is data-related
    2. Schema Extraction: Gets relevant schema
    3. Query Orchestrator: Determines complexity
    4. SQL Generator: Creates SQL
    5. Validator: Multi-level validation
    6. Executor: Runs SQL and presents results

    Args:
        query: Natural language query about the wealth management data

    Returns:
        Query results with detailed agent trace and explanation
    """
    try:
        from core.multi_agent_orchestrator import MultiAgentDataOrchestrator

        orchestrator = MultiAgentDataOrchestrator()
        result = orchestrator.run(query)

        # Format output with trace information
        output_parts = []

        # Add trace information
        traces = result.get("agent_traces", [])
        if traces:
            output_parts.append(orchestrator.format_trace_for_display(traces))
            output_parts.append("")

        # Add SQL if generated
        generated_sql = result.get("generated_sql", "")
        if generated_sql:
            output_parts.append("GENERATED SQL:")
            output_parts.append("-" * 40)
            output_parts.append(generated_sql)
            output_parts.append("")

        # Add validation info
        overall_confidence = result.get("overall_confidence", 0)
        if overall_confidence > 0:
            output_parts.append(f"VALIDATION CONFIDENCE: {overall_confidence:.0%}")
            output_parts.append("")

        # Add final answer
        final_answer = result.get("final_answer", "")
        if final_answer:
            output_parts.append("FINAL ANSWER:")
            output_parts.append("-" * 40)
            output_parts.append(final_answer)
        elif result.get("general_answer"):
            output_parts.append("ANSWER (General Question):")
            output_parts.append("-" * 40)
            output_parts.append(result.get("general_answer"))
        else:
            output_parts.append("No result generated.")

        # Add error if present
        if result.get("error"):
            output_parts.append("")
            output_parts.append(f"ERROR: {result.get('error')}")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Error executing multi-agent query: {str(e)}"


# =============================================================================
# REGISTER TOOLS WITH REGISTRY
# =============================================================================

# Schema extraction tools (Agent2)
tool_registry.register(get_schema_document, "multi_data_agent")
tool_registry.register(get_table_descriptions, "multi_data_agent")
tool_registry.register(extract_table_schema, "multi_data_agent")
tool_registry.register(extract_related_tables, "multi_data_agent")
tool_registry.register(get_column_sample_values, "multi_data_agent")
tool_registry.register(get_database_relationships, "multi_data_agent")

# Query analysis tools
tool_registry.register(analyze_query_requirements, "multi_data_agent")

# Validation helper tools
tool_registry.register(validate_table_references, "multi_data_agent")
tool_registry.register(get_join_syntax_help, "multi_data_agent")

# Main orchestrator tool
tool_registry.register(run_multi_agent_query, "multi_data_agent")


def get_all_multi_data_agent_tools():
    """Get all multi-data-agent tools."""
    return tool_registry.get_tools_by_category("multi_data_agent")
