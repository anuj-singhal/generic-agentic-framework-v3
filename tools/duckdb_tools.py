"""
DuckDB Tools Module for Wealth Management Database
===================================================

Simple, direct tools that return actual data for SQL query generation and execution.
"""

from langchain_core.tools import tool
from typing import Optional
import os

from core.tools_base import tool_registry

# Database configuration
DB_PATH = "agent_ddb.db"

# Import DuckDB
try:
    import duckdb
except ImportError:
    raise ImportError("`duckdb` not installed. Please install using `pip install duckdb`.")


# Global connection - reuse for performance
_db_connection = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get or create a DuckDB connection."""
    global _db_connection
    try:
        if _db_connection is None:
            if not os.path.exists(DB_PATH):
                raise FileNotFoundError(
                    f"Database '{DB_PATH}' not found. Run 'python init_duckdb.py' first."
                )
            _db_connection = duckdb.connect(DB_PATH, read_only=False)
        # Test connection
        _db_connection.execute("SELECT 1")
        return _db_connection
    except Exception as e:
        # Reset connection on error
        _db_connection = None
        raise Exception(f"Database connection error: {str(e)}")


def format_results(result) -> str:
    """Format query results as CSV."""
    try:
        rows = result.fetchall()
        if not rows:
            return "Query executed successfully. No results returned."

        columns = result.description
        column_names = [col[0] for col in columns] if columns else []

        # Format as CSV-like output
        output_lines = [",".join(column_names)]
        for row in rows:
            output_lines.append(",".join(str(val) if val is not None else "NULL" for val in row))

        return "\n".join(output_lines)
    except AttributeError:
        return str(result)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool
def get_database_schema() -> str:
    """
    Get the complete database schema with sample data.
    Use this FIRST to understand what tables and columns are available.

    Returns:
        Complete schema with table structures and sample data
    """
    try:
        conn = get_connection()
        all_tables = ["CLIENTS", "PORTFOLIOS", "ASSETS", "TRANSACTIONS", "HOLDINGS"]

        schema_info = []
        schema_info.append("WEALTH MANAGEMENT DATABASE SCHEMA")
        schema_info.append("="*70)
        schema_info.append("\nTABLES OVERVIEW:")
        schema_info.append("- CLIENTS: Client profiles (10 records)")
        schema_info.append("- PORTFOLIOS: Investment accounts (15 records)")
        schema_info.append("- ASSETS: Tradable instruments (35 records)")
        schema_info.append("- TRANSACTIONS: Trade history (1,200 records)")
        schema_info.append("- HOLDINGS: Current positions (305 records)")

        schema_info.append("\nRELATIONSHIPS:")
        schema_info.append("- PORTFOLIOS.CLIENT_ID -> CLIENTS.CLIENT_ID")
        schema_info.append("- TRANSACTIONS.PORTFOLIO_ID -> PORTFOLIOS.PORTFOLIO_ID")
        schema_info.append("- TRANSACTIONS.ASSET_ID -> ASSETS.ASSET_ID")
        schema_info.append("- HOLDINGS.PORTFOLIO_ID -> PORTFOLIOS.PORTFOLIO_ID")
        schema_info.append("- HOLDINGS.ASSET_ID -> ASSETS.ASSET_ID")

        for table in all_tables:
            schema_info.append(f"\n{'='*70}")
            schema_info.append(f"TABLE: {table}")
            schema_info.append("="*70)

            # Get schema
            desc_result = conn.execute(f"DESCRIBE {table};")
            schema_info.append("\nCOLUMNS:")
            schema_info.append(format_results(desc_result))

            # Get sample data
            sample_result = conn.execute(f"SELECT * FROM {table} LIMIT 3;")
            schema_info.append("\nSAMPLE DATA (3 rows):")
            schema_info.append(format_results(sample_result))

        return "\n".join(schema_info)

    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


@tool
def run_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query and return the results.
    The query will be validated and executed against the DuckDB database.

    Args:
        sql_query: SQL query to execute (SELECT, WITH, etc.)

    Returns:
        Query results in CSV format, or error message if query fails
    """
    try:
        conn = get_connection()

        # Clean query - remove markdown formatting and backticks
        clean_sql = sql_query.strip()
        clean_sql = clean_sql.replace("```sql", "").replace("```", "").replace("`", "").strip()

        # Take only the first statement if multiple
        if ";" in clean_sql:
            clean_sql = clean_sql.split(";")[0].strip()

        if not clean_sql:
            return "Error: Empty SQL query provided"

        # Execute query
        result = conn.execute(clean_sql)

        if result is None:
            return "Query executed successfully. No output."

        return format_results(result)

    except duckdb.ProgrammingError as e:
        return f"SQL Syntax Error: {str(e)}\n\nPlease check your SQL and try again."
    except duckdb.Error as e:
        return f"Database Error: {str(e)}\n\nCheck that table and column names are correct."
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def show_tables() -> str:
    """
    List all tables in the database.

    Returns:
        List of table names
    """
    try:
        conn = get_connection()
        result = conn.execute("SHOW TABLES;")
        return format_results(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def describe_table(table_name: str) -> str:
    """
    Get the structure/schema of a specific table.

    Args:
        table_name: Name of the table (CLIENTS, PORTFOLIOS, ASSETS, TRANSACTIONS, or HOLDINGS)

    Returns:
        Table schema showing column names, types, and constraints
    """
    try:
        conn = get_connection()
        result = conn.execute(f"DESCRIBE {table_name};")
        return format_results(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_sample_data(table_name: str, limit: int = 5) -> str:
    """
    Get sample rows from a table to see what the data looks like.

    Args:
        table_name: Name of the table
        limit: Number of sample rows (default 5)

    Returns:
        Sample data from the table
    """
    try:
        conn = get_connection()
        result = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        return format_results(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def validate_sql(sql_query: str) -> str:
    """
    Validate SQL syntax without executing the query.
    Use this to check if your SQL is correct before running it.

    Args:
        sql_query: SQL query to validate

    Returns:
        "VALID" if the query is syntactically correct, or error message if not
    """
    try:
        conn = get_connection()

        # Clean query
        clean_sql = sql_query.strip().replace("```sql", "").replace("```", "").replace("`", "").strip()

        if ";" in clean_sql:
            clean_sql = clean_sql.split(";")[0].strip()

        if not clean_sql:
            return "ERROR: Empty query"

        # Use EXPLAIN to validate without executing
        conn.execute(f"EXPLAIN {clean_sql}")
        return "VALID: SQL syntax is correct and all tables/columns exist."

    except duckdb.ProgrammingError as e:
        return f"INVALID: Syntax Error - {str(e)}"
    except duckdb.Error as e:
        return f"INVALID: Database Error - {str(e)}"
    except Exception as e:
        return f"INVALID: {str(e)}"


@tool
def get_table_stats(table_name: str) -> str:
    """
    Get statistics about a table (row count, column stats, etc.).

    Args:
        table_name: Name of the table

    Returns:
        Statistical summary of the table
    """
    try:
        conn = get_connection()

        # Get row count
        count_result = conn.execute(f"SELECT COUNT(*) as row_count FROM {table_name};")
        stats = [format_results(count_result)]

        # Get summary stats
        try:
            summary_result = conn.execute(f"SUMMARIZE {table_name};")
            stats.append("\nDETAILED STATISTICS:")
            stats.append(format_results(summary_result))
        except:
            pass  # SUMMARIZE might not work on all table types

        return "\n".join(stats)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# REGISTER TOOLS WITH REGISTRY
# =============================================================================

tool_registry.register(get_database_schema, "duckdb")
tool_registry.register(run_sql_query, "duckdb")
tool_registry.register(show_tables, "duckdb")
tool_registry.register(describe_table, "duckdb")
tool_registry.register(get_sample_data, "duckdb")
tool_registry.register(validate_sql, "duckdb")
tool_registry.register(get_table_stats, "duckdb")


def get_all_duckdb_tools():
    """Get all DuckDB tools."""
    return tool_registry.get_tools_by_category("duckdb")
