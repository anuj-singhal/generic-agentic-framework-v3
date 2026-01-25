"""
Synthetic Data Tools Module for DuckDB Database
================================================

Tools for generating synthetic data using SDV (Synthetic Data Vault) library.
Creates SYNTH_* prefixed tables that maintain referential integrity.
Supports loading table schemas from JSON files and creating tables dynamically.
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import os
import uuid
import json
import glob

from core.tools_base import tool_registry

# Database configuration
DB_PATH = "agent_ddb.db"

# Schema files directory
SCHEMA_DIR = "sample_files/synthetic_data"

# Import DuckDB
try:
    import duckdb
except ImportError:
    raise ImportError("`duckdb` not installed. Please install using `pip install duckdb`.")

# Import pandas (always needed)
try:
    import pandas as pd
except ImportError:
    raise ImportError("`pandas` not installed. Please install using `pip install pandas`.")

# Import SDV (optional - will fail gracefully if not available)
SDV_AVAILABLE = False
SDV_ERROR = None
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError as e:
    SDV_ERROR = f"`sdv` not installed. Please install using `pip install sdv>=1.0.0`. Error: {e}"
except Exception as e:
    SDV_ERROR = f"SDV import failed (possibly due to PyTorch/dependency issues): {e}"

# In-memory session store for tracking generation sessions
_synth_sessions: Dict[str, Dict[str, Any]] = {}

# Cache for loaded schemas
_loaded_schemas: Dict[str, Dict[str, Any]] = {}

# Table relationship definitions (default for wealth management)
TABLE_RELATIONSHIPS = {
    "CLIENTS": {
        "parent_tables": [],
        "child_tables": ["PORTFOLIOS"],
        "primary_key": "CLIENT_ID"
    },
    "PORTFOLIOS": {
        "parent_tables": ["CLIENTS"],
        "child_tables": ["TRANSACTIONS", "HOLDINGS"],
        "primary_key": "PORTFOLIO_ID",
        "foreign_keys": {"CLIENT_ID": "CLIENTS.CLIENT_ID"}
    },
    "ASSETS": {
        "parent_tables": [],
        "child_tables": ["TRANSACTIONS", "HOLDINGS"],
        "primary_key": "ASSET_ID"
    },
    "TRANSACTIONS": {
        "parent_tables": ["PORTFOLIOS", "ASSETS"],
        "child_tables": [],
        "primary_key": "TRANSACTION_ID",
        "foreign_keys": {
            "PORTFOLIO_ID": "PORTFOLIOS.PORTFOLIO_ID",
            "ASSET_ID": "ASSETS.ASSET_ID"
        }
    },
    "HOLDINGS": {
        "parent_tables": ["PORTFOLIOS", "ASSETS"],
        "child_tables": [],
        "primary_key": None,  # Composite key
        "foreign_keys": {
            "PORTFOLIO_ID": "PORTFOLIOS.PORTFOLIO_ID",
            "ASSET_ID": "ASSETS.ASSET_ID"
        }
    }
}


def _load_schema_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a schema JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def _update_relationships_from_schema(schema_data: Dict[str, Any]) -> None:
    """Update TABLE_RELATIONSHIPS from a loaded schema file."""
    global TABLE_RELATIONSHIPS

    if "tables" not in schema_data:
        return

    # Build relationships from schema
    for table in schema_data.get("tables", []):
        table_name = table.get("table_name", "").upper()
        if not table_name:
            continue

        # Remove SYNTH_ prefix if present for relationship mapping
        base_name = table_name.replace("SYNTH_", "")

        # Find primary key
        pk = None
        fks = {}
        parent_tables = []

        for col in table.get("columns", []):
            if col.get("is_primary_key"):
                pk = col.get("column_name")
            if col.get("is_foreign_key") and col.get("references"):
                ref = col.get("references")
                fks[col.get("column_name")] = ref
                # Extract parent table from reference
                parent = ref.split(".")[0].replace("SYNTH_", "")
                if parent not in parent_tables:
                    parent_tables.append(parent)

        # Use parent_dependencies if available
        if "parent_dependencies" in table:
            parent_tables = [p.replace("SYNTH_", "") for p in table.get("parent_dependencies", [])]

        TABLE_RELATIONSHIPS[base_name] = {
            "parent_tables": parent_tables,
            "child_tables": table.get("child_dependents", []),
            "primary_key": pk,
            "foreign_keys": fks
        }

    # Also process relationships section if available
    for rel in schema_data.get("relationships", []):
        parent = rel.get("parent_table", "").replace("SYNTH_", "").upper()
        child = rel.get("child_table", "").replace("SYNTH_", "").upper()

        if parent in TABLE_RELATIONSHIPS:
            if child not in TABLE_RELATIONSHIPS[parent].get("child_tables", []):
                if "child_tables" not in TABLE_RELATIONSHIPS[parent]:
                    TABLE_RELATIONSHIPS[parent]["child_tables"] = []
                TABLE_RELATIONSHIPS[parent]["child_tables"].append(child)


def _map_json_type_to_duckdb(data_type: str) -> str:
    """Map JSON schema data types to DuckDB SQL types."""
    data_type = data_type.upper()

    # Direct mappings
    if "BIGINT" in data_type or "NUMBER(38" in data_type:
        return "BIGINT"
    elif "INT" in data_type:
        return "INTEGER"
    elif "DECIMAL" in data_type or "NUMBER(18" in data_type:
        return "DECIMAL(18,6)"
    elif "FLOAT" in data_type or "DOUBLE" in data_type:
        return "DOUBLE"
    elif "VARCHAR" in data_type or "TEXT" in data_type or "STRING" in data_type:
        return "VARCHAR"
    elif "TIMESTAMP" in data_type:
        return "TIMESTAMP"
    elif "DATE" in data_type:
        return "DATE"
    elif "BOOL" in data_type:
        return "BOOLEAN"
    else:
        return "VARCHAR"


@contextmanager
def get_connection(read_only: bool = True):
    """Get a DuckDB connection with automatic cleanup."""
    conn = None
    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"Database '{DB_PATH}' not found. Run 'python init_duckdb.py' first."
            )
        conn = duckdb.connect(DB_PATH, read_only=read_only)
        yield conn
    except Exception as e:
        raise Exception(f"Database connection error: {str(e)}")
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass


def _get_duckdb_to_sdv_type(duckdb_type: str) -> str:
    """Map DuckDB types to SDV metadata types."""
    duckdb_type = duckdb_type.upper()

    if "INT" in duckdb_type or "NUMBER" in duckdb_type:
        return "numerical"
    elif "FLOAT" in duckdb_type or "DOUBLE" in duckdb_type or "DECIMAL" in duckdb_type:
        return "numerical"
    elif "VARCHAR" in duckdb_type or "TEXT" in duckdb_type or "STRING" in duckdb_type:
        return "categorical"
    elif "DATE" in duckdb_type:
        return "datetime"
    elif "TIMESTAMP" in duckdb_type:
        return "datetime"
    elif "BOOL" in duckdb_type:
        return "boolean"
    else:
        return "categorical"


def _generate_new_ids(table_name: str, count: int, conn) -> List[int]:
    """Generate new unique IDs for synthetic records."""
    pk = TABLE_RELATIONSHIPS.get(table_name, {}).get("primary_key")
    if not pk:
        return list(range(1, count + 1))

    # Get max existing ID
    result = conn.execute(f"SELECT MAX({pk}) FROM {table_name}").fetchone()
    max_id = result[0] if result[0] else 0

    # Check SYNTH table too
    synth_table = f"SYNTH_{table_name}"
    try:
        result = conn.execute(f"SELECT MAX({pk}) FROM {synth_table}").fetchone()
        synth_max = result[0] if result[0] else 0
        max_id = max(max_id, synth_max)
    except:
        pass

    return list(range(max_id + 1, max_id + count + 1))


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool
def check_table_exists(table_name: str) -> str:
    """
    Verify if a source table exists in the database and check if it has data.
    Use this FIRST before attempting to generate synthetic data.

    Args:
        table_name: Name of the table to check (e.g., CLIENTS, PORTFOLIOS)

    Returns:
        Status including: exists/not found, row count, and whether seed data is needed
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            if table_name in tables:
                # Get row count
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

                if count == 0:
                    return json.dumps({
                        "status": "EMPTY",
                        "table": table_name,
                        "exists": True,
                        "row_count": 0,
                        "message": f"Table '{table_name}' exists but is EMPTY. Seed data is required before generating synthetic data.",
                        "action_required": "Use generate_seed_data_prompt() to create seed data, then insert it before generating synthetic data."
                    }, indent=2)
                else:
                    return json.dumps({
                        "status": "OK",
                        "table": table_name,
                        "exists": True,
                        "row_count": count,
                        "message": f"Table '{table_name}' exists with {count} rows. Ready for synthetic data generation."
                    }, indent=2)
            else:
                return json.dumps({
                    "status": "NOT_FOUND",
                    "table": table_name,
                    "exists": False,
                    "row_count": 0,
                    "available_tables": tables,
                    "message": f"Table '{table_name}' not found in database.",
                    "action_required": "Use list_available_schemas() to find schema files, then create_table_from_schema() to create the table."
                }, indent=2)

    except Exception as e:
        return f"Error checking table: {str(e)}"


@tool
def get_table_schema_for_synth(table_name: str) -> str:
    """
    Get the schema of a table with column types for SDV metadata configuration.
    Returns column names, data types, and whether they're primary/foreign keys.

    Args:
        table_name: Name of the table (e.g., CLIENTS, PORTFOLIOS)

    Returns:
        JSON schema information suitable for SDV configuration
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            # Get schema from DESCRIBE
            result = conn.execute(f"DESCRIBE {table_name}").fetchall()

            schema = {
                "table_name": table_name,
                "columns": []
            }

            relationships = TABLE_RELATIONSHIPS.get(table_name, {})
            pk = relationships.get("primary_key")
            fks = relationships.get("foreign_keys", {})

            for row in result:
                col_name = row[0]
                col_type = row[1]

                col_info = {
                    "name": col_name,
                    "duckdb_type": col_type,
                    "sdv_type": _get_duckdb_to_sdv_type(col_type),
                    "is_primary_key": col_name == pk,
                    "is_foreign_key": col_name in fks,
                    "references": fks.get(col_name)
                }
                schema["columns"].append(col_info)

            return json.dumps(schema, indent=2)

    except Exception as e:
        return f"Error getting schema: {str(e)}"


@tool
def get_table_relationships(table_name: str) -> str:
    """
    Get the foreign key relationships for a table.
    Shows parent tables (dependencies) and child tables (dependents).

    Args:
        table_name: Name of the table (e.g., CLIENTS, PORTFOLIOS)

    Returns:
        Relationship information including parent/child tables
    """
    try:
        table_name = table_name.upper().strip()

        if table_name not in TABLE_RELATIONSHIPS:
            return f"Table '{table_name}' not found in relationship definitions."

        rel = TABLE_RELATIONSHIPS[table_name]

        result = {
            "table": table_name,
            "primary_key": rel.get("primary_key"),
            "parent_tables": rel.get("parent_tables", []),
            "child_tables": rel.get("child_tables", []),
            "foreign_keys": rel.get("foreign_keys", {})
        }

        # Add guidance
        if rel.get("parent_tables"):
            result["note"] = f"This table depends on {', '.join(rel['parent_tables'])}. " \
                           f"Ensure SYNTH_{rel['parent_tables'][0]} exists before generating synthetic data."
        else:
            result["note"] = "This table has no parent dependencies. Safe to generate independently."

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error getting relationships: {str(e)}"


@tool
def get_sample_data_for_synth(table_name: str, limit: int = 100) -> str:
    """
    Get sample data from a table for SDV training.
    This data will be used to train the synthetic data generator.

    Args:
        table_name: Name of the source table
        limit: Maximum number of rows to retrieve (default 100)

    Returns:
        Sample data in JSON format suitable for SDV training
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            # Get sample data
            result = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            data = []
            for row in rows:
                record = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    # Convert to JSON-serializable types
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    record[col] = val
                data.append(record)

            return json.dumps({
                "table": table_name,
                "row_count": len(data),
                "columns": columns,
                "sample_data": data
            }, indent=2, default=str)

    except Exception as e:
        return f"Error getting sample data: {str(e)}"


@tool
def analyze_table_dependencies(table_name: str) -> str:
    """
    Analyze the dependency order for generating synthetic data.
    Returns the recommended order to generate tables to maintain referential integrity.

    Args:
        table_name: Target table name

    Returns:
        Recommended generation order with dependency analysis
    """
    try:
        table_name = table_name.upper().strip()

        if table_name not in TABLE_RELATIONSHIPS:
            return f"Table '{table_name}' not found."

        # Build dependency chain
        def get_dependencies(tbl, visited=None):
            if visited is None:
                visited = set()
            if tbl in visited:
                return []
            visited.add(tbl)

            deps = []
            rel = TABLE_RELATIONSHIPS.get(tbl, {})
            for parent in rel.get("parent_tables", []):
                deps.extend(get_dependencies(parent, visited))
                deps.append(parent)
            return deps

        dependencies = get_dependencies(table_name)
        # Remove duplicates while preserving order
        seen = set()
        ordered_deps = []
        for d in dependencies:
            if d not in seen:
                seen.add(d)
                ordered_deps.append(d)

        generation_order = ordered_deps + [table_name]

        result = {
            "target_table": table_name,
            "dependencies": ordered_deps,
            "generation_order": generation_order,
            "steps": []
        }

        for i, tbl in enumerate(generation_order, 1):
            rel = TABLE_RELATIONSHIPS.get(tbl, {})
            step = {
                "step": i,
                "table": tbl,
                "synth_table": f"SYNTH_{tbl}",
                "parent_tables": rel.get("parent_tables", []),
                "requires_fk_values_from": [f"SYNTH_{p}" for p in rel.get("parent_tables", [])]
            }
            result["steps"].append(step)

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error analyzing dependencies: {str(e)}"


@tool
def create_synth_table(table_name: str) -> str:
    """
    Create a SYNTH_* prefixed table with the same structure as the source table.
    The table will be empty and ready to receive synthetic data.

    Args:
        table_name: Source table name (e.g., CLIENTS creates SYNTH_CLIENTS)

    Returns:
        Confirmation of table creation
    """
    try:
        table_name = table_name.upper().strip()
        synth_table = f"SYNTH_{table_name}"

        with get_connection(read_only=False) as conn:
            # Check if source table exists
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            if table_name not in tables:
                return f"Error: Source table '{table_name}' not found."

            # Drop if exists and recreate
            conn.execute(f"DROP TABLE IF EXISTS {synth_table}")

            # Create table with same structure
            conn.execute(f"CREATE TABLE {synth_table} AS SELECT * FROM {table_name} WHERE 1=0")

            return f"[OK] Created table '{synth_table}' with same structure as '{table_name}'."

    except Exception as e:
        return f"Error creating synth table: {str(e)}"


@tool
def generate_synthetic_data(table_name: str, num_rows: int = 10, session_id: Optional[str] = None) -> str:
    """
    Generate synthetic data using SDV's GaussianCopulaSynthesizer.
    The generated data will be stored in the session for later insertion.

    Args:
        table_name: Source table to base synthetic data on
        num_rows: Number of synthetic rows to generate (default 10)
        session_id: Optional session ID for tracking (auto-generated if not provided)

    Returns:
        Session ID and preview of generated data
    """
    try:
        # Check SDV availability first
        if not SDV_AVAILABLE:
            return f"Error: SDV library not available. {SDV_ERROR}"

        table_name = table_name.upper().strip()

        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        with get_connection() as conn:
            # Get real data for training
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()

            if len(df) == 0:
                return f"Error: Source table '{table_name}' is empty."

            # Create SDV metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)

            # Set primary key if exists
            rel = TABLE_RELATIONSHIPS.get(table_name, {})
            pk = rel.get("primary_key")
            if pk and pk in df.columns:
                metadata.update_column(pk, sdtype='id')

            # Create and fit synthesizer
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(df)

            # Generate synthetic data
            synthetic_df = synthesizer.sample(num_rows=num_rows)

            # Generate new unique IDs
            if pk and pk in synthetic_df.columns:
                new_ids = _generate_new_ids(table_name, num_rows, conn)
                synthetic_df[pk] = new_ids

            # Store in session
            _synth_sessions[session_id] = {
                "table_name": table_name,
                "synth_table": f"SYNTH_{table_name}",
                "num_rows": num_rows,
                "data": synthetic_df.to_dict(orient='records'),
                "columns": list(synthetic_df.columns),
                "status": "generated"
            }

            # Preview
            preview = synthetic_df.head(3).to_dict(orient='records')

            return json.dumps({
                "session_id": session_id,
                "table_name": table_name,
                "rows_generated": num_rows,
                "preview": preview,
                "status": "Data generated and stored in session. Use insert_synthetic_data() to save to database."
            }, indent=2, default=str)

    except Exception as e:
        return f"Error generating synthetic data: {str(e)}"


@tool
def insert_synthetic_data(session_id: str) -> str:
    """
    Insert generated synthetic data from a session into the SYNTH_* table.
    The target table must already exist (use create_synth_table first).

    Args:
        session_id: Session ID from generate_synthetic_data

    Returns:
        Confirmation of data insertion
    """
    try:
        if session_id not in _synth_sessions:
            available = list(_synth_sessions.keys()) if _synth_sessions else "none"
            return f"Error: Session '{session_id}' not found. Available sessions: {available}"

        session = _synth_sessions[session_id]
        synth_table = session["synth_table"]
        data = session["data"]
        columns = session["columns"]

        if not data:
            return "Error: No data in session to insert."

        with get_connection(read_only=False) as conn:
            # Check if synth table exists
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            if synth_table not in tables:
                return f"Error: Table '{synth_table}' does not exist. Use create_synth_table() first."

            # Convert to DataFrame and insert
            df = pd.DataFrame(data)

            # Insert data
            conn.execute(f"INSERT INTO {synth_table} SELECT * FROM df")

            # Update session status
            session["status"] = "inserted"

            # Verify
            count = conn.execute(f"SELECT COUNT(*) FROM {synth_table}").fetchone()[0]

            return f"[OK] Inserted {len(data)} rows into '{synth_table}'. Table now has {count} total rows."

    except Exception as e:
        return f"Error inserting data: {str(e)}"


@tool
def list_synth_tables() -> str:
    """
    List all SYNTH_* tables in the database with row counts.

    Returns:
        List of synthetic tables and their row counts
    """
    try:
        with get_connection() as conn:
            result = conn.execute("SHOW TABLES").fetchall()

            synth_tables = []
            for row in result:
                table = row[0]
                if table.upper().startswith("SYNTH_"):
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    synth_tables.append({
                        "table": table,
                        "row_count": count,
                        "source_table": table.replace("SYNTH_", "")
                    })

            if not synth_tables:
                return "No SYNTH_* tables found in the database."

            return json.dumps({
                "synth_tables": synth_tables,
                "total_tables": len(synth_tables)
            }, indent=2)

    except Exception as e:
        return f"Error listing tables: {str(e)}"


@tool
def get_generation_summary(session_id: str) -> str:
    """
    Get a summary of a synthetic data generation session.

    Args:
        session_id: Session ID from generate_synthetic_data

    Returns:
        Complete summary of the generation session
    """
    try:
        if session_id not in _synth_sessions:
            available = list(_synth_sessions.keys()) if _synth_sessions else "none"
            return f"Session '{session_id}' not found. Available sessions: {available}"

        session = _synth_sessions[session_id]

        summary = {
            "session_id": session_id,
            "source_table": session["table_name"],
            "target_table": session["synth_table"],
            "rows_generated": session["num_rows"],
            "columns": session["columns"],
            "status": session["status"]
        }

        # If data exists, add sample
        if session.get("data"):
            summary["sample_records"] = session["data"][:2]

        # Add database state
        with get_connection() as conn:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {session['synth_table']}").fetchone()[0]
                summary["current_synth_table_rows"] = count
            except:
                summary["current_synth_table_rows"] = "Table not created yet"

        return json.dumps(summary, indent=2, default=str)

    except Exception as e:
        return f"Error getting summary: {str(e)}"


# =============================================================================
# SEED DATA GENERATION TOOLS (for empty tables)
# =============================================================================

@tool
def generate_seed_data_prompt(table_name: str, num_rows: int = 5) -> str:
    """
    Generate a structured prompt for the LLM to create seed data for an empty table.
    Use this when a table exists but has no data (row_count = 0).

    The LLM should use this prompt to generate realistic sample data that can be
    inserted into the table using insert_seed_data().

    Args:
        table_name: Name of the empty table
        num_rows: Number of seed rows to generate (default 5, max 10)

    Returns:
        A structured prompt with schema information for the LLM to generate seed data
    """
    try:
        table_name = table_name.upper().strip()
        num_rows = min(max(num_rows, 1), 10)  # Clamp between 1 and 10

        with get_connection() as conn:
            # Check table exists
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            if table_name not in tables:
                return f"Error: Table '{table_name}' does not exist. Create it first using create_table_from_schema()."

            # Get schema
            schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()

            columns = []
            for row in schema_result:
                col_name = row[0]
                col_type = row[1]
                nullable = "NULL" if row[3] == "YES" else "NOT NULL"
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": nullable
                })

        # Get relationship info
        rel = TABLE_RELATIONSHIPS.get(table_name, {})
        pk = rel.get("primary_key")
        fks = rel.get("foreign_keys", {})

        # Try to get sample values from schema files
        sample_values = {}
        for filename, schema_data in _loaded_schemas.items():
            for t in schema_data.get("tables", []):
                if t.get("table_name", "").upper().replace("SYNTH_", "") == table_name:
                    for col in t.get("columns", []):
                        if col.get("sample_values"):
                            sample_values[col.get("column_name")] = col.get("sample_values")

        # Build the prompt
        prompt = {
            "task": "GENERATE_SEED_DATA",
            "table_name": table_name,
            "num_rows": num_rows,
            "schema": columns,
            "primary_key": pk,
            "foreign_keys": fks,
            "sample_values": sample_values,
            "instructions": f"""
Generate {num_rows} realistic seed data records for the {table_name} table.

REQUIREMENTS:
1. Return data as a JSON array of objects
2. Each object must have ALL columns listed in the schema
3. Primary key ({pk}) values should be unique integers starting from 1
4. Data types must match the schema (use ISO format for dates: YYYY-MM-DD)
5. Make the data realistic and varied (different values, not all the same)
6. For foreign keys, use values that would exist in parent tables

SCHEMA:
{json.dumps(columns, indent=2)}

{f"SAMPLE VALUES FOR REFERENCE: {json.dumps(sample_values, indent=2)}" if sample_values else ""}

RESPONSE FORMAT:
Return ONLY a valid JSON array like this:
[
  {{"column1": "value1", "column2": "value2", ...}},
  {{"column1": "value3", "column2": "value4", ...}}
]

Do NOT include any explanation, just the JSON array.
""",
            "response_format": "JSON array of objects with all columns"
        }

        return json.dumps(prompt, indent=2)

    except Exception as e:
        return f"Error generating seed data prompt: {str(e)}"


@tool
def insert_seed_data(table_name: str, seed_data_json: str) -> str:
    """
    Insert LLM-generated seed data into an empty table.
    Use this after generating seed data from the LLM based on generate_seed_data_prompt().

    Args:
        table_name: Name of the table to insert data into
        seed_data_json: JSON array string containing the seed data records

    Returns:
        Confirmation of data insertion with row count
    """
    try:
        table_name = table_name.upper().strip()

        # Parse the JSON data
        try:
            # Clean up the JSON string (remove markdown code blocks if present)
            clean_json = seed_data_json.strip()
            if clean_json.startswith("```"):
                # Remove markdown code block
                lines = clean_json.split("\n")
                clean_json = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            seed_data = json.loads(clean_json)

            if not isinstance(seed_data, list):
                return "Error: seed_data_json must be a JSON array of objects"

            if len(seed_data) == 0:
                return "Error: seed_data_json array is empty"

        except json.JSONDecodeError as e:
            return f"Error parsing seed_data_json: {str(e)}. Make sure it's a valid JSON array."

        with get_connection(read_only=False) as conn:
            # Verify table exists
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            if table_name not in tables:
                return f"Error: Table '{table_name}' does not exist."

            # Get current row count
            current_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Convert to DataFrame
            df = pd.DataFrame(seed_data)

            # Get table columns to ensure correct order
            schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            table_columns = [row[0] for row in schema_result]

            # Reorder DataFrame columns to match table
            missing_cols = set(table_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(table_columns)

            if missing_cols:
                return f"Error: Seed data is missing columns: {missing_cols}"

            if extra_cols:
                # Remove extra columns
                df = df[[c for c in df.columns if c in table_columns]]

            # Reorder to match table schema
            df = df[table_columns]

            # Insert data
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")

            # Verify
            new_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            inserted = new_count - current_count

            return json.dumps({
                "status": "success",
                "table": table_name,
                "rows_inserted": inserted,
                "total_rows": new_count,
                "message": f"Successfully inserted {inserted} seed data rows into '{table_name}'. Table now has {new_count} rows.",
                "next_step": "You can now use generate_synthetic_data() to create more synthetic records based on this seed data."
            }, indent=2)

    except Exception as e:
        return f"Error inserting seed data: {str(e)}"


@tool
def get_table_data_status(table_name: str) -> str:
    """
    Get detailed status of a table including whether it needs seed data.
    This is a comprehensive check that determines the next action needed.

    Args:
        table_name: Name of the table to check

    Returns:
        Detailed status with recommended next action
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            # Check if table exists
            if table_name not in tables:
                # Check if it's in schema files
                schema_file = None
                for filename, schema_data in _loaded_schemas.items():
                    for t in schema_data.get("tables", []):
                        if t.get("table_name", "").upper().replace("SYNTH_", "") == table_name:
                            schema_file = filename
                            break

                return json.dumps({
                    "table": table_name,
                    "status": "NOT_FOUND",
                    "exists": False,
                    "has_data": False,
                    "row_count": 0,
                    "schema_file_available": schema_file,
                    "workflow": "CREATE_AND_SEED",
                    "steps": [
                        f"1. Load schema: load_schema_from_file('{schema_file}')" if schema_file else "1. Find schema file: list_available_schemas()",
                        f"2. Create table: create_table_from_schema('{schema_file}', '{table_name}')" if schema_file else "2. Create table from schema",
                        f"3. Generate seed prompt: generate_seed_data_prompt('{table_name}', 5)",
                        "4. Use LLM to generate seed data from prompt",
                        f"5. Insert seed data: insert_seed_data('{table_name}', '<json_data>')",
                        f"6. Generate synthetic: generate_synthetic_data('{table_name}', <num_rows>)"
                    ]
                }, indent=2)

            # Table exists - check row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            if count == 0:
                return json.dumps({
                    "table": table_name,
                    "status": "EMPTY",
                    "exists": True,
                    "has_data": False,
                    "row_count": 0,
                    "workflow": "SEED_THEN_GENERATE",
                    "steps": [
                        f"1. Generate seed prompt: generate_seed_data_prompt('{table_name}', 5)",
                        "2. Use LLM to generate seed data from the prompt",
                        f"3. Insert seed data: insert_seed_data('{table_name}', '<json_data>')",
                        f"4. Generate synthetic: generate_synthetic_data('{table_name}', <num_rows>)"
                    ]
                }, indent=2)
            else:
                return json.dumps({
                    "table": table_name,
                    "status": "READY",
                    "exists": True,
                    "has_data": True,
                    "row_count": count,
                    "workflow": "GENERATE_DIRECTLY",
                    "steps": [
                        f"1. Create synth table: create_synth_table('{table_name}')",
                        f"2. Generate synthetic: generate_synthetic_data('{table_name}', <num_rows>)",
                        "3. Insert: insert_synthetic_data('<session_id>')"
                    ]
                }, indent=2)

    except Exception as e:
        return f"Error checking table status: {str(e)}"


# =============================================================================
# SCHEMA FILE TOOLS
# =============================================================================

@tool
def list_available_schemas() -> str:
    """
    List all available schema JSON files in the synthetic_data directory.
    These schemas can be used to create new tables in the database.

    Returns:
        List of available schema files with their table definitions
    """
    try:
        schema_files = []

        # Find all JSON files in the schema directory
        pattern = os.path.join(SCHEMA_DIR, "*.json")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                tables = []
                if "tables" in data:
                    for t in data["tables"]:
                        table_name = t.get("table_name", "Unknown")
                        col_count = len(t.get("columns", []))
                        tables.append(f"{table_name} ({col_count} columns)")

                schema_files.append({
                    "filename": filename,
                    "path": file_path,
                    "tables": tables,
                    "table_count": len(tables),
                    "description": data.get("metadata", {}).get("description", "No description")
                })
            except Exception as e:
                schema_files.append({
                    "filename": filename,
                    "path": file_path,
                    "error": str(e)
                })

        if not schema_files:
            return f"No schema files found in '{SCHEMA_DIR}'. Create JSON schema files to define new tables."

        return json.dumps({
            "schema_directory": SCHEMA_DIR,
            "schema_files": schema_files,
            "total_files": len(schema_files)
        }, indent=2)

    except Exception as e:
        return f"Error listing schemas: {str(e)}"


@tool
def load_schema_from_file(filename: str) -> str:
    """
    Load a schema definition from a JSON file.
    This loads the table definitions and relationships into memory.

    Args:
        filename: Name of the JSON file (e.g., 'financial_tables.json')

    Returns:
        Schema details including tables and relationships
    """
    try:
        # Build full path
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        file_path = os.path.join(SCHEMA_DIR, filename)

        if not os.path.exists(file_path):
            return f"Error: Schema file '{filename}' not found in '{SCHEMA_DIR}'"

        # Load schema
        schema_data = _load_schema_file(file_path)

        # Cache it
        _loaded_schemas[filename] = schema_data

        # Update relationships
        _update_relationships_from_schema(schema_data)

        # Prepare summary
        tables_info = []
        for t in schema_data.get("tables", []):
            table_name = t.get("table_name")
            tables_info.append({
                "table_name": table_name,
                "description": t.get("table_description", ""),
                "columns": len(t.get("columns", [])),
                "generation_order": t.get("generation_order", 0),
                "parent_dependencies": t.get("parent_dependencies", [])
            })

        return json.dumps({
            "filename": filename,
            "status": "loaded",
            "tables": tables_info,
            "relationships": len(schema_data.get("relationships", [])),
            "note": "Schema loaded. Use create_table_from_schema() to create tables in the database."
        }, indent=2)

    except Exception as e:
        return f"Error loading schema: {str(e)}"


@tool
def get_schema_table_definition(filename: str, table_name: str) -> str:
    """
    Get the detailed definition of a specific table from a schema file.

    Args:
        filename: Schema JSON file name
        table_name: Name of the table to get definition for

    Returns:
        Complete table definition including columns and relationships
    """
    try:
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        # Check if already loaded
        if filename not in _loaded_schemas:
            file_path = os.path.join(SCHEMA_DIR, filename)
            if not os.path.exists(file_path):
                return f"Error: Schema file '{filename}' not found"
            _loaded_schemas[filename] = _load_schema_file(file_path)

        schema_data = _loaded_schemas[filename]
        table_name = table_name.upper()

        # Find the table
        for t in schema_data.get("tables", []):
            if t.get("table_name", "").upper() == table_name:
                return json.dumps(t, indent=2)

        available = [t.get("table_name") for t in schema_data.get("tables", [])]
        return f"Table '{table_name}' not found in schema. Available: {', '.join(available)}"

    except Exception as e:
        return f"Error getting table definition: {str(e)}"


@tool
def create_table_from_schema(filename: str, table_name: str) -> str:
    """
    Create a new table in the database using a schema definition from a JSON file.
    Use this when a table doesn't exist and you need to create it before generating data.

    Args:
        filename: Schema JSON file name (e.g., 'financial_tables.json')
        table_name: Name of the table to create (will be created without SYNTH_ prefix)

    Returns:
        Confirmation of table creation or error message
    """
    try:
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        # Load schema if not cached
        if filename not in _loaded_schemas:
            file_path = os.path.join(SCHEMA_DIR, filename)
            if not os.path.exists(file_path):
                return f"Error: Schema file '{filename}' not found"
            _loaded_schemas[filename] = _load_schema_file(file_path)
            _update_relationships_from_schema(_loaded_schemas[filename])

        schema_data = _loaded_schemas[filename]
        table_name = table_name.upper()

        # Find the table definition
        table_def = None
        for t in schema_data.get("tables", []):
            t_name = t.get("table_name", "").upper()
            # Match with or without SYNTH_ prefix
            if t_name == table_name or t_name == f"SYNTH_{table_name}" or t_name.replace("SYNTH_", "") == table_name:
                table_def = t
                break

        if not table_def:
            available = [t.get("table_name") for t in schema_data.get("tables", [])]
            return f"Table '{table_name}' not found in schema. Available: {', '.join(available)}"

        # Build CREATE TABLE statement
        columns = []
        for col in table_def.get("columns", []):
            col_name = col.get("column_name")
            data_type = _map_json_type_to_duckdb(col.get("data_type", "VARCHAR"))
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            columns.append(f"{col_name} {data_type} {nullable}")

        create_sql = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n)"

        with get_connection(read_only=False) as conn:
            # Drop if exists
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            # Create table
            conn.execute(create_sql)

            return json.dumps({
                "status": "success",
                "table_name": table_name,
                "columns_created": len(columns),
                "sql_executed": create_sql,
                "note": f"Table '{table_name}' created. You can now generate synthetic data for it."
            }, indent=2)

    except Exception as e:
        return f"Error creating table: {str(e)}"


@tool
def create_tables_with_dependencies(filename: str, target_table: str) -> str:
    """
    Create a table and all its parent dependencies from a schema file.
    Analyzes the dependency chain and creates tables in the correct order.

    Args:
        filename: Schema JSON file name
        target_table: The table you want to create (dependencies will be created first)

    Returns:
        Summary of all tables created
    """
    try:
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        # Load schema
        if filename not in _loaded_schemas:
            file_path = os.path.join(SCHEMA_DIR, filename)
            if not os.path.exists(file_path):
                return f"Error: Schema file '{filename}' not found"
            _loaded_schemas[filename] = _load_schema_file(file_path)
            _update_relationships_from_schema(_loaded_schemas[filename])

        schema_data = _loaded_schemas[filename]
        target_table = target_table.upper()

        # Build table lookup
        tables_by_name = {}
        for t in schema_data.get("tables", []):
            name = t.get("table_name", "").upper().replace("SYNTH_", "")
            tables_by_name[name] = t

        if target_table.replace("SYNTH_", "") not in tables_by_name:
            available = list(tables_by_name.keys())
            return f"Table '{target_table}' not found. Available: {', '.join(available)}"

        # Get generation order from schema or calculate
        def get_creation_order(tbl_name, visited=None):
            if visited is None:
                visited = set()
            if tbl_name in visited:
                return []
            visited.add(tbl_name)

            order = []
            tbl_def = tables_by_name.get(tbl_name.replace("SYNTH_", ""))
            if tbl_def:
                for parent in tbl_def.get("parent_dependencies", []):
                    parent_name = parent.replace("SYNTH_", "")
                    order.extend(get_creation_order(parent_name, visited))
                    if parent_name not in order:
                        order.append(parent_name)
            return order

        creation_order = get_creation_order(target_table)
        creation_order.append(target_table.replace("SYNTH_", ""))

        # Remove duplicates preserving order
        seen = set()
        unique_order = []
        for t in creation_order:
            if t not in seen:
                seen.add(t)
                unique_order.append(t)

        # Create tables
        created = []
        with get_connection(read_only=False) as conn:
            for tbl_name in unique_order:
                tbl_def = tables_by_name.get(tbl_name)
                if not tbl_def:
                    continue

                # Build CREATE TABLE
                columns = []
                for col in tbl_def.get("columns", []):
                    col_name = col.get("column_name")
                    data_type = _map_json_type_to_duckdb(col.get("data_type", "VARCHAR"))
                    nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                    columns.append(f"{col_name} {data_type} {nullable}")

                create_sql = f"CREATE TABLE {tbl_name} (\n    " + ",\n    ".join(columns) + "\n)"

                conn.execute(f"DROP TABLE IF EXISTS {tbl_name}")
                conn.execute(create_sql)
                created.append({
                    "table": tbl_name,
                    "columns": len(columns)
                })

        return json.dumps({
            "status": "success",
            "creation_order": unique_order,
            "tables_created": created,
            "total_tables": len(created),
            "note": "All tables created. You can now generate synthetic data starting from parent tables."
        }, indent=2)

    except Exception as e:
        return f"Error creating tables: {str(e)}"


@tool
def drop_table(table_name: str, confirm: bool = False) -> str:
    """
    Drop (delete) an existing table from the database.
    Use with caution - this permanently removes the table and all its data.

    Args:
        table_name: Name of the table to drop
        confirm: Set to True to confirm deletion (safety check)

    Returns:
        Confirmation of table deletion or error message
    """
    try:
        table_name = table_name.upper().strip()

        if not confirm:
            return json.dumps({
                "warning": f"This will permanently delete table '{table_name}' and all its data.",
                "action_required": "Call drop_table again with confirm=True to proceed.",
                "example": f'drop_table("{table_name}", confirm=True)'
            }, indent=2)

        with get_connection(read_only=False) as conn:
            # Check if table exists
            result = conn.execute("SHOW TABLES").fetchall()
            tables = [row[0].upper() for row in result]

            if table_name not in tables:
                return f"Table '{table_name}' does not exist. Available tables: {', '.join(tables)}"

            # Get row count before dropping
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Drop the table
            conn.execute(f"DROP TABLE {table_name}")

            return json.dumps({
                "status": "success",
                "table_dropped": table_name,
                "rows_deleted": count,
                "note": f"Table '{table_name}' has been permanently deleted."
            }, indent=2)

    except Exception as e:
        return f"Error dropping table: {str(e)}"


@tool
def drop_all_synth_tables(confirm: bool = False) -> str:
    """
    Drop all SYNTH_* prefixed tables from the database.
    Use with caution - this permanently removes all synthetic tables.

    Args:
        confirm: Set to True to confirm deletion (safety check)

    Returns:
        Summary of dropped tables
    """
    try:
        if not confirm:
            with get_connection() as conn:
                result = conn.execute("SHOW TABLES").fetchall()
                synth_tables = [row[0] for row in result if row[0].upper().startswith("SYNTH_")]

            if not synth_tables:
                return "No SYNTH_* tables found in the database."

            return json.dumps({
                "warning": "This will permanently delete ALL synthetic tables:",
                "tables_to_drop": synth_tables,
                "count": len(synth_tables),
                "action_required": "Call drop_all_synth_tables(confirm=True) to proceed."
            }, indent=2)

        dropped = []
        with get_connection(read_only=False) as conn:
            result = conn.execute("SHOW TABLES").fetchall()

            for row in result:
                table = row[0]
                if table.upper().startswith("SYNTH_"):
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    conn.execute(f"DROP TABLE {table}")
                    dropped.append({"table": table, "rows_deleted": count})

        if not dropped:
            return "No SYNTH_* tables found to drop."

        return json.dumps({
            "status": "success",
            "tables_dropped": dropped,
            "total_dropped": len(dropped),
            "note": "All synthetic tables have been removed."
        }, indent=2)

    except Exception as e:
        return f"Error dropping tables: {str(e)}"


# =============================================================================
# REGISTER TOOLS WITH REGISTRY
# =============================================================================

# Core tools
tool_registry.register(check_table_exists, "synthetic_data")
tool_registry.register(get_table_schema_for_synth, "synthetic_data")
tool_registry.register(get_table_relationships, "synthetic_data")
tool_registry.register(get_sample_data_for_synth, "synthetic_data")
tool_registry.register(analyze_table_dependencies, "synthetic_data")
tool_registry.register(create_synth_table, "synthetic_data")
tool_registry.register(generate_synthetic_data, "synthetic_data")
tool_registry.register(insert_synthetic_data, "synthetic_data")
tool_registry.register(list_synth_tables, "synthetic_data")
tool_registry.register(get_generation_summary, "synthetic_data")

# Seed data tools (for empty tables)
tool_registry.register(generate_seed_data_prompt, "synthetic_data")
tool_registry.register(insert_seed_data, "synthetic_data")
tool_registry.register(get_table_data_status, "synthetic_data")

# Schema file tools
tool_registry.register(list_available_schemas, "synthetic_data")
tool_registry.register(load_schema_from_file, "synthetic_data")
tool_registry.register(get_schema_table_definition, "synthetic_data")
tool_registry.register(create_table_from_schema, "synthetic_data")
tool_registry.register(create_tables_with_dependencies, "synthetic_data")

# Table management tools
tool_registry.register(drop_table, "synthetic_data")
tool_registry.register(drop_all_synth_tables, "synthetic_data")


def get_synthetic_data_tools():
    """Get all synthetic data tools."""
    return tool_registry.get_tools_by_category("synthetic_data")
