"""
Database Initialization Script for DuckDB Wealth Management Database
=====================================================================

This script creates the agent_ddb.db DuckDB database with wealth management
tables and loads sample data from CSV files.
"""

import duckdb
import json
import os
from pathlib import Path

# Database configuration
DB_PATH = "agent_ddb.db"
SCHEMA_FILE = "sample_files/wealth_tables.json"
DATA_DIR = "sample_files/wealth_data"


def get_duckdb_type(json_type: str) -> str:
    """Convert JSON data type to DuckDB data type."""
    type_mapping = {
        "NUMBER(38,0)": "BIGINT",
        "NUMBER(18,6)": "DECIMAL(18,6)",
        "VARCHAR(200)": "VARCHAR(200)",
        "VARCHAR(100)": "VARCHAR(100)",
        "VARCHAR(50)": "VARCHAR(50)",
        "VARCHAR(30)": "VARCHAR(30)",
        "VARCHAR(20)": "VARCHAR(20)",
        "VARCHAR(10)": "VARCHAR(10)",
        "VARCHAR(250)": "VARCHAR(250)",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP",
        "TEXT": "TEXT"
    }
    return type_mapping.get(json_type, json_type)


def create_table_ddl(table_info: dict) -> str:
    """Generate CREATE TABLE DDL from table information."""
    table_name = table_info["table_name"]
    columns = table_info["columns"]

    column_defs = []
    primary_keys = []

    for col in columns:
        col_name = col["column_name"]
        col_type = get_duckdb_type(col["data_type"])

        col_def = f"{col_name} {col_type}"

        if not col["nullable"]:
            col_def += " NOT NULL"

        if col.get("is_primary_key", False):
            primary_keys.append(col_name)

        column_defs.append(col_def)

    # Add primary key constraint
    if primary_keys:
        pk_constraint = f"PRIMARY KEY ({', '.join(primary_keys)})"
        column_defs.append(pk_constraint)

    ddl = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  "
    ddl += ",\n  ".join(column_defs)
    ddl += "\n);"

    return ddl


def load_csv_to_table(conn: duckdb.DuckDBPyConnection, table_name: str, csv_path: str):
    """Load CSV data into a DuckDB table."""
    # DuckDB can directly read CSV files
    query = f"""
    INSERT INTO {table_name}
    SELECT * FROM read_csv_auto('{csv_path}', header=true);
    """
    conn.execute(query)

    # Verify data loaded
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  [OK] Loaded {count} rows into {table_name}")


def initialize_database():
    """Initialize the DuckDB database with schema and data."""
    print("="*70)
    print("DuckDB Wealth Management Database Initialization")
    print("="*70)

    # Remove existing database if it exists
    if os.path.exists(DB_PATH):
        print(f"\n[WARN] Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)

    # Create database connection
    print(f"\n[INFO] Creating database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)

    # Load schema definition
    print(f"\n[INFO] Loading schema from: {SCHEMA_FILE}")
    with open(SCHEMA_FILE, 'r') as f:
        schema = json.load(f)

    # Create tables
    print(f"\n[INFO] Creating tables...")
    tables_created = []
    for table_info in schema["tables"]:
        table_name = table_info["table_name"]
        ddl = create_table_ddl(table_info)

        print(f"\n  Creating table: {table_name}")
        print(f"  Description: {table_info['table_description']}")
        conn.execute(ddl)
        tables_created.append(table_name)
        print(f"  [OK] Table {table_name} created successfully")

    # Load data from CSV files
    print(f"\n[INFO] Loading data from CSV files...")
    csv_files = {
        "CLIENTS": "clients.csv",
        "PORTFOLIOS": "portfolios.csv",
        "ASSETS": "assets.csv",
        "TRANSACTIONS": "transactions.csv",
        "HOLDINGS": "holdings.csv"
    }

    for table_name, csv_file in csv_files.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        if os.path.exists(csv_path):
            print(f"\n  Loading {csv_file} into {table_name}...")
            load_csv_to_table(conn, table_name, csv_path)
        else:
            print(f"  [WARN] CSV file not found: {csv_path}")

    # Display summary
    print(f"\n{'='*70}")
    print("Database Summary")
    print("="*70)

    for table_name in tables_created:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  * {table_name}: {count} records")

    # Display sample queries
    print(f"\n{'='*70}")
    print("Sample Queries")
    print("="*70)

    print("\n1. List all clients:")
    print("   SELECT * FROM CLIENTS LIMIT 5;")

    print("\n2. Portfolio summary:")
    print("   SELECT c.FULL_NAME, p.PORTFOLIO_NAME, p.BASE_CURRENCY, p.STATUS")
    print("   FROM PORTFOLIOS p")
    print("   JOIN CLIENTS c ON p.CLIENT_ID = c.CLIENT_ID;")

    print("\n3. Holdings with asset details:")
    print("   SELECT p.PORTFOLIO_NAME, a.SYMBOL, a.ASSET_NAME, h.QUANTITY, h.AVG_COST")
    print("   FROM HOLDINGS h")
    print("   JOIN PORTFOLIOS p ON h.PORTFOLIO_ID = p.PORTFOLIO_ID")
    print("   JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID;")

    print(f"\n{'='*70}")
    print("[SUCCESS] Database initialization completed successfully!")
    print(f"   Database file: {DB_PATH}")
    print("="*70)

    # Close connection
    conn.close()


if __name__ == "__main__":
    try:
        initialize_database()
    except Exception as e:
        print(f"\n[ERROR] Error during initialization: {str(e)}")
        raise
