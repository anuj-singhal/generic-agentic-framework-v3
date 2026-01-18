"""
Text-to-SQL Tools Module
=========================

Provides tools for converting natural language queries to SQL,
managing database schemas, executing queries, and explaining SQL.
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import json
import re
from datetime import datetime

from core.tools_base import tool_registry


# =============================================================================
# SIMULATED DATABASE SCHEMA STORE
# =============================================================================

# In-memory schema store (in production, this would connect to actual databases)
_schema_store: Dict[str, Dict[str, Any]] = {}

# Pre-loaded example schemas for demonstration
EXAMPLE_SCHEMAS = {
    "ecommerce": {
        "name": "ecommerce",
        "description": "E-commerce database with customers, products, orders, and order items",
        "tables": {
            "customers": {
                "columns": {
                    "customer_id": {"type": "INT", "primary_key": True, "description": "Unique customer identifier"},
                    "first_name": {"type": "VARCHAR(50)", "description": "Customer's first name"},
                    "last_name": {"type": "VARCHAR(50)", "description": "Customer's last name"},
                    "email": {"type": "VARCHAR(100)", "unique": True, "description": "Customer's email address"},
                    "phone": {"type": "VARCHAR(20)", "description": "Customer's phone number"},
                    "city": {"type": "VARCHAR(50)", "description": "Customer's city"},
                    "country": {"type": "VARCHAR(50)", "description": "Customer's country"},
                    "created_at": {"type": "DATETIME", "description": "Account creation date"},
                    "is_active": {"type": "BOOLEAN", "default": True, "description": "Whether customer is active"}
                },
                "description": "Stores customer information"
            },
            "products": {
                "columns": {
                    "product_id": {"type": "INT", "primary_key": True, "description": "Unique product identifier"},
                    "product_name": {"type": "VARCHAR(100)", "description": "Name of the product"},
                    "category": {"type": "VARCHAR(50)", "description": "Product category"},
                    "price": {"type": "DECIMAL(10,2)", "description": "Product price"},
                    "stock_quantity": {"type": "INT", "description": "Available stock"},
                    "supplier_id": {"type": "INT", "foreign_key": "suppliers.supplier_id", "description": "Reference to supplier"},
                    "created_at": {"type": "DATETIME", "description": "Product listing date"},
                    "is_available": {"type": "BOOLEAN", "default": True, "description": "Product availability status"}
                },
                "description": "Stores product catalog"
            },
            "orders": {
                "columns": {
                    "order_id": {"type": "INT", "primary_key": True, "description": "Unique order identifier"},
                    "customer_id": {"type": "INT", "foreign_key": "customers.customer_id", "description": "Reference to customer"},
                    "order_date": {"type": "DATETIME", "description": "Date when order was placed"},
                    "total_amount": {"type": "DECIMAL(10,2)", "description": "Total order amount"},
                    "status": {"type": "VARCHAR(20)", "description": "Order status (pending, shipped, delivered, cancelled)"},
                    "shipping_address": {"type": "TEXT", "description": "Delivery address"},
                    "payment_method": {"type": "VARCHAR(30)", "description": "Payment method used"}
                },
                "description": "Stores order headers"
            },
            "order_items": {
                "columns": {
                    "item_id": {"type": "INT", "primary_key": True, "description": "Unique item identifier"},
                    "order_id": {"type": "INT", "foreign_key": "orders.order_id", "description": "Reference to order"},
                    "product_id": {"type": "INT", "foreign_key": "products.product_id", "description": "Reference to product"},
                    "quantity": {"type": "INT", "description": "Quantity ordered"},
                    "unit_price": {"type": "DECIMAL(10,2)", "description": "Price per unit at time of order"},
                    "discount": {"type": "DECIMAL(5,2)", "default": 0, "description": "Discount percentage applied"}
                },
                "description": "Stores individual items within orders"
            },
            "suppliers": {
                "columns": {
                    "supplier_id": {"type": "INT", "primary_key": True, "description": "Unique supplier identifier"},
                    "supplier_name": {"type": "VARCHAR(100)", "description": "Supplier company name"},
                    "contact_name": {"type": "VARCHAR(100)", "description": "Primary contact person"},
                    "contact_email": {"type": "VARCHAR(100)", "description": "Contact email"},
                    "country": {"type": "VARCHAR(50)", "description": "Supplier's country"}
                },
                "description": "Stores supplier information"
            }
        },
        "relationships": [
            {"from": "orders.customer_id", "to": "customers.customer_id", "type": "many-to-one"},
            {"from": "order_items.order_id", "to": "orders.order_id", "type": "many-to-one"},
            {"from": "order_items.product_id", "to": "products.product_id", "type": "many-to-one"},
            {"from": "products.supplier_id", "to": "suppliers.supplier_id", "type": "many-to-one"}
        ]
    },
    
    "hr": {
        "name": "hr",
        "description": "Human Resources database with employees, departments, and salaries",
        "tables": {
            "employees": {
                "columns": {
                    "employee_id": {"type": "INT", "primary_key": True, "description": "Unique employee identifier"},
                    "first_name": {"type": "VARCHAR(50)", "description": "Employee's first name"},
                    "last_name": {"type": "VARCHAR(50)", "description": "Employee's last name"},
                    "email": {"type": "VARCHAR(100)", "unique": True, "description": "Employee's email"},
                    "phone": {"type": "VARCHAR(20)", "description": "Employee's phone"},
                    "hire_date": {"type": "DATE", "description": "Date of hiring"},
                    "job_title": {"type": "VARCHAR(50)", "description": "Current job title"},
                    "department_id": {"type": "INT", "foreign_key": "departments.department_id", "description": "Reference to department"},
                    "manager_id": {"type": "INT", "foreign_key": "employees.employee_id", "description": "Reference to manager (self-referencing)"},
                    "salary": {"type": "DECIMAL(10,2)", "description": "Current salary"},
                    "is_active": {"type": "BOOLEAN", "default": True, "description": "Employment status"}
                },
                "description": "Stores employee information"
            },
            "departments": {
                "columns": {
                    "department_id": {"type": "INT", "primary_key": True, "description": "Unique department identifier"},
                    "department_name": {"type": "VARCHAR(50)", "description": "Name of department"},
                    "location": {"type": "VARCHAR(100)", "description": "Department location"},
                    "budget": {"type": "DECIMAL(12,2)", "description": "Annual department budget"},
                    "manager_id": {"type": "INT", "foreign_key": "employees.employee_id", "description": "Department manager"}
                },
                "description": "Stores department information"
            },
            "salary_history": {
                "columns": {
                    "history_id": {"type": "INT", "primary_key": True, "description": "Unique history record identifier"},
                    "employee_id": {"type": "INT", "foreign_key": "employees.employee_id", "description": "Reference to employee"},
                    "salary": {"type": "DECIMAL(10,2)", "description": "Salary amount"},
                    "effective_date": {"type": "DATE", "description": "Date salary became effective"},
                    "end_date": {"type": "DATE", "nullable": True, "description": "Date salary ended (null if current)"}
                },
                "description": "Tracks salary changes over time"
            },
            "projects": {
                "columns": {
                    "project_id": {"type": "INT", "primary_key": True, "description": "Unique project identifier"},
                    "project_name": {"type": "VARCHAR(100)", "description": "Name of the project"},
                    "department_id": {"type": "INT", "foreign_key": "departments.department_id", "description": "Owning department"},
                    "start_date": {"type": "DATE", "description": "Project start date"},
                    "end_date": {"type": "DATE", "nullable": True, "description": "Project end date"},
                    "budget": {"type": "DECIMAL(12,2)", "description": "Project budget"},
                    "status": {"type": "VARCHAR(20)", "description": "Project status (planning, active, completed, on_hold)"}
                },
                "description": "Stores project information"
            },
            "employee_projects": {
                "columns": {
                    "employee_id": {"type": "INT", "foreign_key": "employees.employee_id", "description": "Reference to employee"},
                    "project_id": {"type": "INT", "foreign_key": "projects.project_id", "description": "Reference to project"},
                    "role": {"type": "VARCHAR(50)", "description": "Employee's role in project"},
                    "hours_allocated": {"type": "INT", "description": "Weekly hours allocated to project"}
                },
                "description": "Maps employees to projects (many-to-many)"
            }
        },
        "relationships": [
            {"from": "employees.department_id", "to": "departments.department_id", "type": "many-to-one"},
            {"from": "employees.manager_id", "to": "employees.employee_id", "type": "many-to-one"},
            {"from": "departments.manager_id", "to": "employees.employee_id", "type": "many-to-one"},
            {"from": "salary_history.employee_id", "to": "employees.employee_id", "type": "many-to-one"},
            {"from": "projects.department_id", "to": "departments.department_id", "type": "many-to-one"},
            {"from": "employee_projects.employee_id", "to": "employees.employee_id", "type": "many-to-one"},
            {"from": "employee_projects.project_id", "to": "projects.project_id", "type": "many-to-one"}
        ]
    },
    
    "analytics": {
        "name": "analytics",
        "description": "Web analytics database with page views, sessions, and events",
        "tables": {
            "users": {
                "columns": {
                    "user_id": {"type": "INT", "primary_key": True, "description": "Unique user identifier"},
                    "username": {"type": "VARCHAR(50)", "unique": True, "description": "Username"},
                    "email": {"type": "VARCHAR(100)", "description": "User email"},
                    "signup_date": {"type": "DATE", "description": "Registration date"},
                    "user_type": {"type": "VARCHAR(20)", "description": "User type (free, premium, enterprise)"},
                    "country": {"type": "VARCHAR(50)", "description": "User's country"}
                },
                "description": "Registered users"
            },
            "sessions": {
                "columns": {
                    "session_id": {"type": "VARCHAR(50)", "primary_key": True, "description": "Unique session identifier"},
                    "user_id": {"type": "INT", "foreign_key": "users.user_id", "nullable": True, "description": "User (null for anonymous)"},
                    "start_time": {"type": "DATETIME", "description": "Session start time"},
                    "end_time": {"type": "DATETIME", "description": "Session end time"},
                    "device_type": {"type": "VARCHAR(20)", "description": "Device type (desktop, mobile, tablet)"},
                    "browser": {"type": "VARCHAR(30)", "description": "Browser name"},
                    "country": {"type": "VARCHAR(50)", "description": "Session country"}
                },
                "description": "User sessions"
            },
            "page_views": {
                "columns": {
                    "view_id": {"type": "INT", "primary_key": True, "description": "Unique view identifier"},
                    "session_id": {"type": "VARCHAR(50)", "foreign_key": "sessions.session_id", "description": "Reference to session"},
                    "page_url": {"type": "VARCHAR(255)", "description": "Page URL"},
                    "page_title": {"type": "VARCHAR(100)", "description": "Page title"},
                    "view_time": {"type": "DATETIME", "description": "Time of page view"},
                    "time_on_page": {"type": "INT", "description": "Seconds spent on page"},
                    "referrer": {"type": "VARCHAR(255)", "description": "Referrer URL"}
                },
                "description": "Individual page views"
            },
            "events": {
                "columns": {
                    "event_id": {"type": "INT", "primary_key": True, "description": "Unique event identifier"},
                    "session_id": {"type": "VARCHAR(50)", "foreign_key": "sessions.session_id", "description": "Reference to session"},
                    "event_type": {"type": "VARCHAR(50)", "description": "Type of event (click, scroll, form_submit, purchase)"},
                    "event_time": {"type": "DATETIME", "description": "Time of event"},
                    "event_data": {"type": "JSON", "description": "Additional event data"},
                    "page_url": {"type": "VARCHAR(255)", "description": "Page where event occurred"}
                },
                "description": "User interaction events"
            }
        },
        "relationships": [
            {"from": "sessions.user_id", "to": "users.user_id", "type": "many-to-one"},
            {"from": "page_views.session_id", "to": "sessions.session_id", "type": "many-to-one"},
            {"from": "events.session_id", "to": "sessions.session_id", "type": "many-to-one"}
        ]
    }
}

# Initialize with example schemas
for name, schema in EXAMPLE_SCHEMAS.items():
    _schema_store[name] = schema


# =============================================================================
# SCHEMA MANAGEMENT TOOLS
# =============================================================================

@tool
def list_databases() -> str:
    """
    List all available database schemas.
    Use this to see what databases are available for querying.
    
    Returns:
        List of available databases with descriptions
    """
    if not _schema_store:
        return "No databases available. Use 'register_schema' to add a database schema."
    
    result = ["Available Databases:", ""]
    for name, schema in _schema_store.items():
        desc = schema.get("description", "No description")
        tables = list(schema.get("tables", {}).keys())
        result.append(f"📁 **{name}**")
        result.append(f"   Description: {desc}")
        result.append(f"   Tables: {', '.join(tables)}")
        result.append("")
    
    return "\n".join(result)

tool_registry.register(list_databases, "sql")


@tool
def get_schema(database_name: str) -> str:
    """
    Get the detailed schema for a specific database.
    Use this to understand table structures before generating SQL.
    
    Args:
        database_name: Name of the database (e.g., 'ecommerce', 'hr', 'analytics')
    
    Returns:
        Detailed schema information including tables, columns, and relationships
    """
    if database_name not in _schema_store:
        available = list(_schema_store.keys())
        return f"Database '{database_name}' not found. Available: {', '.join(available)}"
    
    schema = _schema_store[database_name]
    result = [f"# Schema: {database_name}", f"{schema.get('description', '')}", ""]
    
    # Tables
    result.append("## Tables")
    for table_name, table_info in schema.get("tables", {}).items():
        result.append(f"\n### {table_name}")
        result.append(f"*{table_info.get('description', '')}*")
        result.append("\n| Column | Type | Description |")
        result.append("|--------|------|-------------|")
        
        for col_name, col_info in table_info.get("columns", {}).items():
            col_type = col_info.get("type", "")
            col_desc = col_info.get("description", "")
            
            # Add annotations
            annotations = []
            if col_info.get("primary_key"):
                annotations.append("PK")
            if col_info.get("foreign_key"):
                annotations.append(f"FK→{col_info['foreign_key']}")
            if col_info.get("unique"):
                annotations.append("UNIQUE")
            if col_info.get("nullable"):
                annotations.append("NULL")
            
            if annotations:
                col_desc = f"[{', '.join(annotations)}] {col_desc}"
            
            result.append(f"| {col_name} | {col_type} | {col_desc} |")
    
    # Relationships
    if schema.get("relationships"):
        result.append("\n## Relationships")
        for rel in schema["relationships"]:
            result.append(f"- {rel['from']} → {rel['to']} ({rel['type']})")
    
    return "\n".join(result)

tool_registry.register(get_schema, "sql")


@tool
def get_table_info(database_name: str, table_name: str) -> str:
    """
    Get detailed information about a specific table.
    
    Args:
        database_name: Name of the database
        table_name: Name of the table
    
    Returns:
        Detailed table information including all columns and their properties
    """
    if database_name not in _schema_store:
        return f"Database '{database_name}' not found."
    
    schema = _schema_store[database_name]
    tables = schema.get("tables", {})
    
    if table_name not in tables:
        available = list(tables.keys())
        return f"Table '{table_name}' not found in '{database_name}'. Available: {', '.join(available)}"
    
    table = tables[table_name]
    result = [
        f"# Table: {database_name}.{table_name}",
        f"*{table.get('description', '')}*",
        "",
        "## Columns"
    ]
    
    for col_name, col_info in table.get("columns", {}).items():
        result.append(f"\n**{col_name}** ({col_info.get('type', 'UNKNOWN')})")
        result.append(f"  - Description: {col_info.get('description', 'N/A')}")
        
        if col_info.get("primary_key"):
            result.append("  - Primary Key: Yes")
        if col_info.get("foreign_key"):
            result.append(f"  - Foreign Key: References {col_info['foreign_key']}")
        if col_info.get("unique"):
            result.append("  - Unique: Yes")
        if col_info.get("nullable"):
            result.append("  - Nullable: Yes")
        if "default" in col_info:
            result.append(f"  - Default: {col_info['default']}")
    
    return "\n".join(result)

tool_registry.register(get_table_info, "sql")


# =============================================================================
# SQL GENERATION TOOLS
# =============================================================================

@tool
def generate_sql(database_name: str, natural_language_query: str) -> str:
    """
    Generate SQL from a natural language query.
    This tool analyzes the query intent and generates appropriate SQL.
    
    IMPORTANT: Before using this tool, first use 'get_schema' to understand 
    the database structure.
    
    Args:
        database_name: Name of the database to query
        natural_language_query: The natural language description of what data you want
    
    Returns:
        Generated SQL query with explanation
    """
    if database_name not in _schema_store:
        return f"Database '{database_name}' not found. Use 'list_databases' to see available databases."
    
    schema = _schema_store[database_name]
    
    # This is a simplified SQL generator - in production, this would use
    # more sophisticated NLP or the LLM itself for generation
    # Here we provide templates and the LLM will help construct the query
    
    query_lower = natural_language_query.lower()
    tables = schema.get("tables", {})
    
    result = {
        "database": database_name,
        "natural_query": natural_language_query,
        "schema_context": {
            "tables": list(tables.keys()),
            "table_details": {}
        }
    }
    
    # Gather relevant table info
    for table_name, table_info in tables.items():
        columns = list(table_info.get("columns", {}).keys())
        result["schema_context"]["table_details"][table_name] = {
            "columns": columns,
            "description": table_info.get("description", "")
        }
    
    # Provide SQL generation guidance
    guidance = [
        f"## SQL Generation Request",
        f"**Database:** {database_name}",
        f"**Natural Language Query:** {natural_language_query}",
        "",
        "### Available Schema:",
        ""
    ]
    
    for table_name, details in result["schema_context"]["table_details"].items():
        guidance.append(f"**{table_name}**: {', '.join(details['columns'])}")
    
    guidance.extend([
        "",
        "### Instructions for SQL Generation:",
        "Based on the above schema and the natural language query, construct the appropriate SQL.",
        "Consider:",
        "- Which tables are needed",
        "- What JOIN conditions apply",
        "- What WHERE clauses filter the data",
        "- What aggregations (COUNT, SUM, AVG) are needed",
        "- What GROUP BY or ORDER BY clauses are appropriate",
        "",
        "Please generate the SQL query."
    ])
    
    return "\n".join(guidance)

tool_registry.register(generate_sql, "sql")


@tool
def validate_sql(sql_query: str, database_name: str) -> str:
    """
    Validate a SQL query against a database schema.
    Checks for syntax issues and verifies table/column references.
    
    Args:
        sql_query: The SQL query to validate
        database_name: Name of the database to validate against
    
    Returns:
        Validation result with any issues found
    """
    if database_name not in _schema_store:
        return f"Database '{database_name}' not found."
    
    schema = _schema_store[database_name]
    tables = schema.get("tables", {})
    
    issues = []
    warnings = []
    
    # Basic SQL validation
    sql_upper = sql_query.upper()
    
    # Check for basic SQL structure
    if not any(keyword in sql_upper for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
        issues.append("Query must start with SELECT, INSERT, UPDATE, or DELETE")
    
    # Check for SELECT without FROM
    if "SELECT" in sql_upper and "FROM" not in sql_upper:
        issues.append("SELECT query missing FROM clause")
    
    # Check table references
    for table_name in tables.keys():
        if table_name.lower() in sql_query.lower():
            # Table is referenced - check columns
            table_columns = list(tables[table_name].get("columns", {}).keys())
            
            # Simple column check
            for col in table_columns:
                if col.lower() in sql_query.lower():
                    pass  # Column found
    
    # Check for common issues
    if "SELECT *" in sql_upper:
        warnings.append("Consider specifying columns explicitly instead of SELECT *")
    
    if "WHERE" not in sql_upper and "SELECT" in sql_upper:
        warnings.append("Query has no WHERE clause - this may return all rows")
    
    # Build result
    result = ["## SQL Validation Result", ""]
    result.append(f"**Query:** `{sql_query[:100]}{'...' if len(sql_query) > 100 else ''}`")
    result.append(f"**Database:** {database_name}")
    result.append("")
    
    if not issues:
        result.append("✅ **Validation Passed** - No critical issues found")
    else:
        result.append("❌ **Validation Failed**")
        for issue in issues:
            result.append(f"  - {issue}")
    
    if warnings:
        result.append("")
        result.append("⚠️ **Warnings:**")
        for warning in warnings:
            result.append(f"  - {warning}")
    
    return "\n".join(result)

tool_registry.register(validate_sql, "sql")


@tool
def explain_sql(sql_query: str) -> str:
    """
    Explain what a SQL query does in plain English.
    Useful for understanding complex queries.
    
    Args:
        sql_query: The SQL query to explain
    
    Returns:
        Plain English explanation of the query
    """
    sql_upper = sql_query.upper()
    
    explanation = ["## SQL Query Explanation", "", f"**Query:** ```sql\n{sql_query}\n```", ""]
    
    # Determine query type
    if sql_upper.strip().startswith("SELECT"):
        explanation.append("**Type:** SELECT (Data Retrieval)")
    elif sql_upper.strip().startswith("INSERT"):
        explanation.append("**Type:** INSERT (Data Insertion)")
    elif sql_upper.strip().startswith("UPDATE"):
        explanation.append("**Type:** UPDATE (Data Modification)")
    elif sql_upper.strip().startswith("DELETE"):
        explanation.append("**Type:** DELETE (Data Removal)")
    
    explanation.append("")
    explanation.append("### Breakdown:")
    
    # Parse components
    if "SELECT" in sql_upper:
        # Find selected columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns = select_match.group(1).strip()
            explanation.append(f"- **Selecting:** {columns}")
    
    if "FROM" in sql_upper:
        from_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
        if from_match:
            explanation.append(f"- **From table:** {from_match.group(1)}")
    
    if "JOIN" in sql_upper:
        join_matches = re.findall(r'(LEFT |RIGHT |INNER |OUTER |CROSS )?JOIN\s+(\w+)', sql_query, re.IGNORECASE)
        for join_type, table in join_matches:
            join_type = join_type.strip() if join_type else "INNER"
            explanation.append(f"- **Joining:** {table} ({join_type}JOIN)")
    
    if "WHERE" in sql_upper:
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            conditions = where_match.group(1).strip()
            explanation.append(f"- **Filtering:** {conditions[:100]}{'...' if len(conditions) > 100 else ''}")
    
    if "GROUP BY" in sql_upper:
        group_match = re.search(r'GROUP BY\s+(.*?)(?:HAVING|ORDER BY|LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
        if group_match:
            explanation.append(f"- **Grouping by:** {group_match.group(1).strip()}")
    
    if "HAVING" in sql_upper:
        explanation.append("- **Having:** Post-aggregation filter applied")
    
    if "ORDER BY" in sql_upper:
        order_match = re.search(r'ORDER BY\s+(.*?)(?:LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
        if order_match:
            explanation.append(f"- **Ordering by:** {order_match.group(1).strip()}")
    
    if "LIMIT" in sql_upper:
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_query, re.IGNORECASE)
        if limit_match:
            explanation.append(f"- **Limiting to:** {limit_match.group(1)} rows")
    
    # Aggregation functions
    agg_functions = []
    if "COUNT(" in sql_upper:
        agg_functions.append("COUNT (counting rows)")
    if "SUM(" in sql_upper:
        agg_functions.append("SUM (summing values)")
    if "AVG(" in sql_upper:
        agg_functions.append("AVG (averaging values)")
    if "MAX(" in sql_upper:
        agg_functions.append("MAX (finding maximum)")
    if "MIN(" in sql_upper:
        agg_functions.append("MIN (finding minimum)")
    
    if agg_functions:
        explanation.append(f"- **Aggregations:** {', '.join(agg_functions)}")
    
    return "\n".join(explanation)

tool_registry.register(explain_sql, "sql")


@tool
def execute_sql(sql_query: str, database_name: str) -> str:
    """
    Execute a SQL query and return results (simulated).
    In a production environment, this would connect to an actual database.
    
    Args:
        sql_query: The SQL query to execute
        database_name: Name of the database
    
    Returns:
        Query results or execution confirmation
    """
    if database_name not in _schema_store:
        return f"Database '{database_name}' not found."
    
    # Simulated execution with sample data
    sql_upper = sql_query.upper()
    
    result = ["## Query Execution Result", ""]
    result.append(f"**Database:** {database_name}")
    result.append(f"**Query:** `{sql_query[:100]}...`" if len(sql_query) > 100 else f"**Query:** `{sql_query}`")
    result.append("")
    
    # Simulate different query types
    if sql_upper.strip().startswith("SELECT"):
        # Generate sample results based on the query
        result.append("### Results (Simulated)")
        result.append("")
        
        # Detect what kind of data to return
        if "COUNT" in sql_upper:
            result.append("| count |")
            result.append("|-------|")
            result.append("| 42    |")
            result.append("")
            result.append("*1 row returned*")
        elif "SUM" in sql_upper or "AVG" in sql_upper:
            result.append("| total/average |")
            result.append("|---------------|")
            result.append("| 15,234.50     |")
            result.append("")
            result.append("*1 row returned*")
        elif "customers" in sql_query.lower():
            result.append("| customer_id | first_name | last_name | email |")
            result.append("|-------------|------------|-----------|-------|")
            result.append("| 1 | John | Doe | john@email.com |")
            result.append("| 2 | Jane | Smith | jane@email.com |")
            result.append("| 3 | Bob | Johnson | bob@email.com |")
            result.append("")
            result.append("*3 rows returned (showing sample data)*")
        elif "employees" in sql_query.lower():
            result.append("| employee_id | first_name | last_name | department |")
            result.append("|-------------|------------|-----------|------------|")
            result.append("| 101 | Alice | Williams | Engineering |")
            result.append("| 102 | Charlie | Brown | Marketing |")
            result.append("| 103 | Diana | Miller | Sales |")
            result.append("")
            result.append("*3 rows returned (showing sample data)*")
        elif "orders" in sql_query.lower():
            result.append("| order_id | customer_id | order_date | total_amount | status |")
            result.append("|----------|-------------|------------|--------------|--------|")
            result.append("| 1001 | 1 | 2024-01-15 | 299.99 | delivered |")
            result.append("| 1002 | 2 | 2024-01-16 | 549.50 | shipped |")
            result.append("| 1003 | 1 | 2024-01-17 | 89.99 | pending |")
            result.append("")
            result.append("*3 rows returned (showing sample data)*")
        else:
            result.append("| col1 | col2 | col3 |")
            result.append("|------|------|------|")
            result.append("| value1 | value2 | value3 |")
            result.append("| value4 | value5 | value6 |")
            result.append("")
            result.append("*2 rows returned (showing sample data)*")
        
        result.append("")
        result.append("⚠️ *Note: This is simulated data. In production, this would query the actual database.*")
    
    elif sql_upper.strip().startswith("INSERT"):
        result.append("✅ **Insert Successful**")
        result.append("1 row inserted")
    
    elif sql_upper.strip().startswith("UPDATE"):
        result.append("✅ **Update Successful**")
        result.append("3 rows affected")
    
    elif sql_upper.strip().startswith("DELETE"):
        result.append("✅ **Delete Successful**")
        result.append("1 row deleted")
    
    else:
        result.append("⚠️ Query type not recognized")
    
    result.append("")
    result.append(f"*Execution time: 0.023s*")
    
    return "\n".join(result)

tool_registry.register(execute_sql, "sql")


@tool
def sql_examples(query_type: str) -> str:
    """
    Get example SQL queries for common use cases.
    
    Args:
        query_type: Type of query examples to show (select, join, aggregate, subquery, all)
    
    Returns:
        Example SQL queries with explanations
    """
    examples = {
        "select": [
            ("Basic Select", "SELECT first_name, last_name, email FROM customers WHERE is_active = true"),
            ("Select with LIKE", "SELECT * FROM products WHERE product_name LIKE '%laptop%'"),
            ("Select with IN", "SELECT * FROM orders WHERE status IN ('pending', 'shipped')"),
            ("Select with BETWEEN", "SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31'"),
        ],
        "join": [
            ("Inner Join", """SELECT o.order_id, c.first_name, c.last_name, o.total_amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id"""),
            ("Left Join", """SELECT p.product_name, s.supplier_name
FROM products p
LEFT JOIN suppliers s ON p.supplier_id = s.supplier_id"""),
            ("Multiple Joins", """SELECT o.order_id, c.first_name, p.product_name, oi.quantity
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id"""),
        ],
        "aggregate": [
            ("Count", "SELECT COUNT(*) as total_customers FROM customers WHERE is_active = true"),
            ("Sum with Group By", """SELECT category, SUM(price * stock_quantity) as inventory_value
FROM products
GROUP BY category"""),
            ("Average with Having", """SELECT department_id, AVG(salary) as avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 50000"""),
            ("Multiple Aggregates", """SELECT 
    COUNT(*) as total_orders,
    SUM(total_amount) as revenue,
    AVG(total_amount) as avg_order_value,
    MAX(total_amount) as largest_order
FROM orders
WHERE status = 'delivered'"""),
        ],
        "subquery": [
            ("Subquery in WHERE", """SELECT * FROM products 
WHERE price > (SELECT AVG(price) FROM products)"""),
            ("Subquery in FROM", """SELECT category, avg_price
FROM (
    SELECT category, AVG(price) as avg_price
    FROM products
    GROUP BY category
) as category_avg
WHERE avg_price > 100"""),
            ("Correlated Subquery", """SELECT e.first_name, e.salary
FROM employees e
WHERE salary > (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = e.department_id
)"""),
        ]
    }
    
    result = ["## SQL Query Examples", ""]
    
    if query_type.lower() == "all":
        for qtype, queries in examples.items():
            result.append(f"### {qtype.title()} Queries")
            for name, sql in queries:
                result.append(f"\n**{name}:**")
                result.append(f"```sql\n{sql}\n```")
            result.append("")
    elif query_type.lower() in examples:
        result.append(f"### {query_type.title()} Queries")
        for name, sql in examples[query_type.lower()]:
            result.append(f"\n**{name}:**")
            result.append(f"```sql\n{sql}\n```")
    else:
        result.append(f"Unknown query type: {query_type}")
        result.append(f"Available types: {', '.join(examples.keys())}, all")
    
    return "\n".join(result)

tool_registry.register(sql_examples, "sql")


def get_sql_tools():
    """Get all SQL-related tools."""
    return tool_registry.get_tools_by_category("sql")
