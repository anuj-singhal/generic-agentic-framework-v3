"""
Data Visualization Tools Module
===============================

Comprehensive data visualization tools for creating dashboards from DuckDB data.
Uses Plotly for interactive charts and generates HTML dashboards.
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import os
import json
import uuid
from datetime import datetime

from core.tools_base import tool_registry

# Database configuration
DB_PATH = "agent_ddb.db"
SCHEMA_DIR = "sample_files/synthetic_data"
OUTPUT_DIR = "sample_files/dashboards"

# Import DuckDB
try:
    import duckdb
except ImportError:
    raise ImportError("`duckdb` not installed. Please install using `pip install duckdb`.")

# Import pandas and numpy
try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError("`pandas` and `numpy` not installed.")

# Import Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# In-memory session store for visualization sessions
_viz_sessions: Dict[str, Dict[str, Any]] = {}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def _create_session(table_name: str) -> str:
    """Create a new visualization session."""
    session_id = str(uuid.uuid4())[:8]
    _viz_sessions[session_id] = {
        "table_name": table_name,
        "created_at": datetime.now().isoformat(),
        "visualizations": [],
        "kpis": [],
        "data_cache": {},
        "viz_plan": None,
        "dashboard_title": f"{table_name} Dashboard",
        "dashboard_generated": False
    }
    return session_id


def _safe_json_serialize(obj):
    """Safely serialize objects to JSON-compatible types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


def _get_color_palette():
    """Get a professional color palette for visualizations."""
    return [
        "#2E86AB",  # Blue
        "#A23B72",  # Magenta
        "#F18F01",  # Orange
        "#C73E1D",  # Red
        "#3B1F2B",  # Dark purple
        "#95C623",  # Green
        "#5D5179",  # Purple
        "#F4D35E",  # Yellow
        "#0A2463",  # Navy
        "#FB3640",  # Coral
    ]


# =============================================================================
# DISCOVERY AND SCHEMA TOOLS
# =============================================================================

@tool
def list_tables_for_viz() -> str:
    """
    List all available tables in the database for visualization.
    Shows table names, row counts, and column counts.

    Returns:
        List of tables with metadata for visualization planning
    """
    try:
        with get_connection() as conn:
            result = conn.execute("SHOW TABLES").fetchall()

            tables_info = []
            for row in result:
                table_name = row[0]
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                cols = conn.execute(f"DESCRIBE {table_name}").fetchall()

                # Classify columns
                numeric_cols = []
                categorical_cols = []
                datetime_cols = []

                for col in cols:
                    col_name = col[0]
                    col_type = col[1].upper()
                    if any(t in col_type for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"]):
                        numeric_cols.append(col_name)
                    elif any(t in col_type for t in ["DATE", "TIME", "TIMESTAMP"]):
                        datetime_cols.append(col_name)
                    else:
                        categorical_cols.append(col_name)

                tables_info.append({
                    "table_name": table_name,
                    "row_count": count,
                    "column_count": len(cols),
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols),
                    "datetime_columns": len(datetime_cols),
                    "viz_potential": "high" if count > 10 and len(numeric_cols) > 0 else "medium"
                })

            tables_info.sort(key=lambda x: x["row_count"], reverse=True)

            return json.dumps({
                "total_tables": len(tables_info),
                "tables": tables_info,
                "note": "Select a table to create visualizations using get_table_schema_for_viz()"
            }, indent=2)

    except Exception as e:
        return f"Error listing tables: {str(e)}"


@tool
def get_table_schema_for_viz(table_name: str) -> str:
    """
    Get detailed schema information for visualization planning.
    Includes column types, sample values, and visualization suggestions.

    Args:
        table_name: Name of the table to analyze

    Returns:
        Schema details with visualization recommendations per column
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            # Check table exists
            tables = [r[0].upper() for r in conn.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                return f"Error: Table '{table_name}' not found."

            # Get schema
            schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            columns = []
            for row in schema_result:
                col_name = row[0]
                col_type = row[1].upper()

                # Get sample values and statistics
                sample_query = f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 10"
                samples = [r[0] for r in conn.execute(sample_query).fetchall()]

                # Determine visualization types
                viz_suggestions = []
                if any(t in col_type for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"]):
                    viz_suggestions = ["histogram", "box_plot", "kpi_card", "line_chart", "scatter_plot"]
                    col_class = "numeric"
                elif any(t in col_type for t in ["DATE", "TIME", "TIMESTAMP"]):
                    viz_suggestions = ["line_chart", "area_chart", "timeline"]
                    col_class = "datetime"
                else:
                    unique_count = conn.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}").fetchone()[0]
                    if unique_count <= 15:
                        viz_suggestions = ["bar_chart", "pie_chart", "donut_chart"]
                    else:
                        viz_suggestions = ["bar_chart", "word_cloud", "treemap"]
                    col_class = "categorical"

                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "classification": col_class,
                    "sample_values": [_safe_json_serialize(s) for s in samples[:5]],
                    "suggested_visualizations": viz_suggestions
                })

            return json.dumps({
                "table_name": table_name,
                "row_count": row_count,
                "columns": columns,
                "recommended_dashboard_components": {
                    "kpis": [c["name"] for c in columns if c["classification"] == "numeric"][:4],
                    "charts": [
                        {"type": "bar_chart", "columns": [c["name"] for c in columns if c["classification"] == "categorical"][:2]},
                        {"type": "line_chart", "columns": [c["name"] for c in columns if c["classification"] == "datetime"]},
                        {"type": "pie_chart", "columns": [c["name"] for c in columns if c["classification"] == "categorical"][:1]}
                    ]
                }
            }, indent=2, default=str)

    except Exception as e:
        return f"Error getting schema: {str(e)}"


@tool
def load_schema_relationships(schema_file: Optional[str] = None) -> str:
    """
    Load table relationships from JSON schema files.
    This helps understand how tables relate for cross-table visualizations.

    Args:
        schema_file: Optional specific schema file to load (default: load all)

    Returns:
        Table relationships and foreign key information
    """
    try:
        relationships = []
        tables_info = {}

        # Find schema files
        if schema_file:
            files = [os.path.join(SCHEMA_DIR, schema_file)]
        else:
            files = [os.path.join(SCHEMA_DIR, f) for f in os.listdir(SCHEMA_DIR) if f.endswith('.json')]

        for file_path in files:
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)

            # Extract relationships
            for rel in data.get("relationships", []):
                relationships.append({
                    "source_file": filename,
                    "parent_table": rel.get("parent_table"),
                    "child_table": rel.get("child_table"),
                    "parent_key": rel.get("parent_key"),
                    "foreign_key": rel.get("foreign_key"),
                    "relationship_type": rel.get("relationship_type"),
                    "description": rel.get("description")
                })

            # Extract table info
            for table in data.get("tables", []):
                table_name = table.get("table_name")
                tables_info[table_name] = {
                    "description": table.get("table_description"),
                    "column_count": len(table.get("columns", [])),
                    "source_file": filename
                }

        return json.dumps({
            "schema_files_loaded": len(files),
            "tables_found": len(tables_info),
            "relationships_found": len(relationships),
            "tables": tables_info,
            "relationships": relationships,
            "note": "Use these relationships for JOIN queries in cross-table visualizations"
        }, indent=2)

    except Exception as e:
        return f"Error loading schema relationships: {str(e)}"


# =============================================================================
# DATA ANALYSIS AND PLANNING TOOLS
# =============================================================================

@tool
def analyze_data_for_viz(table_name: str, session_id: Optional[str] = None) -> str:
    """
    Analyze a table's data to determine the best visualizations.
    Creates a visualization session and returns analysis with recommendations.

    Args:
        table_name: Name of the table to analyze
        session_id: Optional existing session ID

    Returns:
        Data analysis with visualization recommendations
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            # Check table exists
            tables = [r[0].upper() for r in conn.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                return f"Error: Table '{table_name}' not found."

            # Load data
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()

            # Create session
            if not session_id or session_id not in _viz_sessions:
                session_id = _create_session(table_name)

            _viz_sessions[session_id]["dataframe"] = df

            # Analyze data
            analysis = {
                "session_id": session_id,
                "table_name": table_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns_analysis": [],
                "suggested_kpis": [],
                "suggested_charts": [],
                "suggested_aggregations": []
            }

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

            # Analyze each column
            for col in df.columns:
                col_analysis = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                    "null_count": int(df[col].isna().sum()),
                    "null_percentage": round(df[col].isna().sum() / len(df) * 100, 2)
                }

                if col in numeric_cols:
                    col_analysis["type"] = "numeric"
                    col_analysis["stats"] = {
                        "min": _safe_json_serialize(df[col].min()),
                        "max": _safe_json_serialize(df[col].max()),
                        "mean": _safe_json_serialize(df[col].mean()),
                        "sum": _safe_json_serialize(df[col].sum())
                    }
                    # Suggest KPIs for numeric columns
                    analysis["suggested_kpis"].append({
                        "column": col,
                        "metrics": ["sum", "avg", "min", "max", "count"]
                    })
                elif col in datetime_cols:
                    col_analysis["type"] = "datetime"
                    col_analysis["range"] = {
                        "min": str(df[col].min()),
                        "max": str(df[col].max())
                    }
                else:
                    col_analysis["type"] = "categorical"
                    if col_analysis["unique_count"] <= 20:
                        col_analysis["top_values"] = df[col].value_counts().head(5).to_dict()

                analysis["columns_analysis"].append(col_analysis)

            # Generate chart suggestions
            # Bar charts for categorical with numeric
            for cat_col in categorical_cols[:3]:
                if df[cat_col].nunique() <= 15:
                    for num_col in numeric_cols[:2]:
                        analysis["suggested_charts"].append({
                            "type": "bar_chart",
                            "title": f"{num_col} by {cat_col}",
                            "x_column": cat_col,
                            "y_column": num_col,
                            "aggregation": "sum"
                        })

            # Pie charts for categorical distribution
            for cat_col in categorical_cols[:2]:
                if df[cat_col].nunique() <= 10:
                    analysis["suggested_charts"].append({
                        "type": "pie_chart",
                        "title": f"{cat_col} Distribution",
                        "column": cat_col
                    })

            # Line charts for time series
            for dt_col in datetime_cols:
                for num_col in numeric_cols[:2]:
                    analysis["suggested_charts"].append({
                        "type": "line_chart",
                        "title": f"{num_col} over Time",
                        "x_column": dt_col,
                        "y_column": num_col
                    })

            # Histograms for numeric distribution
            for num_col in numeric_cols[:3]:
                analysis["suggested_charts"].append({
                    "type": "histogram",
                    "title": f"{num_col} Distribution",
                    "column": num_col
                })

            # Scatter plots for numeric correlations
            if len(numeric_cols) >= 2:
                analysis["suggested_charts"].append({
                    "type": "scatter_plot",
                    "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                    "x_column": numeric_cols[0],
                    "y_column": numeric_cols[1]
                })

            # Suggested aggregations
            for cat_col in categorical_cols[:2]:
                for num_col in numeric_cols[:2]:
                    analysis["suggested_aggregations"].append({
                        "group_by": cat_col,
                        "metric": num_col,
                        "functions": ["SUM", "AVG", "COUNT"]
                    })

            _viz_sessions[session_id]["analysis"] = analysis

            return json.dumps(analysis, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing data: {str(e)}"


@tool
def generate_viz_plan(session_id: str, dashboard_title: Optional[str] = None) -> str:
    """
    Generate a comprehensive visualization plan for the dashboard.
    This creates a structured plan with all charts, KPIs, and layout.

    Args:
        session_id: Visualization session ID
        dashboard_title: Optional custom title for the dashboard

    Returns:
        Detailed visualization plan with SQL queries
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found. Use analyze_data_for_viz first."

        session = _viz_sessions[session_id]
        table_name = session["table_name"]
        analysis = session.get("analysis", {})

        if dashboard_title:
            session["dashboard_title"] = dashboard_title

        # Generate visualization plan
        plan = {
            "session_id": session_id,
            "dashboard_title": session["dashboard_title"],
            "table_name": table_name,
            "layout": {
                "sections": []
            },
            "kpis": [],
            "visualizations": []
        }

        # KPI Section
        kpi_section = {
            "title": "Key Metrics",
            "type": "kpi_row",
            "items": []
        }

        suggested_kpis = analysis.get("suggested_kpis", [])[:4]
        for i, kpi in enumerate(suggested_kpis):
            col = kpi["column"]
            kpi_item = {
                "id": f"kpi_{i+1}",
                "title": f"Total {col}",
                "sql": f"SELECT SUM({col}) as value FROM {table_name}",
                "format": "number"
            }
            kpi_section["items"].append(kpi_item)
            plan["kpis"].append(kpi_item)

        # Add count KPI
        kpi_section["items"].append({
            "id": "kpi_count",
            "title": "Total Records",
            "sql": f"SELECT COUNT(*) as value FROM {table_name}",
            "format": "number"
        })
        plan["layout"]["sections"].append(kpi_section)

        # Charts Section
        chart_section = {
            "title": "Charts",
            "type": "chart_grid",
            "items": []
        }

        suggested_charts = analysis.get("suggested_charts", [])[:8]
        for i, chart in enumerate(suggested_charts):
            viz_item = {
                "id": f"chart_{i+1}",
                "type": chart["type"],
                "title": chart["title"]
            }

            # Generate SQL based on chart type
            if chart["type"] == "bar_chart":
                x_col = chart.get("x_column")
                y_col = chart.get("y_column")
                agg = chart.get("aggregation", "SUM")
                viz_item["sql"] = f"SELECT {x_col}, {agg}({y_col}) as {y_col} FROM {table_name} GROUP BY {x_col} ORDER BY {agg}({y_col}) DESC LIMIT 10"
                viz_item["x_column"] = x_col
                viz_item["y_column"] = y_col

            elif chart["type"] == "pie_chart":
                col = chart.get("column")
                viz_item["sql"] = f"SELECT {col}, COUNT(*) as count FROM {table_name} GROUP BY {col} ORDER BY count DESC LIMIT 10"
                viz_item["column"] = col

            elif chart["type"] == "line_chart":
                x_col = chart.get("x_column")
                y_col = chart.get("y_column")
                viz_item["sql"] = f"SELECT {x_col}, SUM({y_col}) as {y_col} FROM {table_name} GROUP BY {x_col} ORDER BY {x_col}"
                viz_item["x_column"] = x_col
                viz_item["y_column"] = y_col

            elif chart["type"] == "histogram":
                col = chart.get("column")
                viz_item["sql"] = f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL"
                viz_item["column"] = col

            elif chart["type"] == "scatter_plot":
                x_col = chart.get("x_column")
                y_col = chart.get("y_column")
                viz_item["sql"] = f"SELECT {x_col}, {y_col} FROM {table_name} WHERE {x_col} IS NOT NULL AND {y_col} IS NOT NULL LIMIT 1000"
                viz_item["x_column"] = x_col
                viz_item["y_column"] = y_col

            chart_section["items"].append(viz_item)
            plan["visualizations"].append(viz_item)

        plan["layout"]["sections"].append(chart_section)

        # Data Table Section
        table_section = {
            "title": "Data Preview",
            "type": "data_table",
            "sql": f"SELECT * FROM {table_name} LIMIT 100"
        }
        plan["layout"]["sections"].append(table_section)

        session["viz_plan"] = plan

        return json.dumps(plan, indent=2)

    except Exception as e:
        return f"Error generating plan: {str(e)}"


# =============================================================================
# SQL AND DATA COLLECTION TOOLS
# =============================================================================

@tool
def execute_viz_query(session_id: str, sql: str, cache_key: Optional[str] = None) -> str:
    """
    Execute a SQL query and cache the results for visualization.

    Args:
        session_id: Visualization session ID
        sql: SQL query to execute
        cache_key: Optional key to cache results (for reuse)

    Returns:
        Query results with row count and column info
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

            # Cache the result
            if cache_key:
                _viz_sessions[session_id]["data_cache"][cache_key] = df

            # Return summary
            result = {
                "session_id": session_id,
                "sql": sql,
                "row_count": len(df),
                "columns": list(df.columns),
                "cache_key": cache_key,
                "preview": df.head(5).to_dict(orient="records")
            }

            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def collect_all_viz_data(session_id: str) -> str:
    """
    Execute all SQL queries from the visualization plan and cache results.
    This prepares all data needed for the dashboard.

    Args:
        session_id: Visualization session ID

    Returns:
        Summary of collected data
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        plan = session.get("viz_plan")

        if not plan:
            return "Error: No visualization plan found. Use generate_viz_plan first."

        collected = []
        errors = []

        with get_connection() as conn:
            # Collect KPI data
            for kpi in plan.get("kpis", []):
                try:
                    df = conn.execute(kpi["sql"]).fetchdf()
                    session["data_cache"][kpi["id"]] = df
                    collected.append({
                        "id": kpi["id"],
                        "type": "kpi",
                        "rows": len(df),
                        "value": _safe_json_serialize(df.iloc[0, 0]) if len(df) > 0 else None
                    })
                except Exception as e:
                    errors.append({"id": kpi["id"], "error": str(e)})

            # Collect chart data
            for viz in plan.get("visualizations", []):
                try:
                    df = conn.execute(viz["sql"]).fetchdf()
                    session["data_cache"][viz["id"]] = df
                    collected.append({
                        "id": viz["id"],
                        "type": viz["type"],
                        "rows": len(df)
                    })
                except Exception as e:
                    errors.append({"id": viz["id"], "error": str(e)})

            # Collect table data
            for section in plan.get("layout", {}).get("sections", []):
                if section.get("type") == "data_table" and section.get("sql"):
                    try:
                        df = conn.execute(section["sql"]).fetchdf()
                        session["data_cache"]["data_table"] = df
                        collected.append({
                            "id": "data_table",
                            "type": "table",
                            "rows": len(df)
                        })
                    except Exception as e:
                        errors.append({"id": "data_table", "error": str(e)})

        return json.dumps({
            "session_id": session_id,
            "collected_count": len(collected),
            "error_count": len(errors),
            "collected": collected,
            "errors": errors if errors else None,
            "status": "ready" if not errors else "partial"
        }, indent=2)

    except Exception as e:
        return f"Error collecting data: {str(e)}"


# =============================================================================
# VISUALIZATION CREATION TOOLS
# =============================================================================

@tool
def create_bar_chart(session_id: str, title: str, sql: str, x_column: str, y_column: str,
                     orientation: str = "v", color_column: Optional[str] = None) -> str:
    """
    Create a bar chart visualization.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        x_column: Column for X axis
        y_column: Column for Y axis
        orientation: 'v' for vertical, 'h' for horizontal
        color_column: Optional column for color grouping

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed. Run: pip install plotly"

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette()

        if orientation == "h":
            fig = px.bar(df, y=x_column, x=y_column, orientation='h',
                        title=title, color=color_column if color_column else None,
                        color_discrete_sequence=colors)
        else:
            fig = px.bar(df, x=x_column, y=y_column,
                        title=title, color=color_column if color_column else None,
                        color_discrete_sequence=colors)

        fig.update_layout(
            template="plotly_white",
            title_font_size=16,
            showlegend=True if color_column else False
        )

        chart_id = f"bar_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": chart_id,
            "type": "bar_chart",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "bar_chart",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


@tool
def create_line_chart(session_id: str, title: str, sql: str, x_column: str, y_column: str,
                      color_column: Optional[str] = None) -> str:
    """
    Create a line chart visualization.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        x_column: Column for X axis (typically datetime)
        y_column: Column for Y axis
        color_column: Optional column for multiple lines

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette()

        fig = px.line(df, x=x_column, y=y_column, title=title,
                     color=color_column if color_column else None,
                     color_discrete_sequence=colors,
                     markers=True)

        fig.update_layout(
            template="plotly_white",
            title_font_size=16
        )

        chart_id = f"line_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": chart_id,
            "type": "line_chart",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "line_chart",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating line chart: {str(e)}"


@tool
def create_pie_chart(session_id: str, title: str, sql: str, names_column: str, values_column: str,
                     hole: float = 0.0) -> str:
    """
    Create a pie chart or donut chart visualization.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        names_column: Column for slice names
        values_column: Column for slice values
        hole: Size of hole for donut chart (0-0.9, 0 for pie)

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette()

        fig = px.pie(df, names=names_column, values=values_column, title=title,
                    color_discrete_sequence=colors, hole=hole)

        fig.update_layout(
            template="plotly_white",
            title_font_size=16
        )

        chart_type = "donut_chart" if hole > 0 else "pie_chart"
        chart_id = f"pie_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": chart_id,
            "type": chart_type,
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": chart_type,
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating pie chart: {str(e)}"


@tool
def create_histogram(session_id: str, title: str, sql: str, column: str, nbins: int = 30) -> str:
    """
    Create a histogram visualization for distribution analysis.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        column: Column to create histogram for
        nbins: Number of bins (default 30)

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette()

        fig = px.histogram(df, x=column, title=title, nbins=nbins,
                          color_discrete_sequence=colors)

        fig.update_layout(
            template="plotly_white",
            title_font_size=16,
            bargap=0.1
        )

        chart_id = f"hist_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": chart_id,
            "type": "histogram",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "histogram",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating histogram: {str(e)}"


@tool
def create_scatter_plot(session_id: str, title: str, sql: str, x_column: str, y_column: str,
                        color_column: Optional[str] = None, size_column: Optional[str] = None) -> str:
    """
    Create a scatter plot visualization.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        x_column: Column for X axis
        y_column: Column for Y axis
        color_column: Optional column for point colors
        size_column: Optional column for point sizes

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette()

        fig = px.scatter(df, x=x_column, y=y_column, title=title,
                        color=color_column if color_column else None,
                        size=size_column if size_column else None,
                        color_discrete_sequence=colors)

        fig.update_layout(
            template="plotly_white",
            title_font_size=16
        )

        chart_id = f"scatter_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": chart_id,
            "type": "scatter_plot",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "scatter_plot",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


@tool
def create_heatmap(session_id: str, title: str, sql: str, x_column: str, y_column: str,
                   value_column: str) -> str:
    """
    Create a heatmap visualization.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data (should return x, y, value columns)
        x_column: Column for X axis categories
        y_column: Column for Y axis categories
        value_column: Column for heat values

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        # Pivot data for heatmap
        pivot_df = df.pivot(index=y_column, columns=x_column, values=value_column)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale='Blues'
        ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            title_font_size=16
        )

        chart_id = f"heatmap_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": chart_id,
            "type": "heatmap",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "heatmap",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating heatmap: {str(e)}"


@tool
def create_kpi_card(session_id: str, title: str, sql: str, format_type: str = "number",
                    prefix: str = "", suffix: str = "", comparison_sql: Optional[str] = None) -> str:
    """
    Create a KPI card with a metric value.

    Args:
        session_id: Visualization session ID
        title: KPI title
        sql: SQL query that returns a single value
        format_type: 'number', 'currency', 'percentage'
        prefix: Prefix for the value (e.g., '$')
        suffix: Suffix for the value (e.g., '%')
        comparison_sql: Optional SQL for comparison value (for delta)

    Returns:
        Confirmation with KPI ID
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()
            value = df.iloc[0, 0] if len(df) > 0 else 0

            delta = None
            if comparison_sql:
                comp_df = conn.execute(comparison_sql).fetchdf()
                if len(comp_df) > 0:
                    comp_value = comp_df.iloc[0, 0]
                    if comp_value and comp_value != 0:
                        delta = ((value - comp_value) / comp_value) * 100

        # Format value
        if format_type == "currency":
            formatted_value = f"{prefix}${value:,.2f}{suffix}"
        elif format_type == "percentage":
            formatted_value = f"{prefix}{value:.1f}%{suffix}"
        else:
            if isinstance(value, float):
                formatted_value = f"{prefix}{value:,.2f}{suffix}"
            else:
                formatted_value = f"{prefix}{value:,}{suffix}"

        kpi_id = f"kpi_{len(_viz_sessions[session_id]['kpis']) + 1}"
        _viz_sessions[session_id]["kpis"].append({
            "id": kpi_id,
            "title": title,
            "value": _safe_json_serialize(value),
            "formatted_value": formatted_value,
            "delta": _safe_json_serialize(delta) if delta else None,
            "format_type": format_type
        })

        return json.dumps({
            "status": "success",
            "kpi_id": kpi_id,
            "title": title,
            "value": _safe_json_serialize(value),
            "formatted_value": formatted_value,
            "delta": _safe_json_serialize(delta) if delta else None
        }, indent=2)

    except Exception as e:
        return f"Error creating KPI card: {str(e)}"


@tool
def create_data_table(session_id: str, title: str, sql: str, max_rows: int = 100) -> str:
    """
    Create a data table visualization.

    Args:
        session_id: Visualization session ID
        title: Table title
        sql: SQL query for data
        max_rows: Maximum rows to display (default 100)

    Returns:
        Confirmation with table ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        with get_connection() as conn:
            df = conn.execute(f"{sql} LIMIT {max_rows}").fetchdf()

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='#2E86AB',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[df[col].tolist() for col in df.columns],
                fill_color='lavender',
                align='left'
            )
        )])

        fig.update_layout(
            title=title,
            title_font_size=16
        )

        table_id = f"table_{len(_viz_sessions[session_id]['visualizations']) + 1}"
        _viz_sessions[session_id]["visualizations"].append({
            "id": table_id,
            "type": "data_table",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "table_id": table_id,
            "type": "data_table",
            "title": title,
            "data_rows": len(df),
            "columns": list(df.columns)
        }, indent=2)

    except Exception as e:
        return f"Error creating data table: {str(e)}"


# =============================================================================
# DASHBOARD GENERATION TOOLS
# =============================================================================

@tool
def generate_dashboard(session_id: str, output_filename: Optional[str] = None) -> str:
    """
    Generate a complete HTML dashboard from all visualizations in the session.

    Args:
        session_id: Visualization session ID
        output_filename: Optional filename (default: dashboard_<session_id>.html)

    Returns:
        Path to generated dashboard HTML file
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        title = session.get("dashboard_title", "Data Dashboard")
        kpis = session.get("kpis", [])
        visualizations = session.get("visualizations", [])

        if not kpis and not visualizations:
            return "Error: No visualizations created. Create charts first."

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .dashboard-container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(46, 134, 171, 0.3);
        }}
        .dashboard-header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .dashboard-header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .kpi-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        .kpi-title {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .kpi-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .kpi-delta {{
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .kpi-delta.positive {{
            color: #27ae60;
        }}
        .kpi-delta.negative {{
            color: #e74c3c;
        }}
        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 25px;
        }}
        .chart-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .chart-card.full-width {{
            grid-column: 1 / -1;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .charts-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data Source: {session['table_name']}</p>
        </div>
"""

        # Add KPI cards
        if kpis:
            html_content += '        <div class="kpi-container">\n'
            for kpi in kpis:
                delta_html = ""
                if kpi.get("delta") is not None:
                    delta_class = "positive" if kpi["delta"] > 0 else "negative"
                    delta_symbol = "+" if kpi["delta"] > 0 else ""
                    delta_html = f'<div class="kpi-delta {delta_class}">{delta_symbol}{kpi["delta"]:.1f}% vs previous</div>'

                html_content += f"""
            <div class="kpi-card">
                <div class="kpi-title">{kpi['title']}</div>
                <div class="kpi-value">{kpi['formatted_value']}</div>
                {delta_html}
            </div>
"""
            html_content += '        </div>\n'

        # Add charts
        if visualizations:
            html_content += '        <div class="charts-container">\n'
            for i, viz in enumerate(visualizations):
                fig = viz.get("figure")
                if fig:
                    # Update figure layout for better dashboard appearance
                    fig.update_layout(
                        margin=dict(l=40, r=40, t=60, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    full_width = "full-width" if viz["type"] in ["data_table", "heatmap"] else ""
                    html_content += f"""
            <div class="chart-card {full_width}">
                {chart_html}
            </div>
"""
            html_content += '        </div>\n'

        # Footer
        html_content += f"""
        <div class="footer">
            <p>Dashboard generated by Data Visualization Agent | Session: {session_id}</p>
        </div>
    </div>
</body>
</html>
"""

        # Save to file
        if not output_filename:
            output_filename = f"dashboard_{session_id}.html"

        if not output_filename.endswith('.html'):
            output_filename += '.html'

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        session["dashboard_generated"] = True
        session["dashboard_path"] = output_path

        # Get absolute path
        abs_path = os.path.abspath(output_path)

        return json.dumps({
            "status": "success",
            "dashboard_path": abs_path,
            "filename": output_filename,
            "kpi_count": len(kpis),
            "chart_count": len(visualizations),
            "message": f"Dashboard generated successfully! Open the HTML file in a browser to view.",
            "open_command": f"start {abs_path}" if os.name == 'nt' else f"open {abs_path}"
        }, indent=2)

    except Exception as e:
        return f"Error generating dashboard: {str(e)}"


@tool
def generate_dashboard_from_plan(session_id: str, output_filename: Optional[str] = None) -> str:
    """
    Automatically generate a complete dashboard from the visualization plan.
    This executes all planned visualizations and creates the final dashboard.

    Args:
        session_id: Visualization session ID
        output_filename: Optional filename for the dashboard

    Returns:
        Path to generated dashboard
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed. Run: pip install plotly"

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        plan = session.get("viz_plan")

        if not plan:
            return "Error: No visualization plan found. Use generate_viz_plan first."

        results = {"kpis_created": 0, "charts_created": 0, "errors": []}
        colors = _get_color_palette()

        with get_connection() as conn:
            # Create KPIs
            for kpi in plan.get("kpis", []):
                try:
                    df = conn.execute(kpi["sql"]).fetchdf()
                    value = df.iloc[0, 0] if len(df) > 0 else 0

                    if isinstance(value, float):
                        formatted = f"{value:,.2f}"
                    else:
                        formatted = f"{value:,}"

                    session["kpis"].append({
                        "id": kpi["id"],
                        "title": kpi["title"],
                        "value": _safe_json_serialize(value),
                        "formatted_value": formatted,
                        "delta": None,
                        "format_type": kpi.get("format", "number")
                    })
                    results["kpis_created"] += 1
                except Exception as e:
                    results["errors"].append(f"KPI {kpi['id']}: {str(e)}")

            # Create visualizations
            for viz in plan.get("visualizations", []):
                try:
                    df = conn.execute(viz["sql"]).fetchdf()

                    if viz["type"] == "bar_chart":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.bar(df, x=x_col, y=y_col, title=viz["title"],
                                    color_discrete_sequence=colors)

                    elif viz["type"] == "pie_chart":
                        col = viz.get("column")
                        # Assume first column is names, second is values
                        fig = px.pie(df, names=df.columns[0], values=df.columns[1],
                                    title=viz["title"], color_discrete_sequence=colors)

                    elif viz["type"] == "line_chart":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.line(df, x=x_col, y=y_col, title=viz["title"],
                                     color_discrete_sequence=colors, markers=True)

                    elif viz["type"] == "histogram":
                        col = viz.get("column")
                        fig = px.histogram(df, x=col, title=viz["title"],
                                          color_discrete_sequence=colors)

                    elif viz["type"] == "scatter_plot":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.scatter(df, x=x_col, y=y_col, title=viz["title"],
                                        color_discrete_sequence=colors)

                    else:
                        continue

                    fig.update_layout(template="plotly_white", title_font_size=16)

                    session["visualizations"].append({
                        "id": viz["id"],
                        "type": viz["type"],
                        "title": viz["title"],
                        "figure": fig,
                        "data": df.to_dict(orient="records")
                    })
                    results["charts_created"] += 1

                except Exception as e:
                    results["errors"].append(f"Chart {viz['id']}: {str(e)}")

        # Generate the dashboard
        if results["kpis_created"] > 0 or results["charts_created"] > 0:
            dashboard_result = generate_dashboard.invoke({
                "session_id": session_id,
                "output_filename": output_filename
            })
            return dashboard_result
        else:
            return json.dumps({
                "status": "error",
                "message": "No visualizations could be created",
                "errors": results["errors"]
            }, indent=2)

    except Exception as e:
        return f"Error generating dashboard from plan: {str(e)}"


# =============================================================================
# SESSION MANAGEMENT TOOLS
# =============================================================================

@tool
def get_viz_session_info(session_id: str) -> str:
    """
    Get information about a visualization session.

    Args:
        session_id: Visualization session ID

    Returns:
        Session details including created visualizations
    """
    try:
        if session_id not in _viz_sessions:
            available = list(_viz_sessions.keys()) if _viz_sessions else "none"
            return f"Session '{session_id}' not found. Available: {available}"

        session = _viz_sessions[session_id]

        info = {
            "session_id": session_id,
            "table_name": session.get("table_name"),
            "dashboard_title": session.get("dashboard_title"),
            "created_at": session.get("created_at"),
            "kpis_count": len(session.get("kpis", [])),
            "visualizations_count": len(session.get("visualizations", [])),
            "has_plan": session.get("viz_plan") is not None,
            "dashboard_generated": session.get("dashboard_generated", False),
            "dashboard_path": session.get("dashboard_path"),
            "visualizations": [
                {"id": v["id"], "type": v["type"], "title": v["title"]}
                for v in session.get("visualizations", [])
            ],
            "kpis": [
                {"id": k["id"], "title": k["title"], "value": k.get("formatted_value")}
                for k in session.get("kpis", [])
            ]
        }

        return json.dumps(info, indent=2)

    except Exception as e:
        return f"Error getting session info: {str(e)}"


@tool
def list_viz_sessions() -> str:
    """
    List all active visualization sessions.

    Returns:
        List of all visualization sessions
    """
    try:
        if not _viz_sessions:
            return "No active visualization sessions. Use analyze_data_for_viz() to start."

        sessions = []
        for sid, session in _viz_sessions.items():
            sessions.append({
                "session_id": sid,
                "table_name": session.get("table_name"),
                "created_at": session.get("created_at"),
                "visualizations": len(session.get("visualizations", [])),
                "kpis": len(session.get("kpis", [])),
                "dashboard_generated": session.get("dashboard_generated", False)
            })

        return json.dumps({
            "total_sessions": len(sessions),
            "sessions": sessions
        }, indent=2)

    except Exception as e:
        return f"Error listing sessions: {str(e)}"


@tool
def set_dashboard_title(session_id: str, title: str) -> str:
    """
    Set the title for the dashboard.

    Args:
        session_id: Visualization session ID
        title: New dashboard title

    Returns:
        Confirmation
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        _viz_sessions[session_id]["dashboard_title"] = title

        return json.dumps({
            "status": "success",
            "session_id": session_id,
            "dashboard_title": title
        }, indent=2)

    except Exception as e:
        return f"Error setting title: {str(e)}"


@tool
def clear_session_visualizations(session_id: str) -> str:
    """
    Clear all visualizations from a session (to rebuild).

    Args:
        session_id: Visualization session ID

    Returns:
        Confirmation
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        _viz_sessions[session_id]["visualizations"] = []
        _viz_sessions[session_id]["kpis"] = []
        _viz_sessions[session_id]["dashboard_generated"] = False

        return json.dumps({
            "status": "success",
            "session_id": session_id,
            "message": "All visualizations cleared. Ready to create new ones."
        }, indent=2)

    except Exception as e:
        return f"Error clearing session: {str(e)}"


# =============================================================================
# REGISTER TOOLS WITH REGISTRY
# =============================================================================

# Discovery tools
tool_registry.register(list_tables_for_viz, "dataviz")
tool_registry.register(get_table_schema_for_viz, "dataviz")
tool_registry.register(load_schema_relationships, "dataviz")

# Analysis and planning tools
tool_registry.register(analyze_data_for_viz, "dataviz")
tool_registry.register(generate_viz_plan, "dataviz")

# SQL and data collection tools
tool_registry.register(execute_viz_query, "dataviz")
tool_registry.register(collect_all_viz_data, "dataviz")

# Visualization creation tools
tool_registry.register(create_bar_chart, "dataviz")
tool_registry.register(create_line_chart, "dataviz")
tool_registry.register(create_pie_chart, "dataviz")
tool_registry.register(create_histogram, "dataviz")
tool_registry.register(create_scatter_plot, "dataviz")
tool_registry.register(create_heatmap, "dataviz")
tool_registry.register(create_kpi_card, "dataviz")
tool_registry.register(create_data_table, "dataviz")

# Dashboard generation tools
tool_registry.register(generate_dashboard, "dataviz")
tool_registry.register(generate_dashboard_from_plan, "dataviz")

# Session management tools
tool_registry.register(get_viz_session_info, "dataviz")
tool_registry.register(list_viz_sessions, "dataviz")
tool_registry.register(set_dashboard_title, "dataviz")
tool_registry.register(clear_session_visualizations, "dataviz")


def get_dataviz_tools():
    """Get all data visualization tools."""
    return tool_registry.get_tools_by_category("dataviz")
