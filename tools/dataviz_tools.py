"""
Data Visualization Tools Module - Professional Edition
======================================================

Comprehensive data visualization tools for creating professional, interactive
dashboards from DuckDB data. Uses Plotly for charts and generates beautiful
HTML dashboards with dark/light themes.

Features:
- 25+ visualization tools
- Professional KPI cards with trends and sparklines
- Multiple chart types (bar, line, pie, donut, area, scatter, heatmap, treemap, gauge)
- Dark and light theme support
- Automatic visualization planning based on data analysis
- Data table always at the end of dashboard
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


# =============================================================================
# COLOR PALETTES AND THEMES
# =============================================================================

# Professional color palettes
COLOR_PALETTES = {
    "vibrant": [
        "#00D4AA",  # Teal
        "#FF6B6B",  # Coral
        "#4ECDC4",  # Turquoise
        "#FFE66D",  # Yellow
        "#95E1D3",  # Mint
        "#F38181",  # Salmon
        "#AA96DA",  # Lavender
        "#FCBAD3",  # Pink
        "#A8D8EA",  # Light Blue
        "#FF9F43",  # Orange
    ],
    "corporate": [
        "#2E86AB",  # Blue
        "#A23B72",  # Magenta
        "#F18F01",  # Orange
        "#C73E1D",  # Red
        "#95C623",  # Green
        "#5D5179",  # Purple
        "#0A2463",  # Navy
        "#FB3640",  # Coral
        "#1E3888",  # Dark Blue
        "#47A8BD",  # Cyan
    ],
    "modern": [
        "#6C5CE7",  # Purple
        "#00CEC9",  # Teal
        "#FF7675",  # Coral
        "#FDCB6E",  # Yellow
        "#74B9FF",  # Blue
        "#55EFC4",  # Mint
        "#E17055",  # Orange
        "#81ECEC",  # Cyan
        "#A29BFE",  # Lavender
        "#FD79A8",  # Pink
    ],
    "dark_friendly": [
        "#00F5D4",  # Bright Teal
        "#FEE440",  # Bright Yellow
        "#F15BB5",  # Bright Pink
        "#9B5DE5",  # Purple
        "#00BBF9",  # Bright Blue
        "#00F5A0",  # Bright Green
        "#FF6F61",  # Coral
        "#FFD93D",  # Gold
        "#6BCB77",  # Green
        "#4D96FF",  # Blue
    ]
}

# KPI Card colors (for individual cards)
KPI_COLORS = {
    "green": {"bg": "#10B981", "light": "#D1FAE5", "dark": "#065F46"},
    "blue": {"bg": "#3B82F6", "light": "#DBEAFE", "dark": "#1E40AF"},
    "purple": {"bg": "#8B5CF6", "light": "#EDE9FE", "dark": "#5B21B6"},
    "orange": {"bg": "#F59E0B", "light": "#FEF3C7", "dark": "#B45309"},
    "red": {"bg": "#EF4444", "light": "#FEE2E2", "dark": "#B91C1C"},
    "teal": {"bg": "#14B8A6", "light": "#CCFBF1", "dark": "#0F766E"},
    "pink": {"bg": "#EC4899", "light": "#FCE7F3", "dark": "#BE185D"},
    "indigo": {"bg": "#6366F1", "light": "#E0E7FF", "dark": "#4338CA"},
}

# Theme configurations
THEMES = {
    "dark": {
        "bg_color": "#0F172A",
        "card_bg": "#1E293B",
        "text_color": "#F1F5F9",
        "text_secondary": "#94A3B8",
        "border_color": "#334155",
        "grid_color": "#334155",
        "chart_bg": "rgba(30, 41, 59, 0.8)",
    },
    "light": {
        "bg_color": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "text_color": "#1E293B",
        "text_secondary": "#64748B",
        "border_color": "#E2E8F0",
        "grid_color": "#E2E8F0",
        "chart_bg": "rgba(255, 255, 255, 0.9)",
    }
}


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
        "dashboard_generated": False,
        "theme": "dark",  # Default to dark theme for professional look
        "color_palette": "dark_friendly"
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


def _get_color_palette(palette_name: str = "dark_friendly") -> List[str]:
    """Get a color palette by name."""
    return COLOR_PALETTES.get(palette_name, COLOR_PALETTES["dark_friendly"])


def _format_number(value, format_type: str = "number") -> str:
    """Format a number for display."""
    if value is None:
        return "N/A"

    try:
        value = float(value)
        if format_type == "currency":
            if abs(value) >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"${value/1_000:.1f}K"
            else:
                return f"${value:,.2f}"
        elif format_type == "percentage":
            return f"{value:.1f}%"
        elif format_type == "compact":
            if abs(value) >= 1_000_000_000:
                return f"{value/1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                return f"{value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:,.0f}"
        else:
            if isinstance(value, float) and value != int(value):
                return f"{value:,.2f}"
            else:
                return f"{int(value):,}"
    except:
        return str(value)


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
                    viz_suggestions = ["histogram", "box_plot", "kpi_card", "line_chart", "scatter_plot", "gauge", "area_chart"]
                    col_class = "numeric"
                elif any(t in col_type for t in ["DATE", "TIME", "TIMESTAMP"]):
                    viz_suggestions = ["line_chart", "area_chart", "timeline"]
                    col_class = "datetime"
                else:
                    unique_count = conn.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}").fetchone()[0]
                    if unique_count <= 15:
                        viz_suggestions = ["bar_chart", "pie_chart", "donut_chart", "horizontal_bar", "treemap"]
                    else:
                        viz_suggestions = ["bar_chart", "horizontal_bar", "treemap"]
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
                        {"type": "donut_chart", "columns": [c["name"] for c in columns if c["classification"] == "categorical"][:1]}
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
    Creates a visualization session and returns comprehensive analysis with recommendations.

    Args:
        table_name: Name of the table to analyze
        session_id: Optional existing session ID

    Returns:
        Data analysis with visualization recommendations for a complete dashboard
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
                "suggested_aggregations": [],
                "data_story": ""
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
                        "sum": _safe_json_serialize(df[col].sum()),
                        "median": _safe_json_serialize(df[col].median())
                    }
                    # Suggest KPIs for numeric columns
                    analysis["suggested_kpis"].append({
                        "column": col,
                        "metrics": ["sum", "avg", "min", "max", "count"],
                        "suggested_format": "currency" if any(x in col.lower() for x in ["amount", "price", "value", "cost", "revenue", "profit"]) else "number"
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

            # Generate comprehensive chart suggestions
            kpi_colors = list(KPI_COLORS.keys())
            color_idx = 0

            # KPI suggestions with colors
            for i, kpi in enumerate(analysis["suggested_kpis"][:6]):
                kpi["color"] = kpi_colors[i % len(kpi_colors)]

            # Bar charts for categorical with numeric (vertical and horizontal)
            for cat_col in categorical_cols[:4]:
                if df[cat_col].nunique() <= 15:
                    for num_col in numeric_cols[:3]:
                        analysis["suggested_charts"].append({
                            "type": "bar_chart",
                            "title": f"{num_col} by {cat_col}",
                            "x_column": cat_col,
                            "y_column": num_col,
                            "aggregation": "sum"
                        })
                        # Add horizontal bar for variety
                        if df[cat_col].nunique() >= 5:
                            analysis["suggested_charts"].append({
                                "type": "horizontal_bar",
                                "title": f"Top {cat_col} by {num_col}",
                                "x_column": num_col,
                                "y_column": cat_col,
                                "aggregation": "sum"
                            })

            # Donut/Pie charts for categorical distribution
            for cat_col in categorical_cols[:3]:
                if 2 <= df[cat_col].nunique() <= 10:
                    analysis["suggested_charts"].append({
                        "type": "donut_chart",
                        "title": f"{cat_col} Distribution",
                        "column": cat_col
                    })

            # Line charts for time series
            for dt_col in datetime_cols:
                for num_col in numeric_cols[:3]:
                    analysis["suggested_charts"].append({
                        "type": "line_chart",
                        "title": f"{num_col} over Time",
                        "x_column": dt_col,
                        "y_column": num_col
                    })
                    # Add area chart variant
                    analysis["suggested_charts"].append({
                        "type": "area_chart",
                        "title": f"{num_col} Trend",
                        "x_column": dt_col,
                        "y_column": num_col
                    })

            # Histograms for numeric distribution
            for num_col in numeric_cols[:4]:
                analysis["suggested_charts"].append({
                    "type": "histogram",
                    "title": f"{num_col} Distribution",
                    "column": num_col
                })

            # Scatter plots for numeric correlations
            if len(numeric_cols) >= 2:
                for i in range(min(len(numeric_cols)-1, 2)):
                    analysis["suggested_charts"].append({
                        "type": "scatter_plot",
                        "title": f"{numeric_cols[i]} vs {numeric_cols[i+1]}",
                        "x_column": numeric_cols[i],
                        "y_column": numeric_cols[i+1]
                    })

            # Stacked bar for multiple categories
            if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                analysis["suggested_charts"].append({
                    "type": "stacked_bar",
                    "title": f"{numeric_cols[0]} by {categorical_cols[0]} and {categorical_cols[1]}",
                    "x_column": categorical_cols[0],
                    "y_column": numeric_cols[0],
                    "color_column": categorical_cols[1]
                })

            # Treemap for hierarchical data
            for cat_col in categorical_cols[:2]:
                if 3 <= df[cat_col].nunique() <= 20 and len(numeric_cols) > 0:
                    analysis["suggested_charts"].append({
                        "type": "treemap",
                        "title": f"{cat_col} Breakdown",
                        "labels_column": cat_col,
                        "values_column": numeric_cols[0]
                    })

            # Gauge charts for key metrics
            if len(numeric_cols) >= 1:
                analysis["suggested_charts"].append({
                    "type": "gauge",
                    "title": f"Average {numeric_cols[0]}",
                    "column": numeric_cols[0],
                    "metric": "avg"
                })

            # Suggested aggregations
            for cat_col in categorical_cols[:3]:
                for num_col in numeric_cols[:3]:
                    analysis["suggested_aggregations"].append({
                        "group_by": cat_col,
                        "metric": num_col,
                        "functions": ["SUM", "AVG", "COUNT", "MIN", "MAX"]
                    })

            # Generate data story
            analysis["data_story"] = f"""
This dataset contains {len(df):,} records with {len(df.columns)} columns.
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
- Date/Time columns ({len(datetime_cols)}): {', '.join(datetime_cols[:3])}

Recommended dashboard should include:
1. {min(6, len(analysis['suggested_kpis']))} KPI cards showing key metrics
2. {min(8, len(analysis['suggested_charts']))} visualizations covering distributions, comparisons, and trends
3. Data table at the end for detailed exploration
"""

            _viz_sessions[session_id]["analysis"] = analysis

            return json.dumps(analysis, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing data: {str(e)}"


@tool
def generate_viz_plan(session_id: str, dashboard_title: Optional[str] = None, theme: str = "dark") -> str:
    """
    Generate a comprehensive visualization plan for a professional dashboard.
    This creates a structured plan with KPIs, charts, and layout optimized to tell the complete data story.

    Args:
        session_id: Visualization session ID
        dashboard_title: Optional custom title for the dashboard
        theme: 'dark' for professional dark theme (default), 'light' for light theme

    Returns:
        Detailed visualization plan with SQL queries and layout
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found. Use analyze_data_for_viz first."

        session = _viz_sessions[session_id]
        table_name = session["table_name"]
        analysis = session.get("analysis", {})

        if dashboard_title:
            session["dashboard_title"] = dashboard_title

        session["theme"] = theme
        session["color_palette"] = "dark_friendly" if theme == "dark" else "corporate"

        # Generate visualization plan
        plan = {
            "session_id": session_id,
            "dashboard_title": session["dashboard_title"],
            "table_name": table_name,
            "theme": theme,
            "layout": {
                "sections": []
            },
            "kpis": [],
            "visualizations": [],
            "data_table": None
        }

        kpi_colors = list(KPI_COLORS.keys())

        # KPI Section - Up to 6 KPIs
        kpi_section = {
            "title": "Key Performance Indicators",
            "type": "kpi_row",
            "items": []
        }

        suggested_kpis = analysis.get("suggested_kpis", [])[:6]
        for i, kpi in enumerate(suggested_kpis):
            col = kpi["column"]
            color = kpi_colors[i % len(kpi_colors)]
            format_type = kpi.get("suggested_format", "number")

            # Create multiple KPI metrics per column
            kpi_item = {
                "id": f"kpi_{i+1}",
                "title": f"Total {col.replace('_', ' ').title()}",
                "sql": f"SELECT SUM({col}) as value FROM {table_name}",
                "format": format_type,
                "color": color,
                "icon": "chart-line"
            }
            kpi_section["items"].append(kpi_item)
            plan["kpis"].append(kpi_item)

        # Add count KPI
        kpi_section["items"].append({
            "id": "kpi_count",
            "title": "Total Records",
            "sql": f"SELECT COUNT(*) as value FROM {table_name}",
            "format": "compact",
            "color": "teal",
            "icon": "database"
        })
        plan["kpis"].append(kpi_section["items"][-1])

        # Add unique count KPIs for important categorical columns
        cat_cols = [c for c in analysis.get("columns_analysis", []) if c.get("type") == "categorical"]
        for i, cat_col in enumerate(cat_cols[:2]):
            kpi_section["items"].append({
                "id": f"kpi_unique_{i+1}",
                "title": f"Unique {cat_col['name'].replace('_', ' ').title()}",
                "sql": f"SELECT COUNT(DISTINCT {cat_col['name']}) as value FROM {table_name}",
                "format": "compact",
                "color": kpi_colors[(len(suggested_kpis) + i + 1) % len(kpi_colors)],
                "icon": "users"
            })
            plan["kpis"].append(kpi_section["items"][-1])

        plan["layout"]["sections"].append(kpi_section)

        # Charts Section - Select best charts to tell the story
        chart_section = {
            "title": "Analytics & Insights",
            "type": "chart_grid",
            "items": []
        }

        suggested_charts = analysis.get("suggested_charts", [])

        # Select diverse chart types (max 10 charts)
        selected_charts = []
        chart_type_count = {}

        for chart in suggested_charts:
            chart_type = chart["type"]
            if chart_type_count.get(chart_type, 0) < 2:  # Max 2 of each type
                selected_charts.append(chart)
                chart_type_count[chart_type] = chart_type_count.get(chart_type, 0) + 1
                if len(selected_charts) >= 10:
                    break

        for i, chart in enumerate(selected_charts):
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

            elif chart["type"] == "horizontal_bar":
                x_col = chart.get("x_column")
                y_col = chart.get("y_column")
                agg = chart.get("aggregation", "SUM")
                viz_item["sql"] = f"SELECT {y_col}, {agg}({x_col}) as {x_col} FROM {table_name} GROUP BY {y_col} ORDER BY {agg}({x_col}) DESC LIMIT 10"
                viz_item["x_column"] = x_col
                viz_item["y_column"] = y_col

            elif chart["type"] in ["pie_chart", "donut_chart"]:
                col = chart.get("column")
                viz_item["sql"] = f"SELECT {col}, COUNT(*) as count FROM {table_name} GROUP BY {col} ORDER BY count DESC LIMIT 10"
                viz_item["column"] = col

            elif chart["type"] in ["line_chart", "area_chart"]:
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

            elif chart["type"] == "stacked_bar":
                x_col = chart.get("x_column")
                y_col = chart.get("y_column")
                color_col = chart.get("color_column")
                viz_item["sql"] = f"SELECT {x_col}, {color_col}, SUM({y_col}) as {y_col} FROM {table_name} GROUP BY {x_col}, {color_col} ORDER BY {x_col}"
                viz_item["x_column"] = x_col
                viz_item["y_column"] = y_col
                viz_item["color_column"] = color_col

            elif chart["type"] == "treemap":
                labels_col = chart.get("labels_column")
                values_col = chart.get("values_column")
                viz_item["sql"] = f"SELECT {labels_col}, SUM({values_col}) as {values_col} FROM {table_name} GROUP BY {labels_col} ORDER BY SUM({values_col}) DESC LIMIT 15"
                viz_item["labels_column"] = labels_col
                viz_item["values_column"] = values_col

            elif chart["type"] == "gauge":
                col = chart.get("column")
                metric = chart.get("metric", "avg")
                if metric == "avg":
                    viz_item["sql"] = f"SELECT AVG({col}) as value, MIN({col}) as min_val, MAX({col}) as max_val FROM {table_name}"
                else:
                    viz_item["sql"] = f"SELECT SUM({col}) as value FROM {table_name}"
                viz_item["column"] = col

            chart_section["items"].append(viz_item)
            plan["visualizations"].append(viz_item)

        plan["layout"]["sections"].append(chart_section)

        # Data Table Section - ALWAYS at the end
        table_section = {
            "title": "Data Details",
            "type": "data_table",
            "sql": f"SELECT * FROM {table_name} LIMIT 100"
        }
        plan["layout"]["sections"].append(table_section)
        plan["data_table"] = table_section

        session["viz_plan"] = plan

        return json.dumps(plan, indent=2)

    except Exception as e:
        return f"Error generating plan: {str(e)}"


@tool
def set_dashboard_theme(session_id: str, theme: str = "dark", color_palette: str = "dark_friendly") -> str:
    """
    Set the theme and color palette for the dashboard.

    Args:
        session_id: Visualization session ID
        theme: 'dark' for professional dark theme, 'light' for light theme
        color_palette: Color palette name ('vibrant', 'corporate', 'modern', 'dark_friendly')

    Returns:
        Confirmation of theme settings
    """
    try:
        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        if theme not in THEMES:
            return f"Error: Invalid theme. Choose from: {list(THEMES.keys())}"

        if color_palette not in COLOR_PALETTES:
            return f"Error: Invalid palette. Choose from: {list(COLOR_PALETTES.keys())}"

        _viz_sessions[session_id]["theme"] = theme
        _viz_sessions[session_id]["color_palette"] = color_palette

        return json.dumps({
            "status": "success",
            "session_id": session_id,
            "theme": theme,
            "color_palette": color_palette,
            "theme_colors": THEMES[theme],
            "palette_colors": COLOR_PALETTES[color_palette][:5]
        }, indent=2)

    except Exception as e:
        return f"Error setting theme: {str(e)}"


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
            if plan.get("data_table") and plan["data_table"].get("sql"):
                try:
                    df = conn.execute(plan["data_table"]["sql"]).fetchdf()
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

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        if orientation == "h":
            fig = px.bar(df, y=x_column, x=y_column, orientation='h',
                        title=title, color=color_column if color_column else None,
                        color_discrete_sequence=colors)
        else:
            fig = px.bar(df, x=x_column, y=y_column,
                        title=title, color=color_column if color_column else None,
                        color_discrete_sequence=colors)

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"]),
            showlegend=True if color_column else False
        )

        chart_id = f"bar_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
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

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        fig = px.line(df, x=x_column, y=y_column, title=title,
                     color=color_column if color_column else None,
                     color_discrete_sequence=colors,
                     markers=True)

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"])
        )

        chart_id = f"line_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
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
def create_area_chart(session_id: str, title: str, sql: str, x_column: str, y_column: str,
                      color_column: Optional[str] = None, stacked: bool = False) -> str:
    """
    Create an area chart visualization with fill.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        x_column: Column for X axis
        y_column: Column for Y axis
        color_column: Optional column for multiple areas
        stacked: Whether to stack areas (default False)

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        fig = px.area(df, x=x_column, y=y_column, title=title,
                     color=color_column if color_column else None,
                     color_discrete_sequence=colors,
                     line_group=color_column if color_column else None)

        if stacked:
            fig.update_traces(stackgroup='one')

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"])
        )

        chart_id = f"area_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
            "id": chart_id,
            "type": "area_chart",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "area_chart",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating area chart: {str(e)}"


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
        hole: Size of hole for donut chart (0-0.9, 0 for pie, 0.4+ for donut)

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        fig = px.pie(df, names=names_column, values=values_column, title=title,
                    color_discrete_sequence=colors, hole=hole)

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12
        )

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"]),
            showlegend=True,
            legend=dict(font=dict(color=theme["text_color"]))
        )

        chart_type = "donut_chart" if hole > 0 else "pie_chart"
        chart_id = f"pie_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
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
def create_donut_chart(session_id: str, title: str, sql: str, names_column: str, values_column: str,
                       center_text: Optional[str] = None) -> str:
    """
    Create a donut chart with optional center text/value.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        names_column: Column for slice names
        values_column: Column for slice values
        center_text: Optional text to display in center

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        fig = go.Figure(data=[go.Pie(
            labels=df[names_column],
            values=df[values_column],
            hole=0.5,
            marker=dict(colors=colors[:len(df)]),
            textinfo='percent+label',
            textposition='outside'
        )])

        # Add center annotation if provided
        if center_text:
            fig.add_annotation(
                text=center_text,
                x=0.5, y=0.5,
                font=dict(size=24, color=theme["text_color"]),
                showarrow=False
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=theme["text_color"])),
            paper_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"]),
            showlegend=True,
            legend=dict(font=dict(color=theme["text_color"]))
        )

        chart_id = f"donut_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
            "id": chart_id,
            "type": "donut_chart",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "donut_chart",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating donut chart: {str(e)}"


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

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        fig = px.histogram(df, x=column, title=title, nbins=nbins,
                          color_discrete_sequence=colors)

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"]),
            bargap=0.1
        )

        chart_id = f"hist_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
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

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        fig = px.scatter(df, x=x_column, y=y_column, title=title,
                        color=color_column if color_column else None,
                        size=size_column if size_column else None,
                        color_discrete_sequence=colors)

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"])
        )

        chart_id = f"scatter_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
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

        session = _viz_sessions[session_id]
        theme = THEMES.get(session.get("theme", "dark"))

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        # Pivot data for heatmap
        pivot_df = df.pivot(index=y_column, columns=x_column, values=value_column)

        colorscale = 'Viridis' if session.get("theme") == "dark" else 'Blues'

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale=colorscale
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=theme["text_color"])),
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"])
        )

        chart_id = f"heatmap_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
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
def create_treemap(session_id: str, title: str, sql: str, labels_column: str, values_column: str,
                   parent_column: Optional[str] = None) -> str:
    """
    Create a treemap visualization for hierarchical data.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        labels_column: Column for treemap labels
        values_column: Column for treemap values (sizes)
        parent_column: Optional column for hierarchy parent

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        if parent_column:
            fig = px.treemap(df, path=[parent_column, labels_column], values=values_column,
                            title=title, color_discrete_sequence=colors)
        else:
            fig = px.treemap(df, path=[labels_column], values=values_column,
                            title=title, color_discrete_sequence=colors)

        fig.update_layout(
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"])
        )

        chart_id = f"treemap_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
            "id": chart_id,
            "type": "treemap",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "treemap",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating treemap: {str(e)}"


@tool
def create_gauge_chart(session_id: str, title: str, sql: str, value_format: str = "number",
                       min_val: float = 0, max_val: Optional[float] = None) -> str:
    """
    Create a gauge chart for displaying a single metric.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query that returns a single value
        value_format: 'number', 'currency', 'percentage'
        min_val: Minimum value for gauge (default 0)
        max_val: Maximum value for gauge (auto-calculated if not provided)

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        theme = THEMES.get(session.get("theme", "dark"))

        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()
            value = df.iloc[0, 0] if len(df) > 0 else 0

            # Try to get min/max from query if it returns them
            if len(df.columns) >= 3:
                min_val = df.iloc[0, 1] if not pd.isna(df.iloc[0, 1]) else min_val
                max_val = df.iloc[0, 2] if not pd.isna(df.iloc[0, 2]) else max_val

        if max_val is None:
            max_val = value * 1.5 if value > 0 else 100

        formatted_value = _format_number(value, value_format)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'font': {'size': 36, 'color': theme["text_color"]}, 'suffix': '' if value_format == 'number' else ('$' if value_format == 'currency' else '%')},
            title={'text': title, 'font': {'size': 18, 'color': theme["text_color"]}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickcolor': theme["text_secondary"]},
                'bar': {'color': "#00F5D4"},
                'bgcolor': theme["card_bg"],
                'borderwidth': 2,
                'bordercolor': theme["border_color"],
                'steps': [
                    {'range': [min_val, max_val * 0.33], 'color': '#EF4444'},
                    {'range': [max_val * 0.33, max_val * 0.66], 'color': '#F59E0B'},
                    {'range': [max_val * 0.66, max_val], 'color': '#10B981'}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"]),
            height=300
        )

        chart_id = f"gauge_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
            "id": chart_id,
            "type": "gauge",
            "title": title,
            "figure": fig,
            "data": [{"value": value}]
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "gauge",
            "title": title,
            "value": _safe_json_serialize(value),
            "formatted_value": formatted_value
        }, indent=2)

    except Exception as e:
        return f"Error creating gauge chart: {str(e)}"


@tool
def create_stacked_bar_chart(session_id: str, title: str, sql: str, x_column: str, y_column: str,
                             color_column: str, orientation: str = "v") -> str:
    """
    Create a stacked bar chart visualization.

    Args:
        session_id: Visualization session ID
        title: Chart title
        sql: SQL query for data
        x_column: Column for X axis
        y_column: Column for Y axis (values)
        color_column: Column for stack segments
        orientation: 'v' for vertical, 'h' for horizontal

    Returns:
        Confirmation with chart ID
    """
    try:
        if not PLOTLY_AVAILABLE:
            return "Error: Plotly not installed."

        if session_id not in _viz_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _viz_sessions[session_id]
        with get_connection() as conn:
            df = conn.execute(sql).fetchdf()

        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))

        if orientation == "h":
            fig = px.bar(df, y=x_column, x=y_column, color=color_column,
                        orientation='h', title=title, color_discrete_sequence=colors,
                        barmode='stack')
        else:
            fig = px.bar(df, x=x_column, y=y_column, color=color_column,
                        title=title, color_discrete_sequence=colors,
                        barmode='stack')

        fig.update_layout(
            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
            title_font_size=18,
            title_font_color=theme["text_color"],
            paper_bgcolor=theme["chart_bg"],
            plot_bgcolor=theme["chart_bg"],
            font=dict(color=theme["text_color"]),
            legend=dict(font=dict(color=theme["text_color"]))
        )

        chart_id = f"stacked_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
            "id": chart_id,
            "type": "stacked_bar",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records")
        })

        return json.dumps({
            "status": "success",
            "chart_id": chart_id,
            "type": "stacked_bar",
            "title": title,
            "data_rows": len(df)
        }, indent=2)

    except Exception as e:
        return f"Error creating stacked bar chart: {str(e)}"


@tool
def create_kpi_card(session_id: str, title: str, sql: str, format_type: str = "number",
                    color: str = "blue", icon: str = "chart-line",
                    comparison_sql: Optional[str] = None) -> str:
    """
    Create a professional KPI card with optional trend comparison.

    Args:
        session_id: Visualization session ID
        title: KPI title
        sql: SQL query that returns a single value
        format_type: 'number', 'currency', 'percentage', 'compact'
        color: Card color ('green', 'blue', 'purple', 'orange', 'red', 'teal', 'pink', 'indigo')
        icon: Icon name for the card
        comparison_sql: Optional SQL for comparison value (for delta/trend)

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
            delta_value = None
            if comparison_sql:
                try:
                    comp_df = conn.execute(comparison_sql).fetchdf()
                    if len(comp_df) > 0:
                        comp_value = comp_df.iloc[0, 0]
                        if comp_value and comp_value != 0:
                            delta = ((value - comp_value) / comp_value) * 100
                            delta_value = value - comp_value
                except:
                    pass

        formatted_value = _format_number(value, format_type)

        if color not in KPI_COLORS:
            color = "blue"

        kpi_id = f"kpi_{len(_viz_sessions[session_id]['kpis']) + 1}"
        _viz_sessions[session_id]["kpis"].append({
            "id": kpi_id,
            "title": title,
            "value": _safe_json_serialize(value),
            "formatted_value": formatted_value,
            "delta": _safe_json_serialize(delta) if delta else None,
            "delta_value": _safe_json_serialize(delta_value) if delta_value else None,
            "format_type": format_type,
            "color": color,
            "icon": icon
        })

        return json.dumps({
            "status": "success",
            "kpi_id": kpi_id,
            "title": title,
            "value": _safe_json_serialize(value),
            "formatted_value": formatted_value,
            "delta": _safe_json_serialize(delta) if delta else None,
            "color": color
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

        session = _viz_sessions[session_id]
        theme = THEMES.get(session.get("theme", "dark"))

        with get_connection() as conn:
            df = conn.execute(f"{sql} LIMIT {max_rows}").fetchdf()

        # Style based on theme
        if session.get("theme") == "dark":
            header_fill = '#1E40AF'
            cell_fill = ['#1E293B', '#0F172A']
            header_font_color = 'white'
            cell_font_color = '#E2E8F0'
        else:
            header_fill = '#2E86AB'
            cell_fill = ['#F8FAFC', '#FFFFFF']
            header_font_color = 'white'
            cell_font_color = '#1E293B'

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in df.columns],
                fill_color=header_fill,
                font=dict(color=header_font_color, size=13),
                align='left',
                height=40
            ),
            cells=dict(
                values=[df[col].tolist() for col in df.columns],
                fill_color=[cell_fill * (len(df) // 2 + 1)][:len(df)],
                font=dict(color=cell_font_color, size=12),
                align='left',
                height=35
            )
        )])

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=theme["text_color"])),
            paper_bgcolor=theme["chart_bg"],
            margin=dict(l=20, r=20, t=60, b=20)
        )

        table_id = f"table_{len(session['visualizations']) + 1}"
        session["visualizations"].append({
            "id": table_id,
            "type": "data_table",
            "title": title,
            "figure": fig,
            "data": df.to_dict(orient="records"),
            "is_table": True  # Mark as table for sorting in dashboard
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
    Generate a professional HTML dashboard from all visualizations in the session.
    KPIs at top, charts in middle, data tables always at the end.

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
        theme_name = session.get("theme", "dark")
        theme = THEMES[theme_name]

        if not kpis and not visualizations:
            return "Error: No visualizations created. Create charts first."

        # Separate charts from tables - tables go at the end
        charts = [v for v in visualizations if not v.get("is_table", False)]
        tables = [v for v in visualizations if v.get("is_table", False)]

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', sans-serif;
            background: {theme['bg_color']};
            color: {theme['text_color']};
            min-height: 100vh;
            line-height: 1.6;
        }}
        .dashboard-container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Header Styles */
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }}
        .dashboard-header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
        }}
        .dashboard-header h1 {{
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }}
        .dashboard-header p {{
            opacity: 0.9;
            font-size: 1em;
            position: relative;
            z-index: 1;
        }}

        /* KPI Container */
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        /* KPI Card Styles */
        .kpi-card {{
            background: {theme['card_bg']};
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid {theme['border_color']};
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            border-radius: 16px 16px 0 0;
        }}
        .kpi-card.green::before {{ background: linear-gradient(90deg, #10B981, #34D399); }}
        .kpi-card.blue::before {{ background: linear-gradient(90deg, #3B82F6, #60A5FA); }}
        .kpi-card.purple::before {{ background: linear-gradient(90deg, #8B5CF6, #A78BFA); }}
        .kpi-card.orange::before {{ background: linear-gradient(90deg, #F59E0B, #FBBF24); }}
        .kpi-card.red::before {{ background: linear-gradient(90deg, #EF4444, #F87171); }}
        .kpi-card.teal::before {{ background: linear-gradient(90deg, #14B8A6, #2DD4BF); }}
        .kpi-card.pink::before {{ background: linear-gradient(90deg, #EC4899, #F472B6); }}
        .kpi-card.indigo::before {{ background: linear-gradient(90deg, #6366F1, #818CF8); }}

        .kpi-icon {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3em;
            opacity: 0.8;
        }}
        .kpi-card.green .kpi-icon {{ background: rgba(16, 185, 129, 0.2); color: #10B981; }}
        .kpi-card.blue .kpi-icon {{ background: rgba(59, 130, 246, 0.2); color: #3B82F6; }}
        .kpi-card.purple .kpi-icon {{ background: rgba(139, 92, 246, 0.2); color: #8B5CF6; }}
        .kpi-card.orange .kpi-icon {{ background: rgba(245, 158, 11, 0.2); color: #F59E0B; }}
        .kpi-card.red .kpi-icon {{ background: rgba(239, 68, 68, 0.2); color: #EF4444; }}
        .kpi-card.teal .kpi-icon {{ background: rgba(20, 184, 166, 0.2); color: #14B8A6; }}
        .kpi-card.pink .kpi-icon {{ background: rgba(236, 72, 153, 0.2); color: #EC4899; }}
        .kpi-card.indigo .kpi-icon {{ background: rgba(99, 102, 241, 0.2); color: #6366F1; }}

        .kpi-title {{
            color: {theme['text_secondary']};
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        .kpi-value {{
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 8px;
            line-height: 1.2;
        }}
        .kpi-card.green .kpi-value {{ color: #10B981; }}
        .kpi-card.blue .kpi-value {{ color: #3B82F6; }}
        .kpi-card.purple .kpi-value {{ color: #8B5CF6; }}
        .kpi-card.orange .kpi-value {{ color: #F59E0B; }}
        .kpi-card.red .kpi-value {{ color: #EF4444; }}
        .kpi-card.teal .kpi-value {{ color: #14B8A6; }}
        .kpi-card.pink .kpi-value {{ color: #EC4899; }}
        .kpi-card.indigo .kpi-value {{ color: #6366F1; }}

        .kpi-delta {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.9em;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 20px;
        }}
        .kpi-delta.positive {{
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
        }}
        .kpi-delta.negative {{
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
        }}

        /* Section Headers */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 30px 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid {theme['border_color']};
        }}
        .section-header h2 {{
            font-size: 1.3em;
            font-weight: 600;
            color: {theme['text_color']};
        }}
        .section-header i {{
            color: #667eea;
            font-size: 1.2em;
        }}

        /* Charts Container */
        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: {theme['card_bg']};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid {theme['border_color']};
            transition: box-shadow 0.3s ease;
        }}
        .chart-card:hover {{
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        }}
        .chart-card.full-width {{
            grid-column: 1 / -1;
        }}

        /* Tables Section */
        .tables-section {{
            margin-top: 40px;
        }}
        .table-card {{
            background: {theme['card_bg']};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid {theme['border_color']};
            overflow-x: auto;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px 20px;
            color: {theme['text_secondary']};
            font-size: 0.9em;
            margin-top: 40px;
            border-top: 1px solid {theme['border_color']};
        }}
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .charts-container {{
                grid-template-columns: 1fr;
            }}
            .kpi-container {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .dashboard-header h1 {{
                font-size: 1.5em;
            }}
        }}

        /* Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        .kpi-card, .chart-card, .table-card {{
            animation: fadeInUp 0.5s ease-out forwards;
        }}
        .kpi-card:nth-child(1) {{ animation-delay: 0.1s; }}
        .kpi-card:nth-child(2) {{ animation-delay: 0.15s; }}
        .kpi-card:nth-child(3) {{ animation-delay: 0.2s; }}
        .kpi-card:nth-child(4) {{ animation-delay: 0.25s; }}
        .kpi-card:nth-child(5) {{ animation-delay: 0.3s; }}
        .kpi-card:nth-child(6) {{ animation-delay: 0.35s; }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1><i class="fas fa-chart-line"></i> {title}</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} | Data Source: {session['table_name']}</p>
        </div>
"""

        # Add KPI cards
        if kpis:
            html_content += '''
        <div class="section-header">
            <i class="fas fa-tachometer-alt"></i>
            <h2>Key Performance Indicators</h2>
        </div>
        <div class="kpi-container">
'''
            icon_map = {
                'chart-line': 'fa-chart-line',
                'database': 'fa-database',
                'users': 'fa-users',
                'dollar': 'fa-dollar-sign',
                'percent': 'fa-percent',
                'trending': 'fa-arrow-trend-up'
            }

            for kpi in kpis:
                color = kpi.get('color', 'blue')
                icon = icon_map.get(kpi.get('icon', 'chart-line'), 'fa-chart-line')

                delta_html = ""
                if kpi.get("delta") is not None:
                    delta_class = "positive" if kpi["delta"] > 0 else "negative"
                    delta_icon = "fa-arrow-up" if kpi["delta"] > 0 else "fa-arrow-down"
                    delta_html = f'<div class="kpi-delta {delta_class}"><i class="fas {delta_icon}"></i> {abs(kpi["delta"]):.1f}%</div>'

                html_content += f"""
            <div class="kpi-card {color}">
                <div class="kpi-icon"><i class="fas {icon}"></i></div>
                <div class="kpi-title">{kpi['title']}</div>
                <div class="kpi-value">{kpi['formatted_value']}</div>
                {delta_html}
            </div>
"""
            html_content += '        </div>\n'

        # Add charts (excluding tables)
        if charts:
            html_content += '''
        <div class="section-header">
            <i class="fas fa-chart-pie"></i>
            <h2>Analytics & Insights</h2>
        </div>
        <div class="charts-container">
'''
            for i, viz in enumerate(charts):
                fig = viz.get("figure")
                if fig:
                    fig.update_layout(
                        margin=dict(l=40, r=40, t=60, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    html_content += f"""
            <div class="chart-card">
                {chart_html}
            </div>
"""
            html_content += '        </div>\n'

        # Add data tables at the END
        if tables:
            html_content += '''
        <div class="tables-section">
            <div class="section-header">
                <i class="fas fa-table"></i>
                <h2>Data Details</h2>
            </div>
'''
            for viz in tables:
                fig = viz.get("figure")
                if fig:
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=60, b=20),
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    table_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    html_content += f"""
            <div class="table-card">
                {table_html}
            </div>
"""
            html_content += '        </div>\n'

        # Footer
        html_content += f"""
        <div class="footer">
            <p>Dashboard generated by <strong>Data Visualization Agent</strong> | Session: {session_id}</p>
            <p style="margin-top: 8px; font-size: 0.85em;">Powered by Plotly & Python</p>
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
            "theme": theme_name,
            "kpi_count": len(kpis),
            "chart_count": len(charts),
            "table_count": len(tables),
            "message": f"Professional dashboard generated successfully! Open the HTML file in a browser to view.",
            "open_command": f"start {abs_path}" if os.name == 'nt' else f"open {abs_path}"
        }, indent=2)

    except Exception as e:
        return f"Error generating dashboard: {str(e)}"


@tool
def generate_dashboard_from_plan(session_id: str, output_filename: Optional[str] = None) -> str:
    """
    Automatically generate a complete professional dashboard from the visualization plan.
    This executes all planned visualizations and creates the final dashboard with
    KPIs at top, charts in middle, and data tables at the end.

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

        results = {"kpis_created": 0, "charts_created": 0, "tables_created": 0, "errors": []}
        colors = _get_color_palette(session.get("color_palette", "dark_friendly"))
        theme = THEMES.get(session.get("theme", "dark"))
        kpi_colors = list(KPI_COLORS.keys())

        with get_connection() as conn:
            # Create KPIs
            for i, kpi in enumerate(plan.get("kpis", [])):
                try:
                    df = conn.execute(kpi["sql"]).fetchdf()
                    value = df.iloc[0, 0] if len(df) > 0 else 0
                    format_type = kpi.get("format", "number")
                    formatted = _format_number(value, format_type)

                    session["kpis"].append({
                        "id": kpi["id"],
                        "title": kpi["title"],
                        "value": _safe_json_serialize(value),
                        "formatted_value": formatted,
                        "delta": None,
                        "format_type": format_type,
                        "color": kpi.get("color", kpi_colors[i % len(kpi_colors)]),
                        "icon": kpi.get("icon", "chart-line")
                    })
                    results["kpis_created"] += 1
                except Exception as e:
                    results["errors"].append(f"KPI {kpi['id']}: {str(e)}")

            # Create visualizations
            for viz in plan.get("visualizations", []):
                try:
                    df = conn.execute(viz["sql"]).fetchdf()
                    fig = None

                    if viz["type"] == "bar_chart":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.bar(df, x=x_col, y=y_col, title=viz["title"],
                                    color_discrete_sequence=colors)

                    elif viz["type"] == "horizontal_bar":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.bar(df, y=y_col, x=x_col, orientation='h',
                                    title=viz["title"], color_discrete_sequence=colors)

                    elif viz["type"] in ["pie_chart", "donut_chart"]:
                        hole = 0.4 if viz["type"] == "donut_chart" else 0
                        fig = px.pie(df, names=df.columns[0], values=df.columns[1],
                                    title=viz["title"], color_discrete_sequence=colors, hole=hole)
                        fig.update_traces(textposition='inside', textinfo='percent+label')

                    elif viz["type"] == "line_chart":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.line(df, x=x_col, y=y_col, title=viz["title"],
                                     color_discrete_sequence=colors, markers=True)

                    elif viz["type"] == "area_chart":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.area(df, x=x_col, y=y_col, title=viz["title"],
                                     color_discrete_sequence=colors)

                    elif viz["type"] == "histogram":
                        col = viz.get("column")
                        fig = px.histogram(df, x=col, title=viz["title"],
                                          color_discrete_sequence=colors)

                    elif viz["type"] == "scatter_plot":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        fig = px.scatter(df, x=x_col, y=y_col, title=viz["title"],
                                        color_discrete_sequence=colors)

                    elif viz["type"] == "stacked_bar":
                        x_col = viz.get("x_column")
                        y_col = viz.get("y_column")
                        color_col = viz.get("color_column")
                        fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                                    title=viz["title"], color_discrete_sequence=colors,
                                    barmode='stack')

                    elif viz["type"] == "treemap":
                        labels_col = viz.get("labels_column")
                        values_col = viz.get("values_column")
                        fig = px.treemap(df, path=[labels_col], values=values_col,
                                        title=viz["title"], color_discrete_sequence=colors)

                    elif viz["type"] == "gauge":
                        col = viz.get("column")
                        value = df.iloc[0, 0] if len(df) > 0 else 0
                        max_val = df.iloc[0, 2] if len(df.columns) >= 3 and len(df) > 0 else value * 1.5
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=value,
                            title={'text': viz["title"]},
                            gauge={
                                'axis': {'range': [0, max_val]},
                                'bar': {'color': "#00F5D4"},
                                'steps': [
                                    {'range': [0, max_val * 0.33], 'color': '#EF4444'},
                                    {'range': [max_val * 0.33, max_val * 0.66], 'color': '#F59E0B'},
                                    {'range': [max_val * 0.66, max_val], 'color': '#10B981'}
                                ],
                            }
                        ))

                    if fig:
                        fig.update_layout(
                            template="plotly_dark" if session.get("theme") == "dark" else "plotly_white",
                            title_font_size=18,
                            title_font_color=theme["text_color"],
                            paper_bgcolor=theme["chart_bg"],
                            plot_bgcolor=theme["chart_bg"],
                            font=dict(color=theme["text_color"])
                        )

                        session["visualizations"].append({
                            "id": viz["id"],
                            "type": viz["type"],
                            "title": viz["title"],
                            "figure": fig,
                            "data": df.to_dict(orient="records"),
                            "is_table": False
                        })
                        results["charts_created"] += 1

                except Exception as e:
                    results["errors"].append(f"Chart {viz['id']}: {str(e)}")

            # Create data table (always at end)
            if plan.get("data_table") and plan["data_table"].get("sql"):
                try:
                    df = conn.execute(plan["data_table"]["sql"]).fetchdf()

                    if session.get("theme") == "dark":
                        header_fill = '#1E40AF'
                        cell_fill = ['#1E293B', '#0F172A']
                        header_font_color = 'white'
                        cell_font_color = '#E2E8F0'
                    else:
                        header_fill = '#2E86AB'
                        cell_fill = ['#F8FAFC', '#FFFFFF']
                        header_font_color = 'white'
                        cell_font_color = '#1E293B'

                    fig = go.Figure(data=[go.Table(
                        header=dict(
                            values=[f"<b>{col}</b>" for col in df.columns],
                            fill_color=header_fill,
                            font=dict(color=header_font_color, size=13),
                            align='left',
                            height=40
                        ),
                        cells=dict(
                            values=[df[col].tolist() for col in df.columns],
                            fill_color=[cell_fill * (len(df) // 2 + 1)][:len(df)],
                            font=dict(color=cell_font_color, size=12),
                            align='left',
                            height=35
                        )
                    )])

                    fig.update_layout(
                        title=dict(text="Data Details", font=dict(size=18, color=theme["text_color"])),
                        paper_bgcolor=theme["chart_bg"],
                        margin=dict(l=20, r=20, t=60, b=20)
                    )

                    session["visualizations"].append({
                        "id": "data_table",
                        "type": "data_table",
                        "title": "Data Details",
                        "figure": fig,
                        "data": df.to_dict(orient="records"),
                        "is_table": True
                    })
                    results["tables_created"] += 1

                except Exception as e:
                    results["errors"].append(f"Data table: {str(e)}")

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
            "theme": session.get("theme", "dark"),
            "color_palette": session.get("color_palette"),
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
                {"id": k["id"], "title": k["title"], "value": k.get("formatted_value"), "color": k.get("color")}
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
                "theme": session.get("theme", "dark"),
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
tool_registry.register(set_dashboard_theme, "dataviz")

# SQL and data collection tools
tool_registry.register(execute_viz_query, "dataviz")
tool_registry.register(collect_all_viz_data, "dataviz")

# Visualization creation tools
tool_registry.register(create_bar_chart, "dataviz")
tool_registry.register(create_line_chart, "dataviz")
tool_registry.register(create_area_chart, "dataviz")
tool_registry.register(create_pie_chart, "dataviz")
tool_registry.register(create_donut_chart, "dataviz")
tool_registry.register(create_histogram, "dataviz")
tool_registry.register(create_scatter_plot, "dataviz")
tool_registry.register(create_heatmap, "dataviz")
tool_registry.register(create_treemap, "dataviz")
tool_registry.register(create_gauge_chart, "dataviz")
tool_registry.register(create_stacked_bar_chart, "dataviz")
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
