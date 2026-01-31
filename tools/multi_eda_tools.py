"""
Multi-EDA Tools Module
======================

35 tools for the Multi-EDA Agent workflow covering:
- Data loading & schema
- Join & target variable detection
- Structure inspection
- Descriptive statistics
- Distribution visualization
- Segmentation visualization
- Outlier detection
- Correlation analysis
- Deep analysis summaries
- Dashboard generation
- Session management
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import os
import json
import uuid
import base64
from datetime import datetime

from core.tools_base import tool_registry

# Database configuration
DB_PATH = "agent_ddb.db"

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
    raise ImportError("`pandas` and `numpy` not installed. Please install using `pip install pandas numpy`.")

# Import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Optional scipy for advanced stats
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Output directories
VIZ_OUTPUT_DIR = "sample_files/eda_visualizations"
DASHBOARD_OUTPUT_DIR = "sample_files/eda_dashboards"
os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
os.makedirs(DASHBOARD_OUTPUT_DIR, exist_ok=True)

# In-memory session store
_multi_eda_sessions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

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


# =============================================================================
# SESSION MANAGEMENT HELPERS
# =============================================================================

def _create_session(table_name: str) -> str:
    """Create a new multi-EDA session."""
    session_id = str(uuid.uuid4())[:8]
    _multi_eda_sessions[session_id] = {
        "table_name": table_name,
        "tables_loaded": {},
        "dataframe": None,
        "created_at": datetime.now().isoformat(),
        "analyses": {},
        "plots": [],
        "summaries": {},
    }
    return session_id


def _get_session(session_id: str) -> Dict[str, Any]:
    """Get session or raise error."""
    if session_id not in _multi_eda_sessions:
        raise ValueError(f"Session '{session_id}' not found. Use eda_load_table first.")
    return _multi_eda_sessions[session_id]


def _get_df(session_id: str) -> pd.DataFrame:
    """Get the DataFrame from a session."""
    session = _get_session(session_id)
    df = session.get("dataframe")
    if df is None:
        raise ValueError(f"No DataFrame loaded in session '{session_id}'.")
    return df


def _save_plot(session_id: str, agent_name: str, chart_type: str, fig=None) -> str:
    """Save a matplotlib figure and return the file path."""
    if fig is None:
        fig = plt.gcf()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{session_id}_{agent_name}_{chart_type}_{timestamp}.png"
    filepath = os.path.join(VIZ_OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return filepath


# =============================================================================
# TOOL 1-3: DATA LOADING TOOLS
# =============================================================================

@tool
def eda_load_table(table_name: str, row_limit: int = 50000) -> str:
    """
    Load a table from DuckDB into pandas for EDA analysis.
    Returns a session_id for subsequent analysis tools.

    Args:
        table_name: Name of the table to load (e.g., 'TRANSACTIONS')
        row_limit: Maximum rows to load (default 50000)

    Returns:
        Session info with session_id, row count, and column details
    """
    try:
        with get_connection() as conn:
            # Get total count
            count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            total_rows = count_result[0] if count_result else 0

            # Load data
            df = conn.execute(f"SELECT * FROM {table_name} LIMIT {row_limit}").fetchdf()

            session_id = _create_session(table_name)
            session = _multi_eda_sessions[session_id]
            session["dataframe"] = df
            session["tables_loaded"][table_name] = {
                "total_rows": total_rows,
                "loaded_rows": len(df),
                "columns": list(df.columns),
                "col_count": len(df.columns)
            }

            cols_info = []
            for col in df.columns:
                cols_info.append(f"  - {col}: {df[col].dtype}")

            return (
                f"SESSION_ID: {session_id}\n"
                f"Table: {table_name}\n"
                f"Total rows in table: {total_rows}\n"
                f"Rows loaded: {len(df)}\n"
                f"Columns ({len(df.columns)}):\n" + "\n".join(cols_info)
            )
    except Exception as e:
        return f"Error loading table: {str(e)}"


@tool
def eda_get_table_row_count(table_name: str) -> str:
    """
    Get the total row count for a table via SQL COUNT(*).

    Args:
        table_name: Name of the table

    Returns:
        Row count information
    """
    try:
        with get_connection() as conn:
            result = conn.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()
            return f"Table {table_name}: {result[0]} rows"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_load_multiple_tables(table_names: str, row_limit: int = 50000) -> str:
    """
    Load multiple tables from DuckDB. Comma-separated table names.
    Returns a session_id with all tables loaded separately.

    Args:
        table_names: Comma-separated table names (e.g., 'CLIENTS,PORTFOLIOS')
        row_limit: Maximum rows per table (default 50000)

    Returns:
        Session info with all loaded tables
    """
    try:
        tables = [t.strip() for t in table_names.split(",")]
        session_id = _create_session(tables[0])
        session = _multi_eda_sessions[session_id]

        results = []
        with get_connection() as conn:
            for tbl in tables:
                count_result = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
                total_rows = count_result[0] if count_result else 0
                df = conn.execute(f"SELECT * FROM {tbl} LIMIT {row_limit}").fetchdf()

                session["tables_loaded"][tbl] = {
                    "total_rows": total_rows,
                    "loaded_rows": len(df),
                    "columns": list(df.columns),
                    "col_count": len(df.columns),
                    "dataframe": df
                }
                results.append(f"  {tbl}: {len(df)} rows, {len(df.columns)} columns (total: {total_rows})")

        return (
            f"SESSION_ID: {session_id}\n"
            f"Tables loaded ({len(tables)}):\n" + "\n".join(results)
        )
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 4-6: SCHEMA TOOLS
# =============================================================================

@tool
def eda_get_schema_from_json(json_path: str) -> str:
    """
    Load schema definition from a user-provided JSON file.

    Args:
        json_path: Path to the schema JSON file

    Returns:
        Schema information extracted from the JSON
    """
    try:
        with open(json_path, 'r') as f:
            schema = json.load(f)

        lines = ["SCHEMA FROM JSON FILE:", f"Path: {json_path}", ""]
        tables = schema.get("tables", [])
        for table in tables:
            lines.append(f"TABLE: {table.get('table_name', 'unknown')}")
            lines.append(f"  Description: {table.get('table_description', '')}")
            for col in table.get("columns", []):
                pk = " [PK]" if col.get("is_primary_key") else ""
                lines.append(f"  - {col['column_name']} ({col.get('data_type', 'unknown')}){pk}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error loading schema JSON: {str(e)}"


@tool
def eda_get_table_schema(table_name: str) -> str:
    """
    Get table schema from DuckDB including column names, types, and sample data.

    Args:
        table_name: Name of the table

    Returns:
        Schema with column info and sample data
    """
    try:
        with get_connection() as conn:
            desc = conn.execute(f"DESCRIBE {table_name}").fetchdf()
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()

            lines = [f"TABLE: {table_name}", "", "COLUMNS:"]
            for _, row in desc.iterrows():
                lines.append(f"  - {row['column_name']}: {row['column_type']} (null={row['null']})")

            lines.append("")
            lines.append("SAMPLE DATA (3 rows):")
            lines.append(sample.to_string(index=False))

            return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_get_all_tables() -> str:
    """
    List all available tables in the DuckDB database with row counts.

    Returns:
        List of tables with their row counts
    """
    try:
        with get_connection() as conn:
            tables_result = conn.execute("SHOW TABLES").fetchdf()
            lines = ["AVAILABLE TABLES:", ""]
            for _, row in tables_result.iterrows():
                tbl = row.iloc[0]
                cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                cols = conn.execute(f"DESCRIBE {tbl}").fetchdf()
                lines.append(f"  {tbl}: {cnt} rows, {len(cols)} columns")
            return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 7-10: JOIN & TARGET VARIABLE TOOLS
# =============================================================================

@tool
def eda_detect_target_variable(session_id: str) -> str:
    """
    Analyze columns to suggest a target variable for EDA.
    Looks at column names, types, and unique value counts to identify likely targets.

    Args:
        session_id: EDA session ID

    Returns:
        Target variable suggestions with reasoning
    """
    try:
        df = _get_df(session_id)
        suggestions = []

        for col in df.columns:
            score = 0
            reasons = []
            nunique = df[col].nunique()
            dtype = str(df[col].dtype)

            # Low cardinality categorical = likely target
            if nunique <= 10 and nunique >= 2:
                if dtype == 'object' or 'category' in dtype:
                    score += 3
                    reasons.append(f"categorical with {nunique} unique values")
                elif 'int' in dtype:
                    score += 2
                    reasons.append(f"integer with {nunique} unique values (possible label)")

            # Column name hints
            name_lower = col.lower()
            target_keywords = ['type', 'status', 'category', 'class', 'label', 'target',
                               'outcome', 'result', 'flag', 'is_', 'has_']
            for kw in target_keywords:
                if kw in name_lower:
                    score += 2
                    reasons.append(f"name contains '{kw}'")
                    break

            # Binary columns
            if nunique == 2:
                score += 1
                reasons.append("binary column")

            if score >= 2:
                suggestions.append({
                    "column": col,
                    "score": score,
                    "reasons": reasons,
                    "unique_values": nunique,
                    "dtype": dtype
                })

        suggestions.sort(key=lambda x: x["score"], reverse=True)

        if not suggestions:
            return "NO_TARGET_FOUND: No obvious target variable detected. The dataset may be unsupervised."

        lines = ["TARGET VARIABLE SUGGESTIONS:", ""]
        for i, s in enumerate(suggestions[:5], 1):
            lines.append(f"  {i}. {s['column']} (score={s['score']})")
            lines.append(f"     Type: {s['dtype']}, Unique: {s['unique_values']}")
            lines.append(f"     Reasons: {', '.join(s['reasons'])}")

        # Store in session
        session = _get_session(session_id)
        session["analyses"]["target_suggestions"] = suggestions

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_detect_joins(session_id: str, table_names: str) -> str:
    """
    Detect possible join columns between loaded tables by matching column names.

    Args:
        session_id: EDA session ID
        table_names: Comma-separated table names to check joins between

    Returns:
        Detected join columns and relationships
    """
    try:
        session = _get_session(session_id)
        tables = [t.strip() for t in table_names.split(",")]

        joins = []
        for i, t1 in enumerate(tables):
            for t2 in tables[i+1:]:
                t1_info = session["tables_loaded"].get(t1, {})
                t2_info = session["tables_loaded"].get(t2, {})
                t1_cols = set(t1_info.get("columns", []))
                t2_cols = set(t2_info.get("columns", []))

                # Find common column names
                common = t1_cols & t2_cols
                if common:
                    for col in common:
                        joins.append({
                            "table1": t1,
                            "table2": t2,
                            "column": col,
                            "join_sql": f"{t1}.{col} = {t2}.{col}"
                        })

                # Check for ID pattern matches (e.g., CLIENT_ID in PORTFOLIOS -> CLIENTS.CLIENT_ID)
                for c1 in t1_cols:
                    if c1.endswith("_ID"):
                        prefix = c1.replace("_ID", "")
                        if prefix + "S" == t2 or prefix == t2:
                            if c1 in t2_cols:
                                join_entry = {"table1": t1, "table2": t2, "column": c1,
                                              "join_sql": f"{t1}.{c1} = {t2}.{c1}"}
                                if join_entry not in joins:
                                    joins.append(join_entry)

        if not joins:
            return "NO_JOINS_FOUND: No matching join columns detected between tables."

        lines = ["DETECTED JOINS:", ""]
        for j in joins:
            lines.append(f"  {j['table1']} <-> {j['table2']} ON {j['join_sql']}")

        session["analyses"]["detected_joins"] = joins
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_join_tables(session_id: str, join_sql: str) -> str:
    """
    Execute a JOIN query and store the merged DataFrame in the session.

    Args:
        session_id: EDA session ID
        join_sql: Full SQL JOIN query (e.g., 'SELECT * FROM CLIENTS c JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID')

    Returns:
        Information about the joined DataFrame
    """
    try:
        session = _get_session(session_id)
        with get_connection() as conn:
            df = conn.execute(join_sql).fetchdf()

        session["dataframe"] = df
        session["analyses"]["join_sql"] = join_sql

        return (
            f"JOIN RESULT:\n"
            f"  Rows: {len(df)}\n"
            f"  Columns: {len(df.columns)}\n"
            f"  Column list: {', '.join(df.columns)}"
        )
    except Exception as e:
        return f"Error executing join: {str(e)}"


@tool
def eda_validate_target(session_id: str, target_col: str) -> str:
    """
    Validate that a target variable exists and is suitable for analysis.

    Args:
        session_id: EDA session ID
        target_col: Column name to validate as target

    Returns:
        Validation result with target variable statistics
    """
    try:
        df = _get_df(session_id)
        if target_col not in df.columns:
            return f"INVALID: Column '{target_col}' not found. Available: {', '.join(df.columns)}"

        col = df[target_col]
        nunique = col.nunique()
        value_counts = col.value_counts().head(10)

        lines = [
            f"TARGET VALIDATED: {target_col}",
            f"  Type: {col.dtype}",
            f"  Unique values: {nunique}",
            f"  Missing: {col.isna().sum()} ({col.isna().mean()*100:.1f}%)",
            f"  Top values:"
        ]
        for val, cnt in value_counts.items():
            pct = cnt / len(df) * 100
            lines.append(f"    {val}: {cnt} ({pct:.1f}%)")

        session = _get_session(session_id)
        session["analyses"]["target_variable"] = target_col
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 11-14: STRUCTURE INSPECTION TOOLS
# =============================================================================

@tool
def eda_get_shape(session_id: str) -> str:
    """
    Get the shape (rows, columns) of the DataFrame.

    Args:
        session_id: EDA session ID

    Returns:
        DataFrame shape information
    """
    try:
        df = _get_df(session_id)
        return f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_get_dtypes(session_id: str) -> str:
    """
    Get data types for all columns with classification (numeric/categorical/datetime).

    Args:
        session_id: EDA session ID

    Returns:
        Column types with classifications
    """
    try:
        df = _get_df(session_id)
        lines = ["DATA TYPES:", ""]

        numeric = []
        categorical = []
        datetime_cols = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric.append(col)
                lines.append(f"  {col}: {dtype} [NUMERIC]")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                lines.append(f"  {col}: {dtype} [DATETIME]")
            else:
                categorical.append(col)
                lines.append(f"  {col}: {dtype} [CATEGORICAL]")

        lines.append("")
        lines.append(f"Numeric: {len(numeric)} | Categorical: {len(categorical)} | Datetime: {len(datetime_cols)}")

        session = _get_session(session_id)
        session["analyses"]["column_types"] = {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols
        }
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_get_head(session_id: str, n: int = 10) -> str:
    """
    Get the first n rows of the DataFrame.

    Args:
        session_id: EDA session ID
        n: Number of rows to show (default 10)

    Returns:
        First n rows as formatted text
    """
    try:
        df = _get_df(session_id)
        return f"HEAD ({n} rows):\n{df.head(n).to_string()}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_classify_columns(session_id: str) -> str:
    """
    Classify all columns as numeric, categorical, or datetime.
    Stores classification in session for use by other tools.

    Args:
        session_id: EDA session ID

    Returns:
        Column classification summary
    """
    try:
        df = _get_df(session_id)
        numeric = []
        categorical = []
        datetime_cols = []

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric.append(col)
            else:
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    datetime_cols.append(col)
                except (ValueError, TypeError):
                    categorical.append(col)

        classification = {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols
        }

        session = _get_session(session_id)
        session["analyses"]["column_types"] = classification

        lines = [
            "COLUMN CLASSIFICATION:",
            f"  Numeric ({len(numeric)}): {', '.join(numeric) if numeric else 'None'}",
            f"  Categorical ({len(categorical)}): {', '.join(categorical) if categorical else 'None'}",
            f"  Datetime ({len(datetime_cols)}): {', '.join(datetime_cols) if datetime_cols else 'None'}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 15-17: DESCRIPTIVE STATISTICS TOOLS
# =============================================================================

@tool
def eda_describe_numerical(session_id: str) -> str:
    """
    Generate descriptive statistics for all numerical columns.
    Includes count, mean, std, min, quartiles, max, skew, kurtosis.

    Args:
        session_id: EDA session ID

    Returns:
        Descriptive statistics table for numerical columns
    """
    try:
        df = _get_df(session_id)
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return "No numerical columns found."

        desc = numeric_df.describe().T
        desc["skew"] = numeric_df.skew()
        desc["kurtosis"] = numeric_df.kurtosis()
        desc["missing"] = numeric_df.isna().sum()
        desc["missing_pct"] = (numeric_df.isna().mean() * 100).round(2)

        session = _get_session(session_id)
        session["analyses"]["numerical_stats"] = desc.to_dict()
        session["summaries"]["numerical_stats"] = desc.to_string()

        return f"NUMERICAL STATISTICS:\n\n{desc.to_string()}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_describe_categorical(session_id: str) -> str:
    """
    Generate descriptive statistics for all categorical columns.
    Includes unique count, top values, and frequencies.

    Args:
        session_id: EDA session ID

    Returns:
        Descriptive statistics for categorical columns
    """
    try:
        df = _get_df(session_id)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not cat_cols:
            return "No categorical columns found."

        lines = ["CATEGORICAL STATISTICS:", ""]
        stats_data = {}

        for col in cat_cols:
            vc = df[col].value_counts()
            nunique = df[col].nunique()
            missing = df[col].isna().sum()
            top_val = vc.index[0] if len(vc) > 0 else "N/A"
            top_freq = vc.iloc[0] if len(vc) > 0 else 0

            stats_data[col] = {
                "unique": nunique,
                "missing": missing,
                "top": str(top_val),
                "top_freq": int(top_freq)
            }

            lines.append(f"  {col}:")
            lines.append(f"    Unique: {nunique}, Missing: {missing} ({missing/len(df)*100:.1f}%)")
            lines.append(f"    Top: '{top_val}' ({top_freq} occurrences)")
            # Show sparse classes
            sparse = vc[vc / len(df) < 0.05]
            if len(sparse) > 0:
                lines.append(f"    Sparse classes (<5%): {len(sparse)}")
            lines.append("")

        session = _get_session(session_id)
        session["analyses"]["categorical_stats"] = stats_data
        session["summaries"]["categorical_stats"] = "\n".join(lines)

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_generate_stats_table_html(session_id: str) -> str:
    """
    Generate HTML tables for numerical and categorical statistics
    suitable for embedding in the dashboard.

    Args:
        session_id: EDA session ID

    Returns:
        HTML table strings for dashboard embedding
    """
    try:
        df = _get_df(session_id)
        html_parts = []

        # Numerical stats table
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            desc = numeric_df.describe().T
            desc["skew"] = numeric_df.skew()
            desc["kurtosis"] = numeric_df.kurtosis()
            html_parts.append("<h3>Numerical Statistics</h3>")
            html_parts.append(desc.to_html(classes="data-table", float_format="%.2f"))

        # Categorical stats table
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cat_data = []
            for col in cat_cols:
                vc = df[col].value_counts()
                cat_data.append({
                    "Column": col,
                    "Unique": df[col].nunique(),
                    "Missing": df[col].isna().sum(),
                    "Top Value": str(vc.index[0]) if len(vc) > 0 else "N/A",
                    "Top Freq": int(vc.iloc[0]) if len(vc) > 0 else 0
                })
            cat_df = pd.DataFrame(cat_data)
            html_parts.append("<h3>Categorical Statistics</h3>")
            html_parts.append(cat_df.to_html(classes="data-table", index=False))

        html = "\n".join(html_parts)
        session = _get_session(session_id)
        session["analyses"]["stats_html"] = html
        return f"HTML_GENERATED: {len(html)} characters of stats tables"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 18-21: DISTRIBUTION VISUALIZATION TOOLS
# =============================================================================

@tool
def eda_plot_all_histograms(session_id: str) -> str:
    """
    Create a grid of histograms with KDE overlay for ALL numerical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Path to the saved histogram grid image
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return "No numerical columns for histograms."

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        fig.patch.set_facecolor('#1a1a2e')
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            ax.set_facecolor('#16213e')
            data = df[col].dropna()
            ax.hist(data, bins=30, color='#0f3460', edgecolor='#e94560', alpha=0.7)
            try:
                data.plot.kde(ax=ax, color='#e94560', linewidth=2)
            except Exception:
                pass
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='#00ff87', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='#ffd700', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.set_title(col, color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.legend(fontsize=7, facecolor='#16213e', labelcolor='white')

        # Hide unused axes
        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Distribution of All Numerical Columns", color='white', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = _save_plot(session_id, "agent7", "all_histograms", fig)

        session = _get_session(session_id)
        session["plots"].append({"type": "all_histograms", "path": path, "agent": "agent7.1"})
        return f"SAVED: {path}\nPlotted {len(numeric_cols)} histograms with KDE overlay."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_individual_histogram(session_id: str, column: str) -> str:
    """
    Create a detailed histogram for a single numerical column.

    Args:
        session_id: EDA session ID
        column: Column name to plot

    Returns:
        Path to saved histogram image
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        if column not in df.columns:
            return f"Column '{column}' not found."

        data = df[column].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        ax.hist(data, bins=40, color='#0f3460', edgecolor='#e94560', alpha=0.7)
        try:
            data.plot.kde(ax=ax, color='#e94560', linewidth=2, secondary_y=True)
        except Exception:
            pass

        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='#00ff87', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#ffd700', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

        ax.set_title(f"Distribution of {column}", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel(column, color='white')
        ax.set_ylabel("Frequency", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#16213e', labelcolor='white')

        path = _save_plot(session_id, "agent7", f"hist_{column}", fig)
        session = _get_session(session_id)
        session["plots"].append({"type": "individual_histogram", "column": column, "path": path, "agent": "agent7.2"})
        return f"SAVED: {path}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_all_individual_histograms(session_id: str) -> str:
    """
    Create individual detailed histograms for ALL numerical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Paths to all saved histogram images
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return "No numerical columns."

        paths = []
        for col in numeric_cols:
            data = df[col].dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')

            ax.hist(data, bins=40, color='#0f3460', edgecolor='#e94560', alpha=0.7)
            try:
                data.plot.kde(ax=ax, color='#e94560', linewidth=2)
            except Exception:
                pass
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='#00ff87', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='#ffd700', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.set_title(f"Distribution: {col}", color='white', fontsize=12, fontweight='bold')
            ax.tick_params(colors='white')
            ax.legend(fontsize=8, facecolor='#16213e', labelcolor='white')

            path = _save_plot(session_id, "agent7", f"indiv_hist_{col}", fig)
            paths.append(path)

        session = _get_session(session_id)
        for p in paths:
            session["plots"].append({"type": "individual_histogram", "path": p, "agent": "agent7.2"})

        return f"SAVED {len(paths)} individual histograms:\n" + "\n".join(paths)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_countplots(session_id: str) -> str:
    """
    Create countplots (bar charts) for ALL categorical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Paths to saved countplot images
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            return "No categorical columns for countplots."

        paths = []
        sparse_info = []

        for col in cat_cols:
            vc = df[col].value_counts()
            # Limit to top 15 categories for readability
            if len(vc) > 15:
                vc = vc.head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')

            colors = sns.color_palette("rocket", len(vc))
            bars = ax.barh(vc.index.astype(str), vc.values, color=colors, edgecolor='white', linewidth=0.5)

            # Highlight sparse classes
            for i, (val, cnt) in enumerate(vc.items()):
                pct = cnt / len(df) * 100
                if pct < 5:
                    bars[i].set_edgecolor('#ff4444')
                    bars[i].set_linewidth(2)
                    sparse_info.append({"column": col, "value": str(val), "pct": pct})

            ax.set_title(f"Countplot: {col}", color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel("Count", color='white')
            ax.tick_params(colors='white')

            path = _save_plot(session_id, "agent7", f"countplot_{col}", fig)
            paths.append(path)

        session = _get_session(session_id)
        for p in paths:
            session["plots"].append({"type": "countplot", "path": p, "agent": "agent7.3"})
        session["analyses"]["sparse_classes"] = sparse_info

        result = f"SAVED {len(paths)} countplots:\n" + "\n".join(paths)
        if sparse_info:
            result += f"\n\nSPARSE CLASSES (<5%): {len(sparse_info)} found"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 22-24: SEGMENTATION VISUALIZATION TOOLS
# =============================================================================

@tool
def eda_plot_boxplots(session_id: str, target: str) -> str:
    """
    Create boxplots showing numerical column distributions grouped by a target variable.

    Args:
        session_id: EDA session ID
        target: Target/grouping variable column name

    Returns:
        Paths to saved boxplot images
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        if target not in df.columns:
            return f"Target column '{target}' not found."

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if not numeric_cols:
            return "No numerical columns for boxplots."

        paths = []
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')

            plot_df = df[[col, target]].dropna()
            unique_vals = plot_df[target].nunique()
            palette = sns.color_palette("Set2", unique_vals)

            sns.boxplot(data=plot_df, x=target, y=col, ax=ax, palette=palette)
            ax.set_title(f"{col} by {target}", color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel(target, color='white')
            ax.set_ylabel(col, color='white')
            ax.tick_params(colors='white')
            plt.xticks(rotation=45 if unique_vals > 5 else 0)

            path = _save_plot(session_id, "agent8", f"boxplot_{col}_by_{target}", fig)
            paths.append(path)

        session = _get_session(session_id)
        for p in paths:
            session["plots"].append({"type": "boxplot", "path": p, "agent": "agent8.1"})

        return f"SAVED {len(paths)} boxplots:\n" + "\n".join(paths)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_violinplots(session_id: str, target: str) -> str:
    """
    Create violin plots showing numerical distributions with density, grouped by target.

    Args:
        session_id: EDA session ID
        target: Target/grouping variable column name

    Returns:
        Paths to saved violin plot images
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        if target not in df.columns:
            return f"Target column '{target}' not found."

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if not numeric_cols:
            return "No numerical columns for violin plots."

        paths = []
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')

            plot_df = df[[col, target]].dropna()
            unique_vals = plot_df[target].nunique()
            palette = sns.color_palette("husl", unique_vals)

            sns.violinplot(data=plot_df, x=target, y=col, ax=ax, palette=palette, inner="box")
            ax.set_title(f"Violin: {col} by {target}", color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel(target, color='white')
            ax.set_ylabel(col, color='white')
            ax.tick_params(colors='white')
            plt.xticks(rotation=45 if unique_vals > 5 else 0)

            path = _save_plot(session_id, "agent8", f"violin_{col}_by_{target}", fig)
            paths.append(path)

        session = _get_session(session_id)
        for p in paths:
            session["plots"].append({"type": "violinplot", "path": p, "agent": "agent8.2"})

        return f"SAVED {len(paths)} violin plots:\n" + "\n".join(paths)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_lmplots(session_id: str, target: str) -> str:
    """
    Create lmplots (scatter + regression) for pairs of numerical columns, colored by target.
    Limited to first 3 numerical column pairs to avoid excessive plots.

    Args:
        session_id: EDA session ID
        target: Target/grouping variable column name

    Returns:
        Paths to saved lmplot images
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        if target not in df.columns:
            return f"Target column '{target}' not found."

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if len(numeric_cols) < 2:
            return "Need at least 2 numerical columns for lmplots."

        paths = []
        pairs_done = 0
        max_pairs = 3

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if pairs_done >= max_pairs:
                    break
                x_col = numeric_cols[i]
                y_col = numeric_cols[j]

                plot_df = df[[x_col, y_col, target]].dropna()
                if len(plot_df) == 0:
                    continue

                try:
                    g = sns.lmplot(data=plot_df, x=x_col, y=y_col, hue=target,
                                   height=5, aspect=1.5, scatter_kws={"alpha": 0.5, "s": 20})
                    g.fig.patch.set_facecolor('#1a1a2e')
                    for ax in g.axes.flat:
                        ax.set_facecolor('#16213e')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                    g.fig.suptitle(f"{y_col} vs {x_col} by {target}", color='white', y=1.02)

                    path = _save_plot(session_id, "agent8", f"lmplot_{x_col}_{y_col}", g.fig)
                    paths.append(path)
                    pairs_done += 1
                except Exception:
                    continue
            if pairs_done >= max_pairs:
                break

        session = _get_session(session_id)
        for p in paths:
            session["plots"].append({"type": "lmplot", "path": p, "agent": "agent8.3"})

        return f"SAVED {len(paths)} lmplots:\n" + "\n".join(paths)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 25-27: OUTLIER DETECTION TOOLS
# =============================================================================

@tool
def eda_detect_outliers_iqr(session_id: str) -> str:
    """
    Detect outliers using the IQR (Interquartile Range) method for all numerical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Outlier detection results with counts and percentages
    """
    try:
        df = _get_df(session_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return "No numerical columns for outlier detection."

        results = {}
        lines = ["OUTLIER DETECTION (IQR Method):", ""]

        for col in numeric_cols:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = data[(data < lower) | (data > upper)]
            n_outliers = len(outliers)
            pct = n_outliers / len(data) * 100 if len(data) > 0 else 0

            results[col] = {
                "count": n_outliers,
                "percentage": round(pct, 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "iqr": float(IQR)
            }

            severity = "HIGH" if pct > 10 else "MEDIUM" if pct > 5 else "LOW"
            lines.append(f"  {col}: {n_outliers} outliers ({pct:.1f}%) [{severity}]")
            lines.append(f"    Bounds: [{lower:.2f}, {upper:.2f}], IQR: {IQR:.2f}")

        session = _get_session(session_id)
        session["analyses"]["outliers"] = results
        session["summaries"]["outlier_detection"] = "\n".join(lines)

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_outlier_boxplots(session_id: str) -> str:
    """
    Create box plots highlighting outliers for all numerical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Path to saved outlier boxplot image
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return "No numerical columns."

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        fig.patch.set_facecolor('#1a1a2e')
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            ax.set_facecolor('#16213e')
            bp = ax.boxplot(df[col].dropna(), patch_artist=True, flierprops=dict(
                markerfacecolor='#e94560', marker='o', markersize=4))
            bp['boxes'][0].set_facecolor('#0f3460')
            bp['boxes'][0].set_edgecolor('#e94560')
            bp['medians'][0].set_color('#00ff87')
            ax.set_title(col, color='white', fontsize=10)
            ax.tick_params(colors='white')

        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Outlier Detection (Box Plots)", color='white', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = _save_plot(session_id, "agent9", "outlier_boxplots", fig)

        session = _get_session(session_id)
        session["plots"].append({"type": "outlier_boxplots", "path": path, "agent": "agent9"})
        return f"SAVED: {path}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_outlier_scatter(session_id: str) -> str:
    """
    Create scatter plots highlighting outliers for pairs of numerical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Path to saved outlier scatter image
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return "Need at least 2 numerical columns."

        # Use first two numerical columns
        x_col, y_col = numeric_cols[0], numeric_cols[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        # Detect outliers
        x_data = df[x_col].dropna()
        y_data = df[y_col].dropna()
        common_idx = x_data.index.intersection(y_data.index)

        x_q1, x_q3 = x_data.quantile(0.25), x_data.quantile(0.75)
        x_iqr = x_q3 - x_q1
        y_q1, y_q3 = y_data.quantile(0.25), y_data.quantile(0.75)
        y_iqr = y_q3 - y_q1

        is_outlier = (
            (df.loc[common_idx, x_col] < x_q1 - 1.5 * x_iqr) |
            (df.loc[common_idx, x_col] > x_q3 + 1.5 * x_iqr) |
            (df.loc[common_idx, y_col] < y_q1 - 1.5 * y_iqr) |
            (df.loc[common_idx, y_col] > y_q3 + 1.5 * y_iqr)
        )

        normal = df.loc[common_idx][~is_outlier]
        outliers = df.loc[common_idx][is_outlier]

        ax.scatter(normal[x_col], normal[y_col], c='#0f3460', alpha=0.5, s=15, label='Normal')
        ax.scatter(outliers[x_col], outliers[y_col], c='#e94560', alpha=0.8, s=30,
                   marker='x', label=f'Outliers ({len(outliers)})')

        ax.set_title(f"Outlier Scatter: {x_col} vs {y_col}", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel(x_col, color='white')
        ax.set_ylabel(y_col, color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#16213e', labelcolor='white')

        path = _save_plot(session_id, "agent9", "outlier_scatter", fig)
        session = _get_session(session_id)
        session["plots"].append({"type": "outlier_scatter", "path": path, "agent": "agent9"})
        return f"SAVED: {path}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 28-29: CORRELATION TOOLS
# =============================================================================

@tool
def eda_compute_correlations(session_id: str) -> str:
    """
    Compute correlation matrix for all numerical columns.
    Identifies strong correlations (|r| > 0.7).

    Args:
        session_id: EDA session ID

    Returns:
        Correlation matrix and strong correlation pairs
    """
    try:
        df = _get_df(session_id)
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return "Need at least 2 numerical columns for correlation analysis."

        corr = numeric_df.corr()

        # Find strong correlations
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) > 0.7:
                    strong.append({
                        "col1": corr.columns[i],
                        "col2": corr.columns[j],
                        "correlation": round(r, 3)
                    })

        session = _get_session(session_id)
        session["analyses"]["correlations"] = corr.to_dict()
        session["analyses"]["strong_correlations"] = strong
        session["summaries"]["correlation"] = corr.to_string()

        lines = ["CORRELATION MATRIX:", "", corr.to_string()]
        if strong:
            lines.append("")
            lines.append("STRONG CORRELATIONS (|r| > 0.7):")
            for s in strong:
                lines.append(f"  {s['col1']} <-> {s['col2']}: {s['correlation']}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_plot_heatmap(session_id: str) -> str:
    """
    Create a correlation heatmap for all numerical columns.

    Args:
        session_id: EDA session ID

    Returns:
        Path to saved heatmap image
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: matplotlib/seaborn not available."
    try:
        df = _get_df(session_id)
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return "Need at least 2 numerical columns."

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(max(8, len(corr.columns)), max(6, len(corr.columns) * 0.8)))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                     center=0, ax=ax, linewidths=0.5, linecolor='#1a1a2e',
                     cbar_kws={"shrink": 0.8},
                     annot_kws={"size": 9, "color": "white"})

        ax.set_title("Correlation Heatmap", color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        path = _save_plot(session_id, "agent10", "heatmap", fig)
        session = _get_session(session_id)
        session["plots"].append({"type": "heatmap", "path": path, "agent": "agent10"})
        return f"SAVED: {path}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 30: ANALYSIS SUMMARY TOOL
# =============================================================================

@tool
def eda_get_agent_summaries(session_id: str) -> str:
    """
    Collect all agent summaries from the session for deep analysis.
    Returns combined summaries from agents 5-10.

    Args:
        session_id: EDA session ID

    Returns:
        Combined summaries from all completed analyses
    """
    try:
        session = _get_session(session_id)
        summaries = session.get("summaries", {})
        analyses = session.get("analyses", {})

        lines = ["ALL AGENT SUMMARIES:", "=" * 60]

        # Structure info
        df = _get_df(session_id)
        lines.append(f"\nDATASET: {session.get('table_name', 'unknown')}")
        lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        col_types = analyses.get("column_types", {})
        if col_types:
            lines.append(f"Numeric columns: {', '.join(col_types.get('numeric', []))}")
            lines.append(f"Categorical columns: {', '.join(col_types.get('categorical', []))}")
            lines.append(f"Datetime columns: {', '.join(col_types.get('datetime', []))}")

        # Numerical stats
        if "numerical_stats" in summaries:
            lines.append("\n--- NUMERICAL STATISTICS ---")
            lines.append(summaries["numerical_stats"])

        # Categorical stats
        if "categorical_stats" in summaries:
            lines.append("\n--- CATEGORICAL STATISTICS ---")
            lines.append(summaries["categorical_stats"])

        # Outliers
        if "outlier_detection" in summaries:
            lines.append("\n--- OUTLIER DETECTION ---")
            lines.append(summaries["outlier_detection"])

        # Correlations
        if "correlation" in summaries:
            lines.append("\n--- CORRELATIONS ---")
            lines.append(summaries["correlation"])
            strong = analyses.get("strong_correlations", [])
            if strong:
                lines.append(f"\nStrong correlations: {len(strong)}")
                for s in strong:
                    lines.append(f"  {s['col1']} <-> {s['col2']}: {s['correlation']}")

        # Sparse classes
        sparse = analyses.get("sparse_classes", [])
        if sparse:
            lines.append(f"\n--- SPARSE CLASSES ---")
            lines.append(f"Found {len(sparse)} sparse classes (<5%):")
            for s in sparse:
                lines.append(f"  {s['column']}='{s['value']}': {s['pct']:.1f}%")

        # Target variable
        target = analyses.get("target_variable")
        if target:
            lines.append(f"\nTarget variable: {target}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL 31-33: DASHBOARD TOOLS
# =============================================================================

@tool
def eda_generate_dashboard(session_id: str, title: str, all_summaries: str, suggestions: str) -> str:
    """
    Generate a complete HTML dashboard with all plots, summaries, and suggestions.
    Images are embedded as base64 for a self-contained file.

    Args:
        session_id: EDA session ID
        title: Dashboard title
        all_summaries: Combined text summaries from all agents
        suggestions: Data cleaning and feature engineering suggestions

    Returns:
        Path to the generated HTML dashboard
    """
    try:
        session = _get_session(session_id)
        plots = session.get("plots", [])
        df = _get_df(session_id)

        # Encode all plot images as base64
        encoded_plots = []
        for plot in plots:
            path = plot.get("path", "")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                encoded_plots.append({
                    **plot,
                    "base64": b64
                })

        # Group plots by agent
        agent_plots = {}
        for p in encoded_plots:
            agent = p.get("agent", "unknown")
            if agent not in agent_plots:
                agent_plots[agent] = []
            agent_plots[agent].append(p)

        # Build HTML
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table_name = session.get("table_name", "Dataset")

        # Get stats HTML
        stats_html = session.get("analyses", {}).get("stats_html", "")

        html = _build_dashboard_html(
            title=title,
            table_name=table_name,
            shape=df.shape,
            timestamp=timestamp,
            agent_plots=agent_plots,
            all_summaries=all_summaries,
            suggestions=suggestions,
            stats_html=stats_html,
            head_html=df.head(10).to_html(classes="data-table", index=False)
        )

        # Save dashboard
        filename = f"eda_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(DASHBOARD_OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        session["analyses"]["dashboard_path"] = filepath
        return f"DASHBOARD SAVED: {filepath}"
    except Exception as e:
        return f"Error generating dashboard: {str(e)}"


@tool
def eda_embed_image_base64(image_path: str) -> str:
    """
    Convert a PNG image file to base64 string for HTML embedding.

    Args:
        image_path: Path to the PNG image file

    Returns:
        Base64 encoded string of the image
    """
    try:
        if not os.path.exists(image_path):
            return f"Error: File not found: {image_path}"

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:image/png;base64,{b64}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_create_dashboard_section(section_name: str, summary: str) -> str:
    """
    Create a single dashboard section HTML block with summary text.

    Args:
        section_name: Name of the section (e.g., 'Distribution Analysis')
        summary: Summary text for this section

    Returns:
        HTML string for the section
    """
    return f"""
    <section class="agent-section">
        <h2>{section_name}</h2>
        <div class="observation-card">
            <pre>{summary}</pre>
        </div>
    </section>
    """


# =============================================================================
# TOOL 34-35: SESSION MANAGEMENT TOOLS
# =============================================================================

@tool
def eda_get_session_info(session_id: str) -> str:
    """
    Get details about an EDA session including loaded tables, analyses, and plots.

    Args:
        session_id: EDA session ID

    Returns:
        Session details and statistics
    """
    try:
        session = _get_session(session_id)
        lines = [
            f"SESSION: {session_id}",
            f"Table: {session.get('table_name', 'unknown')}",
            f"Created: {session.get('created_at', 'unknown')}",
            f"Tables loaded: {len(session.get('tables_loaded', {}))}",
        ]

        df = session.get("dataframe")
        if df is not None:
            lines.append(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")

        analyses = session.get("analyses", {})
        lines.append(f"Analyses completed: {', '.join(analyses.keys()) if analyses else 'None'}")

        plots = session.get("plots", [])
        lines.append(f"Plots generated: {len(plots)}")

        dashboard_path = analyses.get("dashboard_path")
        if dashboard_path:
            lines.append(f"Dashboard: {dashboard_path}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def eda_list_sessions() -> str:
    """
    List all active Multi-EDA sessions.

    Returns:
        List of active sessions with basic info
    """
    if not _multi_eda_sessions:
        return "No active sessions."

    lines = ["ACTIVE EDA SESSIONS:", ""]
    for sid, session in _multi_eda_sessions.items():
        df = session.get("dataframe")
        shape = f"{df.shape[0]}x{df.shape[1]}" if df is not None else "no data"
        lines.append(f"  {sid}: {session.get('table_name', '?')} ({shape}) - {session.get('created_at', '?')}")

    return "\n".join(lines)


# =============================================================================
# DASHBOARD HTML BUILDER
# =============================================================================

def _parse_deep_analysis_sections(all_summaries: str, suggestions: str) -> str:
    """Parse the deep analysis summaries and suggestions into structured HTML cards."""

    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _extract_bullets(text: str, header: str) -> list:
        """Extract bullet points following a header line."""
        bullets = []
        in_section = False
        for line in text.split("\n"):
            stripped = line.strip()
            if header.upper() in stripped.upper():
                in_section = True
                continue
            if in_section:
                if stripped.startswith("- ") or stripped.startswith("* "):
                    bullets.append(stripped.lstrip("-* ").strip())
                elif stripped and not stripped.startswith("-") and not stripped.startswith("*"):
                    # Hit a new section header or non-bullet line
                    if bullets:
                        break
        return bullets

    # Extract sections from summaries (deep_analysis text)
    key_insights = _extract_bullets(all_summaries, "KEY INSIGHTS")
    cleaning_suggestions = _extract_bullets(suggestions, "DATA CLEANING")
    feature_suggestions = _extract_bullets(suggestions, "FEATURE ENGINEERING")

    # Extract overall assessment
    overall = ""
    for line in all_summaries.split("\n"):
        if line.strip().upper().startswith("OVERALL:"):
            overall = line.strip()[len("OVERALL:"):].strip()
            break

    # Build Key Insights card
    insights_items = "".join(f"<li>{_escape(item)}</li>" for item in key_insights) if key_insights else "<li>No key insights extracted.</li>"
    insights_card = f"""
        <div class="deep-card insights">
            <div class="deep-card-header">
                <div class="deep-card-icon" style="background:rgba(66,165,245,0.15);">&#x1F4A1;</div>
                <div>
                    <div class="deep-card-title">Key Insights</div>
                    <div class="deep-card-count">{len(key_insights)} finding{'s' if len(key_insights) != 1 else ''}</div>
                </div>
            </div>
            <ul>{insights_items}</ul>
        </div>"""

    # Build Data Cleaning card
    cleaning_items = "".join(f"<li>{_escape(item)}</li>" for item in cleaning_suggestions) if cleaning_suggestions else "<li>No cleaning suggestions extracted.</li>"
    cleaning_card = f"""
        <div class="deep-card cleaning">
            <div class="deep-card-header">
                <div class="deep-card-icon" style="background:rgba(239,83,80,0.15);">&#x1F9F9;</div>
                <div>
                    <div class="deep-card-title">Data Cleaning Suggestions</div>
                    <div class="deep-card-count">{len(cleaning_suggestions)} suggestion{'s' if len(cleaning_suggestions) != 1 else ''}</div>
                </div>
            </div>
            <ul>{cleaning_items}</ul>
        </div>"""

    # Build Feature Engineering card
    feature_items = "".join(f"<li>{_escape(item)}</li>" for item in feature_suggestions) if feature_suggestions else "<li>No feature engineering suggestions extracted.</li>"
    feature_card = f"""
        <div class="deep-card features">
            <div class="deep-card-header">
                <div class="deep-card-icon" style="background:rgba(102,187,106,0.15);">&#x2699;</div>
                <div>
                    <div class="deep-card-title">Feature Engineering Suggestions</div>
                    <div class="deep-card-count">{len(feature_suggestions)} suggestion{'s' if len(feature_suggestions) != 1 else ''}</div>
                </div>
            </div>
            <ul>{feature_items}</ul>
        </div>"""

    # Build overall assessment
    overall_html = ""
    if overall:
        overall_html = f"""
        <div class="overall-assessment">
            <h4>&#x1F4CB; Overall Assessment</h4>
            <p>{_escape(overall)}</p>
        </div>"""

    return f"""
        <div class="deep-analysis-grid">
            {insights_card}
            {cleaning_card}
        </div>
        <div class="deep-analysis-grid">
            {feature_card}
        </div>
        {overall_html}
    """


def _build_dashboard_html(title, table_name, shape, timestamp, agent_plots,
                          all_summaries, suggestions, stats_html, head_html):
    """Build the complete dashboard HTML string."""

    # Agent section info
    agent_sections = {
        "agent7.1": {"title": "Histograms (All Numerical)", "icon": "&#x1F4CA;"},
        "agent7.2": {"title": "Individual Histograms", "icon": "&#x1F4C9;"},
        "agent7.3": {"title": "Categorical Countplots", "icon": "&#x1F4F6;"},
        "agent8.1": {"title": "Boxplot Analysis", "icon": "&#x1F4E6;"},
        "agent8.2": {"title": "Violin Plot Analysis", "icon": "&#x1F3BB;"},
        "agent8.3": {"title": "LM Plot Analysis", "icon": "&#x1F4D0;"},
        "agent9":   {"title": "Outlier Detection", "icon": "&#x1F50E;"},
        "agent10":  {"title": "Correlation Analysis", "icon": "&#x1F525;"},
    }

    # Build plot sections HTML
    plot_sections = []
    for agent_key, info in agent_sections.items():
        plots_for_agent = agent_plots.get(agent_key, [])
        if not plots_for_agent:
            continue

        images_html = ""
        for p in plots_for_agent:
            b64 = p.get("base64", "")
            if b64:
                images_html += f'<img src="data:image/png;base64,{b64}" class="chart-img" onclick="openLightbox(this)" />\n'

        plot_sections.append(f"""
        <section id="{agent_key}" class="agent-section">
            <h2>{info['icon']} {info['title']}</h2>
            <div class="chart-grid">
                {images_html}
            </div>
        </section>
        """)

    all_plot_sections = "\n".join(plot_sections)

    # Build navigation links
    nav_links = []
    nav_links.append('<a href="#structure">Structure</a>')
    nav_links.append('<a href="#statistics">Statistics</a>')
    for agent_key, info in agent_sections.items():
        if agent_key in agent_plots:
            nav_links.append(f'<a href="#{agent_key}">{info["title"]}</a>')
    nav_links.append('<a href="#deep-analysis">Deep Analysis</a>')
    nav_html = "\n".join(nav_links)

    # Parse suggestions into structured sections
    deep_sections = _parse_deep_analysis_sections(all_summaries, suggestions)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px 40px;
            border-bottom: 3px solid #e94560;
        }}
        .dashboard-header h1 {{
            font-size: 2em;
            color: #ffffff;
            margin-bottom: 10px;
        }}
        .dashboard-header .meta {{
            color: #9e9e9e;
            font-size: 0.9em;
        }}
        .dashboard-nav {{
            position: fixed;
            left: 0;
            top: 0;
            width: 200px;
            height: 100vh;
            background: #16213e;
            padding: 80px 10px 20px;
            overflow-y: auto;
            border-right: 2px solid #0f3460;
            z-index: 100;
        }}
        .dashboard-nav a {{
            display: block;
            color: #9e9e9e;
            text-decoration: none;
            padding: 8px 12px;
            font-size: 0.85em;
            border-radius: 4px;
            margin-bottom: 4px;
            transition: all 0.2s;
        }}
        .dashboard-nav a:hover {{
            background: #0f3460;
            color: #ffffff;
        }}
        .main-content {{
            margin-left: 210px;
            padding: 20px 40px;
        }}
        .agent-section {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            border-left: 4px solid #e94560;
        }}
        .agent-section h2 {{
            color: #ffffff;
            font-size: 1.4em;
            margin-bottom: 15px;
        }}
        .observation-card {{
            background: #16213e;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #0f3460;
        }}
        .observation-card pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
            color: #b0b0b0;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .chart-img {{
            width: 100%;
            max-height: 350px;
            object-fit: contain;
            border-radius: 6px;
            cursor: pointer;
            transition: transform 0.2s;
            border: 1px solid #0f3460;
        }}
        .chart-img:hover {{
            transform: scale(1.02);
            border-color: #e94560;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 0.85em;
        }}
        .data-table th {{
            background: #0f3460;
            color: #ffffff;
            padding: 10px 8px;
            text-align: left;
            border: 1px solid #16213e;
        }}
        .data-table td {{
            padding: 8px;
            border: 1px solid #16213e;
            background: #1a1a2e;
            color: #d0d0d0;
        }}
        .data-table tr:nth-child(even) td {{
            background: #16213e;
        }}
        .deep-analysis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        @media (max-width: 900px) {{
            .deep-analysis-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .deep-card {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #0f3460;
            transition: border-color 0.2s;
        }}
        .deep-card:hover {{
            border-color: #e94560;
        }}
        .deep-card-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }}
        .deep-card-icon {{
            font-size: 1.5em;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            flex-shrink: 0;
        }}
        .deep-card-title {{
            font-size: 1.05em;
            font-weight: 600;
            color: #ffffff;
        }}
        .deep-card-count {{
            font-size: 0.75em;
            color: #9e9e9e;
            margin-top: 2px;
        }}
        .deep-card ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .deep-card ul li {{
            padding: 8px 0 8px 28px;
            position: relative;
            color: #c0c0c0;
            font-size: 0.88em;
            line-height: 1.5;
            border-bottom: 1px solid rgba(15, 52, 96, 0.5);
        }}
        .deep-card ul li:last-child {{
            border-bottom: none;
        }}
        .deep-card ul li::before {{
            content: '';
            position: absolute;
            left: 8px;
            top: 14px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        .deep-card.insights li::before {{ background: #42a5f5; }}
        .deep-card.cleaning li::before {{ background: #ef5350; }}
        .deep-card.features li::before {{ background: #66bb6a; }}
        .overall-assessment {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 8px;
            padding: 20px 24px;
            margin-top: 20px;
            border-left: 4px solid #e94560;
        }}
        .overall-assessment h4 {{
            color: #e94560;
            margin-bottom: 10px;
            font-size: 1em;
        }}
        .overall-assessment p {{
            color: #c0c0c0;
            font-size: 0.9em;
            line-height: 1.6;
        }}
        .suggestions-card {{
            background: #16213e;
            border-radius: 6px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #0f3460;
        }}
        .suggestions-card h3 {{
            color: #e94560;
            margin: 15px 0 8px;
        }}
        .suggestions-card pre {{
            white-space: pre-wrap;
            color: #b0b0b0;
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
        }}
        .lightbox-overlay {{
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            cursor: pointer;
        }}
        .lightbox-img {{
            max-width: 90vw;
            max-height: 90vh;
            object-fit: contain;
            border-radius: 8px;
            resize: both;
            overflow: auto;
        }}
    </style>
</head>
<body>
    <nav class="dashboard-nav">
        <div style="color:#e94560; font-weight:bold; margin-bottom:15px; font-size:1.1em;">Navigation</div>
        {nav_html}
    </nav>

    <div class="main-content">
        <div class="dashboard-header">
            <h1>{title}</h1>
            <div class="meta">
                Dataset: {table_name} | Shape: {shape[0]} rows x {shape[1]} columns | Generated: {timestamp}
            </div>
        </div>

        <section id="structure" class="agent-section">
            <h2>&#x1F3D7; Structure Inspection</h2>
            <div class="observation-card">
                <pre>Shape: {shape[0]} rows x {shape[1]} columns</pre>
            </div>
            {head_html}
        </section>

        <section id="statistics" class="agent-section">
            <h2>&#x1F4CA; Descriptive Statistics</h2>
            {stats_html if stats_html else '<div class="observation-card"><pre>No statistics computed yet.</pre></div>'}
        </section>

        {all_plot_sections}

        <section id="deep-analysis" class="agent-section">
            <h2>&#x1F9E0; Deep Analysis &amp; Suggestions</h2>
            {deep_sections}
        </section>
    </div>

    <script>
        function openLightbox(img) {{
            var overlay = document.createElement('div');
            overlay.className = 'lightbox-overlay';
            var enlargedImg = document.createElement('img');
            enlargedImg.src = img.src;
            enlargedImg.className = 'lightbox-img';
            overlay.appendChild(enlargedImg);
            overlay.onclick = function() {{ overlay.remove(); }};
            document.body.appendChild(overlay);
        }}
    </script>
</body>
</html>"""


# =============================================================================
# REGISTER ALL TOOLS
# =============================================================================

# Data Loading Tools
tool_registry.register(eda_load_table, "multi_eda")
tool_registry.register(eda_get_table_row_count, "multi_eda")
tool_registry.register(eda_load_multiple_tables, "multi_eda")

# Schema Tools
tool_registry.register(eda_get_schema_from_json, "multi_eda")
tool_registry.register(eda_get_table_schema, "multi_eda")
tool_registry.register(eda_get_all_tables, "multi_eda")

# Join & Target Tools
tool_registry.register(eda_detect_target_variable, "multi_eda")
tool_registry.register(eda_detect_joins, "multi_eda")
tool_registry.register(eda_join_tables, "multi_eda")
tool_registry.register(eda_validate_target, "multi_eda")

# Structure Tools
tool_registry.register(eda_get_shape, "multi_eda")
tool_registry.register(eda_get_dtypes, "multi_eda")
tool_registry.register(eda_get_head, "multi_eda")
tool_registry.register(eda_classify_columns, "multi_eda")

# Statistics Tools
tool_registry.register(eda_describe_numerical, "multi_eda")
tool_registry.register(eda_describe_categorical, "multi_eda")
tool_registry.register(eda_generate_stats_table_html, "multi_eda")

# Distribution Visualization Tools
tool_registry.register(eda_plot_all_histograms, "multi_eda")
tool_registry.register(eda_plot_individual_histogram, "multi_eda")
tool_registry.register(eda_plot_all_individual_histograms, "multi_eda")
tool_registry.register(eda_plot_countplots, "multi_eda")

# Segmentation Visualization Tools
tool_registry.register(eda_plot_boxplots, "multi_eda")
tool_registry.register(eda_plot_violinplots, "multi_eda")
tool_registry.register(eda_plot_lmplots, "multi_eda")

# Outlier Detection Tools
tool_registry.register(eda_detect_outliers_iqr, "multi_eda")
tool_registry.register(eda_plot_outlier_boxplots, "multi_eda")
tool_registry.register(eda_plot_outlier_scatter, "multi_eda")

# Correlation Tools
tool_registry.register(eda_compute_correlations, "multi_eda")
tool_registry.register(eda_plot_heatmap, "multi_eda")

# Analysis Tools
tool_registry.register(eda_get_agent_summaries, "multi_eda")

# Dashboard Tools
tool_registry.register(eda_generate_dashboard, "multi_eda")
tool_registry.register(eda_embed_image_base64, "multi_eda")
tool_registry.register(eda_create_dashboard_section, "multi_eda")

# Session Management Tools
tool_registry.register(eda_get_session_info, "multi_eda")
tool_registry.register(eda_list_sessions, "multi_eda")


def get_all_multi_eda_tools():
    """Get all Multi-EDA tools."""
    return tool_registry.get_tools_by_category("multi_eda")
