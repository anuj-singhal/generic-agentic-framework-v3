"""
Exploratory Data Analysis (EDA) Tools Module
=============================================

Comprehensive EDA tools for analyzing DuckDB tables like a data scientist.
Includes basic statistics, data quality checks, distribution analysis,
correlation analysis, outlier detection, and more.
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
import os
import json
import uuid
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

# Optional imports for advanced analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Output directory for visualization files
VIZ_OUTPUT_DIR = "sample_files/eda_visualizations"
os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

# In-memory session store for EDA sessions
_eda_sessions: Dict[str, Dict[str, Any]] = {}


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
    """Create a new EDA session."""
    session_id = str(uuid.uuid4())[:8]
    _eda_sessions[session_id] = {
        "table_name": table_name,
        "created_at": datetime.now().isoformat(),
        "analyses_completed": [],
        "results": {},
        "summary": {}
    }
    return session_id


def _get_or_create_session(table_name: str, session_id: Optional[str] = None) -> str:
    """Get existing session or create new one."""
    if session_id and session_id in _eda_sessions:
        return session_id
    return _create_session(table_name)


def _store_result(session_id: str, analysis_name: str, result: Dict[str, Any]):
    """Store analysis result in session."""
    if session_id in _eda_sessions:
        _eda_sessions[session_id]["results"][analysis_name] = result
        if analysis_name not in _eda_sessions[session_id]["analyses_completed"]:
            _eda_sessions[session_id]["analyses_completed"].append(analysis_name)


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
    elif isinstance(obj, (pd.Timedelta, np.timedelta64)):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


# =============================================================================
# BASIC EDA TOOLS
# =============================================================================

@tool
def list_tables_for_eda() -> str:
    """
    List all available tables in the database for EDA analysis.
    Shows table names and row counts to help select which table to analyze.

    Returns:
        List of tables with row counts and column counts
    """
    try:
        with get_connection() as conn:
            result = conn.execute("SHOW TABLES").fetchall()

            tables_info = []
            for row in result:
                table_name = row[0]
                # Get row count
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                # Get column count
                cols = conn.execute(f"DESCRIBE {table_name}").fetchall()

                tables_info.append({
                    "table_name": table_name,
                    "row_count": count,
                    "column_count": len(cols),
                    "is_synth_table": table_name.upper().startswith("SYNTH_")
                })

            # Sort by row count descending
            tables_info.sort(key=lambda x: x["row_count"], reverse=True)

            return json.dumps({
                "total_tables": len(tables_info),
                "tables": tables_info,
                "note": "Select a table to begin EDA analysis using get_table_info_for_eda()"
            }, indent=2)

    except Exception as e:
        return f"Error listing tables: {str(e)}"


@tool
def get_table_info_for_eda(table_name: str) -> str:
    """
    Get detailed schema information about a table for EDA.
    Includes column names, data types, and basic metadata.

    Args:
        table_name: Name of the table to analyze

    Returns:
        Detailed table schema with column information and data type classification
    """
    try:
        table_name = table_name.upper().strip()

        with get_connection() as conn:
            # Check table exists
            tables = [r[0].upper() for r in conn.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                return f"Error: Table '{table_name}' not found. Available: {', '.join(tables)}"

            # Get schema
            schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()

            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Classify columns
            columns = []
            numeric_cols = []
            categorical_cols = []
            datetime_cols = []
            text_cols = []

            for row in schema_result:
                col_name = row[0]
                col_type = row[1].upper()
                nullable = row[3] == "YES"

                # Classify column type
                if any(t in col_type for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMBER"]):
                    col_class = "numeric"
                    numeric_cols.append(col_name)
                elif any(t in col_type for t in ["DATE", "TIME", "TIMESTAMP"]):
                    col_class = "datetime"
                    datetime_cols.append(col_name)
                elif "BOOL" in col_type:
                    col_class = "boolean"
                    categorical_cols.append(col_name)
                else:
                    col_class = "categorical"
                    categorical_cols.append(col_name)

                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": nullable,
                    "classification": col_class
                })

            return json.dumps({
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": columns,
                "column_classification": {
                    "numeric": numeric_cols,
                    "categorical": categorical_cols,
                    "datetime": datetime_cols
                },
                "recommended_analyses": [
                    "get_basic_statistics" if numeric_cols else None,
                    "check_missing_values",
                    "check_duplicates",
                    "analyze_categorical_columns" if categorical_cols else None,
                    "analyze_numerical_columns" if numeric_cols else None,
                    "analyze_datetime_columns" if datetime_cols else None,
                    "analyze_correlations" if len(numeric_cols) > 1 else None
                ]
            }, indent=2, default=str)

    except Exception as e:
        return f"Error getting table info: {str(e)}"


@tool
def load_table_to_pandas(table_name: str, limit: Optional[int] = None, session_id: Optional[str] = None) -> str:
    """
    Load a table into a pandas DataFrame for EDA.
    Creates an EDA session to track all analyses.

    Args:
        table_name: Name of the table to load
        limit: Optional row limit (default: all rows, max 100000 for safety)
        session_id: Optional existing session ID

    Returns:
        Session ID and DataFrame info (shape, dtypes, memory usage)
    """
    try:
        table_name = table_name.upper().strip()
        limit = min(limit or 100000, 100000)  # Cap at 100k rows

        with get_connection() as conn:
            # Check table exists
            tables = [r[0].upper() for r in conn.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                return f"Error: Table '{table_name}' not found."

            # Load data
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"

            df = conn.execute(query).fetchdf()

            # Create or get session
            session_id = _get_or_create_session(table_name, session_id)

            # Store DataFrame in session
            _eda_sessions[session_id]["dataframe"] = df
            _eda_sessions[session_id]["row_count"] = len(df)
            _eda_sessions[session_id]["column_count"] = len(df.columns)

            # Get dtype info
            dtype_summary = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                if dtype not in dtype_summary:
                    dtype_summary[dtype] = []
                dtype_summary[dtype].append(col)

            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

            return json.dumps({
                "session_id": session_id,
                "table_name": table_name,
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": list(df.columns),
                "dtypes_summary": dtype_summary,
                "memory_usage_mb": round(memory_mb, 2),
                "note": "Data loaded. Use session_id for subsequent EDA operations."
            }, indent=2)

    except Exception as e:
        return f"Error loading table: {str(e)}"


@tool
def get_basic_statistics(session_id: str) -> str:
    """
    Calculate basic descriptive statistics for all numeric columns.
    Includes count, mean, std, min, 25%, 50%, 75%, max.

    Args:
        session_id: EDA session ID from load_table_to_pandas

    Returns:
        Descriptive statistics for all numeric columns
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found. Load table first."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session. Use load_table_to_pandas first."

        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return json.dumps({
                "session_id": session_id,
                "message": "No numeric columns found in the table.",
                "available_columns": list(df.columns)
            }, indent=2)

        # Calculate statistics
        stats_df = numeric_df.describe()

        # Add additional statistics
        additional_stats = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            additional_stats[col] = {
                "count": int(stats_df.loc["count", col]),
                "mean": _safe_json_serialize(stats_df.loc["mean", col]),
                "std": _safe_json_serialize(stats_df.loc["std", col]),
                "min": _safe_json_serialize(stats_df.loc["min", col]),
                "25%": _safe_json_serialize(stats_df.loc["25%", col]),
                "50%": _safe_json_serialize(stats_df.loc["50%", col]),
                "75%": _safe_json_serialize(stats_df.loc["75%", col]),
                "max": _safe_json_serialize(stats_df.loc["max", col]),
                "range": _safe_json_serialize(stats_df.loc["max", col] - stats_df.loc["min", col]),
                "iqr": _safe_json_serialize(stats_df.loc["75%", col] - stats_df.loc["25%", col]),
                "variance": _safe_json_serialize(col_data.var()) if len(col_data) > 0 else None,
                "null_count": int(df[col].isna().sum()),
                "null_percentage": round(df[col].isna().sum() / len(df) * 100, 2)
            }

            # Add skewness and kurtosis if scipy available
            if SCIPY_AVAILABLE and len(col_data) > 2:
                additional_stats[col]["skewness"] = _safe_json_serialize(stats.skew(col_data))
                additional_stats[col]["kurtosis"] = _safe_json_serialize(stats.kurtosis(col_data))

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "numeric_columns_count": len(numeric_df.columns),
            "statistics": additional_stats
        }

        # Store result
        _store_result(session_id, "basic_statistics", result)

        return json.dumps(result, indent=2, default=_safe_json_serialize)

    except Exception as e:
        return f"Error calculating statistics: {str(e)}"


@tool
def check_missing_values(session_id: str) -> str:
    """
    Analyze missing values in the dataset.
    Shows count, percentage, and pattern of missing values per column.

    Args:
        session_id: EDA session ID

    Returns:
        Missing value analysis for each column
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Calculate missing values
        missing_info = []
        total_cells = len(df) * len(df.columns)
        total_missing = 0

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            total_missing += missing_count
            missing_pct = round(missing_count / len(df) * 100, 2)

            # Determine severity
            if missing_pct == 0:
                severity = "none"
            elif missing_pct < 5:
                severity = "low"
            elif missing_pct < 20:
                severity = "moderate"
            elif missing_pct < 50:
                severity = "high"
            else:
                severity = "critical"

            missing_info.append({
                "column": col,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "present_count": len(df) - missing_count,
                "severity": severity
            })

        # Sort by missing percentage descending
        missing_info.sort(key=lambda x: x["missing_percentage"], reverse=True)

        # Columns with missing values
        cols_with_missing = [m for m in missing_info if m["missing_count"] > 0]

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "overall_missing_percentage": round(total_missing / total_cells * 100, 2),
            "columns_with_missing": len(cols_with_missing),
            "columns_complete": len(df.columns) - len(cols_with_missing),
            "missing_by_column": missing_info,
            "recommendations": []
        }

        # Add recommendations
        for m in cols_with_missing:
            if m["severity"] == "critical":
                result["recommendations"].append(
                    f"Column '{m['column']}' has {m['missing_percentage']}% missing - consider dropping or imputation"
                )
            elif m["severity"] == "high":
                result["recommendations"].append(
                    f"Column '{m['column']}' has {m['missing_percentage']}% missing - investigate cause"
                )

        _store_result(session_id, "missing_values", result)

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error checking missing values: {str(e)}"


@tool
def check_duplicates(session_id: str, subset: Optional[str] = None) -> str:
    """
    Check for duplicate rows in the dataset.
    Can check all columns or a specific subset.

    Args:
        session_id: EDA session ID
        subset: Optional comma-separated column names to check for duplicates

    Returns:
        Duplicate analysis including count and examples
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Parse subset columns
        check_cols = None
        if subset:
            check_cols = [c.strip().upper() for c in subset.split(",")]
            # Validate columns exist
            df_cols_upper = [c.upper() for c in df.columns]
            invalid = [c for c in check_cols if c not in df_cols_upper]
            if invalid:
                return f"Error: Columns not found: {invalid}"
            # Map back to actual column names
            col_map = {c.upper(): c for c in df.columns}
            check_cols = [col_map[c] for c in check_cols]

        # Find duplicates
        if check_cols:
            duplicates = df[df.duplicated(subset=check_cols, keep=False)]
            dup_count = df.duplicated(subset=check_cols, keep="first").sum()
        else:
            duplicates = df[df.duplicated(keep=False)]
            dup_count = df.duplicated(keep="first").sum()

        # Get sample duplicates
        sample_dups = []
        if len(duplicates) > 0:
            sample_df = duplicates.head(10)
            for _, row in sample_df.iterrows():
                sample_dups.append({col: _safe_json_serialize(row[col]) for col in df.columns})

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "total_rows": len(df),
            "columns_checked": check_cols if check_cols else "all columns",
            "duplicate_rows": int(dup_count),
            "duplicate_percentage": round(dup_count / len(df) * 100, 2),
            "unique_rows": len(df) - int(dup_count),
            "rows_in_duplicate_groups": len(duplicates),
            "sample_duplicates": sample_dups[:5] if sample_dups else [],
            "recommendation": "Consider removing duplicates" if dup_count > 0 else "No duplicates found"
        }

        _store_result(session_id, "duplicates", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error checking duplicates: {str(e)}"


@tool
def check_data_types(session_id: str) -> str:
    """
    Analyze data types and potential type issues.
    Detects mixed types, potential type conversions, and type mismatches.

    Args:
        session_id: EDA session ID

    Returns:
        Data type analysis with potential issues and recommendations
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        type_analysis = []

        for col in df.columns:
            col_data = df[col].dropna()
            current_dtype = str(df[col].dtype)

            analysis = {
                "column": col,
                "current_dtype": current_dtype,
                "non_null_count": len(col_data),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
                "potential_issues": [],
                "recommendations": []
            }

            # Check for potential numeric columns stored as object
            if current_dtype == "object":
                # Try to convert to numeric
                numeric_conversion = pd.to_numeric(col_data, errors="coerce")
                successful = numeric_conversion.notna().sum()
                if len(col_data) > 0 and successful / len(col_data) > 0.9:
                    analysis["potential_issues"].append("Possibly numeric data stored as string")
                    analysis["recommendations"].append(f"Consider converting to numeric type")
                    analysis["numeric_conversion_rate"] = round(successful / len(col_data) * 100, 2)

                # Check for potential datetime
                try:
                    datetime_conversion = pd.to_datetime(col_data.head(100), errors="coerce")
                    if datetime_conversion.notna().sum() > 50:
                        analysis["potential_issues"].append("Possibly datetime stored as string")
                        analysis["recommendations"].append("Consider converting to datetime type")
                except:
                    pass

                # Check for potential boolean
                unique_vals = set(col_data.astype(str).str.lower().unique())
                bool_indicators = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
                if len(unique_vals) <= 3 and unique_vals.issubset(bool_indicators):
                    analysis["potential_issues"].append("Possibly boolean data")
                    analysis["recommendations"].append("Consider converting to boolean type")

            # Check for low cardinality numeric (might be categorical)
            if np.issubdtype(df[col].dtype, np.number):
                if analysis["unique_count"] < 10 and len(col_data) > 100:
                    analysis["potential_issues"].append("Low cardinality numeric - might be categorical")
                    analysis["recommendations"].append("Consider treating as categorical for analysis")

            # Check for high cardinality in object columns
            if current_dtype == "object" and len(col_data) > 0:
                cardinality_ratio = analysis["unique_count"] / len(col_data)
                if cardinality_ratio > 0.9:
                    analysis["potential_issues"].append("High cardinality - possibly ID or free text")
                    analysis["cardinality_ratio"] = round(cardinality_ratio, 4)

            type_analysis.append(analysis)

        # Summarize by dtype
        dtype_summary = {}
        for a in type_analysis:
            dtype = a["current_dtype"]
            if dtype not in dtype_summary:
                dtype_summary[dtype] = []
            dtype_summary[dtype].append(a["column"])

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "total_columns": len(df.columns),
            "dtype_summary": dtype_summary,
            "columns_with_issues": [a["column"] for a in type_analysis if a["potential_issues"]],
            "detailed_analysis": type_analysis
        }

        _store_result(session_id, "data_types", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error checking data types: {str(e)}"


@tool
def get_unique_value_counts(session_id: str, column_name: Optional[str] = None, top_n: int = 20) -> str:
    """
    Get unique value counts for categorical columns.
    Shows frequency distribution and percentage for each unique value.

    Args:
        session_id: EDA session ID
        column_name: Specific column to analyze (optional, analyzes all categorical if not specified)
        top_n: Number of top values to show (default 20)

    Returns:
        Value counts and frequency distribution
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        results = {}

        if column_name:
            columns = [column_name.upper()]
            # Map to actual column name
            col_map = {c.upper(): c for c in df.columns}
            if columns[0] not in col_map:
                return f"Error: Column '{column_name}' not found."
            columns = [col_map[columns[0]]]
        else:
            # Get categorical columns (object, category, bool)
            columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            # Also include low-cardinality numeric
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 20:
                    columns.append(col)

        for col in columns:
            value_counts = df[col].value_counts(dropna=False)
            total = len(df)

            top_values = []
            for val, count in value_counts.head(top_n).items():
                top_values.append({
                    "value": _safe_json_serialize(val) if pd.notna(val) else "NULL",
                    "count": int(count),
                    "percentage": round(count / total * 100, 2)
                })

            results[col] = {
                "total_unique": int(df[col].nunique(dropna=False)),
                "most_common": top_values[0] if top_values else None,
                "least_common": {
                    "value": _safe_json_serialize(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    "count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
                },
                "top_values": top_values,
                "values_shown": len(top_values),
                "values_not_shown": max(0, int(df[col].nunique(dropna=False)) - top_n)
            }

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "columns_analyzed": len(results),
            "value_distributions": results
        }

        _store_result(session_id, "unique_values", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error getting unique values: {str(e)}"


@tool
def detect_outliers(session_id: str, method: str = "iqr", threshold: float = 1.5) -> str:
    """
    Detect outliers in numeric columns using statistical methods.

    Args:
        session_id: EDA session ID
        method: Detection method - 'iqr' (Interquartile Range) or 'zscore' (Z-Score)
        threshold: Threshold for outlier detection (IQR multiplier or Z-score threshold)

    Returns:
        Outlier analysis for each numeric column
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return json.dumps({
                "session_id": session_id,
                "message": "No numeric columns found for outlier detection."
            }, indent=2)

        outlier_analysis = []

        for col in numeric_df.columns:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            analysis = {
                "column": col,
                "method": method,
                "threshold": threshold,
                "total_values": len(col_data)
            }

            if method.lower() == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers_low = col_data[col_data < lower_bound]
                outliers_high = col_data[col_data > upper_bound]

                analysis.update({
                    "Q1": _safe_json_serialize(Q1),
                    "Q3": _safe_json_serialize(Q3),
                    "IQR": _safe_json_serialize(IQR),
                    "lower_bound": _safe_json_serialize(lower_bound),
                    "upper_bound": _safe_json_serialize(upper_bound),
                    "outliers_below": len(outliers_low),
                    "outliers_above": len(outliers_high),
                    "total_outliers": len(outliers_low) + len(outliers_high),
                    "outlier_percentage": round((len(outliers_low) + len(outliers_high)) / len(col_data) * 100, 2)
                })

                # Sample outlier values
                if len(outliers_low) > 0:
                    analysis["sample_low_outliers"] = [_safe_json_serialize(v) for v in outliers_low.head(5).tolist()]
                if len(outliers_high) > 0:
                    analysis["sample_high_outliers"] = [_safe_json_serialize(v) for v in outliers_high.head(5).tolist()]

            elif method.lower() == "zscore":
                if SCIPY_AVAILABLE:
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers = col_data[z_scores > threshold]
                else:
                    mean = col_data.mean()
                    std = col_data.std()
                    z_scores = np.abs((col_data - mean) / std) if std > 0 else np.zeros(len(col_data))
                    outliers = col_data[z_scores > threshold]

                analysis.update({
                    "mean": _safe_json_serialize(col_data.mean()),
                    "std": _safe_json_serialize(col_data.std()),
                    "total_outliers": len(outliers),
                    "outlier_percentage": round(len(outliers) / len(col_data) * 100, 2),
                    "sample_outliers": [_safe_json_serialize(v) for v in outliers.head(10).tolist()]
                })

            outlier_analysis.append(analysis)

        # Summary
        total_outliers = sum(a.get("total_outliers", 0) for a in outlier_analysis)
        cols_with_outliers = [a["column"] for a in outlier_analysis if a.get("total_outliers", 0) > 0]

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "method": method,
            "threshold": threshold,
            "columns_analyzed": len(outlier_analysis),
            "columns_with_outliers": cols_with_outliers,
            "total_outliers_found": total_outliers,
            "detailed_analysis": outlier_analysis
        }

        _store_result(session_id, "outliers", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error detecting outliers: {str(e)}"


# =============================================================================
# ADVANCED EDA TOOLS
# =============================================================================

@tool
def analyze_distributions(session_id: str) -> str:
    """
    Analyze the distribution of numeric columns.
    Includes skewness, kurtosis, normality tests, and distribution type suggestions.

    Args:
        session_id: EDA session ID

    Returns:
        Distribution analysis for each numeric column
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return json.dumps({
                "session_id": session_id,
                "message": "No numeric columns found for distribution analysis."
            }, indent=2)

        distribution_analysis = []

        for col in numeric_df.columns:
            col_data = df[col].dropna()

            if len(col_data) < 3:
                continue

            analysis = {
                "column": col,
                "count": len(col_data),
                "mean": _safe_json_serialize(col_data.mean()),
                "median": _safe_json_serialize(col_data.median()),
                "mode": _safe_json_serialize(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                "std": _safe_json_serialize(col_data.std()),
                "variance": _safe_json_serialize(col_data.var())
            }

            # Calculate skewness and kurtosis
            if SCIPY_AVAILABLE:
                skewness = stats.skew(col_data)
                kurtosis_val = stats.kurtosis(col_data)

                analysis["skewness"] = _safe_json_serialize(skewness)
                analysis["kurtosis"] = _safe_json_serialize(kurtosis_val)

                # Interpret skewness
                if abs(skewness) < 0.5:
                    analysis["skewness_interpretation"] = "approximately symmetric"
                elif skewness > 0:
                    analysis["skewness_interpretation"] = "right-skewed (positive)"
                else:
                    analysis["skewness_interpretation"] = "left-skewed (negative)"

                # Interpret kurtosis
                if abs(kurtosis_val) < 0.5:
                    analysis["kurtosis_interpretation"] = "mesokurtic (normal-like)"
                elif kurtosis_val > 0:
                    analysis["kurtosis_interpretation"] = "leptokurtic (heavy tails)"
                else:
                    analysis["kurtosis_interpretation"] = "platykurtic (light tails)"

                # Normality test (Shapiro-Wilk for small samples)
                if len(col_data) <= 5000:
                    try:
                        stat, p_value = stats.shapiro(col_data.head(5000))
                        analysis["normality_test"] = {
                            "test": "Shapiro-Wilk",
                            "statistic": _safe_json_serialize(stat),
                            "p_value": _safe_json_serialize(p_value),
                            "is_normal": p_value > 0.05,
                            "interpretation": "Likely normal distribution" if p_value > 0.05 else "Not normally distributed"
                        }
                    except:
                        pass

            # Suggest distribution type
            if analysis.get("skewness") is not None:
                skew = abs(analysis["skewness"]) if analysis["skewness"] else 0
                if skew < 0.5 and analysis.get("normality_test", {}).get("is_normal", False):
                    analysis["suggested_distribution"] = "Normal"
                elif analysis["skewness"] > 1:
                    analysis["suggested_distribution"] = "Log-normal or Exponential"
                elif analysis["skewness"] < -1:
                    analysis["suggested_distribution"] = "Left-skewed (Beta or Weibull)"
                else:
                    analysis["suggested_distribution"] = "Unknown - further investigation needed"

            # Percentiles
            analysis["percentiles"] = {
                "1%": _safe_json_serialize(col_data.quantile(0.01)),
                "5%": _safe_json_serialize(col_data.quantile(0.05)),
                "10%": _safe_json_serialize(col_data.quantile(0.10)),
                "25%": _safe_json_serialize(col_data.quantile(0.25)),
                "50%": _safe_json_serialize(col_data.quantile(0.50)),
                "75%": _safe_json_serialize(col_data.quantile(0.75)),
                "90%": _safe_json_serialize(col_data.quantile(0.90)),
                "95%": _safe_json_serialize(col_data.quantile(0.95)),
                "99%": _safe_json_serialize(col_data.quantile(0.99))
            }

            distribution_analysis.append(analysis)

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "columns_analyzed": len(distribution_analysis),
            "scipy_available": SCIPY_AVAILABLE,
            "distribution_analysis": distribution_analysis
        }

        _store_result(session_id, "distributions", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing distributions: {str(e)}"


@tool
def analyze_correlations(session_id: str, method: str = "pearson", threshold: float = 0.5) -> str:
    """
    Calculate correlation matrix for numeric columns.
    Identifies highly correlated pairs and potential multicollinearity.

    Args:
        session_id: EDA session ID
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
        threshold: Threshold for highlighting high correlations (default 0.5)

    Returns:
        Correlation matrix and analysis of correlated pairs
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return json.dumps({
                "session_id": session_id,
                "message": "Need at least 2 numeric columns for correlation analysis."
            }, indent=2)

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)

        # Convert to serializable format
        corr_dict = {}
        for col in corr_matrix.columns:
            corr_dict[col] = {
                c: _safe_json_serialize(corr_matrix.loc[col, c])
                for c in corr_matrix.columns
            }

        # Find highly correlated pairs
        high_correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) >= threshold:
                    high_correlations.append({
                        "column_1": col1,
                        "column_2": col2,
                        "correlation": _safe_json_serialize(corr_val),
                        "strength": "strong" if abs(corr_val) >= 0.7 else "moderate",
                        "direction": "positive" if corr_val > 0 else "negative"
                    })

        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Find columns with many high correlations (potential multicollinearity)
        col_correlation_count = {}
        for hc in high_correlations:
            for col in [hc["column_1"], hc["column_2"]]:
                col_correlation_count[col] = col_correlation_count.get(col, 0) + 1

        multicollinearity_risk = [
            {"column": col, "high_correlation_count": count}
            for col, count in col_correlation_count.items()
            if count >= 2
        ]
        multicollinearity_risk.sort(key=lambda x: x["high_correlation_count"], reverse=True)

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "method": method,
            "threshold": threshold,
            "columns_analyzed": len(numeric_df.columns),
            "correlation_matrix": corr_dict,
            "high_correlations": high_correlations,
            "high_correlation_count": len(high_correlations),
            "multicollinearity_risk": multicollinearity_risk,
            "recommendations": []
        }

        # Add recommendations
        if multicollinearity_risk:
            result["recommendations"].append(
                f"Columns {[m['column'] for m in multicollinearity_risk[:3]]} may have multicollinearity issues"
            )
        if len(high_correlations) > 0:
            result["recommendations"].append(
                f"Found {len(high_correlations)} pairs with correlation >= {threshold}"
            )

        _store_result(session_id, "correlations", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing correlations: {str(e)}"


@tool
def analyze_categorical_columns(session_id: str, max_categories: int = 50) -> str:
    """
    Deep analysis of categorical columns.
    Includes frequency analysis, entropy, imbalance ratio, and rare categories.

    Args:
        session_id: EDA session ID
        max_categories: Maximum categories to show details for (default 50)

    Returns:
        Detailed analysis of each categorical column
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Get categorical columns
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if not cat_cols:
            return json.dumps({
                "session_id": session_id,
                "message": "No categorical columns found in the dataset."
            }, indent=2)

        categorical_analysis = []

        for col in cat_cols:
            col_data = df[col].dropna()
            value_counts = df[col].value_counts(dropna=False)

            analysis = {
                "column": col,
                "total_values": len(df),
                "non_null_values": len(col_data),
                "null_count": int(df[col].isna().sum()),
                "unique_categories": int(df[col].nunique()),
                "cardinality_ratio": round(df[col].nunique() / len(df), 4) if len(df) > 0 else 0
            }

            # Most and least common
            if len(value_counts) > 0:
                analysis["most_common"] = {
                    "value": str(value_counts.index[0]),
                    "count": int(value_counts.iloc[0]),
                    "percentage": round(value_counts.iloc[0] / len(df) * 100, 2)
                }
                analysis["least_common"] = {
                    "value": str(value_counts.index[-1]),
                    "count": int(value_counts.iloc[-1]),
                    "percentage": round(value_counts.iloc[-1] / len(df) * 100, 2)
                }

            # Imbalance ratio (most common / least common)
            if len(value_counts) > 1 and value_counts.iloc[-1] > 0:
                analysis["imbalance_ratio"] = round(value_counts.iloc[0] / value_counts.iloc[-1], 2)

            # Entropy (measure of uniformity)
            if len(col_data) > 0:
                probs = value_counts / len(df)
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
                max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
                analysis["entropy"] = _safe_json_serialize(entropy)
                analysis["normalized_entropy"] = _safe_json_serialize(entropy / max_entropy) if max_entropy > 0 else 0

            # Rare categories (< 1% of data)
            rare_threshold = len(df) * 0.01
            rare_categories = value_counts[value_counts < rare_threshold]
            analysis["rare_categories_count"] = len(rare_categories)
            analysis["rare_categories_percentage"] = round(len(rare_categories) / len(value_counts) * 100, 2) if len(value_counts) > 0 else 0

            # Category distribution (top categories)
            if len(value_counts) <= max_categories:
                analysis["category_distribution"] = [
                    {"value": str(v), "count": int(c), "percentage": round(c/len(df)*100, 2)}
                    for v, c in value_counts.items()
                ]
            else:
                analysis["category_distribution"] = [
                    {"value": str(v), "count": int(c), "percentage": round(c/len(df)*100, 2)}
                    for v, c in value_counts.head(max_categories).items()
                ]
                analysis["categories_not_shown"] = len(value_counts) - max_categories

            # Classify cardinality
            if analysis["unique_categories"] == 2:
                analysis["cardinality_type"] = "binary"
            elif analysis["unique_categories"] <= 10:
                analysis["cardinality_type"] = "low"
            elif analysis["unique_categories"] <= 50:
                analysis["cardinality_type"] = "medium"
            else:
                analysis["cardinality_type"] = "high"

            categorical_analysis.append(analysis)

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "categorical_columns_count": len(cat_cols),
            "categorical_analysis": categorical_analysis
        }

        _store_result(session_id, "categorical_analysis", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing categorical columns: {str(e)}"


@tool
def analyze_numerical_columns(session_id: str) -> str:
    """
    Deep analysis of numerical columns.
    Includes range analysis, zero/negative values, precision analysis, and binning suggestions.

    Args:
        session_id: EDA session ID

    Returns:
        Detailed analysis of each numerical column
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return json.dumps({
                "session_id": session_id,
                "message": "No numerical columns found in the dataset."
            }, indent=2)

        numerical_analysis = []

        for col in numeric_df.columns:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            analysis = {
                "column": col,
                "dtype": str(df[col].dtype),
                "total_values": len(df),
                "non_null_values": len(col_data),
                "null_count": int(df[col].isna().sum()),
                "null_percentage": round(df[col].isna().sum() / len(df) * 100, 2)
            }

            # Basic stats
            analysis["statistics"] = {
                "mean": _safe_json_serialize(col_data.mean()),
                "median": _safe_json_serialize(col_data.median()),
                "std": _safe_json_serialize(col_data.std()),
                "min": _safe_json_serialize(col_data.min()),
                "max": _safe_json_serialize(col_data.max()),
                "range": _safe_json_serialize(col_data.max() - col_data.min()),
                "sum": _safe_json_serialize(col_data.sum())
            }

            # Value characteristics
            analysis["value_characteristics"] = {
                "unique_count": int(col_data.nunique()),
                "zero_count": int((col_data == 0).sum()),
                "zero_percentage": round((col_data == 0).sum() / len(col_data) * 100, 2),
                "negative_count": int((col_data < 0).sum()),
                "negative_percentage": round((col_data < 0).sum() / len(col_data) * 100, 2),
                "positive_count": int((col_data > 0).sum()),
                "positive_percentage": round((col_data > 0).sum() / len(col_data) * 100, 2)
            }

            # Check if likely integer
            if col_data.dtype in ['float64', 'float32']:
                is_integer_like = (col_data == col_data.round()).all()
                analysis["value_characteristics"]["is_integer_like"] = bool(is_integer_like)

            # Coefficient of variation
            if col_data.mean() != 0:
                cv = col_data.std() / abs(col_data.mean())
                analysis["coefficient_of_variation"] = _safe_json_serialize(cv)

            # Range analysis
            range_val = col_data.max() - col_data.min()
            if range_val > 0:
                analysis["range_analysis"] = {
                    "range": _safe_json_serialize(range_val),
                    "range_to_mean_ratio": _safe_json_serialize(range_val / abs(col_data.mean())) if col_data.mean() != 0 else None,
                    "suggested_bins": min(int(np.sqrt(len(col_data))), 50)
                }

            # Detect if likely ID column
            is_sequential = (col_data.sort_values().diff().dropna() == 1).all() if len(col_data) > 1 else False
            is_unique = col_data.nunique() == len(col_data)
            analysis["likely_id_column"] = bool(is_sequential or (is_unique and col_data.dtype in ['int64', 'int32']))

            # Precision analysis for floats
            if col_data.dtype in ['float64', 'float32']:
                # Check decimal places
                decimal_places = col_data.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                analysis["precision"] = {
                    "max_decimal_places": int(decimal_places.max()),
                    "common_decimal_places": int(decimal_places.mode().iloc[0]) if len(decimal_places.mode()) > 0 else 0
                }

            numerical_analysis.append(analysis)

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "numerical_columns_count": len(numeric_df.columns),
            "numerical_analysis": numerical_analysis
        }

        _store_result(session_id, "numerical_analysis", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing numerical columns: {str(e)}"


@tool
def analyze_datetime_columns(session_id: str) -> str:
    """
    Analyze datetime columns for temporal patterns.
    Includes range, gaps, seasonality indicators, and time-based statistics.

    Args:
        session_id: EDA session ID

    Returns:
        Temporal analysis for each datetime column
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Get datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns.tolist()

        # Also check object columns that might be datetime
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
            except:
                pass

        if not datetime_cols:
            return json.dumps({
                "session_id": session_id,
                "message": "No datetime columns found in the dataset."
            }, indent=2)

        datetime_analysis = []

        for col in datetime_cols:
            try:
                col_data = pd.to_datetime(df[col], errors='coerce').dropna()

                if len(col_data) == 0:
                    continue

                analysis = {
                    "column": col,
                    "total_values": len(df),
                    "non_null_values": len(col_data),
                    "null_count": int(df[col].isna().sum() + (len(df) - len(df[col].dropna())))
                }

                # Range
                analysis["range"] = {
                    "min": col_data.min().isoformat(),
                    "max": col_data.max().isoformat(),
                    "span_days": (col_data.max() - col_data.min()).days
                }

                # Unique dates
                analysis["unique_dates"] = int(col_data.dt.date.nunique())

                # Time components distribution
                analysis["time_distribution"] = {
                    "years": sorted(col_data.dt.year.unique().tolist()),
                    "year_count": int(col_data.dt.year.nunique()),
                    "month_distribution": col_data.dt.month.value_counts().sort_index().to_dict(),
                    "weekday_distribution": col_data.dt.dayofweek.value_counts().sort_index().to_dict()
                }

                # Check for patterns
                if len(col_data) > 1:
                    sorted_dates = col_data.sort_values()
                    gaps = sorted_dates.diff().dropna()

                    if len(gaps) > 0:
                        analysis["gap_analysis"] = {
                            "min_gap": str(gaps.min()),
                            "max_gap": str(gaps.max()),
                            "mean_gap": str(gaps.mean()),
                            "median_gap": str(gaps.median())
                        }

                # Recent data check
                today = pd.Timestamp.now()
                days_since_last = (today - col_data.max()).days
                analysis["recency"] = {
                    "days_since_last_record": days_since_last,
                    "is_recent": days_since_last < 30
                }

                # Completeness for date range
                if analysis["range"]["span_days"] > 0:
                    expected_days = analysis["range"]["span_days"] + 1
                    actual_days = analysis["unique_dates"]
                    analysis["date_completeness"] = round(actual_days / expected_days * 100, 2)

                datetime_analysis.append(analysis)

            except Exception as e:
                datetime_analysis.append({
                    "column": col,
                    "error": str(e)
                })

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "datetime_columns_count": len(datetime_analysis),
            "datetime_analysis": datetime_analysis
        }

        _store_result(session_id, "datetime_analysis", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing datetime columns: {str(e)}"


@tool
def check_data_quality(session_id: str) -> str:
    """
    Comprehensive data quality assessment.
    Calculates a data quality score based on completeness, uniqueness, validity, and consistency.

    Args:
        session_id: EDA session ID

    Returns:
        Data quality report with scores and recommendations
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        quality_metrics = {
            "completeness": {},
            "uniqueness": {},
            "validity": {},
            "consistency": {}
        }

        # 1. Completeness - measure of missing values
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells * 100

        quality_metrics["completeness"] = {
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "score": round(completeness_score, 2),
            "columns_with_missing": [col for col in df.columns if df[col].isna().any()]
        }

        # 2. Uniqueness - measure of duplicate rows
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = (len(df) - duplicate_rows) / len(df) * 100

        quality_metrics["uniqueness"] = {
            "total_rows": len(df),
            "duplicate_rows": int(duplicate_rows),
            "score": round(uniqueness_score, 2)
        }

        # 3. Validity - check for invalid/unexpected values
        validity_issues = []
        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            # Check for infinity
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validity_issues.append({
                    "column": col,
                    "issue": "Contains infinity values",
                    "count": int(inf_count)
                })

            # Check for unexpected negative values in typically positive columns
            if any(kw in col.lower() for kw in ["count", "quantity", "amount", "price", "value", "total"]):
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    validity_issues.append({
                        "column": col,
                        "issue": "Unexpected negative values",
                        "count": int(neg_count)
                    })

        validity_score = 100 - (len(validity_issues) / max(len(df.columns), 1) * 100)
        quality_metrics["validity"] = {
            "issues_found": len(validity_issues),
            "score": round(validity_score, 2),
            "issues": validity_issues[:10]  # Top 10 issues
        }

        # 4. Consistency - check for consistent formatting
        consistency_issues = []

        for col in df.select_dtypes(include=["object"]).columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Check for mixed case issues
            if col_data.str.isupper().any() and col_data.str.islower().any():
                consistency_issues.append({
                    "column": col,
                    "issue": "Mixed case values",
                    "sample": col_data.head(5).tolist()
                })

            # Check for leading/trailing whitespace
            whitespace_count = (col_data != col_data.str.strip()).sum()
            if whitespace_count > 0:
                consistency_issues.append({
                    "column": col,
                    "issue": "Values with leading/trailing whitespace",
                    "count": int(whitespace_count)
                })

        consistency_score = 100 - (len(consistency_issues) / max(len(df.columns), 1) * 100)
        quality_metrics["consistency"] = {
            "issues_found": len(consistency_issues),
            "score": round(consistency_score, 2),
            "issues": consistency_issues[:10]
        }

        # Overall quality score (weighted average)
        overall_score = (
            completeness_score * 0.35 +
            uniqueness_score * 0.25 +
            validity_score * 0.25 +
            consistency_score * 0.15
        )

        # Quality grade
        if overall_score >= 90:
            grade = "A"
            interpretation = "Excellent data quality"
        elif overall_score >= 80:
            grade = "B"
            interpretation = "Good data quality with minor issues"
        elif overall_score >= 70:
            grade = "C"
            interpretation = "Acceptable data quality, some issues need attention"
        elif overall_score >= 60:
            grade = "D"
            interpretation = "Poor data quality, significant issues present"
        else:
            grade = "F"
            interpretation = "Critical data quality issues, not suitable for analysis"

        # Recommendations
        recommendations = []
        if completeness_score < 90:
            recommendations.append("Address missing values through imputation or removal")
        if uniqueness_score < 95:
            recommendations.append("Investigate and remove duplicate records")
        if validity_score < 90:
            recommendations.append("Review and fix invalid/unexpected values")
        if consistency_score < 90:
            recommendations.append("Standardize data formatting for consistency")

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "overall_quality_score": round(overall_score, 2),
            "grade": grade,
            "interpretation": interpretation,
            "dimension_scores": {
                "completeness": quality_metrics["completeness"]["score"],
                "uniqueness": quality_metrics["uniqueness"]["score"],
                "validity": quality_metrics["validity"]["score"],
                "consistency": quality_metrics["consistency"]["score"]
            },
            "detailed_metrics": quality_metrics,
            "recommendations": recommendations
        }

        _store_result(session_id, "data_quality", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error checking data quality: {str(e)}"


@tool
def analyze_column_relationships(session_id: str) -> str:
    """
    Analyze relationships between columns.
    Detects potential functional dependencies, key columns, and relationships.

    Args:
        session_id: EDA session ID

    Returns:
        Analysis of column relationships and potential dependencies
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        relationships = {
            "potential_keys": [],
            "functional_dependencies": [],
            "categorical_numeric_relationships": [],
            "column_pairs_analysis": []
        }

        # 1. Find potential key columns
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            null_count = df[col].isna().sum()

            if unique_ratio == 1 and null_count == 0:
                relationships["potential_keys"].append({
                    "column": col,
                    "type": "primary_key",
                    "reason": "100% unique, no nulls"
                })
            elif unique_ratio > 0.95 and null_count == 0:
                relationships["potential_keys"].append({
                    "column": col,
                    "type": "candidate_key",
                    "unique_ratio": round(unique_ratio, 4),
                    "reason": f"{round(unique_ratio*100, 1)}% unique, no nulls"
                })

        # 2. Find functional dependencies (A -> B)
        # A column functionally determines another if each value of A maps to exactly one value of B
        cat_cols = df.select_dtypes(include=["object", "category"]).columns[:10]  # Limit for performance

        for col_a in cat_cols:
            for col_b in cat_cols:
                if col_a == col_b:
                    continue

                # Check if col_a -> col_b
                grouped = df.groupby(col_a)[col_b].nunique()
                if (grouped == 1).all():
                    relationships["functional_dependencies"].append({
                        "determinant": col_a,
                        "dependent": col_b,
                        "relationship": f"{col_a} -> {col_b}",
                        "note": f"Each value of {col_a} maps to exactly one value of {col_b}"
                    })

        # 3. Categorical-Numeric relationships
        cat_cols_all = df.select_dtypes(include=["object", "category"]).columns[:5]
        num_cols = df.select_dtypes(include=[np.number]).columns[:5]

        for cat_col in cat_cols_all:
            if df[cat_col].nunique() > 20:  # Skip high cardinality
                continue

            for num_col in num_cols:
                # Calculate group statistics
                group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])

                # Check if means differ significantly
                overall_mean = df[num_col].mean()
                if overall_mean != 0:
                    max_deviation = (group_stats['mean'] - overall_mean).abs().max() / abs(overall_mean)
                    if max_deviation > 0.2:  # 20% deviation threshold
                        relationships["categorical_numeric_relationships"].append({
                            "categorical_column": cat_col,
                            "numeric_column": num_col,
                            "relationship_strength": round(max_deviation, 4),
                            "note": f"Means of {num_col} vary by {round(max_deviation*100, 1)}% across {cat_col} groups"
                        })

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "potential_key_columns": len(relationships["potential_keys"]),
            "functional_dependencies_found": len(relationships["functional_dependencies"]),
            "relationships": relationships
        }

        _store_result(session_id, "column_relationships", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing relationships: {str(e)}"


@tool
def generate_eda_plan(session_id: str) -> str:
    """
    Generate a customized EDA plan based on the data characteristics.
    Uses analysis results to suggest domain-specific analyses.

    Args:
        session_id: EDA session ID

    Returns:
        Customized EDA plan with step-by-step recommendations
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Analyze data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # Check for object columns that might be datetime
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                pd.to_datetime(df[col].head(10), errors='raise')
                datetime_cols.append(col)
            except:
                pass

        # Completed analyses
        completed = session.get("analyses_completed", [])

        # Build EDA plan
        plan = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "data_profile": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(cat_cols),
                "datetime_columns": len(datetime_cols)
            },
            "completed_analyses": completed,
            "recommended_analyses": [],
            "domain_specific_analyses": [],
            "advanced_analyses": []
        }

        # Basic analyses (always recommended)
        basic = [
            ("get_basic_statistics", "Basic statistics for numeric columns", "basic_statistics" not in completed and len(numeric_cols) > 0),
            ("check_missing_values", "Missing value analysis", "missing_values" not in completed),
            ("check_duplicates", "Duplicate detection", "duplicates" not in completed),
            ("check_data_types", "Data type analysis", "data_types" not in completed),
            ("check_data_quality", "Data quality assessment", "data_quality" not in completed)
        ]

        for tool_name, description, should_run in basic:
            if should_run:
                plan["recommended_analyses"].append({
                    "tool": tool_name,
                    "description": description,
                    "priority": "high"
                })

        # Column-specific analyses
        if len(numeric_cols) > 0 and "numerical_analysis" not in completed:
            plan["recommended_analyses"].append({
                "tool": "analyze_numerical_columns",
                "description": "Deep numerical analysis",
                "priority": "medium"
            })

        if len(cat_cols) > 0 and "categorical_analysis" not in completed:
            plan["recommended_analyses"].append({
                "tool": "analyze_categorical_columns",
                "description": "Categorical column analysis",
                "priority": "medium"
            })

        if len(datetime_cols) > 0 and "datetime_analysis" not in completed:
            plan["recommended_analyses"].append({
                "tool": "analyze_datetime_columns",
                "description": "Temporal analysis",
                "priority": "medium"
            })

        # Advanced analyses
        if len(numeric_cols) > 1 and "correlations" not in completed:
            plan["advanced_analyses"].append({
                "tool": "analyze_correlations",
                "description": "Correlation analysis between numeric columns",
                "priority": "medium"
            })

        if len(numeric_cols) > 0 and "distributions" not in completed:
            plan["advanced_analyses"].append({
                "tool": "analyze_distributions",
                "description": "Distribution analysis with normality tests",
                "priority": "medium"
            })

        if len(numeric_cols) > 0 and "outliers" not in completed:
            plan["advanced_analyses"].append({
                "tool": "detect_outliers",
                "description": "Outlier detection",
                "priority": "medium"
            })

        if "column_relationships" not in completed:
            plan["advanced_analyses"].append({
                "tool": "analyze_column_relationships",
                "description": "Column relationship analysis",
                "priority": "low"
            })

        # Domain-specific suggestions based on column names
        col_names_lower = [c.lower() for c in df.columns]

        # Financial domain
        if any(kw in " ".join(col_names_lower) for kw in ["price", "amount", "cost", "revenue", "profit", "transaction"]):
            plan["domain_specific_analyses"].append({
                "domain": "Financial",
                "suggestions": [
                    "Analyze transaction patterns over time",
                    "Check for outliers in monetary values",
                    "Look for seasonal patterns in transactions"
                ]
            })

        # Customer/User domain
        if any(kw in " ".join(col_names_lower) for kw in ["customer", "user", "client", "member"]):
            plan["domain_specific_analyses"].append({
                "domain": "Customer Analytics",
                "suggestions": [
                    "Segment customers by behavior",
                    "Analyze customer distribution by geography",
                    "Check customer activity patterns"
                ]
            })

        # Time series domain
        if datetime_cols:
            plan["domain_specific_analyses"].append({
                "domain": "Time Series",
                "suggestions": [
                    "Check for trends over time",
                    "Identify seasonality patterns",
                    "Analyze data completeness over time periods"
                ]
            })

        _store_result(session_id, "eda_plan", plan)

        return json.dumps(plan, indent=2)

    except Exception as e:
        return f"Error generating EDA plan: {str(e)}"


@tool
def execute_custom_analysis(session_id: str, analysis_code: str) -> str:
    """
    Execute custom pandas analysis code on the loaded DataFrame.
    Allows flexible analysis beyond predefined tools.

    Args:
        session_id: EDA session ID
        analysis_code: Python code using 'df' as the DataFrame variable
                      Example: "df.groupby('COUNTRY')['AMOUNT'].mean()"

    Returns:
        Result of the custom analysis
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Safety check - limit allowed operations
        dangerous_keywords = ["exec", "eval", "import", "open", "write", "delete", "drop", "remove", "system", "os."]
        code_lower = analysis_code.lower()
        for kw in dangerous_keywords:
            if kw in code_lower:
                return f"Error: '{kw}' operation not allowed for safety reasons."

        # Create safe execution environment
        safe_globals = {
            "df": df,
            "pd": pd,
            "np": np,
            "__builtins__": {
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "range": range,
                "enumerate": enumerate,
                "zip": zip
            }
        }

        # Execute the analysis
        result = eval(analysis_code, safe_globals)

        # Convert result to JSON-serializable format
        if isinstance(result, pd.DataFrame):
            output = {
                "type": "DataFrame",
                "shape": result.shape,
                "columns": list(result.columns),
                "data": result.head(100).to_dict(orient="records")
            }
        elif isinstance(result, pd.Series):
            output = {
                "type": "Series",
                "name": result.name,
                "length": len(result),
                "data": result.head(100).to_dict()
            }
        else:
            output = {
                "type": type(result).__name__,
                "value": _safe_json_serialize(result)
            }

        return json.dumps({
            "session_id": session_id,
            "analysis_code": analysis_code,
            "result": output
        }, indent=2, default=str)

    except Exception as e:
        return f"Error executing analysis: {str(e)}"


@tool
def get_eda_summary(session_id: str) -> str:
    """
    Generate a comprehensive EDA summary report.
    Collects all analysis results and provides key insights.

    Args:
        session_id: EDA session ID

    Returns:
        Complete EDA summary with all findings and recommendations
    """
    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Build comprehensive summary
        summary = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "generated_at": datetime.now().isoformat(),
            "data_overview": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            },
            "column_summary": {
                "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
                "datetime": df.select_dtypes(include=["datetime64"]).columns.tolist(),
                "boolean": df.select_dtypes(include=["bool"]).columns.tolist()
            },
            "analyses_completed": session.get("analyses_completed", []),
            "key_findings": [],
            "data_issues": [],
            "recommendations": []
        }

        results = session.get("results", {})

        # Extract key findings from each analysis

        # Missing values findings
        if "missing_values" in results:
            mv = results["missing_values"]
            if mv.get("overall_missing_percentage", 0) > 5:
                summary["data_issues"].append({
                    "issue": "Significant missing data",
                    "details": f"{mv['overall_missing_percentage']}% of data is missing",
                    "affected_columns": mv.get("columns_with_missing", [])[:5]
                })

        # Duplicate findings
        if "duplicates" in results:
            dup = results["duplicates"]
            if dup.get("duplicate_percentage", 0) > 1:
                summary["data_issues"].append({
                    "issue": "Duplicate rows detected",
                    "details": f"{dup['duplicate_percentage']}% duplicate rows ({dup['duplicate_rows']} rows)"
                })

        # Data quality score
        if "data_quality" in results:
            dq = results["data_quality"]
            summary["key_findings"].append({
                "finding": "Data Quality Assessment",
                "score": dq.get("overall_quality_score"),
                "grade": dq.get("grade"),
                "interpretation": dq.get("interpretation")
            })
            summary["recommendations"].extend(dq.get("recommendations", []))

        # Correlation findings
        if "correlations" in results:
            corr = results["correlations"]
            if corr.get("high_correlation_count", 0) > 0:
                summary["key_findings"].append({
                    "finding": "High correlations detected",
                    "count": corr["high_correlation_count"],
                    "pairs": corr.get("high_correlations", [])[:3]
                })

        # Outlier findings
        if "outliers" in results:
            out = results["outliers"]
            if out.get("total_outliers_found", 0) > 0:
                summary["key_findings"].append({
                    "finding": "Outliers detected",
                    "total": out["total_outliers_found"],
                    "columns_affected": out.get("columns_with_outliers", [])
                })

        # Distribution findings
        if "distributions" in results:
            dist = results["distributions"]
            non_normal = [
                a["column"] for a in dist.get("distribution_analysis", [])
                if a.get("normality_test", {}).get("is_normal") == False
            ]
            if non_normal:
                summary["key_findings"].append({
                    "finding": "Non-normal distributions detected",
                    "columns": non_normal[:5]
                })

        # Generate final recommendations
        if not summary["recommendations"]:
            summary["recommendations"] = [
                "Review missing value patterns and decide on imputation strategy",
                "Investigate outliers to determine if they are valid data points",
                "Consider feature engineering based on correlation analysis"
            ]

        # Analysis coverage
        total_possible = 12  # Number of main analysis tools
        completed = len(summary["analyses_completed"])
        summary["analysis_coverage"] = {
            "completed": completed,
            "total_possible": total_possible,
            "percentage": round(completed / total_possible * 100, 1)
        }

        _store_result(session_id, "eda_summary", summary)

        return json.dumps(summary, indent=2, default=str)

    except Exception as e:
        return f"Error generating summary: {str(e)}"


@tool
def get_session_info(session_id: str) -> str:
    """
    Get information about an EDA session.
    Shows what analyses have been run and current state.

    Args:
        session_id: EDA session ID

    Returns:
        Session information including completed analyses
    """
    try:
        if session_id not in _eda_sessions:
            available = list(_eda_sessions.keys()) if _eda_sessions else "none"
            return f"Session '{session_id}' not found. Available sessions: {available}"

        session = _eda_sessions[session_id]

        info = {
            "session_id": session_id,
            "table_name": session.get("table_name"),
            "created_at": session.get("created_at"),
            "row_count": session.get("row_count"),
            "column_count": session.get("column_count"),
            "analyses_completed": session.get("analyses_completed", []),
            "analyses_count": len(session.get("analyses_completed", [])),
            "results_available": list(session.get("results", {}).keys())
        }

        return json.dumps(info, indent=2)

    except Exception as e:
        return f"Error getting session info: {str(e)}"


@tool
def list_eda_sessions() -> str:
    """
    List all active EDA sessions.

    Returns:
        List of all EDA sessions with their basic info
    """
    try:
        if not _eda_sessions:
            return "No active EDA sessions. Use load_table_to_pandas() to start a new session."

        sessions = []
        for sid, session in _eda_sessions.items():
            sessions.append({
                "session_id": sid,
                "table_name": session.get("table_name"),
                "created_at": session.get("created_at"),
                "analyses_completed": len(session.get("analyses_completed", []))
            })

        return json.dumps({
            "total_sessions": len(sessions),
            "sessions": sessions
        }, indent=2)

    except Exception as e:
        return f"Error listing sessions: {str(e)}"


# =============================================================================
# VISUALIZATION TOOLS
# =============================================================================

def _get_viz_path(session_id: str, chart_type: str, suffix: str = "") -> str:
    """Generate a unique path for saving visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{session_id}_{chart_type}_{suffix}_{timestamp}.png" if suffix else f"{session_id}_{chart_type}_{timestamp}.png"
    return os.path.join(VIZ_OUTPUT_DIR, filename)


def _setup_plot_style(figsize=(12, 8), style="darkgrid"):
    """Set up matplotlib/seaborn plot style."""
    if not VISUALIZATION_AVAILABLE:
        return None

    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    sns.set_palette("husl")
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


@tool
def plot_histogram(session_id: str, column_name: Optional[str] = None, bins: int = 30, show_kde: bool = True) -> str:
    """
    Create histogram plots for numeric columns to analyze distributions.
    Shows frequency distribution with optional KDE (Kernel Density Estimation) overlay.

    Args:
        session_id: EDA session ID
        column_name: Specific column to plot (optional, plots all numeric if not specified)
        bins: Number of bins for histogram (default 30)
        show_kde: Whether to show KDE curve overlay (default True)

    Returns:
        Path to saved visualization file(s) and distribution insights
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries (matplotlib, seaborn) not installed. Run: pip install matplotlib seaborn"

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Get columns to plot
        if column_name:
            column_name = column_name.upper()
            col_map = {c.upper(): c for c in df.columns}
            if column_name not in col_map:
                return f"Error: Column '{column_name}' not found."
            columns = [col_map[column_name]]
        else:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            return "Error: No numeric columns found for histogram."

        saved_files = []
        insights = []

        # Create individual histograms
        for col in columns[:6]:  # Limit to 6 columns
            fig, ax = plt.subplots(figsize=(10, 6))

            col_data = df[col].dropna()

            # Plot histogram with KDE
            if show_kde and len(col_data) > 1:
                sns.histplot(col_data, bins=bins, kde=True, ax=ax, color='#3498db', alpha=0.7)
            else:
                sns.histplot(col_data, bins=bins, ax=ax, color='#3498db', alpha=0.7)

            # Add statistics annotations
            mean_val = col_data.mean()
            median_val = col_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')

            ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()

            # Save figure
            filepath = _get_viz_path(session_id, "histogram", col)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            saved_files.append(filepath)

            # Add insight
            skewness = stats.skew(col_data) if SCIPY_AVAILABLE else None
            insights.append({
                "column": col,
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "skewness": round(skewness, 4) if skewness else None,
                "distribution_shape": "right-skewed" if skewness and skewness > 0.5 else "left-skewed" if skewness and skewness < -0.5 else "approximately normal"
            })

        result = {
            "session_id": session_id,
            "visualization_type": "histogram",
            "columns_plotted": len(saved_files),
            "saved_files": saved_files,
            "insights": insights,
            "note": "Histograms show the frequency distribution. Red dashed line = Mean, Green dash-dot = Median."
        }

        _store_result(session_id, "viz_histograms", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating histogram: {str(e)}"


@tool
def plot_boxplot(session_id: str, column_name: Optional[str] = None, group_by: Optional[str] = None) -> str:
    """
    Create box plots to visualize distributions and identify outliers.
    Shows median, quartiles, and outliers for numeric columns.

    Args:
        session_id: EDA session ID
        column_name: Specific numeric column to plot (optional)
        group_by: Categorical column to group by (optional, creates grouped boxplot)

    Returns:
        Path to saved visualization and outlier insights
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        saved_files = []
        insights = []

        # Get numeric columns
        if column_name:
            col_map = {c.upper(): c for c in df.columns}
            if column_name.upper() not in col_map:
                return f"Error: Column '{column_name}' not found."
            numeric_cols = [col_map[column_name.upper()]]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return "Error: No numeric columns found for boxplot."

        # Grouped boxplot
        if group_by:
            col_map = {c.upper(): c for c in df.columns}
            if group_by.upper() not in col_map:
                return f"Error: Group column '{group_by}' not found."
            group_col = col_map[group_by.upper()]

            for col in numeric_cols[:4]:  # Limit to 4 columns
                fig, ax = plt.subplots(figsize=(12, 6))

                sns.boxplot(data=df, x=group_col, y=col, ax=ax, palette="husl")
                ax.set_title(f'Box Plot: {col} by {group_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(group_col, fontsize=12)
                ax.set_ylabel(col, fontsize=12)
                plt.xticks(rotation=45, ha='right')

                filepath = _get_viz_path(session_id, "boxplot_grouped", f"{col}_by_{group_col}")
                plt.tight_layout()
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                saved_files.append(filepath)
        else:
            # Single boxplot for all numeric columns
            if len(numeric_cols) > 1:
                # Normalize data for comparison
                fig, ax = plt.subplots(figsize=(14, 8))

                # Create boxplot for all numeric columns
                df_numeric = df[numeric_cols].dropna()

                # Standardize for visualization if ranges differ significantly
                df_plot = (df_numeric - df_numeric.mean()) / df_numeric.std()

                sns.boxplot(data=df_plot, ax=ax, palette="husl")
                ax.set_title('Box Plots - All Numeric Columns (Standardized)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Standardized Value', fontsize=12)
                plt.xticks(rotation=45, ha='right')

                filepath = _get_viz_path(session_id, "boxplot", "all_numeric")
                plt.tight_layout()
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                saved_files.append(filepath)

            # Individual boxplots
            for col in numeric_cols[:4]:
                fig, ax = plt.subplots(figsize=(8, 6))

                col_data = df[col].dropna()

                # Box plot with individual points
                sns.boxplot(y=col_data, ax=ax, color='#3498db', width=0.3)
                sns.stripplot(y=col_data, ax=ax, color='#e74c3c', alpha=0.3, size=3)

                # Calculate outlier info
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]

                ax.set_title(f'Box Plot: {col}', fontsize=14, fontweight='bold')
                ax.set_ylabel(col, fontsize=12)

                filepath = _get_viz_path(session_id, "boxplot", col)
                plt.tight_layout()
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                saved_files.append(filepath)

                insights.append({
                    "column": col,
                    "Q1": round(Q1, 4),
                    "median": round(col_data.median(), 4),
                    "Q3": round(Q3, 4),
                    "IQR": round(IQR, 4),
                    "outlier_count": len(outliers),
                    "outlier_percentage": round(len(outliers) / len(col_data) * 100, 2)
                })

        result = {
            "session_id": session_id,
            "visualization_type": "boxplot",
            "columns_plotted": len(numeric_cols),
            "grouped_by": group_by,
            "saved_files": saved_files,
            "outlier_insights": insights,
            "note": "Box plots show median (line), IQR (box), and outliers (points beyond whiskers)."
        }

        _store_result(session_id, "viz_boxplots", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating boxplot: {str(e)}"


@tool
def plot_bar_chart(session_id: str, column_name: str, top_n: int = 15, horizontal: bool = False) -> str:
    """
    Create bar charts to visualize categorical variable distributions.
    Shows value counts for categorical columns.

    Args:
        session_id: EDA session ID
        column_name: Categorical column to plot
        top_n: Number of top categories to show (default 15)
        horizontal: Whether to create horizontal bar chart (default False)

    Returns:
        Path to saved visualization and category insights
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Find column
        col_map = {c.upper(): c for c in df.columns}
        if column_name.upper() not in col_map:
            return f"Error: Column '{column_name}' not found."

        col = col_map[column_name.upper()]

        # Get value counts
        value_counts = df[col].value_counts().head(top_n)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = sns.color_palette("husl", len(value_counts))

        if horizontal:
            bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels(value_counts.index)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ax.invert_yaxis()  # Highest at top

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, value_counts.values)):
                ax.text(val + max(value_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:,}', va='center', fontsize=10)
        else:
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

            # Add value labels
            for bar, val in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values) * 0.01,
                       f'{val:,}', ha='center', fontsize=10)

        ax.set_title(f'Distribution of {col} (Top {top_n})', fontsize=14, fontweight='bold')

        # Save figure
        filepath = _get_viz_path(session_id, "bar_chart", col)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Calculate insights
        total = value_counts.sum()
        top_category = value_counts.index[0]
        top_pct = value_counts.iloc[0] / total * 100

        result = {
            "session_id": session_id,
            "visualization_type": "bar_chart",
            "column": col,
            "saved_file": filepath,
            "insights": {
                "total_records": int(total),
                "unique_categories": int(df[col].nunique()),
                "top_category": str(top_category),
                "top_category_count": int(value_counts.iloc[0]),
                "top_category_percentage": round(top_pct, 2),
                "categories_shown": len(value_counts),
                "categories_hidden": max(0, df[col].nunique() - top_n)
            }
        }

        _store_result(session_id, f"viz_bar_{col}", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


@tool
def plot_scatter(session_id: str, x_column: str, y_column: str, color_by: Optional[str] = None, size_by: Optional[str] = None) -> str:
    """
    Create scatter plots for bivariate analysis between two numeric columns.
    Optionally color or size points by another variable.

    Args:
        session_id: EDA session ID
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_by: Optional column to color points by (categorical recommended)
        size_by: Optional numeric column to size points by

    Returns:
        Path to saved visualization and correlation insights
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        col_map = {c.upper(): c for c in df.columns}

        # Validate columns
        if x_column.upper() not in col_map or y_column.upper() not in col_map:
            return f"Error: Column not found. Available: {list(df.columns)}"

        x_col = col_map[x_column.upper()]
        y_col = col_map[y_column.upper()]

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter_kwargs = {"alpha": 0.6, "edgecolors": "white", "linewidth": 0.5}

        if color_by and color_by.upper() in col_map:
            color_col = col_map[color_by.upper()]
            unique_colors = df[color_col].nunique()

            if unique_colors <= 10:
                # Categorical coloring
                for i, cat in enumerate(df[color_col].unique()):
                    mask = df[color_col] == cat
                    size = df[col_map[size_by.upper()]][mask] if size_by and size_by.upper() in col_map else 50
                    ax.scatter(df[x_col][mask], df[y_col][mask], s=size, label=str(cat), **scatter_kwargs)
                ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Continuous coloring
                size = df[col_map[size_by.upper()]] if size_by and size_by.upper() in col_map else 50
                scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], s=size, cmap='viridis', **scatter_kwargs)
                plt.colorbar(scatter, label=color_col)
        else:
            size = df[col_map[size_by.upper()]] if size_by and size_by.upper() in col_map else 50
            ax.scatter(df[x_col], df[y_col], s=size, c='#3498db', **scatter_kwargs)

        # Add regression line
        try:
            x_clean = df[x_col].dropna()
            y_clean = df[y_col].dropna()

            # Align the data
            mask = df[x_col].notna() & df[y_col].notna()
            x_vals = df.loc[mask, x_col]
            y_vals = df.loc[mask, y_col]

            if len(x_vals) > 2:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend Line')
        except:
            pass

        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')

        # Calculate correlation
        mask = df[x_col].notna() & df[y_col].notna()
        correlation = df.loc[mask, [x_col, y_col]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save figure
        filepath = _get_viz_path(session_id, "scatter", f"{x_col}_vs_{y_col}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Determine relationship strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            relationship = "strong"
        elif abs_corr >= 0.4:
            relationship = "moderate"
        elif abs_corr >= 0.2:
            relationship = "weak"
        else:
            relationship = "very weak/no"

        result = {
            "session_id": session_id,
            "visualization_type": "scatter_plot",
            "x_column": x_col,
            "y_column": y_col,
            "color_by": color_by,
            "size_by": size_by,
            "saved_file": filepath,
            "insights": {
                "correlation": round(correlation, 4),
                "correlation_type": "positive" if correlation > 0 else "negative",
                "relationship_strength": relationship,
                "data_points": int(mask.sum())
            }
        }

        _store_result(session_id, f"viz_scatter_{x_col}_{y_col}", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


@tool
def plot_correlation_heatmap(session_id: str, method: str = "pearson", annotate: bool = True) -> str:
    """
    Create a correlation heatmap for all numeric columns.
    Visualizes the correlation matrix with color-coded cells.

    Args:
        session_id: EDA session ID
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
        annotate: Whether to show correlation values on cells (default True)

    Returns:
        Path to saved heatmap and high correlation pairs
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return "Error: Need at least 2 numeric columns for correlation heatmap."

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)

        # Create heatmap
        n_cols = len(corr_matrix.columns)
        fig_size = max(10, n_cols * 0.8)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

        # Create mask for upper triangle (optional - show full matrix)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        # Generate heatmap
        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annotate and n_cols <= 15,  # Only annotate if not too many columns
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            fmt='.2f',
            ax=ax
        )

        ax.set_title(f'Correlation Heatmap ({method.capitalize()})', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Save figure
        filepath = _get_viz_path(session_id, "correlation_heatmap", method)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Find high correlations
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= 0.5:
                    high_correlations.append({
                        "column_1": corr_matrix.columns[i],
                        "column_2": corr_matrix.columns[j],
                        "correlation": round(corr, 4),
                        "strength": "strong" if abs(corr) >= 0.7 else "moderate"
                    })

        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        result = {
            "session_id": session_id,
            "visualization_type": "correlation_heatmap",
            "method": method,
            "saved_file": filepath,
            "columns_analyzed": len(numeric_df.columns),
            "high_correlations": high_correlations[:10],  # Top 10
            "total_high_correlations": len(high_correlations),
            "note": "Red = positive correlation, Blue = negative correlation. Values closer to 1 or -1 indicate stronger relationships."
        }

        _store_result(session_id, "viz_correlation_heatmap", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating correlation heatmap: {str(e)}"


@tool
def plot_pairplot(session_id: str, columns: Optional[str] = None, hue: Optional[str] = None, max_columns: int = 5) -> str:
    """
    Create a pairplot (scatter matrix) for multivariate analysis.
    Shows relationships between all pairs of numeric variables.

    Args:
        session_id: EDA session ID
        columns: Comma-separated list of columns (optional, uses top numeric columns)
        hue: Categorical column to color points by (optional)
        max_columns: Maximum number of columns to include (default 5)

    Returns:
        Path to saved pairplot visualization
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        col_map = {c.upper(): c for c in df.columns}

        # Get columns to plot
        if columns:
            col_list = [c.strip() for c in columns.split(",")]
            plot_cols = []
            for c in col_list:
                if c.upper() in col_map:
                    plot_cols.append(col_map[c.upper()])
        else:
            # Get top numeric columns by variance
            numeric_df = df.select_dtypes(include=[np.number])
            variance = numeric_df.var().sort_values(ascending=False)
            plot_cols = variance.head(max_columns).index.tolist()

        if len(plot_cols) < 2:
            return "Error: Need at least 2 columns for pairplot."

        plot_cols = plot_cols[:max_columns]  # Limit columns

        # Prepare data for pairplot
        plot_df = df[plot_cols].dropna()

        # Add hue column if specified
        hue_col = None
        if hue and hue.upper() in col_map:
            hue_col = col_map[hue.upper()]
            if hue_col not in plot_cols:
                plot_df = df[plot_cols + [hue_col]].dropna()

        # Create pairplot
        fig_size = max(8, len(plot_cols) * 2.5)

        if hue_col and plot_df[hue_col].nunique() <= 6:
            g = sns.pairplot(plot_df, hue=hue_col, palette="husl", diag_kind="kde",
                           plot_kws={'alpha': 0.6}, height=fig_size/len(plot_cols))
        else:
            g = sns.pairplot(plot_df[plot_cols], diag_kind="kde",
                           plot_kws={'alpha': 0.6, 'color': '#3498db'},
                           height=fig_size/len(plot_cols))

        g.fig.suptitle('Pairplot - Bivariate Analysis', y=1.02, fontsize=14, fontweight='bold')

        # Save figure
        filepath = _get_viz_path(session_id, "pairplot", "multivariate")
        plt.savefig(filepath, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()

        result = {
            "session_id": session_id,
            "visualization_type": "pairplot",
            "columns_plotted": plot_cols,
            "hue_column": hue_col,
            "data_points": len(plot_df),
            "saved_file": filepath,
            "note": "Diagonal shows distributions (KDE), off-diagonal shows scatter plots between pairs. Useful for identifying clusters and relationships."
        }

        _store_result(session_id, "viz_pairplot", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating pairplot: {str(e)}"


@tool
def plot_class_distribution(session_id: str, column_name: str, show_percentages: bool = True) -> str:
    """
    Visualize class distribution for categorical columns to identify imbalanced/sparse classes.
    Shows both count and percentage distribution with imbalance indicators.

    Args:
        session_id: EDA session ID
        column_name: Categorical column to analyze
        show_percentages: Whether to show percentage labels (default True)

    Returns:
        Path to saved visualization and class imbalance analysis
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        col_map = {c.upper(): c for c in df.columns}
        if column_name.upper() not in col_map:
            return f"Error: Column '{column_name}' not found."

        col = col_map[column_name.upper()]

        # Get value counts
        value_counts = df[col].value_counts()
        total = len(df)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Bar chart
        colors = []
        rare_threshold = total * 0.05  # 5% threshold for rare classes
        for count in value_counts.values:
            if count < rare_threshold:
                colors.append('#e74c3c')  # Red for rare/sparse classes
            elif count < total * 0.1:
                colors.append('#f39c12')  # Orange for moderately rare
            else:
                colors.append('#3498db')  # Blue for normal

        bars = ax1.bar(range(len(value_counts)), value_counts.values, color=colors)
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax1.set_xlabel(col, fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Class Distribution: {col}', fontsize=14, fontweight='bold')

        # Add percentage labels
        if show_percentages:
            for bar, count in zip(bars, value_counts.values):
                pct = count / total * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values) * 0.01,
                        f'{pct:.1f}%', ha='center', fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Sparse (<5%)'),
            Patch(facecolor='#f39c12', label='Rare (5-10%)'),
            Patch(facecolor='#3498db', label='Normal (>10%)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Pie chart for proportions
        explode = [0.05 if c < rare_threshold else 0 for c in value_counts.values]

        wedges, texts, autotexts = ax2.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
            explode=explode,
            colors=sns.color_palette("husl", len(value_counts)),
            startangle=90
        )
        ax2.set_title(f'Class Proportions: {col}', fontsize=14, fontweight='bold')

        # Save figure
        filepath = _get_viz_path(session_id, "class_distribution", col)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Calculate imbalance metrics
        imbalance_ratio = value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf')
        sparse_classes = [str(idx) for idx, count in value_counts.items() if count < rare_threshold]
        rare_classes = [str(idx) for idx, count in value_counts.items() if rare_threshold <= count < total * 0.1]

        # Gini impurity for imbalance
        proportions = value_counts / total
        gini = 1 - sum(proportions ** 2)

        result = {
            "session_id": session_id,
            "visualization_type": "class_distribution",
            "column": col,
            "saved_file": filepath,
            "class_counts": {str(k): int(v) for k, v in value_counts.items()},
            "imbalance_analysis": {
                "total_classes": len(value_counts),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "gini_impurity": round(gini, 4),
                "is_imbalanced": imbalance_ratio > 3,
                "majority_class": str(value_counts.index[0]),
                "majority_percentage": round(value_counts.iloc[0] / total * 100, 2),
                "minority_class": str(value_counts.index[-1]),
                "minority_percentage": round(value_counts.iloc[-1] / total * 100, 2),
                "sparse_classes": sparse_classes,
                "sparse_count": len(sparse_classes),
                "rare_classes": rare_classes
            },
            "recommendations": []
        }

        # Add recommendations
        if imbalance_ratio > 10:
            result["recommendations"].append("Severe class imbalance detected. Consider SMOTE, undersampling, or class weights.")
        elif imbalance_ratio > 3:
            result["recommendations"].append("Moderate class imbalance. Consider stratified sampling or balanced class weights.")

        if sparse_classes:
            result["recommendations"].append(f"Sparse classes found ({len(sparse_classes)}): Consider grouping or special handling for: {', '.join(sparse_classes[:5])}")

        _store_result(session_id, f"viz_class_dist_{col}", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating class distribution plot: {str(e)}"


@tool
def plot_violin(session_id: str, column_name: str, group_by: Optional[str] = None) -> str:
    """
    Create violin plots combining box plots and KDE for distribution visualization.
    Shows probability density and summary statistics.

    Args:
        session_id: EDA session ID
        column_name: Numeric column to plot
        group_by: Optional categorical column to group by

    Returns:
        Path to saved violin plot
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        col_map = {c.upper(): c for c in df.columns}

        if column_name.upper() not in col_map:
            return f"Error: Column '{column_name}' not found."

        col = col_map[column_name.upper()]

        # Create violin plot
        fig, ax = plt.subplots(figsize=(12, 8))

        if group_by and group_by.upper() in col_map:
            group_col = col_map[group_by.upper()]

            # Limit groups if too many
            unique_groups = df[group_col].nunique()
            if unique_groups > 10:
                top_groups = df[group_col].value_counts().head(10).index
                plot_df = df[df[group_col].isin(top_groups)]
            else:
                plot_df = df

            sns.violinplot(data=plot_df, x=group_col, y=col, ax=ax, palette="husl", inner="box")
            ax.set_xlabel(group_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            title = f'Violin Plot: {col} by {group_col}'
        else:
            sns.violinplot(y=df[col].dropna(), ax=ax, color='#3498db', inner="box")
            title = f'Violin Plot: {col}'

        ax.set_ylabel(col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Save figure
        suffix = f"{col}_by_{group_by}" if group_by else col
        filepath = _get_viz_path(session_id, "violin", suffix)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        result = {
            "session_id": session_id,
            "visualization_type": "violin_plot",
            "column": col,
            "grouped_by": group_by,
            "saved_file": filepath,
            "note": "Violin plots show probability density (width) and box plot statistics (inner box). Wider sections indicate higher probability."
        }

        _store_result(session_id, f"viz_violin_{col}", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error creating violin plot: {str(e)}"


@tool
def generate_eda_visual_report(session_id: str, include_all: bool = True) -> str:
    """
    Generate a comprehensive visual EDA report with multiple charts and a summary table.
    Creates histograms, boxplots, correlation heatmap, and summary statistics table.

    Args:
        session_id: EDA session ID
        include_all: Whether to include all visualizations (default True)

    Returns:
        Summary of generated visualizations and table report
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        generated_files = []

        # 1. Create summary statistics table
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 0:
            # Create styled summary table
            stats_df = numeric_df.describe().T
            stats_df['missing'] = df[numeric_df.columns].isna().sum()
            stats_df['missing_pct'] = (df[numeric_df.columns].isna().sum() / len(df) * 100).round(2)

            if SCIPY_AVAILABLE:
                stats_df['skewness'] = [stats.skew(df[col].dropna()) for col in numeric_df.columns]

            # Create table figure
            fig, ax = plt.subplots(figsize=(16, min(len(stats_df) * 0.5 + 2, 12)))
            ax.axis('off')
            ax.axis('tight')

            # Format the dataframe for display
            display_df = stats_df.round(2)

            table = ax.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                rowLabels=display_df.index,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # Color header row
            for col_idx in range(len(display_df.columns)):
                table[(0, col_idx)].set_facecolor('#3498db')
                table[(0, col_idx)].set_text_props(color='white', weight='bold')

            ax.set_title('Summary Statistics Table', fontsize=14, fontweight='bold', pad=20)

            filepath = _get_viz_path(session_id, "summary_table", "statistics")
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append({"type": "summary_table", "file": filepath})

        # 2. Generate histograms for top numeric columns
        if len(numeric_df.columns) > 0:
            cols_to_plot = numeric_df.columns[:6]
            n_cols = len(cols_to_plot)
            n_rows = (n_cols + 2) // 3

            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
            axes = axes.flatten() if n_cols > 1 else [axes]

            for i, col in enumerate(cols_to_plot):
                col_data = df[col].dropna()
                sns.histplot(col_data, kde=True, ax=axes[i], color='#3498db', alpha=0.7)
                axes[i].axvline(col_data.mean(), color='red', linestyle='--', label='Mean')
                axes[i].axvline(col_data.median(), color='green', linestyle='-.', label='Median')
                axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
                axes[i].legend(fontsize=8)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle('Distribution of Numeric Variables', fontsize=14, fontweight='bold', y=1.02)

            filepath = _get_viz_path(session_id, "distributions_grid", "all")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append({"type": "distributions_grid", "file": filepath})

        # 3. Correlation heatmap
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

            sns.heatmap(
                corr_matrix, mask=mask, annot=len(corr_matrix.columns) <= 12,
                cmap=sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True),
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, fmt='.2f', ax=ax
            )
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')

            filepath = _get_viz_path(session_id, "correlation", "matrix")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append({"type": "correlation_heatmap", "file": filepath})

        # 4. Box plots grid
        if len(numeric_df.columns) > 0:
            cols_to_plot = numeric_df.columns[:8]
            n_cols = len(cols_to_plot)
            n_rows = (n_cols + 3) // 4

            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
            axes = axes.flatten() if n_cols > 1 else [axes]

            for i, col in enumerate(cols_to_plot):
                col_data = df[col].dropna()
                sns.boxplot(y=col_data, ax=axes[i], color='#3498db')
                axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle('Box Plots - Outlier Detection', fontsize=14, fontweight='bold', y=1.02)

            filepath = _get_viz_path(session_id, "boxplots_grid", "all")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append({"type": "boxplots_grid", "file": filepath})

        # 5. Categorical distributions
        cat_cols = df.select_dtypes(include=["object", "category"]).columns[:4]
        if len(cat_cols) > 0:
            n_cats = len(cat_cols)
            fig, axes = plt.subplots(1, min(n_cats, 4), figsize=(5 * min(n_cats, 4), 6))
            if n_cats == 1:
                axes = [axes]

            for i, col in enumerate(cat_cols):
                value_counts = df[col].value_counts().head(10)
                colors = sns.color_palette("husl", len(value_counts))
                sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], hue=value_counts.index, palette=colors, legend=False)
                axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
                axes[i].set_xlabel('Count')

            fig.suptitle('Top Categories Distribution', fontsize=14, fontweight='bold', y=1.02)

            filepath = _get_viz_path(session_id, "categorical_grid", "all")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append({"type": "categorical_distributions", "file": filepath})

        # 6. Missing values heatmap
        missing_pct = df.isna().sum() / len(df) * 100
        cols_with_missing = missing_pct[missing_pct > 0]

        if len(cols_with_missing) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Sample data for missing pattern visualization
            sample_size = min(100, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df

            sns.heatmap(sample_df.isna().T, cmap='YlOrRd', cbar_kws={'label': 'Missing'}, ax=ax)
            ax.set_title('Missing Values Pattern (Sample)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Row Index')
            ax.set_ylabel('Column')

            filepath = _get_viz_path(session_id, "missing_pattern", "heatmap")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append({"type": "missing_values_pattern", "file": filepath})

        result = {
            "session_id": session_id,
            "table_name": session["table_name"],
            "report_type": "comprehensive_visual_eda",
            "generated_at": datetime.now().isoformat(),
            "visualizations_generated": len(generated_files),
            "files": generated_files,
            "data_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_df.columns),
                "categorical_columns": len(df.select_dtypes(include=["object", "category"]).columns),
                "missing_values_total": int(df.isna().sum().sum()),
                "missing_percentage": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
            },
            "note": "Visual EDA report generated. Files saved to sample_files/eda_visualizations/"
        }

        _store_result(session_id, "visual_eda_report", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error generating visual EDA report: {str(e)}"


def _image_to_base64(filepath: str) -> str:
    """Convert an image file to base64 string."""
    import base64
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _df_to_html_table(df, title: str = "", max_rows: int = 100) -> str:
    """Convert a DataFrame to a styled HTML table."""
    if len(df) > max_rows:
        df = df.head(max_rows)

    html = f"""
    <div class="table-container">
        {f'<h3>{title}</h3>' if title else ''}
        <table class="styled-table">
            <thead>
                <tr>
                    <th></th>
                    {''.join(f'<th>{col}</th>' for col in df.columns)}
                </tr>
            </thead>
            <tbody>
    """

    for idx, row in df.iterrows():
        html += f"<tr><td class='row-header'>{idx}</td>"
        for val in row:
            if isinstance(val, float):
                html += f"<td>{val:.4f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>"

    html += """
            </tbody>
        </table>
    </div>
    """
    return html


@tool
def generate_eda_html_report(session_id: str, open_browser: bool = True) -> str:
    """
    Generate a comprehensive HTML EDA report with embedded charts and interactive tables.
    This report can be viewed in any browser and displays all visualizations inline.

    Args:
        session_id: EDA session ID
        open_browser: Whether to automatically open the report in browser (default True)

    Returns:
        Path to the generated HTML report file
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization libraries not installed."

    try:
        if session_id not in _eda_sessions:
            return f"Error: Session '{session_id}' not found."

        session = _eda_sessions[session_id]
        df = session.get("dataframe")

        if df is None:
            return "Error: No DataFrame in session."

        table_name = session["table_name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize HTML content
        html_sections = []
        embedded_images = []

        # ==================== HEADER ====================
        html_header = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EDA Report - {table_name}</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    color: #e0e0e0;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{ max-width: 1400px; margin: 0 auto; }}

                /* Header */
                .header {{
                    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    border: 1px solid #0f3460;
                }}
                .header h1 {{
                    color: #00d4aa;
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }}
                .header .subtitle {{
                    color: #94a3b8;
                    font-size: 1.1rem;
                }}
                .header .meta {{
                    display: flex;
                    gap: 30px;
                    margin-top: 20px;
                    flex-wrap: wrap;
                }}
                .header .meta-item {{
                    background: rgba(0, 212, 170, 0.1);
                    padding: 10px 20px;
                    border-radius: 8px;
                    border: 1px solid #00d4aa;
                }}
                .header .meta-item .label {{ color: #94a3b8; font-size: 0.85rem; }}
                .header .meta-item .value {{ color: #00d4aa; font-size: 1.3rem; font-weight: bold; }}

                /* Sections */
                .section {{
                    background: rgba(30, 41, 59, 0.8);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 25px;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
                    border: 1px solid #334155;
                }}
                .section h2 {{
                    color: #00d4aa;
                    font-size: 1.5rem;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #00d4aa;
                }}
                .section h3 {{
                    color: #74b9ff;
                    font-size: 1.2rem;
                    margin: 15px 0 10px 0;
                }}

                /* Charts Grid */
                .chart-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                }}
                .chart-container {{
                    background: #1e293b;
                    border-radius: 10px;
                    padding: 15px;
                    border: 1px solid #334155;
                }}
                .chart-container img {{
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                }}
                .chart-container .chart-title {{
                    color: #f1f5f9;
                    font-size: 1rem;
                    font-weight: 600;
                    margin-bottom: 10px;
                    text-align: center;
                }}

                /* Full Width Charts */
                .chart-full {{
                    grid-column: 1 / -1;
                }}
                .chart-full img {{
                    max-width: 900px;
                    margin: 0 auto;
                    display: block;
                }}

                /* Tables */
                .table-container {{
                    overflow-x: auto;
                    margin: 15px 0;
                }}
                .styled-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9rem;
                    background: #1e293b;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .styled-table thead tr {{
                    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
                    color: #1a1a2e;
                    text-align: left;
                    font-weight: bold;
                }}
                .styled-table th, .styled-table td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #334155;
                }}
                .styled-table tbody tr:hover {{
                    background: rgba(0, 212, 170, 0.1);
                }}
                .styled-table .row-header {{
                    background: #0f3460;
                    color: #00d4aa;
                    font-weight: 600;
                }}

                /* Quality Score */
                .quality-score {{
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    padding: 20px;
                    background: #1e293b;
                    border-radius: 10px;
                    margin: 15px 0;
                }}
                .quality-grade {{
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: white;
                }}
                .grade-A {{ background: linear-gradient(135deg, #10b981, #059669); }}
                .grade-B {{ background: linear-gradient(135deg, #3b82f6, #2563eb); }}
                .grade-C {{ background: linear-gradient(135deg, #f59e0b, #d97706); }}
                .grade-D {{ background: linear-gradient(135deg, #ef4444, #dc2626); }}
                .grade-F {{ background: linear-gradient(135deg, #991b1b, #7f1d1d); }}

                /* Insights */
                .insight-card {{
                    background: rgba(0, 212, 170, 0.1);
                    border: 1px solid #00d4aa;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .insight-card.warning {{
                    background: rgba(245, 158, 11, 0.1);
                    border-color: #f59e0b;
                }}
                .insight-card.error {{
                    background: rgba(239, 68, 68, 0.1);
                    border-color: #ef4444;
                }}

                /* Navigation */
                .nav {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #1e293b;
                    border-radius: 10px;
                    padding: 15px;
                    border: 1px solid #334155;
                    z-index: 1000;
                }}
                .nav a {{
                    display: block;
                    color: #94a3b8;
                    text-decoration: none;
                    padding: 5px 10px;
                    font-size: 0.85rem;
                }}
                .nav a:hover {{ color: #00d4aa; }}

                /* Footer */
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #64748b;
                    font-size: 0.85rem;
                }}
            </style>
        </head>
        <body>
        <div class="container">
        """

        # ==================== REPORT HEADER ====================
        numeric_df = df.select_dtypes(include=[np.number])
        cat_df = df.select_dtypes(include=["object", "category"])

        html_header += f"""
        <div class="header">
            <h1>📊 EDA Report: {table_name}</h1>
            <p class="subtitle">Comprehensive Exploratory Data Analysis with Visualizations</p>
            <div class="meta">
                <div class="meta-item">
                    <div class="label">Total Rows</div>
                    <div class="value">{len(df):,}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Total Columns</div>
                    <div class="value">{len(df.columns)}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Numeric Columns</div>
                    <div class="value">{len(numeric_df.columns)}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Categorical Columns</div>
                    <div class="value">{len(cat_df.columns)}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Missing Values</div>
                    <div class="value">{df.isna().sum().sum():,}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Generated</div>
                    <div class="value">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
                </div>
            </div>
        </div>

        <nav class="nav">
            <strong style="color: #00d4aa;">📑 Sections</strong>
            <a href="#summary">Summary Statistics</a>
            <a href="#distributions">Distributions</a>
            <a href="#correlations">Correlations</a>
            <a href="#outliers">Outliers</a>
            <a href="#categorical">Categorical Analysis</a>
            <a href="#quality">Data Quality</a>
        </nav>
        """

        html_sections.append(html_header)

        # ==================== SUMMARY STATISTICS TABLE ====================
        if len(numeric_df.columns) > 0:
            stats_df = numeric_df.describe().T
            stats_df['missing'] = df[numeric_df.columns].isna().sum()
            stats_df['missing_%'] = (df[numeric_df.columns].isna().sum() / len(df) * 100).round(2)

            if SCIPY_AVAILABLE:
                stats_df['skewness'] = [round(stats.skew(df[col].dropna()), 4) for col in numeric_df.columns]

            summary_html = f"""
            <div class="section" id="summary">
                <h2>📈 Summary Statistics</h2>
                {_df_to_html_table(stats_df.round(4), "")}
            </div>
            """
            html_sections.append(summary_html)

        # ==================== DISTRIBUTION HISTOGRAMS ====================
        if len(numeric_df.columns) > 0:
            dist_html = '<div class="section" id="distributions"><h2>📊 Distribution Analysis</h2><div class="chart-grid">'

            for col in numeric_df.columns[:8]:
                fig, ax = plt.subplots(figsize=(8, 5))
                col_data = df[col].dropna()

                sns.histplot(col_data, kde=True, ax=ax, color='#00d4aa', alpha=0.7)
                ax.axvline(col_data.mean(), color='#ff6b6b', linestyle='--', linewidth=2, label=f'Mean: {col_data.mean():.2f}')
                ax.axvline(col_data.median(), color='#ffd93d', linestyle='-.', linewidth=2, label=f'Median: {col_data.median():.2f}')
                ax.set_title(f'{col}', fontsize=12, fontweight='bold', color='#333')
                ax.legend(fontsize=9)
                ax.set_facecolor('#f8fafc')
                fig.patch.set_facecolor('white')

                # Save to memory and convert to base64
                from io import BytesIO
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                buf.seek(0)
                import base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                dist_html += f'''
                <div class="chart-container">
                    <div class="chart-title">{col}</div>
                    <img src="data:image/png;base64,{img_base64}" alt="{col} histogram">
                </div>
                '''

            dist_html += '</div></div>'
            html_sections.append(dist_html)

        # ==================== CORRELATION HEATMAP ====================
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

            sns.heatmap(corr_matrix, mask=mask, annot=len(corr_matrix.columns) <= 12,
                       cmap=cmap, center=0, vmin=-1, vmax=1, square=True,
                       linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            fig.patch.set_facecolor('white')

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')

            # Find high correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= 0.5:
                        high_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")

            corr_html = f'''
            <div class="section" id="correlations">
                <h2>🔗 Correlation Analysis</h2>
                <div class="chart-grid">
                    <div class="chart-container chart-full">
                        <div class="chart-title">Correlation Heatmap</div>
                        <img src="data:image/png;base64,{img_base64}" alt="Correlation Heatmap">
                    </div>
                </div>
                {"<h3>High Correlations (|r| ≥ 0.5)</h3><div class='insight-card'>" + "<br>".join(high_corr) + "</div>" if high_corr else ""}
            </div>
            '''
            html_sections.append(corr_html)

        # ==================== BOX PLOTS (OUTLIERS) ====================
        if len(numeric_df.columns) > 0:
            outlier_html = '<div class="section" id="outliers"><h2>📦 Box Plots & Outlier Detection</h2><div class="chart-grid">'

            outlier_insights = []
            for col in numeric_df.columns[:6]:
                fig, ax = plt.subplots(figsize=(6, 5))
                col_data = df[col].dropna()

                sns.boxplot(y=col_data, ax=ax, color='#74b9ff', width=0.4)
                ax.set_title(f'{col}', fontsize=12, fontweight='bold')
                fig.patch.set_facecolor('white')

                # Calculate outliers
                Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    outlier_insights.append(f"{col}: {len(outliers)} outliers ({len(outliers)/len(col_data)*100:.1f}%)")

                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                outlier_html += f'''
                <div class="chart-container">
                    <div class="chart-title">{col}</div>
                    <img src="data:image/png;base64,{img_base64}" alt="{col} boxplot">
                </div>
                '''

            outlier_html += '</div>'
            if outlier_insights:
                outlier_html += '<h3>Outlier Summary</h3><div class="insight-card warning">' + '<br>'.join(outlier_insights) + '</div>'
            outlier_html += '</div>'
            html_sections.append(outlier_html)

        # ==================== CATEGORICAL DISTRIBUTIONS ====================
        if len(cat_df.columns) > 0:
            cat_html = '<div class="section" id="categorical"><h2>📊 Categorical Distributions</h2><div class="chart-grid">'

            for col in cat_df.columns[:6]:
                value_counts = df[col].value_counts().head(10)

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = sns.color_palette("husl", len(value_counts))
                bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors)
                ax.set_yticks(range(len(value_counts)))
                ax.set_yticklabels([str(v)[:30] for v in value_counts.index])
                ax.invert_yaxis()
                ax.set_xlabel('Count')
                ax.set_title(f'{col} (Top 10)', fontsize=12, fontweight='bold')
                fig.patch.set_facecolor('white')

                # Add value labels
                for bar, val in zip(bars, value_counts.values):
                    ax.text(val + max(value_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:,}', va='center', fontsize=9)

                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                cat_html += f'''
                <div class="chart-container">
                    <div class="chart-title">{col}</div>
                    <img src="data:image/png;base64,{img_base64}" alt="{col} distribution">
                </div>
                '''

            cat_html += '</div></div>'
            html_sections.append(cat_html)

        # ==================== DATA QUALITY ====================
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        duplicates = df.duplicated().sum()
        uniqueness = (len(df) - duplicates) / len(df) * 100

        overall_score = completeness * 0.5 + uniqueness * 0.5

        if overall_score >= 90:
            grade, grade_class = "A", "grade-A"
        elif overall_score >= 80:
            grade, grade_class = "B", "grade-B"
        elif overall_score >= 70:
            grade, grade_class = "C", "grade-C"
        elif overall_score >= 60:
            grade, grade_class = "D", "grade-D"
        else:
            grade, grade_class = "F", "grade-F"

        quality_html = f'''
        <div class="section" id="quality">
            <h2>✅ Data Quality Assessment</h2>
            <div class="quality-score">
                <div class="quality-grade {grade_class}">{grade}</div>
                <div>
                    <h3 style="color: #00d4aa; margin-bottom: 10px;">Overall Score: {overall_score:.1f}%</h3>
                    <p>Completeness: {completeness:.1f}% | Uniqueness: {uniqueness:.1f}%</p>
                    <p style="color: #94a3b8;">Missing Values: {int(missing_cells):,} | Duplicate Rows: {int(duplicates):,}</p>
                </div>
            </div>
        </div>
        '''
        html_sections.append(quality_html)

        # ==================== FOOTER ====================
        html_footer = f'''
        <div class="footer">
            <p>Generated by EDA Agent | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Powered by Seaborn, Matplotlib, and Pandas</p>
        </div>
        </div>
        </body>
        </html>
        '''
        html_sections.append(html_footer)

        # ==================== SAVE HTML FILE ====================
        html_content = '\n'.join(html_sections)

        # Ensure output directory exists
        html_output_dir = os.path.join(VIZ_OUTPUT_DIR, "reports")
        os.makedirs(html_output_dir, exist_ok=True)

        html_filepath = os.path.join(html_output_dir, f"eda_report_{table_name}_{timestamp}.html")
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Open in browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(html_filepath)}')

        result = {
            "session_id": session_id,
            "table_name": table_name,
            "report_type": "html_eda_report",
            "html_file": html_filepath,
            "generated_at": datetime.now().isoformat(),
            "sections": ["Summary Statistics", "Distributions", "Correlations", "Outliers", "Categorical", "Data Quality"],
            "data_summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "quality_score": round(overall_score, 1),
                "quality_grade": grade
            },
            "note": f"HTML report generated and opened in browser. File: {html_filepath}"
        }

        _store_result(session_id, "html_eda_report", result)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        import traceback
        return f"Error generating HTML report: {str(e)}\n{traceback.format_exc()}"


# =============================================================================
# REGISTER TOOLS WITH REGISTRY
# =============================================================================

# Basic EDA tools
tool_registry.register(list_tables_for_eda, "eda")
tool_registry.register(get_table_info_for_eda, "eda")
tool_registry.register(load_table_to_pandas, "eda")
tool_registry.register(get_basic_statistics, "eda")
tool_registry.register(check_missing_values, "eda")
tool_registry.register(check_duplicates, "eda")
tool_registry.register(check_data_types, "eda")
tool_registry.register(get_unique_value_counts, "eda")
tool_registry.register(detect_outliers, "eda")

# Advanced EDA tools
tool_registry.register(analyze_distributions, "eda")
tool_registry.register(analyze_correlations, "eda")
tool_registry.register(analyze_categorical_columns, "eda")
tool_registry.register(analyze_numerical_columns, "eda")
tool_registry.register(analyze_datetime_columns, "eda")
tool_registry.register(check_data_quality, "eda")
tool_registry.register(analyze_column_relationships, "eda")

# Planning and custom analysis
tool_registry.register(generate_eda_plan, "eda")
tool_registry.register(execute_custom_analysis, "eda")

# Summary and session management
tool_registry.register(get_eda_summary, "eda")
tool_registry.register(get_session_info, "eda")
tool_registry.register(list_eda_sessions, "eda")

# Visualization tools
tool_registry.register(plot_histogram, "eda")
tool_registry.register(plot_boxplot, "eda")
tool_registry.register(plot_bar_chart, "eda")
tool_registry.register(plot_scatter, "eda")
tool_registry.register(plot_correlation_heatmap, "eda")
tool_registry.register(plot_pairplot, "eda")
tool_registry.register(plot_class_distribution, "eda")
tool_registry.register(plot_violin, "eda")
tool_registry.register(generate_eda_visual_report, "eda")
tool_registry.register(generate_eda_html_report, "eda")


def get_eda_tools():
    """Get all EDA tools."""
    return tool_registry.get_tools_by_category("eda")
