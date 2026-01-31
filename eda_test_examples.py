"""
Test Examples for Multi-EDA Agent
=================================

Comprehensive test scenarios for the 12-agent Multi-EDA workflow covering:
- Simple: Single table EDA, general questions, basic analysis
- Medium: Multi-step EDA, specific analysis requests, target variable selection
- Complex: Multi-table EDA with joins, full pipeline with dashboard, edge cases

Usage:
    1. Run specific tests: python run_tests.py --category multi_eda_simple
    2. Run all Multi-EDA tests: python run_tests.py --agent multi_eda_agent
    3. Direct execution: python eda_test_examples.py
"""

# =============================================================================
# SIMPLE EXAMPLES
# Single table, basic intent, straightforward EDA requests
# =============================================================================

MULTI_EDA_SIMPLE_EXAMPLES = [
    # General questions (should be answered without EDA)
    {
        "name": "Multi-EDA General Question - What is EDA",
        "agent": "multi_eda_agent",
        "query": "What is Exploratory Data Analysis?",
        "expected_tools": [],
        "description": "General question - Agent1 should classify as GENERAL_QUESTION and answer directly without running EDA pipeline"
    },
    {
        "name": "Multi-EDA General Question - Available Tables",
        "agent": "multi_eda_agent",
        "query": "What tables are available in this database?",
        "expected_tools": [],
        "description": "General question about schema - should be answered by Agent1 using schema context"
    },

    # Single table - small dataset
    {
        "name": "Multi-EDA Simple - Clients Table",
        "agent": "multi_eda_agent",
        "query": "Do EDA on the CLIENTS table",
        "expected_tools": ["eda_load_table", "eda_get_table_schema", "eda_classify_columns",
                           "eda_describe_numerical", "eda_describe_categorical",
                           "eda_plot_all_histograms", "eda_detect_outliers_iqr",
                           "eda_compute_correlations", "eda_generate_dashboard"],
        "description": "Full EDA on small table (10 rows, 6 columns). Tests all 12 agents on minimal data."
    },
    {
        "name": "Multi-EDA Simple - Assets Table",
        "agent": "multi_eda_agent",
        "query": "Perform EDA on ASSETS",
        "expected_tools": ["eda_load_table", "eda_get_table_schema", "eda_classify_columns",
                           "eda_describe_categorical", "eda_plot_countplots",
                           "eda_generate_dashboard"],
        "description": "EDA on mostly-categorical table (35 rows). Tests handling when few numerical columns exist."
    },
    {
        "name": "Multi-EDA Simple - Portfolios Table",
        "agent": "multi_eda_agent",
        "query": "Analyze the PORTFOLIOS table",
        "expected_tools": ["eda_load_table", "eda_classify_columns",
                           "eda_describe_categorical", "eda_generate_dashboard"],
        "description": "EDA on mixed table (15 rows, 6 columns). Has categorical target candidates (STATUS)."
    },

    # Explicit target variable
    {
        "name": "Multi-EDA Simple - Explicit Target",
        "agent": "multi_eda_agent",
        "query": "Do EDA on CLIENTS with RISK_PROFILE as the target variable",
        "expected_tools": ["eda_load_table", "eda_validate_target",
                           "eda_plot_boxplots", "eda_plot_violinplots",
                           "eda_generate_dashboard"],
        "description": "EDA with explicitly specified target. Agent4 should validate and use RISK_PROFILE for segmentation."
    },

    # Row limit specification
    {
        "name": "Multi-EDA Simple - With Row Limit",
        "agent": "multi_eda_agent",
        "query": "Do EDA on the TRANSACTIONS table, limit to 500 rows",
        "expected_tools": ["eda_load_table", "eda_classify_columns",
                           "eda_describe_numerical", "eda_generate_dashboard"],
        "description": "EDA with explicit row limit. Agent3 should parse '500 rows' and load accordingly."
    },
]


# =============================================================================
# MEDIUM EXAMPLES
# Specific analysis focus, multi-step requests, target detection
# =============================================================================

MULTI_EDA_MEDIUM_EXAMPLES = [
    # Transactions - large dataset, numeric-heavy
    {
        "name": "Multi-EDA Medium - Transactions Full",
        "agent": "multi_eda_agent",
        "query": "Perform a complete EDA on the TRANSACTIONS table including distribution analysis and outlier detection",
        "expected_tools": ["eda_load_table", "eda_get_table_schema", "eda_classify_columns",
                           "eda_describe_numerical", "eda_describe_categorical",
                           "eda_plot_all_histograms", "eda_plot_all_individual_histograms",
                           "eda_plot_countplots", "eda_detect_outliers_iqr",
                           "eda_plot_outlier_boxplots", "eda_compute_correlations",
                           "eda_plot_heatmap", "eda_generate_dashboard"],
        "description": "Full pipeline on 1200-row table. All 12 agents should run. Tests histogram, outlier, and correlation on QUANTITY/PRICE/FEES."
    },

    # Holdings - numeric-heavy with outlier potential
    {
        "name": "Multi-EDA Medium - Holdings Analysis",
        "agent": "multi_eda_agent",
        "query": "Do exploratory data analysis on HOLDINGS table, focus on numerical patterns",
        "expected_tools": ["eda_load_table", "eda_describe_numerical",
                           "eda_plot_all_histograms", "eda_detect_outliers_iqr",
                           "eda_compute_correlations", "eda_plot_heatmap",
                           "eda_generate_dashboard"],
        "description": "EDA on HOLDINGS (305 rows). QUANTITY and AVG_COST should show interesting distributions and possible outliers."
    },

    # Target variable auto-detection
    {
        "name": "Multi-EDA Medium - Auto Target Detection",
        "agent": "multi_eda_agent",
        "query": "Run EDA on TRANSACTIONS and automatically detect the best target variable",
        "expected_tools": ["eda_load_table", "eda_detect_target_variable",
                           "eda_classify_columns", "eda_plot_boxplots",
                           "eda_plot_violinplots", "eda_generate_dashboard"],
        "description": "Tests Agent4 target auto-detection. TRANSACTION_TYPE (BUY/SELL) should be detected as target. Segmentation agents (8.1-8.3) should run."
    },

    # Distribution focus
    {
        "name": "Multi-EDA Medium - Distribution Focus",
        "agent": "multi_eda_agent",
        "query": "Analyze the distributions of all columns in the TRANSACTIONS table and create visualizations",
        "expected_tools": ["eda_load_table", "eda_plot_all_histograms",
                           "eda_plot_all_individual_histograms", "eda_plot_countplots",
                           "eda_generate_dashboard"],
        "description": "Tests Agent7 sub-agents (7.1 all histograms, 7.2 individual histograms, 7.3 countplots). Should produce grid and individual charts."
    },

    # Correlation focus
    {
        "name": "Multi-EDA Medium - Correlation Analysis",
        "agent": "multi_eda_agent",
        "query": "Do EDA on HOLDINGS focusing on correlations between QUANTITY and AVG_COST",
        "expected_tools": ["eda_load_table", "eda_compute_correlations",
                           "eda_plot_heatmap", "eda_generate_dashboard"],
        "description": "Tests Agent10 correlation analysis. Should detect correlation between QUANTITY and AVG_COST and generate heatmap."
    },

    # Segmentation with specified target
    {
        "name": "Multi-EDA Medium - Segmentation by Transaction Type",
        "agent": "multi_eda_agent",
        "query": "Do EDA on TRANSACTIONS with TRANSACTION_TYPE as target. Show boxplots and violin plots.",
        "expected_tools": ["eda_load_table", "eda_validate_target",
                           "eda_plot_boxplots", "eda_plot_violinplots",
                           "eda_plot_lmplots", "eda_generate_dashboard"],
        "description": "Tests Agent8 segmentation sub-agents with explicit BUY/SELL target. Boxplots and violins for QUANTITY/PRICE/FEES by TRANSACTION_TYPE."
    },

    # Deep analysis focus
    {
        "name": "Multi-EDA Medium - Deep Analysis Suggestions",
        "agent": "multi_eda_agent",
        "query": "Analyze TRANSACTIONS table and provide data cleaning and feature engineering suggestions",
        "expected_tools": ["eda_load_table", "eda_describe_numerical",
                           "eda_detect_outliers_iqr", "eda_compute_correlations",
                           "eda_get_agent_summaries", "eda_generate_dashboard"],
        "description": "Tests Agent11 deep analysis. Should provide cleaning suggestions (outliers, missing data) and feature engineering suggestions (interaction features, binning)."
    },

    # Outlier-focused analysis
    {
        "name": "Multi-EDA Medium - Outlier Detection",
        "agent": "multi_eda_agent",
        "query": "Do EDA on HOLDINGS and find all outliers. Show me outlier boxplots and scatter plots.",
        "expected_tools": ["eda_load_table", "eda_detect_outliers_iqr",
                           "eda_plot_outlier_boxplots", "eda_plot_outlier_scatter",
                           "eda_generate_dashboard"],
        "description": "Tests Agent9 outlier detection. AVG_COST and QUANTITY in HOLDINGS likely have outliers from high-value positions."
    },
]


# =============================================================================
# COMPLEX EXAMPLES
# Multi-table, full dashboard, edge cases, comprehensive pipelines
# =============================================================================

MULTI_EDA_COMPLEX_EXAMPLES = [
    # Full pipeline - single table with dashboard
    {
        "name": "Multi-EDA Complex - Full TRANSACTIONS Pipeline",
        "agent": "multi_eda_agent",
        "query": """Do a complete EDA on the TRANSACTIONS table with all visualizations and generate an HTML dashboard.
Include:
- Structure inspection
- Descriptive statistics
- All distribution charts (histograms, countplots)
- Segmentation analysis
- Outlier detection
- Correlation analysis
- Deep analysis with suggestions
- Final dashboard""",
        "expected_tools": ["eda_load_table", "eda_get_table_schema",
                           "eda_classify_columns", "eda_get_shape", "eda_get_dtypes", "eda_get_head",
                           "eda_describe_numerical", "eda_describe_categorical", "eda_generate_stats_table_html",
                           "eda_plot_all_histograms", "eda_plot_all_individual_histograms", "eda_plot_countplots",
                           "eda_detect_target_variable",
                           "eda_plot_boxplots", "eda_plot_violinplots", "eda_plot_lmplots",
                           "eda_detect_outliers_iqr", "eda_plot_outlier_boxplots", "eda_plot_outlier_scatter",
                           "eda_compute_correlations", "eda_plot_heatmap",
                           "eda_get_agent_summaries", "eda_generate_dashboard"],
        "description": "Complete 12-agent pipeline on TRANSACTIONS (1200 rows). All agents run, dashboard generated with embedded plots. Verifies full workflow end-to-end."
    },

    # Multi-table with joins
    {
        "name": "Multi-EDA Complex - Multi-Table Join EDA",
        "agent": "multi_eda_agent",
        "query": "Do EDA on CLIENTS and PORTFOLIOS tables together, joining them on CLIENT_ID",
        "expected_tools": ["eda_load_multiple_tables", "eda_detect_joins",
                           "eda_join_tables", "eda_classify_columns",
                           "eda_describe_numerical", "eda_describe_categorical",
                           "eda_plot_all_histograms", "eda_plot_countplots",
                           "eda_generate_dashboard"],
        "description": "Tests multi-table workflow. Agent3 loads both tables, Agent4 detects CLIENT_ID join and merges. EDA runs on joined result."
    },

    # Three-table join
    {
        "name": "Multi-EDA Complex - Three-Table Join",
        "agent": "multi_eda_agent",
        "query": "Perform EDA on CLIENTS, PORTFOLIOS, and HOLDINGS tables joined together",
        "expected_tools": ["eda_load_multiple_tables", "eda_detect_joins",
                           "eda_join_tables", "eda_classify_columns",
                           "eda_describe_numerical", "eda_describe_categorical",
                           "eda_detect_target_variable",
                           "eda_plot_all_histograms", "eda_plot_countplots",
                           "eda_detect_outliers_iqr", "eda_compute_correlations",
                           "eda_generate_dashboard"],
        "description": "Three-table join EDA. Agent4 must chain joins: CLIENTS->PORTFOLIOS via CLIENT_ID, then PORTFOLIOS->HOLDINGS via PORTFOLIO_ID."
    },

    # Full pipeline - HOLDINGS with dashboard
    {
        "name": "Multi-EDA Complex - Full HOLDINGS with Dashboard",
        "agent": "multi_eda_agent",
        "query": """Analyze the HOLDINGS table completely:
1. Load all 305 rows
2. Inspect structure and data types
3. Generate statistics for QUANTITY and AVG_COST
4. Create all distribution visualizations
5. Detect and visualize outliers
6. Compute correlation matrix with heatmap
7. Provide cleaning and feature engineering suggestions
8. Generate a complete HTML dashboard""",
        "expected_tools": ["eda_load_table", "eda_get_shape", "eda_get_dtypes", "eda_get_head",
                           "eda_classify_columns",
                           "eda_describe_numerical", "eda_generate_stats_table_html",
                           "eda_plot_all_histograms", "eda_plot_all_individual_histograms",
                           "eda_detect_outliers_iqr", "eda_plot_outlier_boxplots",
                           "eda_compute_correlations", "eda_plot_heatmap",
                           "eda_get_agent_summaries", "eda_generate_dashboard"],
        "description": "Complete pipeline on HOLDINGS. Numeric-heavy table should produce rich statistics, histograms, outlier plots, and correlation heatmap."
    },

    # Edge: no target variable - should still complete
    {
        "name": "Multi-EDA Complex - No Target Variable (Unsupervised)",
        "agent": "multi_eda_agent",
        "query": "Do EDA on the ASSETS table. There is no target variable for this dataset.",
        "expected_tools": ["eda_load_table", "eda_detect_target_variable",
                           "eda_classify_columns",
                           "eda_describe_categorical", "eda_plot_countplots",
                           "eda_generate_dashboard"],
        "description": "Tests unsupervised path. ASSETS is mostly categorical with no obvious target. Agent4 should detect NO_TARGET, Agent8 segmentation should be skipped."
    },

    # Edge: categorical-only table
    {
        "name": "Multi-EDA Complex - Categorical-Heavy Table",
        "agent": "multi_eda_agent",
        "query": "Perform complete EDA on the CLIENTS table and generate an HTML dashboard with all charts",
        "expected_tools": ["eda_load_table", "eda_classify_columns",
                           "eda_describe_categorical", "eda_plot_countplots",
                           "eda_detect_target_variable",
                           "eda_generate_dashboard"],
        "description": "CLIENTS has few numeric columns. Tests graceful handling of sparse numerical data - histograms/correlations may be minimal, countplots should dominate."
    },

    # Large dataset with row limit
    {
        "name": "Multi-EDA Complex - Large Dataset with Limit",
        "agent": "multi_eda_agent",
        "query": "Do a complete EDA on TRANSACTIONS table, limit to 200 rows for faster processing, and generate dashboard",
        "expected_tools": ["eda_load_table", "eda_classify_columns",
                           "eda_describe_numerical", "eda_describe_categorical",
                           "eda_plot_all_histograms", "eda_detect_outliers_iqr",
                           "eda_compute_correlations", "eda_generate_dashboard"],
        "description": "Tests row limit parsing. Agent3 should load only 200 of 1200 rows. Full pipeline still runs on subset."
    },

    # Multi-table no valid joins - should stop early
    {
        "name": "Multi-EDA Complex - No Valid Joins (Stop Early)",
        "agent": "multi_eda_agent",
        "query": "Do EDA on CLIENTS and ASSETS tables together",
        "expected_tools": ["eda_load_multiple_tables", "eda_detect_joins"],
        "description": "CLIENTS and ASSETS share no join columns. Agent4 should detect NO_JOINS_FOUND and set stop_reason. Pipeline should stop after Agent4."
    },

    # Comprehensive with explicit instructions
    {
        "name": "Multi-EDA Complex - Comprehensive Analysis Request",
        "agent": "multi_eda_agent",
        "query": """I need a thorough exploratory data analysis of the TRANSACTIONS table.

Requirements:
- Show me the shape and data types
- Compute mean, median, std for all numeric columns
- Find any outliers using IQR method
- Check correlations between PRICE, QUANTITY, and FEES
- Use TRANSACTION_TYPE as the target variable for segmentation
- Create boxplots and violin plots for each numeric column grouped by TRANSACTION_TYPE
- Create scatter plots showing relationships between key variables
- Summarize data quality issues
- Suggest feature engineering opportunities
- Generate an HTML dashboard with all charts embedded""",
        "expected_tools": ["eda_load_table", "eda_get_shape", "eda_get_dtypes",
                           "eda_classify_columns", "eda_validate_target",
                           "eda_describe_numerical", "eda_describe_categorical",
                           "eda_generate_stats_table_html",
                           "eda_plot_all_histograms", "eda_plot_all_individual_histograms",
                           "eda_plot_countplots",
                           "eda_plot_boxplots", "eda_plot_violinplots", "eda_plot_lmplots",
                           "eda_detect_outliers_iqr", "eda_plot_outlier_boxplots",
                           "eda_plot_outlier_scatter",
                           "eda_compute_correlations", "eda_plot_heatmap",
                           "eda_get_agent_summaries", "eda_generate_dashboard"],
        "description": "Maximum coverage test. All 12 agents + sub-agents run. Explicit target, all chart types, full dashboard. Validates complete workflow with user-specified requirements."
    },
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_multi_eda_examples():
    """Get all Multi-EDA test examples."""
    return (
        MULTI_EDA_SIMPLE_EXAMPLES +
        MULTI_EDA_MEDIUM_EXAMPLES +
        MULTI_EDA_COMPLEX_EXAMPLES
    )


def get_multi_eda_examples_by_category(category: str):
    """Get Multi-EDA examples by category name."""
    categories = {
        "multi_eda_simple": MULTI_EDA_SIMPLE_EXAMPLES,
        "multi_eda_medium": MULTI_EDA_MEDIUM_EXAMPLES,
        "multi_eda_complex": MULTI_EDA_COMPLEX_EXAMPLES,
        "multi_eda_all": get_all_multi_eda_examples(),
        "multi_eda": get_all_multi_eda_examples(),
    }
    return categories.get(category.lower(), [])


def get_multi_eda_examples_by_tool(tool_name: str):
    """Get Multi-EDA examples that use a specific tool."""
    return [ex for ex in get_all_multi_eda_examples()
            if tool_name in ex.get("expected_tools", [])]


def print_example_summary():
    """Print summary of all Multi-EDA examples."""
    print("=" * 70)
    print("MULTI-EDA AGENT TEST EXAMPLES SUMMARY")
    print("=" * 70)
    print(f"\nSimple Examples:   {len(MULTI_EDA_SIMPLE_EXAMPLES)}")
    print(f"Medium Examples:   {len(MULTI_EDA_MEDIUM_EXAMPLES)}")
    print(f"Complex Examples:  {len(MULTI_EDA_COMPLEX_EXAMPLES)}")
    print(f"\nTotal Examples:    {len(get_all_multi_eda_examples())}")
    print("=" * 70)

    # Tool coverage
    all_tools = set()
    for ex in get_all_multi_eda_examples():
        all_tools.update(ex.get("expected_tools", []))

    print(f"\nTools Covered: {len(all_tools)}")
    print("-" * 40)

    viz_tools = sorted([t for t in all_tools if "plot" in t or "heatmap" in t or "dashboard" in t])
    data_tools = sorted([t for t in all_tools if t not in viz_tools])

    if viz_tools:
        print("\nVisualization Tools:")
        for t in viz_tools:
            count = len(get_multi_eda_examples_by_tool(t))
            print(f"  - {t}: {count} examples")

    if data_tools:
        print("\nData/Analysis Tools:")
        for t in data_tools:
            count = len(get_multi_eda_examples_by_tool(t))
            print(f"  - {t}: {count} examples")

    # Agent coverage analysis
    print("\n" + "-" * 40)
    print("AGENT COVERAGE:")
    agent_coverage = {
        "Agent1 (Intent)": any("General Question" in ex["name"] for ex in get_all_multi_eda_examples()),
        "Agent2 (Schema)": any("eda_get_table_schema" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent3 (Loading)": any("eda_load_table" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent4 (Join/Target)": any("eda_detect_target_variable" in ex.get("expected_tools", []) or "eda_detect_joins" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent5 (Structure)": any("eda_get_shape" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent6 (Stats)": any("eda_describe_numerical" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent7 (Distribution)": any("eda_plot_all_histograms" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent8 (Segmentation)": any("eda_plot_boxplots" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent9 (Outlier)": any("eda_detect_outliers_iqr" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent10 (Correlation)": any("eda_compute_correlations" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent11 (Deep)": any("eda_get_agent_summaries" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
        "Agent12 (Dashboard)": any("eda_generate_dashboard" in ex.get("expected_tools", []) for ex in get_all_multi_eda_examples()),
    }
    for agent, covered in agent_coverage.items():
        status = "COVERED" if covered else "NOT COVERED"
        print(f"  {agent}: {status}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print_example_summary()

    print("\n" + "=" * 70)
    print("SAMPLE QUERIES FOR TESTING")
    print("=" * 70)

    print("\n[SIMPLE] Basic EDA:")
    print(f"  {MULTI_EDA_SIMPLE_EXAMPLES[2]['query']}")

    print("\n[MEDIUM] Full Transactions EDA:")
    print(f"  {MULTI_EDA_MEDIUM_EXAMPLES[0]['query']}")

    print("\n[COMPLEX] Full Pipeline with Dashboard:")
    print(f"  {MULTI_EDA_COMPLEX_EXAMPLES[0]['query'][:120]}...")

    print("\n[COMPLEX] Multi-Table Join:")
    print(f"  {MULTI_EDA_COMPLEX_EXAMPLES[1]['query']}")

    print("\n" + "=" * 70)
    print("To run these tests:")
    print("  python run_tests.py --category multi_eda_simple")
    print("  python run_tests.py --category multi_eda_medium")
    print("  python run_tests.py --category multi_eda_complex")
    print("  python run_tests.py --agent multi_eda_agent")
    print("=" * 70)
