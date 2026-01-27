"""
Test Examples for EDA Agent with Visualization
===============================================

This file contains comprehensive test scenarios for the EDA Agent,
covering all visualization tools (histograms, boxplots, scatter plots,
correlation heatmaps, pairplots, class distribution, violin plots)
and summary table generation.

Usage:
    1. Run specific tests: python run_tests.py --category eda_simple
    2. Run all EDA tests: python run_tests.py --agent eda_agent
    3. Direct execution: python test_examples_eda.py
"""

# =============================================================================
# EDA SIMPLE EXAMPLES (Single visualization tool tests)
# =============================================================================

EDA_VIZ_SIMPLE_EXAMPLES = [
    # Histogram - Basic distribution
    {
        "name": "EDA - Basic Histogram",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a histogram for the PRICE column showing its distribution.",
        "expected_tools": ["load_table_to_pandas", "plot_histogram"],
        "description": "Create a single histogram with KDE for price distribution"
    },
    {
        "name": "EDA - Multi-Column Histogram",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create histograms for all numeric columns to see their distributions.",
        "expected_tools": ["load_table_to_pandas", "plot_histogram"],
        "description": "Create histograms for all numeric columns"
    },

    # Box Plot - Outlier detection
    {
        "name": "EDA - Basic Boxplot",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a box plot for the QUANTITY column to identify outliers.",
        "expected_tools": ["load_table_to_pandas", "plot_boxplot"],
        "description": "Create boxplot for outlier visualization"
    },
    {
        "name": "EDA - Grouped Boxplot",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a box plot of PRICE grouped by TRANSACTION_TYPE to compare distributions.",
        "expected_tools": ["load_table_to_pandas", "plot_boxplot"],
        "description": "Create grouped boxplot comparing categories"
    },

    # Bar Chart - Categorical distribution
    {
        "name": "EDA - Bar Chart Categories",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a bar chart showing the distribution of TRANSACTION_TYPE.",
        "expected_tools": ["load_table_to_pandas", "plot_bar_chart"],
        "description": "Create bar chart for categorical column"
    },
    {
        "name": "EDA - Horizontal Bar Chart",
        "agent": "eda_agent",
        "query": "Load the CLIENTS table and create a horizontal bar chart showing the distribution of COUNTRY.",
        "expected_tools": ["load_table_to_pandas", "plot_bar_chart"],
        "description": "Create horizontal bar chart for countries"
    },

    # Scatter Plot - Bivariate analysis
    {
        "name": "EDA - Basic Scatter Plot",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a scatter plot of PRICE vs QUANTITY to see if there's a relationship.",
        "expected_tools": ["load_table_to_pandas", "plot_scatter"],
        "description": "Create scatter plot for two numeric columns"
    },

    # Correlation Heatmap
    {
        "name": "EDA - Correlation Heatmap",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a correlation heatmap to see relationships between numeric columns.",
        "expected_tools": ["load_table_to_pandas", "plot_correlation_heatmap"],
        "description": "Create correlation matrix visualization"
    },

    # Violin Plot
    {
        "name": "EDA - Violin Plot",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a violin plot for the FEES column.",
        "expected_tools": ["load_table_to_pandas", "plot_violin"],
        "description": "Create violin plot showing distribution density"
    },

    # Class Distribution
    {
        "name": "EDA - Class Distribution",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and analyze the class distribution of TRANSACTION_TYPE to check for imbalance.",
        "expected_tools": ["load_table_to_pandas", "plot_class_distribution"],
        "description": "Visualize class imbalance in categorical column"
    },
]

# =============================================================================
# EDA MEDIUM EXAMPLES (Multiple tools, analysis + visualization)
# =============================================================================

EDA_VIZ_MEDIUM_EXAMPLES = [
    # Statistics + Visualization
    {
        "name": "EDA - Stats with Histogram",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table, get basic statistics for numeric columns, and create histograms to visualize the distributions.",
        "expected_tools": ["load_table_to_pandas", "get_basic_statistics", "plot_histogram"],
        "description": "Combine statistical analysis with visual distribution"
    },

    # Missing Values + Visualization
    {
        "name": "EDA - Missing Values Analysis",
        "agent": "eda_agent",
        "query": "Load the HOLDINGS table, check for missing values, and create a box plot to visualize QUANTITY distribution.",
        "expected_tools": ["load_table_to_pandas", "check_missing_values", "plot_boxplot"],
        "description": "Data quality check with visualization"
    },

    # Correlation Analysis + Heatmap
    {
        "name": "EDA - Full Correlation Analysis",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table, perform correlation analysis, identify highly correlated pairs, and create a correlation heatmap.",
        "expected_tools": ["load_table_to_pandas", "analyze_correlations", "plot_correlation_heatmap"],
        "description": "Statistical correlation with visual heatmap"
    },

    # Categorical Analysis + Charts
    {
        "name": "EDA - Categorical Deep Dive",
        "agent": "eda_agent",
        "query": "Load the CLIENTS table, analyze all categorical columns including entropy and imbalance, then create bar charts for COUNTRY and RISK_PROFILE.",
        "expected_tools": ["load_table_to_pandas", "analyze_categorical_columns", "plot_bar_chart"],
        "description": "Deep categorical analysis with visualization"
    },

    # Outlier Detection + Box Plots
    {
        "name": "EDA - Outlier Visualization",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table, detect outliers using the IQR method, and create box plots to visualize the outliers in PRICE and QUANTITY.",
        "expected_tools": ["load_table_to_pandas", "detect_outliers", "plot_boxplot"],
        "description": "Outlier detection with visual confirmation"
    },

    # Distribution Analysis + Histograms
    {
        "name": "EDA - Distribution Deep Dive",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table, analyze distributions including skewness and normality tests, and create histograms for the top 4 numeric columns.",
        "expected_tools": ["load_table_to_pandas", "analyze_distributions", "plot_histogram"],
        "description": "Distribution statistics with visual histograms"
    },

    # Bivariate Analysis with Scatter
    {
        "name": "EDA - Bivariate Relationships",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create scatter plots to analyze the relationship between PRICE and QUANTITY, colored by TRANSACTION_TYPE.",
        "expected_tools": ["load_table_to_pandas", "plot_scatter"],
        "description": "Bivariate scatter with categorical coloring"
    },

    # Pairplot Multi-Variable
    {
        "name": "EDA - Pairwise Analysis",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create a pairplot to visualize relationships between PRICE, QUANTITY, and FEES.",
        "expected_tools": ["load_table_to_pandas", "plot_pairplot"],
        "description": "Multi-variable pairwise scatter matrix"
    },

    # Class Imbalance Detection
    {
        "name": "EDA - Imbalance Detection",
        "agent": "eda_agent",
        "query": "Load the CLIENTS table, analyze the RISK_PROFILE column for class imbalance, and create a class distribution chart showing sparse classes.",
        "expected_tools": ["load_table_to_pandas", "analyze_categorical_columns", "plot_class_distribution"],
        "description": "Identify and visualize class imbalance"
    },

    # Grouped Violin Plots
    {
        "name": "EDA - Grouped Distribution",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and create violin plots of PRICE grouped by TRANSACTION_TYPE to compare buy vs sell distributions.",
        "expected_tools": ["load_table_to_pandas", "plot_violin"],
        "description": "Grouped violin plot comparison"
    },
]

# =============================================================================
# EDA COMPLEX EXAMPLES (Full workflow with multiple visualizations)
# =============================================================================

EDA_VIZ_COMPLEX_EXAMPLES = [
    # Complete Visual EDA Report
    {
        "name": "EDA - Complete Visual Report",
        "agent": "eda_agent",
        "query": "Perform a complete EDA on the TRANSACTIONS table with all visualizations. Include histograms, box plots, correlation heatmap, and generate a comprehensive visual report with summary tables.",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics", "check_missing_values",
            "analyze_correlations", "plot_histogram", "plot_boxplot",
            "plot_correlation_heatmap", "generate_eda_visual_report"
        ],
        "description": "Full EDA workflow with complete visual report"
    },

    # Multi-Table Visual Comparison
    {
        "name": "EDA - Multi-Table Analysis",
        "agent": "eda_agent",
        "query": "Load both CLIENTS and PORTFOLIOS tables. For CLIENTS, create bar charts of COUNTRY and RISK_PROFILE. For PORTFOLIOS, create histograms and box plots. Compare the distributions visually.",
        "expected_tools": [
            "load_table_to_pandas", "plot_bar_chart", "plot_histogram", "plot_boxplot"
        ],
        "description": "Visual analysis across multiple tables"
    },

    # Transaction Pattern Analysis
    {
        "name": "EDA - Transaction Patterns",
        "agent": "eda_agent",
        "query": """Analyze TRANSACTIONS table comprehensively:
        1. Get basic statistics and check data quality
        2. Create histograms for PRICE and QUANTITY distributions
        3. Create box plots to identify outliers
        4. Create scatter plot of PRICE vs QUANTITY colored by TRANSACTION_TYPE
        5. Create correlation heatmap
        6. Analyze class distribution of TRANSACTION_TYPE
        7. Generate the complete visual EDA report with summary table""",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics", "check_data_quality",
            "plot_histogram", "plot_boxplot", "plot_scatter",
            "plot_correlation_heatmap", "plot_class_distribution",
            "generate_eda_visual_report"
        ],
        "description": "Comprehensive transaction analysis with all visualizations"
    },

    # Client Risk Analysis
    {
        "name": "EDA - Client Risk Profiling",
        "agent": "eda_agent",
        "query": """Perform a complete EDA on CLIENTS table focusing on risk analysis:
        1. Load the data and get basic info
        2. Analyze categorical columns (COUNTRY, RISK_PROFILE, KYC_STATUS)
        3. Check for class imbalance in RISK_PROFILE
        4. Create bar charts for all categorical columns
        5. Create class distribution plot for RISK_PROFILE showing any sparse classes
        6. Generate a visual summary report""",
        "expected_tools": [
            "load_table_to_pandas", "get_table_info_for_eda",
            "analyze_categorical_columns", "plot_bar_chart",
            "plot_class_distribution", "generate_eda_visual_report"
        ],
        "description": "Client categorical analysis with imbalance detection"
    },

    # Holdings Portfolio Analysis
    {
        "name": "EDA - Holdings Deep Dive",
        "agent": "eda_agent",
        "query": """Comprehensive EDA on HOLDINGS table:
        1. Load data and examine structure
        2. Get statistics for QUANTITY and AVG_COST
        3. Analyze distributions and check for normality
        4. Detect outliers in both QUANTITY and AVG_COST
        5. Create histograms with KDE for all numeric columns
        6. Create box plots showing outliers
        7. Create scatter plot of QUANTITY vs AVG_COST
        8. Generate correlation heatmap
        9. Create the full visual report with summary table""",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics",
            "analyze_distributions", "detect_outliers",
            "plot_histogram", "plot_boxplot", "plot_scatter",
            "plot_correlation_heatmap", "generate_eda_visual_report"
        ],
        "description": "Deep holdings analysis with all statistical visualizations"
    },

    # Bivariate and Multivariate Analysis
    {
        "name": "EDA - Multivariate Exploration",
        "agent": "eda_agent",
        "query": """Perform multivariate analysis on TRANSACTIONS:
        1. Load the TRANSACTIONS table
        2. Analyze correlations between all numeric columns
        3. Create a pairplot for PRICE, QUANTITY, FEES, and TRANSACTION_ID
        4. Create scatter plots for the most correlated pairs
        5. Create violin plots for PRICE and QUANTITY grouped by TRANSACTION_TYPE
        6. Create the correlation heatmap
        7. Summarize the key relationships found""",
        "expected_tools": [
            "load_table_to_pandas", "analyze_correlations",
            "plot_pairplot", "plot_scatter", "plot_violin",
            "plot_correlation_heatmap", "get_eda_summary"
        ],
        "description": "Full multivariate relationship analysis"
    },

    # Data Quality with Visualizations
    {
        "name": "EDA - Quality Assessment Visual",
        "agent": "eda_agent",
        "query": """Comprehensive data quality assessment for TRANSACTIONS with visualizations:
        1. Load the table
        2. Check for missing values
        3. Check for duplicates
        4. Check data types for potential issues
        5. Run the data quality assessment (get the quality score)
        6. Create box plots to visualize potential data issues
        7. Create histograms to see distribution anomalies
        8. Generate visual report with quality summary table""",
        "expected_tools": [
            "load_table_to_pandas", "check_missing_values", "check_duplicates",
            "check_data_types", "check_data_quality",
            "plot_boxplot", "plot_histogram", "generate_eda_visual_report"
        ],
        "description": "Data quality checks with visual evidence"
    },

    # Portfolio Comparison
    {
        "name": "EDA - Portfolio Visual Comparison",
        "agent": "eda_agent",
        "query": """Analyze PORTFOLIOS table visually:
        1. Load the data
        2. Get basic statistics
        3. Create bar chart for STATUS distribution
        4. Create bar chart for BASE_CURRENCY distribution
        5. Create class distribution plot to check for any imbalanced categories
        6. Analyze datetime columns for INCEPTION_DATE patterns
        7. Generate comprehensive visual report""",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics",
            "plot_bar_chart", "plot_class_distribution",
            "analyze_datetime_columns", "generate_eda_visual_report"
        ],
        "description": "Portfolio categorical and temporal visual analysis"
    },

    # Assets Analysis
    {
        "name": "EDA - Assets Deep Analysis",
        "agent": "eda_agent",
        "query": """Complete EDA on ASSETS table with all visualizations:
        1. Load the data and get table info
        2. Analyze categorical columns (ASSET_TYPE, CURRENCY, EXCHANGE)
        3. Check unique values for each categorical column
        4. Create bar charts for top asset types and currencies
        5. Create class distribution to identify sparse asset categories
        6. Check data quality score
        7. Generate the full visual EDA report with charts and summary table""",
        "expected_tools": [
            "load_table_to_pandas", "get_table_info_for_eda",
            "analyze_categorical_columns", "get_unique_value_counts",
            "plot_bar_chart", "plot_class_distribution",
            "check_data_quality", "generate_eda_visual_report"
        ],
        "description": "Comprehensive assets categorical visual analysis"
    },

    # Full EDA Pipeline
    {
        "name": "EDA - Complete Pipeline",
        "agent": "eda_agent",
        "query": """Execute the complete EDA pipeline on TRANSACTIONS table with ALL visualizations and analyses:

        PHASE 1 - Discovery:
        - List all tables
        - Get table info and column classification
        - Load the data

        PHASE 2 - Basic Analysis:
        - Get basic statistics
        - Check missing values
        - Check duplicates
        - Check data types

        PHASE 3 - Deep Analysis:
        - Analyze numerical columns
        - Analyze distributions
        - Detect outliers
        - Analyze correlations

        PHASE 4 - Quality:
        - Check data quality and get score

        PHASE 5 - Visualizations:
        - Create histograms for all numeric columns
        - Create box plots for outlier visualization
        - Create bar chart for TRANSACTION_TYPE
        - Create scatter plot PRICE vs QUANTITY
        - Create correlation heatmap
        - Create pairplot
        - Create class distribution for TRANSACTION_TYPE
        - Create violin plot for PRICE by TRANSACTION_TYPE
        - Generate complete visual report with summary table

        PHASE 6 - Summary:
        - Get comprehensive EDA summary""",
        "expected_tools": [
            "list_tables_for_eda", "get_table_info_for_eda", "load_table_to_pandas",
            "get_basic_statistics", "check_missing_values", "check_duplicates", "check_data_types",
            "analyze_numerical_columns", "analyze_distributions", "detect_outliers", "analyze_correlations",
            "check_data_quality",
            "plot_histogram", "plot_boxplot", "plot_bar_chart", "plot_scatter",
            "plot_correlation_heatmap", "plot_pairplot", "plot_class_distribution", "plot_violin",
            "generate_eda_visual_report", "get_eda_summary"
        ],
        "description": "Complete EDA pipeline with all 30 tools"
    },
]

# =============================================================================
# EDA EDGE CASES (Error handling, empty data, etc.)
# =============================================================================

EDA_EDGE_CASES = [
    {
        "name": "EDA - Invalid Column Name",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a histogram for the column INVALID_COLUMN_NAME.",
        "expected_tools": ["load_table_to_pandas", "plot_histogram"],
        "description": "Handle invalid column name gracefully"
    },
    {
        "name": "EDA - Scatter with Categorical",
        "agent": "eda_agent",
        "query": "Load CLIENTS and create a scatter plot with COUNTRY as both axes.",
        "expected_tools": ["load_table_to_pandas", "plot_scatter"],
        "description": "Handle non-numeric columns in scatter plot"
    },
    {
        "name": "EDA - Empty Session",
        "agent": "eda_agent",
        "query": "Create a histogram using session_id 'nonexistent_session'.",
        "expected_tools": ["plot_histogram"],
        "description": "Handle invalid session ID"
    },
    {
        "name": "EDA - Single Value Column",
        "agent": "eda_agent",
        "query": "Load a small subset of TRANSACTIONS (limit 10) and try to create visualizations to see how it handles limited data.",
        "expected_tools": ["load_table_to_pandas", "plot_histogram", "plot_boxplot"],
        "description": "Handle small dataset edge cases"
    },
]

# =============================================================================
# VISUALIZATION-FOCUSED EXAMPLES (Testing specific chart features)
# =============================================================================

EDA_VISUALIZATION_EXAMPLES = [
    # Histogram Features
    {
        "name": "VIZ - Histogram with Custom Bins",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a histogram for PRICE with 50 bins to see finer detail in the distribution.",
        "expected_tools": ["load_table_to_pandas", "plot_histogram"],
        "description": "Histogram with custom bin count"
    },
    {
        "name": "VIZ - Histogram without KDE",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a simple histogram for QUANTITY without the KDE curve overlay.",
        "expected_tools": ["load_table_to_pandas", "plot_histogram"],
        "description": "Histogram without KDE overlay"
    },

    # Scatter Features
    {
        "name": "VIZ - Colored Scatter",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a scatter plot of PRICE vs QUANTITY with points colored by TRANSACTION_TYPE.",
        "expected_tools": ["load_table_to_pandas", "plot_scatter"],
        "description": "Scatter plot with categorical coloring"
    },
    {
        "name": "VIZ - Sized Scatter",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a scatter plot of PRICE vs QUANTITY where point size represents FEES.",
        "expected_tools": ["load_table_to_pandas", "plot_scatter"],
        "description": "Scatter plot with size encoding"
    },

    # Correlation Features
    {
        "name": "VIZ - Spearman Correlation",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a correlation heatmap using Spearman correlation instead of Pearson.",
        "expected_tools": ["load_table_to_pandas", "plot_correlation_heatmap"],
        "description": "Correlation heatmap with Spearman method"
    },

    # Pairplot Features
    {
        "name": "VIZ - Pairplot with Hue",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and create a pairplot for PRICE, QUANTITY, and FEES, colored by TRANSACTION_TYPE.",
        "expected_tools": ["load_table_to_pandas", "plot_pairplot"],
        "description": "Pairplot with categorical hue"
    },

    # Bar Chart Features
    {
        "name": "VIZ - Top N Bar Chart",
        "agent": "eda_agent",
        "query": "Load ASSETS and create a horizontal bar chart showing only the top 5 asset types.",
        "expected_tools": ["load_table_to_pandas", "plot_bar_chart"],
        "description": "Bar chart with limited categories"
    },

    # Class Distribution Features
    {
        "name": "VIZ - Sparse Class Detection",
        "agent": "eda_agent",
        "query": "Load CLIENTS and create a class distribution chart for KYC_STATUS, highlighting any sparse or rare classes in red.",
        "expected_tools": ["load_table_to_pandas", "plot_class_distribution"],
        "description": "Class distribution with sparse class highlighting"
    },

    # Visual Report Features
    {
        "name": "VIZ - Summary Table Only",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS and generate the visual EDA report, which should include a formatted summary statistics table.",
        "expected_tools": ["load_table_to_pandas", "generate_eda_visual_report"],
        "description": "Visual report with summary table"
    },
]

# =============================================================================
# HTML REPORT EXAMPLES (Interactive browser-based reports with embedded charts)
# =============================================================================

EDA_HTML_REPORT_EXAMPLES = [
    # Basic HTML Report
    {
        "name": "HTML - Basic Report",
        "agent": "eda_agent",
        "query": "Load the TRANSACTIONS table and generate an interactive HTML EDA report that I can view in my browser with all charts embedded.",
        "expected_tools": ["load_table_to_pandas", "generate_eda_html_report"],
        "description": "Generate basic HTML report with embedded visualizations"
    },

    # Full Analysis with HTML Output
    {
        "name": "HTML - Full Analysis Report",
        "agent": "eda_agent",
        "query": """Perform a complete EDA on TRANSACTIONS:
        1. Load the data
        2. Get basic statistics
        3. Check for missing values
        4. Analyze correlations
        5. Generate an HTML report that shows all charts and tables in my browser.""",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics",
            "check_missing_values", "analyze_correlations",
            "generate_eda_html_report"
        ],
        "description": "Complete analysis with interactive HTML report"
    },

    # Client Analysis HTML Report
    {
        "name": "HTML - Client Analysis Report",
        "agent": "eda_agent",
        "query": "Load CLIENTS table, analyze all columns (categorical and numerical), check data quality, and generate an HTML report showing all findings with charts and tables that opens in my browser.",
        "expected_tools": [
            "load_table_to_pandas", "analyze_categorical_columns",
            "check_data_quality", "generate_eda_html_report"
        ],
        "description": "Client data HTML report with quality assessment"
    },

    # Portfolio Analysis HTML Report
    {
        "name": "HTML - Portfolio Dashboard",
        "agent": "eda_agent",
        "query": "Analyze PORTFOLIOS table - get statistics, check distributions, analyze categorical columns like STATUS and BASE_CURRENCY, then generate an interactive HTML dashboard that shows all visualizations embedded in one page.",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics",
            "analyze_distributions", "analyze_categorical_columns",
            "generate_eda_html_report"
        ],
        "description": "Portfolio data interactive HTML dashboard"
    },

    # Holdings Analysis HTML Report
    {
        "name": "HTML - Holdings Visual Report",
        "agent": "eda_agent",
        "query": """Comprehensive HOLDINGS analysis with visual HTML output:
        1. Load the data
        2. Get statistics for QUANTITY and AVG_COST
        3. Detect outliers
        4. Analyze correlations between numeric columns
        5. Generate an HTML report with all charts and summary tables viewable in browser.""",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics",
            "detect_outliers", "analyze_correlations",
            "generate_eda_html_report"
        ],
        "description": "Holdings analysis with embedded HTML charts"
    },

    # Assets Categorical HTML Report
    {
        "name": "HTML - Assets Categories Report",
        "agent": "eda_agent",
        "query": "Load ASSETS table, analyze categorical columns (ASSET_TYPE, CURRENCY, EXCHANGE), check for class imbalance, and generate an HTML report showing all bar charts and distribution tables in browser.",
        "expected_tools": [
            "load_table_to_pandas", "analyze_categorical_columns",
            "get_unique_value_counts", "generate_eda_html_report"
        ],
        "description": "Assets categorical analysis with HTML charts"
    },

    # Multi-Table Comparison HTML
    {
        "name": "HTML - Multi-Table Comparison",
        "agent": "eda_agent",
        "query": "Load both CLIENTS and TRANSACTIONS tables, perform EDA on each, then generate separate HTML reports for each table so I can compare them visually in my browser.",
        "expected_tools": [
            "load_table_to_pandas", "get_basic_statistics",
            "generate_eda_html_report"
        ],
        "description": "Multiple HTML reports for table comparison"
    },

    # Complete Pipeline with HTML Output
    {
        "name": "HTML - Complete EDA Pipeline",
        "agent": "eda_agent",
        "query": """Execute the full EDA pipeline on TRANSACTIONS with HTML output:

        ANALYSIS:
        - Load data and get structure info
        - Get basic statistics
        - Check missing values and duplicates
        - Analyze distributions (skewness, normality)
        - Detect outliers using IQR
        - Analyze correlations
        - Check data quality score

        OUTPUT:
        - Generate an interactive HTML report with ALL charts embedded
        - The report should open in my browser automatically
        - Include summary statistics table, distribution histograms, correlation heatmap, and box plots for outliers""",
        "expected_tools": [
            "load_table_to_pandas", "get_table_info_for_eda",
            "get_basic_statistics", "check_missing_values", "check_duplicates",
            "analyze_distributions", "detect_outliers", "analyze_correlations",
            "check_data_quality", "generate_eda_html_report"
        ],
        "description": "Complete EDA pipeline with comprehensive HTML report"
    },

    # Quality-Focused HTML Report
    {
        "name": "HTML - Data Quality Dashboard",
        "agent": "eda_agent",
        "query": "I want to assess the data quality of TRANSACTIONS table. Check for missing values, duplicates, data types, and run the quality assessment. Then generate an HTML report showing the quality score, any issues found, and visualizations of the data distributions.",
        "expected_tools": [
            "load_table_to_pandas", "check_missing_values", "check_duplicates",
            "check_data_types", "check_data_quality", "generate_eda_html_report"
        ],
        "description": "Data quality assessment with HTML dashboard"
    },

    # Outlier Investigation HTML
    {
        "name": "HTML - Outlier Investigation Report",
        "agent": "eda_agent",
        "query": "Load TRANSACTIONS, detect outliers in all numeric columns using both IQR and Z-score methods, and generate an HTML report that shows box plots highlighting the outliers along with summary statistics.",
        "expected_tools": [
            "load_table_to_pandas", "detect_outliers",
            "get_basic_statistics", "generate_eda_html_report"
        ],
        "description": "Outlier analysis with visual HTML evidence"
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_eda_examples():
    """Get all EDA examples combined."""
    return (
        EDA_VIZ_SIMPLE_EXAMPLES +
        EDA_VIZ_MEDIUM_EXAMPLES +
        EDA_VIZ_COMPLEX_EXAMPLES +
        EDA_EDGE_CASES +
        EDA_VISUALIZATION_EXAMPLES +
        EDA_HTML_REPORT_EXAMPLES
    )


def get_eda_examples_by_category(category: str):
    """Get EDA examples by category name."""
    categories = {
        "eda_simple": EDA_VIZ_SIMPLE_EXAMPLES,
        "eda_medium": EDA_VIZ_MEDIUM_EXAMPLES,
        "eda_complex": EDA_VIZ_COMPLEX_EXAMPLES,
        "eda_edge": EDA_EDGE_CASES,
        "eda_viz": EDA_VISUALIZATION_EXAMPLES,
        "eda_html": EDA_HTML_REPORT_EXAMPLES,
        "eda_all": get_all_eda_examples()
    }
    return categories.get(category.lower(), [])


def get_eda_examples_by_tool(tool_name: str):
    """Get EDA examples that use a specific tool."""
    all_examples = get_all_eda_examples()
    return [ex for ex in all_examples if tool_name in ex.get("expected_tools", [])]


def print_example_summary():
    """Print summary of all EDA examples."""
    print("=" * 70)
    print("EDA VISUALIZATION TEST EXAMPLES SUMMARY")
    print("=" * 70)
    print(f"\nSimple Examples:        {len(EDA_VIZ_SIMPLE_EXAMPLES)}")
    print(f"Medium Examples:        {len(EDA_VIZ_MEDIUM_EXAMPLES)}")
    print(f"Complex Examples:       {len(EDA_VIZ_COMPLEX_EXAMPLES)}")
    print(f"Edge Cases:             {len(EDA_EDGE_CASES)}")
    print(f"Visualization Examples: {len(EDA_VISUALIZATION_EXAMPLES)}")
    print(f"HTML Report Examples:   {len(EDA_HTML_REPORT_EXAMPLES)}")
    print(f"\nTotal Examples:         {len(get_all_eda_examples())}")
    print("=" * 70)

    # Count tools coverage
    all_tools = set()
    for ex in get_all_eda_examples():
        all_tools.update(ex.get("expected_tools", []))

    print(f"\nTools Covered: {len(all_tools)}")
    print("-" * 40)

    viz_tools = [t for t in all_tools if 'plot' in t or 'visual' in t]
    other_tools = [t for t in all_tools if 'plot' not in t and 'visual' not in t]

    print("\nVisualization Tools:")
    for t in sorted(viz_tools):
        count = len(get_eda_examples_by_tool(t))
        print(f"  - {t}: {count} examples")

    print("\nAnalysis Tools:")
    for t in sorted(other_tools):
        count = len(get_eda_examples_by_tool(t))
        print(f"  - {t}: {count} examples")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print_example_summary()

    print("\n" + "=" * 70)
    print("SAMPLE QUERIES FOR TESTING")
    print("=" * 70)

    # Print a few sample queries
    print("\n[SIMPLE] Basic Histogram:")
    print(f"  {EDA_VIZ_SIMPLE_EXAMPLES[0]['query']}")

    print("\n[MEDIUM] Correlation + Heatmap:")
    print(f"  {EDA_VIZ_MEDIUM_EXAMPLES[2]['query']}")

    print("\n[COMPLEX] Complete Visual Report:")
    print(f"  {EDA_VIZ_COMPLEX_EXAMPLES[0]['query']}")

    print("\n[HTML] Interactive Browser Report (RECOMMENDED):")
    print(f"  {EDA_HTML_REPORT_EXAMPLES[0]['query']}")

    print("\n" + "=" * 70)
    print("To run these tests:")
    print("  python run_tests.py --category eda_viz_simple")
    print("  python run_tests.py --category eda_viz_medium")
    print("  python run_tests.py --category eda_viz_complex")
    print("  python run_tests.py --category eda_viz_html    # HTML reports")
    print("  python run_tests.py --agent eda_agent")
    print("=" * 70)
