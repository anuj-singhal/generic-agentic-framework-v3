"""
Advanced Test Examples for Data Visualization Agent
====================================================

This file contains advanced test scenarios for the dataviz_agent that demonstrate:
- Multi-table analysis and cross-table dashboards
- Country maps and geographic visualizations
- 3D scatter plots and 3D bar charts
- Time series analysis with transactions
- Bubble charts showing multi-dimensional data
- Complex cross-table KPIs and visualizations

IMPORTANT: These examples use EXPLICIT chart creation calls to ensure specific chart types
are created. The agent should NOT rely solely on auto-generation for advanced charts.

Database Schema:
- CLIENTS (10 records): CLIENT_ID, FULL_NAME, COUNTRY, RISK_PROFILE, ONBOARDING_DATE, KYC_STATUS
- PORTFOLIOS (15 records): PORTFOLIO_ID, CLIENT_ID, PORTFOLIO_NAME, BASE_CURRENCY, INCEPTION_DATE, STATUS
- ASSETS (35 records): ASSET_ID, SYMBOL, ASSET_NAME, ASSET_TYPE, CURRENCY, EXCHANGE
- TRANSACTIONS (1200 records): TRANSACTION_ID, PORTFOLIO_ID, ASSET_ID, TRADE_DATE, TRANSACTION_TYPE, QUANTITY, PRICE, FEES
- HOLDINGS (305 records): PORTFOLIO_ID, ASSET_ID, QUANTITY, AVG_COST, LAST_UPDATED

Usage:
    1. Run the Streamlit app: streamlit run app.py
    2. Select the dataviz_agent
    3. Copy/paste these examples into the chat
"""

# =============================================================================
# MULTI-TABLE EXECUTIVE DASHBOARD - EXPLICIT CHART CREATION
# =============================================================================

MULTI_TABLE_EXAMPLES = [
    # Executive Dashboard with ALL chart types explicitly specified
    {
        "name": "Executive Wealth Dashboard - Explicit Charts",
        "agent": "dataviz_agent",
        "query": """Create a wealth management executive dashboard using CLIENTS, PORTFOLIOS, HOLDINGS, ASSETS, and TRANSACTIONS tables.

STEP-BY-STEP INSTRUCTIONS:

1. First, call analyze_multi_table_for_viz("CLIENTS,PORTFOLIOS,HOLDINGS,ASSETS,TRANSACTIONS") to get a session_id

2. Create EXACTLY 6 KPI cards using create_kpi_card() with these SQLs:
   - "Total AUM" (green): SELECT ROUND(SUM(QUANTITY * AVG_COST), 0) as value FROM HOLDINGS
   - "Active Clients" (blue): SELECT COUNT(*) as value FROM CLIENTS WHERE KYC_STATUS = 'Approved'
   - "Total Portfolios" (purple): SELECT COUNT(*) as value FROM PORTFOLIOS WHERE STATUS = 'Active'
   - "Trading Volume" (orange): SELECT ROUND(SUM(QUANTITY * PRICE), 0) as value FROM TRANSACTIONS
   - "Total Fees" (teal): SELECT ROUND(SUM(FEES), 2) as value FROM TRANSACTIONS
   - "Avg Portfolio Value" (pink): SELECT ROUND(AVG(portfolio_value), 0) as value FROM (SELECT p.PORTFOLIO_ID, SUM(h.QUANTITY * h.AVG_COST) as portfolio_value FROM PORTFOLIOS p JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY p.PORTFOLIO_ID)

3. Create a COUNTRY MAP using create_country_map():
   SQL: SELECT c.COUNTRY, SUM(h.QUANTITY * h.AVG_COST) as total_aum FROM CLIENTS c JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY c.COUNTRY
   country_column="COUNTRY", value_column="total_aum", color_scale="Viridis"
   Title: "AUM by Country"

4. Create a BAR CHART using create_bar_chart():
   SQL: SELECT c.FULL_NAME, SUM(h.QUANTITY * h.AVG_COST) as total_value FROM CLIENTS c JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY c.FULL_NAME ORDER BY total_value DESC LIMIT 5
   x_column="FULL_NAME", y_column="total_value"
   Title: "Top 5 Clients by AUM"

5. Create a DONUT CHART using create_donut_chart():
   SQL: SELECT RISK_PROFILE, COUNT(*) as count FROM CLIENTS GROUP BY RISK_PROFILE
   names_column="RISK_PROFILE", values_column="count"
   Title: "Clients by Risk Profile"

6. Create a TREEMAP using create_treemap():
   SQL: SELECT a.ASSET_TYPE, SUM(h.QUANTITY * h.AVG_COST) as total_value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE
   labels_column="ASSET_TYPE", values_column="total_value"
   Title: "AUM by Asset Type"

7. Create a LINE CHART using create_line_chart():
   SQL: SELECT DATE_TRUNC('month', TRADE_DATE) as month, SUM(QUANTITY * PRICE) as volume FROM TRANSACTIONS GROUP BY DATE_TRUNC('month', TRADE_DATE) ORDER BY month
   x_column="month", y_column="volume"
   Title: "Monthly Trading Volume"

8. Create a DATA TABLE using create_data_table():
   SQL: SELECT c.FULL_NAME, c.COUNTRY, c.RISK_PROFILE, COUNT(DISTINCT p.PORTFOLIO_ID) as portfolios, ROUND(SUM(h.QUANTITY * h.AVG_COST), 0) as total_aum FROM CLIENTS c LEFT JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID LEFT JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY c.CLIENT_ID, c.FULL_NAME, c.COUNTRY, c.RISK_PROFILE ORDER BY total_aum DESC
   Title: "Client Summary"

9. Generate the dashboard using generate_dashboard(session_id, "executive_wealth_dashboard")

IMPORTANT: Use dark theme. The file path should use forward slashes when returning to user.""",
        "expected_tools": [
            "analyze_multi_table_for_viz",
            "create_kpi_card",
            "create_country_map",
            "create_bar_chart",
            "create_donut_chart",
            "create_treemap",
            "create_line_chart",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "Executive dashboard with explicit chart creation - includes map, bar, donut, treemap, line charts"
    },

    # Client-Centric Dashboard with Map Focus
    {
        "name": "Client Geographic Dashboard",
        "agent": "dataviz_agent",
        "query": """Create a client-focused geographic dashboard using CLIENTS, PORTFOLIOS, and HOLDINGS.

Follow these EXACT steps:

1. analyze_multi_table_for_viz("CLIENTS,PORTFOLIOS,HOLDINGS") to get session_id

2. Create 4 KPI cards:
   - "Total Clients" (blue): SELECT COUNT(*) FROM CLIENTS
   - "Countries Served" (green): SELECT COUNT(DISTINCT COUNTRY) FROM CLIENTS
   - "Total AUM" (purple, currency format): SELECT SUM(QUANTITY * AVG_COST) FROM HOLDINGS
   - "Avg AUM per Client" (orange, currency format): SELECT AVG(client_aum) FROM (SELECT c.CLIENT_ID, SUM(h.QUANTITY * h.AVG_COST) as client_aum FROM CLIENTS c JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY c.CLIENT_ID)

3. Create COUNTRY MAP (MOST IMPORTANT):
   create_country_map(session_id,
     title="Client Distribution by AUM",
     sql="SELECT c.COUNTRY, SUM(h.QUANTITY * h.AVG_COST) as total_aum FROM CLIENTS c JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY c.COUNTRY",
     country_column="COUNTRY",
     value_column="total_aum",
     color_scale="Plasma")

4. Create COLORFUL BAR CHART:
   create_colorful_bar_chart(session_id,
     title="AUM by Country",
     sql="SELECT c.COUNTRY, SUM(h.QUANTITY * h.AVG_COST) as total_aum FROM CLIENTS c JOIN PORTFOLIOS p ON c.CLIENT_ID = p.CLIENT_ID JOIN HOLDINGS h ON p.PORTFOLIO_ID = h.PORTFOLIO_ID GROUP BY c.COUNTRY ORDER BY total_aum DESC",
     x_column="COUNTRY",
     y_column="total_aum",
     orientation="v")

5. Create DONUT chart for risk profile:
   create_donut_chart(session_id,
     title="Risk Profile Distribution",
     sql="SELECT RISK_PROFILE, COUNT(*) as count FROM CLIENTS GROUP BY RISK_PROFILE",
     names_column="RISK_PROFILE",
     values_column="count")

6. Create DATA TABLE:
   create_data_table(session_id,
     title="Client Details",
     sql="SELECT FULL_NAME, COUNTRY, RISK_PROFILE, KYC_STATUS FROM CLIENTS ORDER BY FULL_NAME")

7. generate_dashboard(session_id, "client_geographic_dashboard")

Use dark theme with Plasma color scale for the map.""",
        "expected_tools": [
            "analyze_multi_table_for_viz",
            "create_kpi_card",
            "create_country_map",
            "create_colorful_bar_chart",
            "create_donut_chart",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "Geographic dashboard with country choropleth map and colorful bar charts"
    },
]

# =============================================================================
# 3D VISUALIZATION EXAMPLES
# =============================================================================

THREE_D_EXAMPLES = [
    # 3D Portfolio Analysis
    {
        "name": "3D Portfolio Analysis Dashboard",
        "agent": "dataviz_agent",
        "query": """Create a dashboard with 3D visualizations for HOLDINGS and ASSETS.

EXACT STEPS:

1. analyze_multi_table_for_viz("HOLDINGS,ASSETS") to get session_id

2. Create 4 KPIs:
   - "Total Positions" (blue): SELECT COUNT(*) FROM HOLDINGS
   - "Total Value" (green, currency): SELECT SUM(QUANTITY * AVG_COST) FROM HOLDINGS
   - "Unique Assets" (purple): SELECT COUNT(DISTINCT ASSET_ID) FROM HOLDINGS
   - "Avg Position Size" (orange, currency): SELECT AVG(QUANTITY * AVG_COST) FROM HOLDINGS

3. Create 3D SCATTER chart (CRITICAL):
   create_3d_scatter(session_id,
     title="Holdings 3D View: Quantity vs Avg Cost vs Total Value",
     sql="SELECT h.QUANTITY, h.AVG_COST, (h.QUANTITY * h.AVG_COST) as total_value, a.ASSET_TYPE FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID WHERE h.QUANTITY > 0 LIMIT 200",
     x_column="QUANTITY",
     y_column="AVG_COST",
     z_column="total_value",
     color_column="ASSET_TYPE")

4. Create BUBBLE chart:
   create_bubble_chart(session_id,
     title="Asset Analysis: Quantity vs Cost (Size = Value)",
     sql="SELECT h.QUANTITY, h.AVG_COST, (h.QUANTITY * h.AVG_COST) as total_value, a.SYMBOL FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID ORDER BY total_value DESC LIMIT 50",
     x_column="QUANTITY",
     y_column="AVG_COST",
     size_column="total_value",
     text_column="SYMBOL")

5. Create TREEMAP for asset allocation:
   create_treemap(session_id,
     title="Holdings by Asset Type",
     sql="SELECT a.ASSET_TYPE, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE",
     labels_column="ASSET_TYPE",
     values_column="value")

6. Create DATA TABLE:
   create_data_table(session_id,
     title="Top Holdings",
     sql="SELECT a.SYMBOL, a.ASSET_TYPE, h.QUANTITY, h.AVG_COST, ROUND(h.QUANTITY * h.AVG_COST, 2) as total_value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID ORDER BY total_value DESC LIMIT 20")

7. generate_dashboard(session_id, "portfolio_3d_dashboard")

Use dark theme.""",
        "expected_tools": [
            "analyze_multi_table_for_viz",
            "create_kpi_card",
            "create_3d_scatter",
            "create_bubble_chart",
            "create_treemap",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "3D scatter plot and bubble chart dashboard for portfolio analysis"
    },
]

# =============================================================================
# TIME SERIES EXAMPLES
# =============================================================================

TIME_SERIES_EXAMPLES = [
    # Trading Activity Time Series
    {
        "name": "Trading Activity Time Series Dashboard",
        "agent": "dataviz_agent",
        "query": """Create a time series dashboard analyzing TRANSACTIONS.

EXACT STEPS:

1. analyze_data_for_viz("TRANSACTIONS") to get session_id

2. Create 5 KPIs:
   - "Total Transactions" (blue): SELECT COUNT(*) FROM TRANSACTIONS
   - "Total Volume" (green, currency): SELECT SUM(QUANTITY * PRICE) FROM TRANSACTIONS
   - "Total Fees" (purple, currency): SELECT SUM(FEES) FROM TRANSACTIONS
   - "Avg Transaction" (orange, currency): SELECT AVG(QUANTITY * PRICE) FROM TRANSACTIONS
   - "Buy/Sell Ratio" (teal): SELECT ROUND(CAST(SUM(CASE WHEN TRANSACTION_TYPE='BUY' THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(SUM(CASE WHEN TRANSACTION_TYPE='SELL' THEN 1 ELSE 0 END), 0), 2) FROM TRANSACTIONS

3. Create LINE CHART for volume over time:
   create_line_chart(session_id,
     title="Daily Trading Volume",
     sql="SELECT TRADE_DATE, SUM(QUANTITY * PRICE) as volume FROM TRANSACTIONS GROUP BY TRADE_DATE ORDER BY TRADE_DATE",
     x_column="TRADE_DATE",
     y_column="volume")

4. Create AREA CHART for cumulative volume:
   create_area_chart(session_id,
     title="Cumulative Trading Volume",
     sql="SELECT TRADE_DATE, SUM(SUM(QUANTITY * PRICE)) OVER (ORDER BY TRADE_DATE) as cumulative_volume FROM TRANSACTIONS GROUP BY TRADE_DATE ORDER BY TRADE_DATE",
     x_column="TRADE_DATE",
     y_column="cumulative_volume")

5. Create STACKED BAR for buy vs sell:
   create_stacked_bar_chart(session_id,
     title="Buy vs Sell Volume by Month",
     sql="SELECT DATE_TRUNC('month', TRADE_DATE) as month, TRANSACTION_TYPE, SUM(QUANTITY * PRICE) as volume FROM TRANSACTIONS GROUP BY DATE_TRUNC('month', TRADE_DATE), TRANSACTION_TYPE ORDER BY month",
     x_column="month",
     y_column="volume",
     color_column="TRANSACTION_TYPE")

6. Create HISTOGRAM for transaction sizes:
   create_histogram(session_id,
     title="Transaction Size Distribution",
     sql="SELECT (QUANTITY * PRICE) as transaction_value FROM TRANSACTIONS",
     column="transaction_value",
     nbins=30)

7. Create DATA TABLE:
   create_data_table(session_id,
     title="Recent Transactions",
     sql="SELECT TRADE_DATE, TRANSACTION_TYPE, QUANTITY, PRICE, ROUND(QUANTITY*PRICE, 2) as value, FEES FROM TRANSACTIONS ORDER BY TRADE_DATE DESC LIMIT 50")

8. generate_dashboard(session_id, "trading_activity_dashboard")

Dark theme.""",
        "expected_tools": [
            "analyze_data_for_viz",
            "create_kpi_card",
            "create_line_chart",
            "create_area_chart",
            "create_stacked_bar_chart",
            "create_histogram",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "Time series dashboard with line charts, area charts, and stacked bars"
    },
]

# =============================================================================
# ASSET ANALYSIS WITH TREEMAPS
# =============================================================================

TREEMAP_EXAMPLES = [
    # Asset Catalog with Treemap
    {
        "name": "Asset Allocation Treemap Dashboard",
        "agent": "dataviz_agent",
        "query": """Create an asset allocation dashboard using ASSETS and HOLDINGS with treemaps.

STEPS:

1. analyze_multi_table_for_viz("ASSETS,HOLDINGS") to get session_id

2. Create 4 KPIs:
   - "Total Assets" (blue): SELECT COUNT(*) FROM ASSETS
   - "Asset Types" (green): SELECT COUNT(DISTINCT ASSET_TYPE) FROM ASSETS
   - "Total Value" (purple, currency): SELECT SUM(QUANTITY * AVG_COST) FROM HOLDINGS
   - "Exchanges" (orange): SELECT COUNT(DISTINCT EXCHANGE) FROM ASSETS

3. Create TREEMAP for asset type allocation:
   create_treemap(session_id,
     title="Portfolio Allocation by Asset Type",
     sql="SELECT a.ASSET_TYPE, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE ORDER BY value DESC",
     labels_column="ASSET_TYPE",
     values_column="value")

4. Create second TREEMAP for exchange distribution:
   create_treemap(session_id,
     title="Holdings by Exchange",
     sql="SELECT a.EXCHANGE, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.EXCHANGE ORDER BY value DESC",
     labels_column="EXCHANGE",
     values_column="value")

5. Create COLORFUL BAR for top assets:
   create_colorful_bar_chart(session_id,
     title="Top 10 Assets by Value",
     sql="SELECT a.SYMBOL, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.SYMBOL ORDER BY value DESC LIMIT 10",
     x_column="SYMBOL",
     y_column="value")

6. Create DONUT for asset type count:
   create_donut_chart(session_id,
     title="Assets by Type",
     sql="SELECT ASSET_TYPE, COUNT(*) as count FROM ASSETS GROUP BY ASSET_TYPE",
     names_column="ASSET_TYPE",
     values_column="count")

7. Create DATA TABLE:
   create_data_table(session_id,
     title="Asset Catalog",
     sql="SELECT SYMBOL, ASSET_NAME, ASSET_TYPE, CURRENCY, EXCHANGE FROM ASSETS ORDER BY ASSET_TYPE, SYMBOL")

8. generate_dashboard(session_id, "asset_allocation_dashboard")

Dark theme.""",
        "expected_tools": [
            "analyze_multi_table_for_viz",
            "create_kpi_card",
            "create_treemap",
            "create_colorful_bar_chart",
            "create_donut_chart",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "Asset allocation dashboard with multiple treemaps and colorful charts"
    },
]

# =============================================================================
# WATERFALL AND GAUGE EXAMPLES
# =============================================================================

WATERFALL_GAUGE_EXAMPLES = [
    # Portfolio Breakdown with Waterfall
    {
        "name": "Portfolio Value Waterfall Dashboard",
        "agent": "dataviz_agent",
        "query": """Create a dashboard showing portfolio value breakdown using waterfall and gauge charts.

STEPS:

1. analyze_multi_table_for_viz("HOLDINGS,ASSETS") to get session_id

2. Create 4 KPIs:
   - "Total AUM" (green, currency): SELECT SUM(QUANTITY * AVG_COST) FROM HOLDINGS
   - "Equity Value" (blue, currency): SELECT SUM(h.QUANTITY * h.AVG_COST) FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID WHERE a.ASSET_TYPE = 'Equity'
   - "ETF Value" (purple, currency): SELECT SUM(h.QUANTITY * h.AVG_COST) FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID WHERE a.ASSET_TYPE = 'ETF'
   - "Crypto Value" (orange, currency): SELECT SUM(h.QUANTITY * h.AVG_COST) FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID WHERE a.ASSET_TYPE = 'Crypto'

3. Create WATERFALL chart:
   create_waterfall_chart(session_id,
     title="AUM Breakdown by Asset Type",
     sql="SELECT a.ASSET_TYPE, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE ORDER BY value DESC",
     x_column="ASSET_TYPE",
     y_column="value")

4. Create GAUGE for portfolio health:
   create_gauge_chart(session_id,
     title="Average Position Value",
     sql="SELECT AVG(QUANTITY * AVG_COST) as value, 0 as min_val, MAX(QUANTITY * AVG_COST) as max_val FROM HOLDINGS",
     value_format="currency")

5. Create BAR chart for comparison:
   create_bar_chart(session_id,
     title="Value by Asset Type",
     sql="SELECT a.ASSET_TYPE, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE ORDER BY value DESC",
     x_column="ASSET_TYPE",
     y_column="value")

6. Create DONUT for allocation:
   create_donut_chart(session_id,
     title="Asset Type Allocation",
     sql="SELECT a.ASSET_TYPE, SUM(h.QUANTITY * h.AVG_COST) as value FROM HOLDINGS h JOIN ASSETS a ON h.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE",
     names_column="ASSET_TYPE",
     values_column="value")

7. generate_dashboard(session_id, "portfolio_waterfall_dashboard")

Dark theme.""",
        "expected_tools": [
            "analyze_multi_table_for_viz",
            "create_kpi_card",
            "create_waterfall_chart",
            "create_gauge_chart",
            "create_bar_chart",
            "create_donut_chart",
            "generate_dashboard"
        ],
        "description": "Waterfall and gauge chart dashboard for portfolio breakdown"
    },
]

# =============================================================================
# HEATMAP EXAMPLES
# =============================================================================

HEATMAP_EXAMPLES = [
    # Trading Pattern Heatmap
    {
        "name": "Trading Heatmap Dashboard",
        "agent": "dataviz_agent",
        "query": """Create a trading pattern heatmap dashboard using TRANSACTIONS and ASSETS.

STEPS:

1. analyze_multi_table_for_viz("TRANSACTIONS,ASSETS") to get session_id

2. Create 4 KPIs:
   - "Total Trades" (blue): SELECT COUNT(*) FROM TRANSACTIONS
   - "Total Volume" (green, currency): SELECT SUM(QUANTITY * PRICE) FROM TRANSACTIONS
   - "Avg Trade Size" (purple, currency): SELECT AVG(QUANTITY * PRICE) FROM TRANSACTIONS
   - "Active Assets" (orange): SELECT COUNT(DISTINCT ASSET_ID) FROM TRANSACTIONS

3. Create HEATMAP for trading by asset type and month:
   create_heatmap(session_id,
     title="Trading Volume: Month vs Asset Type",
     sql="SELECT EXTRACT(MONTH FROM t.TRADE_DATE) as month, a.ASSET_TYPE, SUM(t.QUANTITY * t.PRICE) as volume FROM TRANSACTIONS t JOIN ASSETS a ON t.ASSET_ID = a.ASSET_ID GROUP BY EXTRACT(MONTH FROM t.TRADE_DATE), a.ASSET_TYPE ORDER BY month, a.ASSET_TYPE",
     x_column="ASSET_TYPE",
     y_column="month",
     value_column="volume")

4. Create STACKED BAR for buy/sell by type:
   create_stacked_bar_chart(session_id,
     title="Buy vs Sell by Asset Type",
     sql="SELECT a.ASSET_TYPE, t.TRANSACTION_TYPE, SUM(t.QUANTITY * t.PRICE) as volume FROM TRANSACTIONS t JOIN ASSETS a ON t.ASSET_ID = a.ASSET_ID GROUP BY a.ASSET_TYPE, t.TRANSACTION_TYPE",
     x_column="ASSET_TYPE",
     y_column="volume",
     color_column="TRANSACTION_TYPE")

5. Create LINE chart for trends:
   create_line_chart(session_id,
     title="Monthly Trading Trend",
     sql="SELECT DATE_TRUNC('month', TRADE_DATE) as month, SUM(QUANTITY * PRICE) as volume FROM TRANSACTIONS GROUP BY DATE_TRUNC('month', TRADE_DATE) ORDER BY month",
     x_column="month",
     y_column="volume")

6. Create DATA TABLE:
   create_data_table(session_id,
     title="Trading Summary by Asset",
     sql="SELECT a.SYMBOL, a.ASSET_TYPE, COUNT(t.TRANSACTION_ID) as trades, SUM(t.QUANTITY * t.PRICE) as volume FROM TRANSACTIONS t JOIN ASSETS a ON t.ASSET_ID = a.ASSET_ID GROUP BY a.SYMBOL, a.ASSET_TYPE ORDER BY volume DESC LIMIT 20")

7. generate_dashboard(session_id, "trading_heatmap_dashboard")

Dark theme.""",
        "expected_tools": [
            "analyze_multi_table_for_viz",
            "create_kpi_card",
            "create_heatmap",
            "create_stacked_bar_chart",
            "create_line_chart",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "Heatmap dashboard for trading pattern analysis"
    },
]

# =============================================================================
# SIMPLE SINGLE-TABLE EXAMPLES
# =============================================================================

SINGLE_TABLE_EXAMPLES = [
    # Simple Clients Dashboard
    {
        "name": "Simple Clients Dashboard",
        "agent": "dataviz_agent",
        "query": """Create a simple dashboard for the CLIENTS table.

STEPS:
1. analyze_data_for_viz("CLIENTS") to get session_id
2. Create 3 KPIs:
   - "Total Clients" (blue): SELECT COUNT(*) FROM CLIENTS
   - "Approved KYC" (green): SELECT COUNT(*) FROM CLIENTS WHERE KYC_STATUS = 'Approved'
   - "Countries" (purple): SELECT COUNT(DISTINCT COUNTRY) FROM CLIENTS

3. Create BAR chart:
   create_bar_chart(session_id, "Clients by Country",
     sql="SELECT COUNTRY, COUNT(*) as count FROM CLIENTS GROUP BY COUNTRY ORDER BY count DESC",
     x_column="COUNTRY", y_column="count")

4. Create DONUT chart:
   create_donut_chart(session_id, "Risk Profile Distribution",
     sql="SELECT RISK_PROFILE, COUNT(*) as count FROM CLIENTS GROUP BY RISK_PROFILE",
     names_column="RISK_PROFILE", values_column="count")

5. Create DATA TABLE:
   create_data_table(session_id, "All Clients",
     sql="SELECT * FROM CLIENTS ORDER BY FULL_NAME")

6. generate_dashboard(session_id, "simple_clients_dashboard")

Dark theme.""",
        "expected_tools": [
            "analyze_data_for_viz",
            "create_kpi_card",
            "create_bar_chart",
            "create_donut_chart",
            "create_data_table",
            "generate_dashboard"
        ],
        "description": "Simple single-table dashboard with bar and donut charts"
    },
]

# =============================================================================
# COMBINE ALL EXAMPLES
# =============================================================================

DATA_VIZ_ADVANCED_EXAMPLES = (
    MULTI_TABLE_EXAMPLES +
    THREE_D_EXAMPLES +
    TIME_SERIES_EXAMPLES +
    TREEMAP_EXAMPLES +
    WATERFALL_GAUGE_EXAMPLES +
    HEATMAP_EXAMPLES +
    SINGLE_TABLE_EXAMPLES
)

EXAMPLE_CATEGORIES = {
    "multi_table": MULTI_TABLE_EXAMPLES,
    "3d": THREE_D_EXAMPLES,
    "time_series": TIME_SERIES_EXAMPLES,
    "treemap": TREEMAP_EXAMPLES,
    "waterfall_gauge": WATERFALL_GAUGE_EXAMPLES,
    "heatmap": HEATMAP_EXAMPLES,
    "single_table": SINGLE_TABLE_EXAMPLES,
    "all": DATA_VIZ_ADVANCED_EXAMPLES
}


def list_examples():
    """Print all available examples grouped by category."""
    print("\n" + "=" * 80)
    print("DATA VISUALIZATION ADVANCED TEST EXAMPLES")
    print("=" * 80)

    for category, examples in EXAMPLE_CATEGORIES.items():
        if category == "all":
            continue
        print(f"\n{category.upper().replace('_', ' ')} ({len(examples)} examples):")
        print("-" * 40)
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex['name']}")
            print(f"     {ex['description']}")

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {len(DATA_VIZ_ADVANCED_EXAMPLES)} advanced examples")
    print("=" * 80)


def get_example_by_name(name: str):
    """Get a specific example by name."""
    for example in DATA_VIZ_ADVANCED_EXAMPLES:
        if example["name"].lower() == name.lower():
            return example
    return None


def get_examples_by_category(category: str):
    """Get all examples in a category."""
    return EXAMPLE_CATEGORIES.get(category.lower(), [])


if __name__ == "__main__":
    list_examples()
