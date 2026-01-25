"""
Test Examples for the Agentic AI Framework
==========================================

This file contains a variety of test scenarios ranging from simple to complex,
demonstrating the ReAct pattern with multiple tools and agents.

Usage:
    1. Run the Streamlit app: streamlit run app.py
    2. Copy/paste these examples into the chat
    3. Or run this file directly: python test_examples.py
"""

# =============================================================================
# SIMPLE EXAMPLES (Single tool, straightforward tasks)
# =============================================================================

SIMPLE_EXAMPLES = [
    # Math - Calculator
    {
        "name": "Basic Calculation",
        "agent": "math_specialist",
        "query": "What is 15% of 850?",
        "expected_tools": ["calculator"],
        "description": "Simple percentage calculation"
    },
    {
        "name": "Scientific Calculation",
        "agent": "math_specialist",
        "query": "Calculate the square root of 144 plus the cube of 5",
        "expected_tools": ["calculator"],
        "description": "Mathematical expression with multiple operations"
    },
    
    # Math - Unit Conversion
    {
        "name": "Temperature Conversion",
        "agent": "math_specialist",
        "query": "Convert 98.6 degrees Fahrenheit to Celsius",
        "expected_tools": ["unit_converter"],
        "description": "Simple unit conversion"
    },
    {
        "name": "Distance Conversion",
        "agent": "math_specialist",
        "query": "How many kilometers is 26.2 miles?",
        "expected_tools": ["unit_converter"],
        "description": "Marathon distance conversion"
    },
    
    # DateTime - Current Time
    {
        "name": "Current DateTime",
        "agent": "general_assistant",
        "query": "What is today's date and time?",
        "expected_tools": ["get_current_datetime"],
        "description": "Get current date and time"
    },
    
    # Text - Analysis
    {
        "name": "Text Statistics",
        "agent": "data_analyst",
        "query": "Analyze this text and give me statistics: 'The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.'",
        "expected_tools": ["text_analyzer"],
        "description": "Basic text analysis"
    },
    
    # Text - Transformation
    {
        "name": "Text Transform",
        "agent": "data_analyst",
        "query": "Convert 'hello world from the agentic framework' to title case",
        "expected_tools": ["text_transformer"],
        "description": "Simple text transformation"
    },
    
    # Task - Create
    {
        "name": "Create Single Task",
        "agent": "task_manager",
        "query": "Create a high priority task called 'Review quarterly report' with description 'Analyze Q4 financial data'",
        "expected_tools": ["create_task"],
        "description": "Create a single task"
    },
    
    # Knowledge - Search
    {
        "name": "Knowledge Query",
        "agent": "researcher",
        "query": "What is Python programming language?",
        "expected_tools": ["knowledge_base_search"],
        "description": "Simple knowledge base lookup"
    },
    
    # Data - List Operations
    {
        "name": "Sort List",
        "agent": "data_analyst",
        "query": "Sort this list alphabetically: banana, apple, cherry, date, elderberry",
        "expected_tools": ["list_operations"],
        "description": "Simple list sorting"
    },
]

# =============================================================================
# MEDIUM EXAMPLES (Multiple tools, multi-step reasoning)
# =============================================================================

MEDIUM_EXAMPLES = [
    # Math combinations
    {
        "name": "Multi-Step Calculation",
        "agent": "math_specialist",
        "query": "I'm traveling 150 miles. If my car gets 30 miles per gallon and gas costs $3.50 per gallon, how much will the trip cost? Also convert the distance to kilometers.",
        "expected_tools": ["calculator", "unit_converter"],
        "description": "Combines calculation with unit conversion"
    },
    {
        "name": "Compound Interest",
        "agent": "math_specialist",
        "query": "If I invest $10,000 at 5% annual interest compounded yearly, how much will I have after 10 years? Use the formula: Principal * (1 + rate)^years",
        "expected_tools": ["calculator"],
        "description": "Financial calculation requiring formula application"
    },
    
    # DateTime combinations
    {
        "name": "Date Planning",
        "agent": "general_assistant",
        "query": "Today is the start date. I need to plan a project that takes 45 days. What will be the end date? And how many days until December 31, 2025?",
        "expected_tools": ["get_current_datetime", "add_days_to_date", "calculate_date_difference"],
        "description": "Multiple date operations"
    },
    {
        "name": "Event Countdown",
        "agent": "general_assistant",
        "query": "Calculate how many days are between January 1, 2025 and July 4, 2025, then add 30 days to July 4, 2025 to find a follow-up date",
        "expected_tools": ["calculate_date_difference", "add_days_to_date"],
        "description": "Date difference and addition"
    },
    
    # Text + Data combinations
    {
        "name": "Text Analysis Pipeline",
        "agent": "data_analyst",
        "query": "Analyze this paragraph for statistics, then transform it to uppercase: 'Machine learning is transforming how we interact with technology. From virtual assistants to recommendation systems, AI is everywhere.'",
        "expected_tools": ["text_analyzer", "text_transformer"],
        "description": "Chained text operations"
    },
    {
        "name": "Data Processing",
        "agent": "data_analyst",
        "query": "I have this JSON: '{\"employees\": [{\"name\": \"John\", \"age\": 30}, {\"name\": \"Jane\", \"age\": 25}]}'. Extract the employees array, then tell me how many items are in it.",
        "expected_tools": ["json_parser", "list_operations"],
        "description": "JSON parsing and list analysis"
    },
    
    # Task Management
    {
        "name": "Task Workflow",
        "agent": "task_manager",
        "query": "Create three tasks: 'Design mockups' (high priority), 'Write documentation' (medium priority), and 'Code review' (low priority). Then list all pending tasks.",
        "expected_tools": ["create_task", "list_tasks"],
        "description": "Multiple task creation and listing"
    },
    {
        "name": "Task Status Update",
        "agent": "task_manager",
        "query": "First, list all tasks to see what we have. Then update the first task to 'in_progress' status.",
        "expected_tools": ["list_tasks", "update_task_status"],
        "description": "Query and update tasks"
    },
    
    # Research combinations
    {
        "name": "Research and Analyze",
        "agent": "researcher",
        "query": "Search for information about LangGraph, then analyze the text you find to give me word count and other statistics.",
        "expected_tools": ["knowledge_base_search", "text_analyzer"],
        "description": "Search and analyze results"
    },
    
    # Cross-domain
    {
        "name": "Recipe Scaling",
        "agent": "general_assistant",
        "query": "A recipe for 4 people needs 2.5 cups of flour. I'm cooking for 7 people. How many cups do I need? Also convert that to milliliters (1 cup = 236.588 ml).",
        "expected_tools": ["calculator", "unit_converter"],
        "description": "Practical math application"
    },
]

# =============================================================================
# COMPLEX EXAMPLES (Multi-agent capable, complex reasoning, many tools)
# =============================================================================

COMPLEX_EXAMPLES = [
    # Comprehensive Project Planning
    {
        "name": "Project Planning Suite",
        "agent": "general_assistant",
        "query": """I'm starting a new software project. Help me:
1. Get today's date as the start date
2. Calculate the end date if the project takes 90 days
3. Create tasks for: 'Requirements gathering' (high), 'Development' (high), 'Testing' (medium), 'Deployment' (medium)
4. List all the tasks we created
5. Search for information about 'agent' to help with the project""",
        "expected_tools": ["get_current_datetime", "add_days_to_date", "create_task", "list_tasks", "knowledge_base_search"],
        "description": "Full project setup with dates, tasks, and research"
    },
    
    # Financial Analysis
    {
        "name": "Investment Analysis",
        "agent": "math_specialist",
        "query": """Analyze this investment scenario:
1. Initial investment: $50,000
2. Calculate 7% annual return after 5 years (compound interest: P * (1.07)^5)
3. Calculate the total gain (final - initial)
4. What percentage gain is that? ((gain/initial) * 100)
5. Convert the final amount from USD to approximate EUR (assume 1 USD = 0.92 EUR, so multiply by 0.92)""",
        "expected_tools": ["calculator"],
        "description": "Multi-step financial calculations"
    },
    
    # Data Pipeline
    {
        "name": "Data Analysis Pipeline",
        "agent": "data_analyst",
        "query": """Process this data:
1. Parse this JSON: '{"products": ["laptop", "phone", "tablet", "watch", "headphones"], "prices": [999, 699, 449, 299, 199]}'
2. Extract the products list and sort them alphabetically
3. Calculate the average price: (999 + 699 + 449 + 299 + 199) / 5
4. Calculate the total inventory value
5. Analyze this product description text: 'Our premium electronics lineup features cutting-edge technology designed for modern consumers.'""",
        "expected_tools": ["json_parser", "list_operations", "calculator", "text_analyzer"],
        "description": "Complete data processing pipeline"
    },
    
    # Event Planning
    {
        "name": "Conference Planning",
        "agent": "general_assistant",
        "query": """Help me plan a tech conference:
1. The conference is on 2025-06-15. How many days from today until the conference?
2. Registration deadline should be 30 days before the conference. What date is that?
3. Create these tasks:
   - 'Book venue' (high priority)
   - 'Send invitations' (high priority)  
   - 'Arrange catering' (medium priority)
   - 'Prepare presentations' (medium priority)
   - 'Set up registration' (high priority)
4. List all tasks
5. Calculate the budget: Venue $5000 + Catering $3000 + Marketing $2000 + Miscellaneous $1500""",
        "expected_tools": ["get_current_datetime", "calculate_date_difference", "add_days_to_date", "create_task", "list_tasks", "calculator"],
        "description": "Complete event planning workflow"
    },
    
    # Research Report
    {
        "name": "Technology Research Report",
        "agent": "researcher",
        "query": """Create a mini research report:
1. Search for information about 'Python' programming
2. Search for information about 'ReAct' AI pattern
3. Search for information about 'Streamlit'
4. Analyze the combined information for text statistics
5. Create a task to 'Write full research report' with high priority""",
        "expected_tools": ["knowledge_base_search", "text_analyzer", "create_task"],
        "description": "Multi-topic research with documentation"
    },
    
    # Fitness Tracker Calculations
    {
        "name": "Fitness Progress Analysis",
        "agent": "general_assistant",
        "query": """Analyze my fitness progress:
1. I ran 5.5 miles today. Convert that to kilometers.
2. My run took 48 minutes. Calculate my pace in minutes per mile (48/5.5).
3. Also calculate my speed in miles per hour (5.5 / (48/60)).
4. Convert my speed to km/h.
5. I weighed 180 pounds at the start. Convert to kg.
6. Create a task 'Log weekly fitness summary' with medium priority.
7. What's today's date for my fitness log?""",
        "expected_tools": ["unit_converter", "calculator", "create_task", "get_current_datetime"],
        "description": "Comprehensive fitness calculations and tracking"
    },
    
    # Business Metrics
    {
        "name": "Business Dashboard",
        "agent": "data_analyst",
        "query": """Calculate business metrics from this data:
1. Parse: '{"sales": [12000, 15000, 18000, 14000, 20000], "months": ["Jan", "Feb", "Mar", "Apr", "May"]}'
2. Calculate total sales (sum all values)
3. Calculate average monthly sales
4. Calculate month-over-month growth from Jan to May: ((20000-12000)/12000)*100
5. Sort the months by their sales values: 'Jan:12000, Feb:15000, Mar:18000, Apr:14000, May:20000'
6. Analyze this executive summary: 'Q1 performance exceeded expectations with strong growth in digital channels. Customer acquisition costs decreased by 15% while lifetime value increased.'
7. Create task 'Prepare Q2 forecast' with high priority""",
        "expected_tools": ["json_parser", "calculator", "list_operations", "text_analyzer", "create_task"],
        "description": "Full business analytics workflow"
    },
    
    # Travel Planning
    {
        "name": "Travel Itinerary",
        "agent": "general_assistant",
        "query": """Plan my trip:
1. Departure date: 2025-03-15. Arrival date: 2025-03-22. How many days is the trip?
2. Flight is 6,500 km. Convert to miles.
3. Budget calculation: Flights $800 + Hotel ($150 * 7 nights) + Food ($75 * 7 days) + Activities $500
4. Convert total budget to Euros (multiply by 0.92)
5. Create tasks: 'Book flights' (high), 'Reserve hotel' (high), 'Plan activities' (medium), 'Pack luggage' (low)
6. List all travel tasks
7. Add 14 days to return date for post-trip report deadline""",
        "expected_tools": ["calculate_date_difference", "unit_converter", "calculator", "create_task", "list_tasks", "add_days_to_date"],
        "description": "Complete travel planning with budget and tasks"
    },
    
    # Content Creation Workflow
    {
        "name": "Content Production Pipeline",
        "agent": "general_assistant",
        "query": """Set up my content creation workflow:
1. Get today's date as the start
2. Content calendar: First post in 7 days, second in 14 days, third in 21 days - calculate all three dates
3. Create tasks: 'Research topics' (high), 'Write draft' (high), 'Create graphics' (medium), 'Schedule posts' (medium), 'Engage with comments' (low)
4. Analyze this content brief: 'Target audience is tech professionals aged 25-45. Focus on practical tutorials and industry insights. Maintain professional but approachable tone. Include code examples where relevant.'
5. Transform the brief to uppercase for emphasis
6. List all tasks
7. Search for information about Python to include in first post""",
        "expected_tools": ["get_current_datetime", "add_days_to_date", "create_task", "text_analyzer", "text_transformer", "list_tasks", "knowledge_base_search"],
        "description": "Full content workflow setup"
    },
    
    # Scientific Calculations
    {
        "name": "Physics Problem Solver",
        "agent": "math_specialist",
        "query": """Solve these physics problems:
1. Calculate kinetic energy: KE = 0.5 * mass * velocity^2, where mass=10kg and velocity=15m/s
2. Convert the result from Joules to calories (1 Joule = 0.239006 calories): multiply by 0.239006
3. Calculate gravitational potential energy: PE = mass * g * height, where mass=10kg, g=9.8, height=20m
4. Calculate total mechanical energy (KE + PE)
5. If an object falls from 20m, calculate final velocity using v = sqrt(2 * g * h)
6. Convert the final velocity from m/s to km/h (multiply by 3.6)""",
        "expected_tools": ["calculator"],
        "description": "Multi-step physics calculations"
    },
]

# =============================================================================
# EDGE CASE EXAMPLES (Testing robustness)
# =============================================================================

EDGE_CASES = [
    {
        "name": "Empty Input Handling",
        "agent": "general_assistant",
        "query": "Analyze this text: ''",
        "expected_tools": ["text_analyzer"],
        "description": "Handle empty input"
    },
    {
        "name": "Invalid JSON",
        "agent": "data_analyst",
        "query": "Parse this JSON: '{invalid json here}'",
        "expected_tools": ["json_parser"],
        "description": "Handle malformed JSON"
    },
    {
        "name": "Unknown Unit",
        "agent": "math_specialist",
        "query": "Convert 100 frobnicators to widgets",
        "expected_tools": ["unit_converter"],
        "description": "Handle unknown units"
    },
    {
        "name": "Division by Zero",
        "agent": "math_specialist",
        "query": "Calculate 100 / 0",
        "expected_tools": ["calculator"],
        "description": "Handle division by zero"
    },
    {
        "name": "Invalid Date Format",
        "agent": "general_assistant",
        "query": "Calculate days between 'tomorrow' and 'next week'",
        "expected_tools": ["calculate_date_difference"],
        "description": "Handle invalid date formats"
    },
    {
        "name": "Very Large Numbers",
        "agent": "math_specialist",
        "query": "Calculate 999999999999 * 888888888888",
        "expected_tools": ["calculator"],
        "description": "Handle large number calculations"
    },
    {
        "name": "Complex Nested JSON",
        "agent": "data_analyst",
        "query": "Parse this and extract 'data.users.0.name': '{\"data\": {\"users\": [{\"name\": \"Alice\", \"role\": \"admin\"}, {\"name\": \"Bob\", \"role\": \"user\"}]}}'",
        "expected_tools": ["json_parser"],
        "description": "Handle deeply nested JSON"
    },
]

# =============================================================================
# CONVERSATIONAL EXAMPLES (Natural language queries)
# =============================================================================

CONVERSATIONAL_EXAMPLES = [
    {
        "name": "Casual Math",
        "agent": "general_assistant",
        "query": "Hey, I need to split a $247.50 restaurant bill between 5 friends. How much does each person owe including a 20% tip?",
        "expected_tools": ["calculator"],
        "description": "Natural language math problem"
    },
    {
        "name": "Friendly Task Request",
        "agent": "task_manager",
        "query": "Can you help me remember to buy groceries? It's pretty important, I keep forgetting!",
        "expected_tools": ["create_task"],
        "description": "Casual task creation"
    },
    {
        "name": "Curious Question",
        "agent": "researcher",
        "query": "I've been hearing a lot about AI agents lately. What exactly are they and how do they work?",
        "expected_tools": ["knowledge_base_search"],
        "description": "Natural knowledge query"
    },
    {
        "name": "Planning Help",
        "agent": "general_assistant",
        "query": "I'm trying to figure out when to schedule my vacation. If I leave on March 1st 2025 and want to be back before my meeting on March 20th 2025, how long of a trip can I take?",
        "expected_tools": ["calculate_date_difference"],
        "description": "Natural date planning"
    },
    {
        "name": "Quick Conversion",
        "agent": "math_specialist",
        "query": "The recipe says 350°F but my oven only shows Celsius. What temperature should I set it to?",
        "expected_tools": ["unit_converter"],
        "description": "Practical conversion request"
    },
]


# =============================================================================
# SQL / TEXT-TO-SQL EXAMPLES
# =============================================================================

SQL_SIMPLE_EXAMPLES = [
    {
        "name": "List Databases",
        "agent": "sql_specialist",
        "query": "What databases are available?",
        "expected_tools": ["list_databases"],
        "description": "List all available database schemas"
    },
    {
        "name": "Get Schema",
        "agent": "sql_specialist",
        "query": "Show me the schema for the ecommerce database",
        "expected_tools": ["get_schema"],
        "description": "View database schema"
    },
    {
        "name": "Table Info",
        "agent": "sql_specialist",
        "query": "What columns are in the customers table in the ecommerce database?",
        "expected_tools": ["get_table_info"],
        "description": "Get detailed table information"
    },
    {
        "name": "SQL Examples",
        "agent": "sql_specialist",
        "query": "Show me some examples of JOIN queries",
        "expected_tools": ["sql_examples"],
        "description": "Get example SQL queries"
    },
    {
        "name": "Explain SQL",
        "agent": "sql_specialist",
        "query": "Explain this SQL: SELECT c.first_name, COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.first_name",
        "expected_tools": ["explain_sql"],
        "description": "Explain a SQL query in plain English"
    },
]

SQL_MEDIUM_EXAMPLES = [
    {
        "name": "Simple Text-to-SQL",
        "agent": "sql_specialist",
        "query": "Write a SQL query to get all customers from the ecommerce database who are from the USA",
        "expected_tools": ["get_schema", "generate_sql"],
        "description": "Basic natural language to SQL"
    },
    {
        "name": "Count Query",
        "agent": "sql_specialist",
        "query": "How many orders are in pending status? Use the ecommerce database.",
        "expected_tools": ["get_schema", "generate_sql", "execute_sql"],
        "description": "Count with filter"
    },
    {
        "name": "Join Query",
        "agent": "sql_specialist",
        "query": "I need a query that shows customer names along with their order totals from the ecommerce database",
        "expected_tools": ["get_schema", "generate_sql"],
        "description": "Query requiring JOIN"
    },
    {
        "name": "HR Database Query",
        "agent": "sql_specialist",
        "query": "Show me all employees in the Engineering department with their salaries, using the hr database",
        "expected_tools": ["get_schema", "generate_sql"],
        "description": "Query on HR database"
    },
    {
        "name": "Aggregation Query",
        "agent": "sql_specialist",
        "query": "What is the average order value by customer country in the ecommerce database?",
        "expected_tools": ["get_schema", "generate_sql"],
        "description": "Aggregation with grouping"
    },
    {
        "name": "Validate and Execute",
        "agent": "sql_specialist",
        "query": "Validate this SQL and then execute it: SELECT product_name, price FROM products WHERE price > 100 ORDER BY price DESC",
        "expected_tools": ["validate_sql", "execute_sql"],
        "description": "Validate and run a query"
    },
]

SQL_COMPLEX_EXAMPLES = [
    {
        "name": "Multi-Table Analysis",
        "agent": "sql_analyst",
        "query": """Using the ecommerce database:
1. First show me the database schema
2. Write a query to find the top 5 customers by total order value
3. The query should show customer name, email, number of orders, and total spent
4. Explain what the query does
5. Execute it to see sample results""",
        "expected_tools": ["get_schema", "generate_sql", "explain_sql", "execute_sql"],
        "description": "Complete text-to-SQL workflow with analysis"
    },
    {
        "name": "HR Analytics",
        "agent": "sql_analyst",
        "query": """I need to analyze the HR database:
1. Show me the schema first
2. Write a query to find the average salary by department
3. Also write a query to find employees who earn more than their department's average
4. Explain both queries""",
        "expected_tools": ["get_schema", "generate_sql", "explain_sql"],
        "description": "HR analytics with subqueries"
    },
    {
        "name": "Sales Report Query",
        "agent": "sql_analyst",
        "query": """Create a comprehensive sales report query for the ecommerce database that shows:
- Monthly revenue totals
- Number of orders per month
- Average order value per month
- Best selling product category per month
Group by year and month, order by date descending""",
        "expected_tools": ["get_schema", "generate_sql", "validate_sql", "explain_sql"],
        "description": "Complex reporting query"
    },
    {
        "name": "Customer Segmentation",
        "agent": "sql_analyst",
        "query": """Using the ecommerce database, help me segment customers:
1. Find customers who have made more than 3 orders (loyal customers)
2. Find customers who haven't ordered in the last 6 months (at-risk)
3. Find customers with order total > $1000 (high value)
Create SQL for each segment and explain the logic""",
        "expected_tools": ["get_schema", "generate_sql", "explain_sql"],
        "description": "Customer segmentation queries"
    },
    {
        "name": "Web Analytics Query",
        "agent": "sql_analyst",
        "query": """Explore the analytics database and write queries to:
1. Find the top 10 most visited pages
2. Calculate the bounce rate (sessions with only 1 page view)
3. Find the average session duration by device type
4. Identify the most common user journey (page sequence)""",
        "expected_tools": ["get_schema", "generate_sql", "explain_sql", "execute_sql"],
        "description": "Web analytics queries"
    },
    {
        "name": "Cross-Database Understanding",
        "agent": "sql_analyst",
        "query": """I'm new to these databases. Please:
1. List all available databases
2. Show me the schema for each one
3. Give me one useful example query for each database
4. Explain what business questions each database can answer""",
        "expected_tools": ["list_databases", "get_schema", "sql_examples"],
        "description": "Complete database exploration"
    },
    {
        "name": "Performance Query",
        "agent": "sql_analyst",
        "query": """Using the hr database, create a performance analysis:
1. Find employees and their project assignments
2. Calculate total hours allocated per employee across all projects
3. Find employees assigned to more than 2 projects
4. Identify departments with the most active projects
Write optimized SQL for each and explain your approach""",
        "expected_tools": ["get_schema", "generate_sql", "validate_sql", "explain_sql"],
        "description": "HR performance analysis"
    },
    {
        "name": "Natural Language Complex",
        "agent": "sql_analyst",
        "query": """I'm a business analyst and I need to answer this question:
'Which product categories are underperforming in terms of order volume but have high inventory?'
Use the ecommerce database. Walk me through your approach, show the SQL, and explain the results.""",
        "expected_tools": ["get_schema", "generate_sql", "explain_sql", "execute_sql", "calculator"],
        "description": "Business question to SQL with analysis"
    },
]


# =============================================================================
# NAME MATCHING EXAMPLES
# =============================================================================

NAME_MATCHING_SIMPLE_EXAMPLES = [
    {
        "name": "Load Names",
        "agent": "name_matcher",
        "query": """Load these company names for matching:
["Emirates NBD", "Emirates NBD PJSC", "Emirates NBD Group", "Emirates NBD Bank", 
"DEWA", "DEWA Authority", "Dubai Electricity and Water", "Dubai Electricity and Water Authority",
"Etisalat", "Emirates Telecommunications", "Etisalat UAE", "Emirates Telecom Group"]""",
        "expected_tools": ["load_names_for_matching"],
        "description": "Load a small list of names for matching"
    },
    {
        "name": "Session Info",
        "agent": "name_matcher",
        "query": "Show me information about the current name matching session",
        "expected_tools": ["get_session_info"],
        "description": "View session statistics"
    },
    {
        "name": "Analyze Name",
        "agent": "name_matcher",
        "query": "Analyze how the name 'Emirates NBD PJSC' will be processed for matching",
        "expected_tools": ["analyze_name"],
        "description": "Understand name processing"
    },
    {
        "name": "Simple Match",
        "agent": "name_matcher",
        "query": "Find all names that match 'Emirates NBD Bank' from the loaded list",
        "expected_tools": ["find_matching_names"],
        "description": "Find matches for a single name"
    },
]

NAME_MATCHING_MEDIUM_EXAMPLES = [
    {
        "name": "Load and Match",
        "agent": "name_matcher",
        "query": """First load these names:
["Abu Dhabi Commercial Bank", "ADCB", "ADCB Bank", "Abu Dhabi Commercial Bank PJSC",
"First Abu Dhabi Bank", "FAB", "FAB Bank", "First Abu Dhabi Bank PJSC",
"Dubai Islamic Bank", "DIB", "DIB Bank", "Dubai Islamic Bank PJSC"]

Then find all matches for "Abu Dhabi Commercial Bank" """,
        "expected_tools": ["load_names_for_matching", "find_matching_names"],
        "description": "Load names and find matches"
    },
    {
        "name": "Batch Matching",
        "agent": "name_matcher",
        "query": """Load these names:
["Emaar Properties", "Emaar", "Emaar Properties PJSC", "Emaar Development",
"DAMAC Properties", "DAMAC", "DAMAC Holding", "DAMAC Real Estate",
"Nakheel", "Nakheel Properties", "Nakheel PJSC", "Nakheel Development"]

Then batch match these canonical names: ["Emaar Properties", "DAMAC Properties", "Nakheel"]""",
        "expected_tools": ["load_names_for_matching", "batch_match_names"],
        "description": "Match multiple canonical names at once"
    },
    {
        "name": "Create Mapping",
        "agent": "name_matcher",
        "query": """Load these names:
["DP World", "DP World Limited", "DP World PJSC", "Dubai Ports World",
"Jebel Ali Free Zone", "JAFZA", "Jafza", "JAFZA Authority"]

Create a canonical mapping for "DP World" showing all its variations""",
        "expected_tools": ["load_names_for_matching", "create_canonical_mapping"],
        "description": "Create a structured mapping"
    },
    {
        "name": "Threshold Adjustment",
        "agent": "name_matcher",
        "query": """Load these names:
["Al Futtaim Group", "Al-Futtaim", "Alfuttaim", "Al Futtaim Holdings",
"Majid Al Futtaim", "MAF", "Majid Al Futtaim Holding", "MAF Holdings"]

Find matches for "Al Futtaim" with a low threshold of 0.5 to catch more variations""",
        "expected_tools": ["load_names_for_matching", "find_matching_names"],
        "description": "Adjust matching threshold"
    },
]

NAME_MATCHING_COMPLEX_EXAMPLES = [
    {
        "name": "Large Batch Processing",
        "agent": "name_matcher",
        "query": """I have a large list of UAE company names. Load them in batches:

Batch 1:
["Emirates NBD", "Emirates NBD PJSC", "Emirates NBD Bank", "ENBD",
"First Abu Dhabi Bank", "FAB", "FAB Bank", "First AD Bank",
"Abu Dhabi Commercial Bank", "ADCB", "ADCB PJSC", "AD Commercial Bank",
"Dubai Islamic Bank", "DIB", "DIB PJSC", "DI Bank",
"Mashreq Bank", "Mashreq", "Mashreq PJSC", "Mashreqbank"]

Batch 2 (append to same session):
["DEWA", "Dubai Electricity", "Dubai Electricity and Water", "Dubai Electricity and Water Authority",
"Etisalat", "Emirates Telecommunications", "Etisalat UAE", "E& UAE",
"Du", "Emirates Integrated Telecommunications", "EITC", "Du Telecom",
"ADNOC", "Abu Dhabi National Oil", "Abu Dhabi National Oil Company", "ADNOC Group"]

Then:
1. Show session info
2. Create bulk mappings for: ["Emirates NBD", "First Abu Dhabi Bank", "DEWA", "Etisalat", "ADNOC"]
3. Analyze why "E& UAE" might not match "Etisalat" """,
        "expected_tools": ["load_names_for_matching", "get_session_info", "bulk_create_mappings", "analyze_name"],
        "description": "Process large lists in batches"
    },
    {
        "name": "Full Standardization Workflow",
        "agent": "name_data_processor",
        "query": """I need to standardize company names. Here's my data:

Names list:
["Emaar Properties", "EMAAR", "Emaar Properties PJSC", "Emaar Development LLC",
"Dubai Holding", "Dubai Holding LLC", "DH Group", "Dubai Holding Group",
"Meraas", "Meraas Holding", "Meraas Development", "Meraas PJSC",
"Aldar Properties", "Aldar", "ALDAR PJSC", "Aldar Development",
"Mubadala", "Mubadala Investment", "Mubadala Investment Company", "Mubadala PJSC"]

Canonical names I want to use:
["Emaar Properties", "Dubai Holding", "Meraas", "Aldar Properties", "Mubadala"]

Please:
1. Load all the names
2. Create mappings for each canonical name
3. Output a summary showing which variations map to which canonical name
4. Identify any names that don't match any canonical (orphans)""",
        "expected_tools": ["load_names_for_matching", "bulk_create_mappings", "get_session_info"],
        "description": "Complete standardization workflow"
    },
    {
        "name": "Analyze Matching Quality",
        "agent": "name_matcher",
        "query": """Load these potentially tricky names:
["Bank ABC", "ABC Bank", "Arab Banking Corporation", "Arab Banking Corp",
"Commercial Bank of Dubai", "CBD", "CB Dubai", "Commercial Bank Dubai",
"National Bank of Fujairah", "NBF", "NB Fujairah", "Fujairah National Bank",
"Sharjah Islamic Bank", "SIB", "SI Bank", "Islamic Bank of Sharjah"]

Then:
1. Analyze how "Bank ABC" and "Arab Banking Corporation" are processed
2. Find matches for "Bank ABC" - explain why some might not match
3. Find matches for "Commercial Bank of Dubai" with threshold 0.55
4. Create a mapping for "National Bank of Fujairah"
5. Explain what makes these names challenging to match""",
        "expected_tools": ["load_names_for_matching", "analyze_name", "find_matching_names", "create_canonical_mapping"],
        "description": "Analyze matching quality and edge cases"
    },
    {
        "name": "Government Entity Matching",
        "agent": "name_matcher",
        "query": """Load UAE government entities:
["Dubai Municipality", "DM", "Dubai Mun", "Municipality of Dubai",
"Roads and Transport Authority", "RTA", "RTA Dubai", "Dubai RTA",
"Dubai Health Authority", "DHA", "DH Authority", "Health Authority Dubai",
"Department of Economic Development", "DED", "Dubai DED", "Economic Development Dept",
"Dubai Land Department", "DLD", "Land Dept Dubai", "Dubai Land Dept",
"Knowledge and Human Development Authority", "KHDA", "KHD Authority"]

Find matches for each of these canonical names:
1. Dubai Municipality
2. Roads and Transport Authority  
3. Dubai Health Authority
4. Department of Economic Development
5. Dubai Land Department
6. Knowledge and Human Development Authority

Use a threshold of 0.60 to catch abbreviations like RTA, DHA, DED""",
        "expected_tools": ["load_names_for_matching", "batch_match_names"],
        "description": "Match government entities with abbreviations"
    },
]


# =============================================================================
# WEALTH PORTFOLIO / DUCKDB DATA AGENT EXAMPLES
# =============================================================================

WEALTH_SIMPLE_EXAMPLES = [
    {
        "name": "List All Clients",
        "agent": "data_agent",
        "query": "Show me all clients in the database",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Simple query to list all clients"
    },
    {
        "name": "Active Portfolios",
        "agent": "data_agent",
        "query": "Show me all active portfolios",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Filter portfolios by active status"
    },
    {
        "name": "Client Risk Profiles",
        "agent": "data_agent",
        "query": "What are the different risk profiles of our clients?",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Query distinct risk profiles"
    },
    {
        "name": "Get Sample Assets",
        "agent": "data_agent",
        "query": "Show me sample data from the assets table",
        "expected_tools": ["get_sample_data"],
        "description": "View sample asset records"
    },
    {
        "name": "Describe Transactions Table",
        "agent": "data_agent",
        "query": "What is the structure of the transactions table?",
        "expected_tools": ["describe_table"],
        "description": "Get table schema information"
    },
]

WEALTH_MEDIUM_EXAMPLES = [
    {
        "name": "Clients with Portfolios",
        "agent": "data_agent",
        "query": "Show me all clients along with their portfolio names and currencies",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "JOIN clients and portfolios tables"
    },
    {
        "name": "Portfolio Holdings Summary",
        "agent": "data_agent",
        "query": "For each portfolio, show me the total number of different assets they hold",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Aggregate holdings by portfolio with COUNT"
    },
    {
        "name": "Recent Transactions",
        "agent": "data_agent",
        "query": "Show me the 10 most recent transactions with client names, portfolio names, and asset symbols",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Multi-table JOIN with ORDER BY and LIMIT"
    },
    {
        "name": "Assets by Type",
        "agent": "data_agent",
        "query": "How many assets do we have of each type (Equity, ETF, Crypto)?",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "GROUP BY with COUNT aggregation"
    },
    {
        "name": "Client Transaction Activity",
        "agent": "data_agent",
        "query": "Which clients have made transactions, and how many transactions has each client made?",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "JOIN with aggregation and GROUP BY"
    },
    {
        "name": "Holdings with Asset Details",
        "agent": "data_agent",
        "query": "Show current holdings with asset symbols, names, quantities, and average cost",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "JOIN holdings with assets table"
    },
]

WEALTH_COMPLEX_EXAMPLES = [
    {
        "name": "Portfolio Valuation Analysis",
        "agent": "data_agent",
        "query": """Calculate the total value of each portfolio based on current holdings.
Show client name, portfolio name, base currency, and total portfolio value (quantity * average cost).
Order by total value descending.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Complex query with CTE, multiple JOINs, and aggregation"
    },
    {
        "name": "Client Portfolio Diversification",
        "agent": "data_agent",
        "query": """For each client, show:
1. Number of portfolios they own
2. Total number of unique assets across all their portfolios
3. Number of different asset types (Equity, ETF, Crypto) they hold
Group by client and order by number of assets descending.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Multi-level aggregation with multiple JOINs and CTEs"
    },
    {
        "name": "Trading Activity Report",
        "agent": "data_agent",
        "query": """Generate a trading activity report showing:
- Client name
- Total number of buy transactions
- Total number of sell transactions
- Total transaction count
- Most frequently traded asset symbol
Only include clients with more than 5 transactions. Order by total transactions descending.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Complex aggregation with conditional counts and filtering"
    },
    {
        "name": "Top Holdings by Portfolio",
        "agent": "data_agent",
        "query": """For each portfolio, identify the top 3 holdings by total value (quantity * avg_cost).
Show portfolio name, asset symbol, asset name, quantity, average cost, and total value.
Use a window function or CTE to rank holdings within each portfolio.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Window functions or CTEs with ranking"
    },
    {
        "name": "Risk Profile Distribution Analysis",
        "agent": "data_agent",
        "query": """Analyze the distribution of clients by risk profile and show:
1. Risk profile category
2. Number of clients in each category
3. Number of portfolios managed by clients in each category
4. Average portfolio value per risk profile
Include KYC approved clients only. Order by risk profile.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Multi-table JOIN with multiple aggregations and filtering"
    },
    {
        "name": "Asset Allocation by Client",
        "agent": "data_agent",
        "query": """Create a comprehensive asset allocation report showing:
For client 'Anuj Singhal':
- Asset type breakdown (percentage of total portfolio value by asset type)
- Top 5 holdings by value
- Total portfolio value
- Number of different assets held

Use CTEs to organize the calculation logic.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Complex CTE-based query with percentage calculations"
    },
    {
        "name": "Transaction Volume Trends",
        "agent": "data_agent",
        "query": """Analyze transaction patterns:
1. Total transaction volume by month (count of transactions)
2. Total value traded by month (sum of quantity * price)
3. Buy vs Sell ratio for each month
Show results for the most recent 6 months of data.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Time-based analysis with date functions and aggregations"
    },
    {
        "name": "Comprehensive Client Summary",
        "agent": "data_agent",
        "query": """Create a comprehensive summary for each client showing:
- Client details (name, country, risk profile, KYC status)
- Number of portfolios owned
- Total assets across all portfolios
- Total portfolio value (sum of all holdings value)
- Most recent transaction date
- Most frequently held asset type

Order by total portfolio value descending. Use multiple CTEs to organize the logic.""",
        "expected_tools": ["get_database_schema", "run_sql_query"],
        "description": "Master query with multiple CTEs combining all aspects"
    },
]


# =============================================================================
# MULTI-DATA-AGENT EXAMPLES (Advanced Multi-Agent Workflow)
# =============================================================================

MULTI_DATA_AGENT_SIMPLE_EXAMPLES = [
    {
        "name": "Multi-Agent Simple Query",
        "agent": "multi_data_agent",
        "query": "Show me all clients in the database",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Simple query through multi-agent workflow"
    },
    {
        "name": "Multi-Agent General Question",
        "agent": "multi_data_agent",
        "query": "What is SQL?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "General question - should be answered by LLM directly"
    },
    {
        "name": "Get Schema Document",
        "agent": "multi_data_agent",
        "query": "Show me the database schema document",
        "expected_tools": ["get_schema_document"],
        "description": "Get the complete schema definition"
    },
    {
        "name": "Get Table Descriptions",
        "agent": "multi_data_agent",
        "query": "What tables are available and what do they contain?",
        "expected_tools": ["get_table_descriptions"],
        "description": "Get summary of all tables"
    },
]

MULTI_DATA_AGENT_MEDIUM_EXAMPLES = [
    {
        "name": "Multi-Agent Join Query",
        "agent": "multi_data_agent",
        "query": "Show me all clients along with their portfolio names",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Medium complexity query through multi-agent workflow"
    },
    {
        "name": "Analyze Query Requirements",
        "agent": "multi_data_agent",
        "query": "What tables and complexity would be needed to answer: 'Show top 5 clients by portfolio value'",
        "expected_tools": ["analyze_query_requirements"],
        "description": "Analyze query requirements before execution"
    },
    {
        "name": "Extract Related Tables",
        "agent": "multi_data_agent",
        "query": "Get the detailed schema for CLIENTS and PORTFOLIOS tables",
        "expected_tools": ["extract_related_tables"],
        "description": "Extract schema for multiple related tables"
    },
    {
        "name": "Get Join Syntax Help",
        "agent": "multi_data_agent",
        "query": "How do I join the CLIENTS and PORTFOLIOS tables?",
        "expected_tools": ["get_join_syntax_help"],
        "description": "Get correct JOIN syntax for two tables"
    },
]

MULTI_DATA_AGENT_COMPLEX_EXAMPLES = [
    {
        "name": "Complex Portfolio Analysis",
        "agent": "multi_data_agent",
        "query": """Calculate the total portfolio value for each client.
Show client name, number of portfolios, total holdings value (quantity * avg_cost),
and rank clients by total value. Only include clients with approved KYC status.""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Complex query with CTEs and multiple aggregations"
    },
    {
        "name": "Trading Pattern Analysis",
        "agent": "multi_data_agent",
        "query": """Analyze trading patterns:
1. Show total buy vs sell transactions per client
2. Calculate average transaction value per client
3. Identify clients with more sell than buy transactions
4. Order by total transaction count""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Multi-part analysis query"
    },
    {
        "name": "Asset Diversification Report",
        "agent": "multi_data_agent",
        "query": """Create a diversification report showing:
- Number of unique asset types per client
- Percentage of portfolio in each asset type (Equity, ETF, Crypto)
- Top 3 most held assets per client
Use CTEs to organize the calculation logic.""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Complex diversification analysis"
    },
    {
        "name": "Full Workflow Demonstration",
        "agent": "multi_data_agent",
        "query": """I need a comprehensive analysis:
1. First show me what tables are available
2. Then show top 5 clients by total transaction volume
3. Include their risk profiles and KYC status
4. Show their most frequently traded asset

This should trigger the full 6-agent workflow with validation.""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Demonstrates full multi-agent workflow"
    },
    {
        "name": "Validation Retry Test",
        "agent": "multi_data_agent",
        "query": """Write a query to calculate the monthly transaction growth rate for each portfolio,
comparing each month to the previous month. Use window functions for the calculation.
Show portfolio name, month, transaction count, and growth percentage.""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Complex query that may trigger validation retries"
    },
]


# =============================================================================
# INTENT CLASSIFICATION EXAMPLES (Conversation Continuity)
# =============================================================================
# These examples demonstrate the intent classification feature:
# - NEW_DATA_QUERY: Fresh query requiring SQL generation
# - MODIFIED_QUERY: Modify/filter previous query (needs new SQL with context)
# - FOLLOWUP_QUESTION: Question about previous results (answer from cache, no SQL)
# - GENERAL_QUESTION: Non-data question
#
# Run examples in sequence within the same conversation_id to test caching.

# =============================================================================
# MODIFIED QUERY EXAMPLES (Need new SQL based on previous context)
# =============================================================================

MODIFIED_QUERY_SIMPLE_EXAMPLES = [
    # Conversation 1: Simple client query with modifications
    {
        "name": "Modified Simple - Initial Query",
        "agent": "multi_data_agent",
        "query": "Show me all clients in the database",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv1",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Modified Simple - Filter by Country",
        "agent": "multi_data_agent",
        "query": "Show me only the ones from USA",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - filter previous results by country (needs new SQL)",
        "conversation_id": "conv1",
        "sequence": 2,
        "intent_type": "MODIFIED_QUERY"
    },
    {
        "name": "Modified Simple - Sort Results",
        "agent": "multi_data_agent",
        "query": "Sort them by name alphabetically",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - sort previous results (needs new SQL)",
        "conversation_id": "conv1",
        "sequence": 3,
        "intent_type": "MODIFIED_QUERY"
    },
    # Conversation 2: Portfolio modifications
    {
        "name": "Modified Simple - Portfolio Query",
        "agent": "multi_data_agent",
        "query": "Show me all portfolios",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv2",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Modified Simple - Add Columns",
        "agent": "multi_data_agent",
        "query": "Add the client name to those results",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - expand with additional columns (needs new SQL with JOIN)",
        "conversation_id": "conv2",
        "sequence": 2,
        "intent_type": "MODIFIED_QUERY"
    },
]

MODIFIED_QUERY_MEDIUM_EXAMPLES = [
    # Conversation 3: Client-Portfolio analysis with modifications
    {
        "name": "Modified Medium - Initial Join Query",
        "agent": "multi_data_agent",
        "query": "Show me all clients along with their portfolio names and currencies",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv3",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Modified Medium - Aggregate Previous",
        "agent": "multi_data_agent",
        "query": "Now group by client and count their portfolios",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - aggregate from previous context (needs new SQL)",
        "conversation_id": "conv3",
        "sequence": 2,
        "intent_type": "MODIFIED_QUERY"
    },
    {
        "name": "Modified Medium - Filter Aggregation",
        "agent": "multi_data_agent",
        "query": "Show only clients with more than one portfolio",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - filter the aggregated results (needs new SQL with HAVING)",
        "conversation_id": "conv3",
        "sequence": 3,
        "intent_type": "MODIFIED_QUERY"
    },
    # Conversation 4: Transaction modifications
    {
        "name": "Modified Medium - Transaction Query",
        "agent": "multi_data_agent",
        "query": "Show me the top 10 transactions by value (quantity * price)",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv4",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Modified Medium - Add Client Info",
        "agent": "multi_data_agent",
        "query": "For those transactions, add the client names who made them",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - expand with client details (needs new SQL with JOINs)",
        "conversation_id": "conv4",
        "sequence": 2,
        "intent_type": "MODIFIED_QUERY"
    },
    {
        "name": "Modified Medium - Group by Type",
        "agent": "multi_data_agent",
        "query": "Break this down by transaction type (BUY vs SELL)",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - group previous results (needs new SQL with GROUP BY)",
        "conversation_id": "conv4",
        "sequence": 3,
        "intent_type": "MODIFIED_QUERY"
    },
]

MODIFIED_QUERY_COMPLEX_EXAMPLES = [
    # Conversation 5: Complex portfolio modifications
    {
        "name": "Modified Complex - Portfolio Valuation",
        "agent": "multi_data_agent",
        "query": """Calculate the total portfolio value for each client based on their holdings.
Show client name, number of portfolios, and total holdings value (quantity * avg_cost).""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial complex query - NEW_DATA_QUERY intent",
        "conversation_id": "conv5",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Modified Complex - Rank and Limit",
        "agent": "multi_data_agent",
        "query": "Rank them by total value descending and show only top 5",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - rank and limit (needs new SQL with ORDER BY/LIMIT)",
        "conversation_id": "conv5",
        "sequence": 2,
        "intent_type": "MODIFIED_QUERY"
    },
    {
        "name": "Modified Complex - Drill Down",
        "agent": "multi_data_agent",
        "query": "For the top client, show their individual asset holdings with values",
        "expected_tools": ["run_multi_agent_query"],
        "description": "MODIFIED_QUERY - drill down (needs new SQL for specific client)",
        "conversation_id": "conv5",
        "sequence": 3,
        "intent_type": "MODIFIED_QUERY"
    },
]

# =============================================================================
# FOLLOWUP QUESTION EXAMPLES (Answer from cache, no new SQL)
# =============================================================================

FOLLOWUP_QUESTION_SIMPLE_EXAMPLES = [
    # Conversation 6: Questions about client results
    {
        "name": "Followup Question - Initial Query",
        "agent": "multi_data_agent",
        "query": "Show me all clients in the database",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv6",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Followup Question - Count",
        "agent": "multi_data_agent",
        "query": "How many clients are there in total?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - count from cached results (no SQL needed)",
        "conversation_id": "conv6",
        "sequence": 2,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    {
        "name": "Followup Question - Which One",
        "agent": "multi_data_agent",
        "query": "Which one is from UAE?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - identify from cached results (no SQL needed)",
        "conversation_id": "conv6",
        "sequence": 3,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    # Conversation 7: Questions about portfolio results
    {
        "name": "Followup Question - Portfolio Initial",
        "agent": "multi_data_agent",
        "query": "Show me all portfolios with their values",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv7",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Followup Question - Highest Value",
        "agent": "multi_data_agent",
        "query": "Which one has the highest value?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - find max from cached results (no SQL needed)",
        "conversation_id": "conv7",
        "sequence": 2,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    {
        "name": "Followup Question - Summarize",
        "agent": "multi_data_agent",
        "query": "Summarize the results",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - summarize cached results (no SQL needed)",
        "conversation_id": "conv7",
        "sequence": 3,
        "intent_type": "FOLLOWUP_QUESTION"
    },
]

FOLLOWUP_QUESTION_MEDIUM_EXAMPLES = [
    # Conversation 8: Questions about transaction analysis
    {
        "name": "Followup Question Medium - Transaction Initial",
        "agent": "multi_data_agent",
        "query": "Show me the last 20 transactions with their values",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial query - NEW_DATA_QUERY intent",
        "conversation_id": "conv8",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Followup Question Medium - Total Value",
        "agent": "multi_data_agent",
        "query": "What is the total value of all these transactions?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - sum from cached results (no SQL needed)",
        "conversation_id": "conv8",
        "sequence": 2,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    {
        "name": "Followup Question Medium - Average",
        "agent": "multi_data_agent",
        "query": "What's the average transaction value?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - average from cached results (no SQL needed)",
        "conversation_id": "conv8",
        "sequence": 3,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    {
        "name": "Followup Question Medium - Explain",
        "agent": "multi_data_agent",
        "query": "Explain what these results show",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - explain cached results (no SQL needed)",
        "conversation_id": "conv8",
        "sequence": 4,
        "intent_type": "FOLLOWUP_QUESTION"
    },
]

FOLLOWUP_QUESTION_COMPLEX_EXAMPLES = [
    # Conversation 9: Complex analysis questions
    {
        "name": "Followup Question Complex - Risk Analysis Initial",
        "agent": "multi_data_agent",
        "query": """Show me clients grouped by risk profile with:
- Number of clients per profile
- Total portfolio value per profile""",
        "expected_tools": ["run_multi_agent_query"],
        "description": "Initial complex query - NEW_DATA_QUERY intent",
        "conversation_id": "conv9",
        "sequence": 1,
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "Followup Question Complex - Compare Profiles",
        "agent": "multi_data_agent",
        "query": "Which risk profile has the most clients?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - compare from cached results (no SQL needed)",
        "conversation_id": "conv9",
        "sequence": 2,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    {
        "name": "Followup Question Complex - Insight",
        "agent": "multi_data_agent",
        "query": "What insights can you derive from this data?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - analyze cached results (no SQL needed)",
        "conversation_id": "conv9",
        "sequence": 3,
        "intent_type": "FOLLOWUP_QUESTION"
    },
    {
        "name": "Followup Question Complex - Pattern",
        "agent": "multi_data_agent",
        "query": "Is there a correlation between risk profile and portfolio value?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "FOLLOWUP_QUESTION - pattern analysis from cached results (no SQL needed)",
        "conversation_id": "conv9",
        "sequence": 4,
        "intent_type": "FOLLOWUP_QUESTION"
    },
]

# =============================================================================
# NEW QUERY EXAMPLES (Fresh queries that should not reference previous context)
# =============================================================================

NEW_QUERY_EXAMPLES = [
    {
        "name": "New Query - Unrelated Topic",
        "agent": "multi_data_agent",
        "query": "What assets are available in the database?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "NEW_DATA_QUERY - unrelated to previous queries",
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "New Query - Different Domain",
        "agent": "multi_data_agent",
        "query": "Show me all cryptocurrency assets",
        "expected_tools": ["run_multi_agent_query"],
        "description": "NEW_DATA_QUERY - different data domain",
        "intent_type": "NEW_DATA_QUERY"
    },
    {
        "name": "New Query - General Question",
        "agent": "multi_data_agent",
        "query": "What is the difference between ETF and Equity?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "GENERAL_QUESTION - answered directly without SQL",
        "intent_type": "GENERAL_QUESTION"
    },
    {
        "name": "New Query - SQL Syntax Question",
        "agent": "multi_data_agent",
        "query": "How do I use GROUP BY with HAVING in SQL?",
        "expected_tools": ["run_multi_agent_query"],
        "description": "GENERAL_QUESTION - SQL concept explanation",
        "intent_type": "GENERAL_QUESTION"
    },
]

# Backward compatibility aliases
FOLLOWUP_SIMPLE_EXAMPLES = MODIFIED_QUERY_SIMPLE_EXAMPLES + FOLLOWUP_QUESTION_SIMPLE_EXAMPLES
FOLLOWUP_MEDIUM_EXAMPLES = MODIFIED_QUERY_MEDIUM_EXAMPLES + FOLLOWUP_QUESTION_MEDIUM_EXAMPLES
FOLLOWUP_COMPLEX_EXAMPLES = MODIFIED_QUERY_COMPLEX_EXAMPLES + FOLLOWUP_QUESTION_COMPLEX_EXAMPLES


# =============================================================================
# SYNTHETIC DATA GENERATION EXAMPLES
# =============================================================================

SYNTH_SIMPLE_EXAMPLES = [
    {
        "name": "Check Table Exists",
        "agent": "synthetic_data_agent",
        "query": "Check if the CLIENTS table exists in the database",
        "expected_tools": ["check_table_exists"],
        "description": "Verify source table exists before generation"
    },
    {
        "name": "Get Table Schema",
        "agent": "synthetic_data_agent",
        "query": "Show me the schema of the CLIENTS table for synthetic data generation",
        "expected_tools": ["get_table_schema_for_synth"],
        "description": "Get schema with SDV type mappings"
    },
    {
        "name": "Get Table Relationships",
        "agent": "synthetic_data_agent",
        "query": "What are the relationships for the PORTFOLIOS table?",
        "expected_tools": ["get_table_relationships"],
        "description": "View parent/child table dependencies"
    },
    {
        "name": "Get Sample Data",
        "agent": "synthetic_data_agent",
        "query": "Show me sample data from the ASSETS table for training",
        "expected_tools": ["get_sample_data_for_synth"],
        "description": "Get training data for SDV"
    },
    {
        "name": "List Synth Tables",
        "agent": "synthetic_data_agent",
        "query": "List all existing SYNTH_* tables in the database",
        "expected_tools": ["list_synth_tables"],
        "description": "View all generated synthetic tables"
    },
    {
        "name": "Analyze Dependencies",
        "agent": "synthetic_data_agent",
        "query": "What is the generation order for the TRANSACTIONS table?",
        "expected_tools": ["analyze_table_dependencies"],
        "description": "Get dependency chain and generation order"
    },
]

SYNTH_MEDIUM_EXAMPLES = [
    {
        "name": "Generate Simple Synthetic Clients",
        "agent": "synthetic_data_agent",
        "query": "Generate 5 synthetic clients for testing purposes",
        "expected_tools": ["check_table_exists", "get_table_schema_for_synth", "get_table_relationships", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Full workflow for generating synthetic clients (no dependencies)"
    },
    {
        "name": "Generate Synthetic Assets",
        "agent": "synthetic_data_agent",
        "query": "Create 10 synthetic asset records in the database",
        "expected_tools": ["check_table_exists", "get_table_schema_for_synth", "get_table_relationships", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Generate synthetic assets (independent table)"
    },
    {
        "name": "Check and Generate",
        "agent": "synthetic_data_agent",
        "query": """I want to generate synthetic data for CLIENTS:
1. First check if the table exists
2. Show me the schema
3. Check any relationships
4. Then generate 3 synthetic records""",
        "expected_tools": ["check_table_exists", "get_table_schema_for_synth", "get_table_relationships", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Step-by-step synthetic data generation"
    },
    {
        "name": "View Generation Summary",
        "agent": "synthetic_data_agent",
        "query": """Generate 5 synthetic clients and then show me a summary of what was generated""",
        "expected_tools": ["check_table_exists", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Generate and summarize results"
    },
    {
        "name": "Analyze Before Generate",
        "agent": "synthetic_data_agent",
        "query": """Before generating data for PORTFOLIOS, analyze the dependencies.
What tables need to exist first?""",
        "expected_tools": ["analyze_table_dependencies", "get_table_relationships"],
        "description": "Dependency analysis before generation"
    },
]

SYNTH_COMPLEX_EXAMPLES = [
    {
        "name": "Generate Portfolios with Dependencies",
        "agent": "synthetic_data_agent",
        "query": """Generate 5 synthetic portfolios.
Note: PORTFOLIOS depends on CLIENTS, so make sure SYNTH_CLIENTS exists first.
If not, generate synthetic clients first, then generate portfolios.""",
        "expected_tools": ["analyze_table_dependencies", "list_synth_tables", "check_table_exists", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Handle table dependencies during generation"
    },
    {
        "name": "Full Dependency Chain",
        "agent": "synthetic_data_agent",
        "query": """I need to generate synthetic TRANSACTIONS data.
Analyze the full dependency chain and tell me what tables need to be generated first.
Then generate the entire chain: CLIENTS -> PORTFOLIOS -> TRANSACTIONS with 5 records each.""",
        "expected_tools": ["analyze_table_dependencies", "list_synth_tables", "check_table_exists", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Multi-table generation with dependency chain"
    },
    {
        "name": "Complete Test Data Set",
        "agent": "synthetic_data_agent",
        "query": """Create a complete synthetic test dataset:
1. First analyze dependencies for all tables
2. Generate 10 synthetic CLIENTS
3. Generate 15 synthetic ASSETS
4. Generate 20 synthetic PORTFOLIOS
5. List all SYNTH_* tables to confirm""",
        "expected_tools": ["analyze_table_dependencies", "check_table_exists", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data", "list_synth_tables"],
        "description": "Generate multiple related synthetic tables"
    },
    {
        "name": "Validate and Generate Holdings",
        "agent": "synthetic_data_agent",
        "query": """Generate synthetic HOLDINGS data:
1. Check what the HOLDINGS table looks like (schema)
2. Analyze its dependencies (needs PORTFOLIOS and ASSETS)
3. Check if SYNTH_PORTFOLIOS and SYNTH_ASSETS exist
4. If they exist, generate 25 synthetic holdings
5. Show generation summary""",
        "expected_tools": ["get_table_schema_for_synth", "analyze_table_dependencies", "list_synth_tables", "check_table_exists", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Complex generation with validation"
    },
    {
        "name": "Incremental Generation",
        "agent": "synthetic_data_agent",
        "query": """I already have some SYNTH_CLIENTS. Generate 5 more synthetic clients
and add them to the existing SYNTH_CLIENTS table. Show the total count after.""",
        "expected_tools": ["list_synth_tables", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Add more records to existing synthetic table"
    },
    {
        "name": "Full Pipeline Verification",
        "agent": "synthetic_data_agent",
        "query": """Execute a full synthetic data generation pipeline:
1. List any existing SYNTH_* tables
2. Analyze the dependency order for TRANSACTIONS table
3. For each table in the dependency chain:
   a. Check if source table exists
   b. Get its schema
   c. Create SYNTH_* table if needed
   d. Generate 3 synthetic records
   e. Insert into database
4. Show final summary of all SYNTH_* tables""",
        "expected_tools": ["list_synth_tables", "analyze_table_dependencies", "check_table_exists", "get_table_schema_for_synth", "create_synth_table", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Complete pipeline with verification at each step"
    },
]


# =============================================================================
# SYNTHETIC DATA - SCHEMA FILE EXAMPLES (Creating new tables)
# =============================================================================

SYNTH_SCHEMA_SIMPLE_EXAMPLES = [
    {
        "name": "List Available Schemas",
        "agent": "synthetic_data_agent",
        "query": "What schema files are available for creating new tables?",
        "expected_tools": ["list_available_schemas"],
        "description": "List all JSON schema files in the synthetic_data directory"
    },
    {
        "name": "Load Financial Schema",
        "agent": "synthetic_data_agent",
        "query": "Load the financial_transactions.json schema file and show me what tables it contains",
        "expected_tools": ["load_schema_from_file"],
        "description": "Load a schema file and view available tables"
    },
    {
        "name": "View Table Definition",
        "agent": "synthetic_data_agent",
        "query": "Show me the detailed definition of the ACCOUNTS table from financial_transactions.json",
        "expected_tools": ["load_schema_from_file", "get_schema_table_definition"],
        "description": "View specific table definition from schema file"
    },
    {
        "name": "Check Non-Existent Table",
        "agent": "synthetic_data_agent",
        "query": "Does the ACCOUNTS table exist in the database?",
        "expected_tools": ["check_table_exists"],
        "description": "Check for a table that doesn't exist"
    },
]

SYNTH_SCHEMA_MEDIUM_EXAMPLES = [
    {
        "name": "Create Single Table from Schema",
        "agent": "synthetic_data_agent",
        "query": """Create the MERCHANTS table from the financial_transactions.json schema file.
This table has no dependencies so it can be created independently.""",
        "expected_tools": ["check_table_exists", "list_available_schemas", "load_schema_from_file", "create_table_from_schema"],
        "description": "Create a single table from schema definition"
    },
    {
        "name": "Create Table with Dependencies",
        "agent": "synthetic_data_agent",
        "query": """Create the CARDS table from financial_transactions.json.
Note: CARDS depends on ACCOUNTS, so both should be created.""",
        "expected_tools": ["check_table_exists", "load_schema_from_file", "create_tables_with_dependencies"],
        "description": "Create table and its dependencies from schema"
    },
    {
        "name": "Generate Data for New Table",
        "agent": "synthetic_data_agent",
        "query": """I want to generate synthetic ACCOUNTS data, but the table doesn't exist yet.
1. Check if ACCOUNTS exists
2. If not, find the schema file and create the table
3. Then generate 10 synthetic accounts""",
        "expected_tools": ["check_table_exists", "list_available_schemas", "load_schema_from_file", "create_table_from_schema", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Full workflow for non-existent table"
    },
    {
        "name": "Drop and Recreate Table",
        "agent": "synthetic_data_agent",
        "query": """Drop the MERCHANTS table if it exists, then recreate it from the financial_transactions.json schema""",
        "expected_tools": ["check_table_exists", "drop_table", "load_schema_from_file", "create_table_from_schema"],
        "description": "Drop existing table and recreate from schema"
    },
]

SYNTH_SCHEMA_COMPLEX_EXAMPLES = [
    {
        "name": "Full Financial Tables Setup",
        "agent": "synthetic_data_agent",
        "query": """Set up the complete financial transactions database:
1. List available schemas
2. Load the financial_transactions.json schema
3. Create all tables in the correct dependency order
4. Show the tables that were created""",
        "expected_tools": ["list_available_schemas", "load_schema_from_file", "create_tables_with_dependencies", "check_table_exists"],
        "description": "Create all tables from a schema file"
    },
    {
        "name": "Generate Card Transactions Pipeline",
        "agent": "synthetic_data_agent",
        "query": """Generate synthetic card transaction data:
1. The CARD_TRANSACTIONS table doesn't exist - create it from financial_transactions.json
2. This should also create: ACCOUNTS, CARDS, MERCHANTS (dependencies)
3. Generate data for each table in order:
   - 10 ACCOUNTS
   - 5 MERCHANTS
   - 15 CARDS
   - 50 CARD_TRANSACTIONS
4. Show summary of all created tables""",
        "expected_tools": ["check_table_exists", "load_schema_from_file", "create_tables_with_dependencies", "generate_synthetic_data", "insert_synthetic_data", "list_synth_tables"],
        "description": "Full pipeline: create tables and generate data"
    },
    {
        "name": "Clean and Rebuild",
        "agent": "synthetic_data_agent",
        "query": """Clean up and rebuild the synthetic data environment:
1. List all existing SYNTH_* tables
2. Drop all SYNTH_* tables
3. List available schema files
4. Create ACCOUNTS and CARDS tables from financial_transactions.json
5. Generate 5 synthetic accounts and 10 synthetic cards
6. Show final state""",
        "expected_tools": ["list_synth_tables", "drop_all_synth_tables", "list_available_schemas", "load_schema_from_file", "create_tables_with_dependencies", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Clean environment and rebuild with new schema"
    },
    {
        "name": "Mixed Schema Generation",
        "agent": "synthetic_data_agent",
        "query": """I need test data from both wealth management and financial transactions:
1. Generate 5 synthetic CLIENTS (existing table)
2. Create ACCOUNTS from financial_transactions.json and generate 5 records
3. Create MERCHANTS and generate 10 records
4. List all synthetic data in the database""",
        "expected_tools": ["check_table_exists", "create_synth_table", "generate_synthetic_data", "load_schema_from_file", "create_table_from_schema", "insert_synthetic_data", "list_synth_tables"],
        "description": "Generate data from multiple schema sources"
    },
    {
        "name": "Complete Banking Test Suite",
        "agent": "synthetic_data_agent",
        "query": """Create a complete test suite for a banking application:
1. Check what financial tables exist vs what's available in schema files
2. Create all missing tables from financial_transactions.json
3. Generate realistic test data:
   - 20 ACCOUNTS with different types and statuses
   - 10 MERCHANTS across different categories
   - 30 CARDS linked to accounts
   - 100 BANK_TRANSACTIONS
   - 200 CARD_TRANSACTIONS
   - 50 ACCOUNT_BALANCES
4. Provide a summary of the generated test data""",
        "expected_tools": ["list_available_schemas", "check_table_exists", "load_schema_from_file", "create_tables_with_dependencies", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Complete test data generation for banking domain"
    },
]


# =============================================================================
# SYNTHETIC DATA - SEED DATA EXAMPLES (Empty Table Handling)
# =============================================================================

SYNTH_SEED_SIMPLE_EXAMPLES = [
    {
        "name": "Check Table Data Status",
        "agent": "synthetic_data_agent",
        "query": "Check the data status of the CLIENTS table - does it have data?",
        "expected_tools": ["get_table_data_status"],
        "description": "Get detailed status including row count and workflow guidance"
    },
    {
        "name": "Check Empty Table",
        "agent": "synthetic_data_agent",
        "query": "Check if the MERCHANTS table exists and whether it has any data",
        "expected_tools": ["check_table_exists"],
        "description": "Verify table exists and check if empty (returns OK/EMPTY/NOT_FOUND)"
    },
    {
        "name": "Generate Seed Data Prompt",
        "agent": "synthetic_data_agent",
        "query": "Generate a seed data prompt for the CLIENTS table to create 5 sample records",
        "expected_tools": ["get_table_schema_for_synth", "generate_seed_data_prompt"],
        "description": "Create LLM prompt for generating seed data"
    },
    {
        "name": "View Table Status Details",
        "agent": "synthetic_data_agent",
        "query": "What is the current status of the PORTFOLIOS table? Show me the workflow I should follow.",
        "expected_tools": ["get_table_data_status"],
        "description": "Get workflow guidance based on table status"
    },
]

SYNTH_SEED_MEDIUM_EXAMPLES = [
    {
        "name": "Empty Table Seed Generation",
        "agent": "synthetic_data_agent",
        "query": """The MERCHANTS table is empty. I need to:
1. Check its current status
2. Get its schema
3. Generate a seed data prompt for 5 records
4. Use the prompt to understand what data to generate""",
        "expected_tools": ["check_table_exists", "get_table_schema_for_synth", "generate_seed_data_prompt"],
        "description": "Prepare seed data generation for empty table"
    },
    {
        "name": "Insert Seed Data",
        "agent": "synthetic_data_agent",
        "query": """Insert this seed data into the MERCHANTS table:
[
    {"MERCHANT_ID": 1, "MERCHANT_NAME": "Amazon", "CATEGORY": "Retail", "COUNTRY": "USA", "MCC_CODE": "5411"},
    {"MERCHANT_ID": 2, "MERCHANT_NAME": "Uber", "CATEGORY": "Transportation", "COUNTRY": "USA", "MCC_CODE": "4121"},
    {"MERCHANT_ID": 3, "MERCHANT_NAME": "Carrefour", "CATEGORY": "Retail", "COUNTRY": "UAE", "MCC_CODE": "5411"}
]""",
        "expected_tools": ["insert_seed_data"],
        "description": "Insert LLM-generated seed data into table"
    },
    {
        "name": "Seed Then Synthesize",
        "agent": "synthetic_data_agent",
        "query": """For the ACCOUNTS table:
1. Check if it exists and has data
2. If empty, generate a seed data prompt
3. After seeding (assume done), generate 10 synthetic records""",
        "expected_tools": ["check_table_exists", "get_table_data_status", "generate_seed_data_prompt", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Full workflow: check -> seed -> synthesize"
    },
    {
        "name": "Multi-Table Status Check",
        "agent": "synthetic_data_agent",
        "query": """Check the data status of these tables and tell me what needs seeding:
1. CLIENTS
2. ASSETS
3. MERCHANTS
4. ACCOUNTS""",
        "expected_tools": ["get_table_data_status", "check_table_exists"],
        "description": "Identify which tables need seed data"
    },
]

SYNTH_SEED_COMPLEX_EXAMPLES = [
    {
        "name": "Complete Empty Table Workflow",
        "agent": "synthetic_data_agent",
        "query": """I have an empty CARDS table. Execute the complete workflow:
1. Check the table exists and is empty
2. Get the table schema
3. Generate a seed data prompt for 5 records
4. Based on the schema, generate appropriate JSON seed data
5. Insert the seed data
6. Then generate 20 additional synthetic records
7. Show the generation summary""",
        "expected_tools": ["check_table_exists", "get_table_schema_for_synth", "generate_seed_data_prompt", "insert_seed_data", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Full workflow for empty table: seed -> synthesize"
    },
    {
        "name": "Seed Dependencies Chain",
        "agent": "synthetic_data_agent",
        "query": """I need to generate CARD_TRANSACTIONS but all parent tables are empty.
1. Check status of ACCOUNTS, CARDS, MERCHANTS (dependencies)
2. For each empty table, generate seed data prompts
3. After seeding parents, generate synthetic CARD_TRANSACTIONS
The workflow should handle the dependency order correctly.""",
        "expected_tools": ["get_table_data_status", "analyze_table_dependencies", "generate_seed_data_prompt", "insert_seed_data", "generate_synthetic_data", "insert_synthetic_data"],
        "description": "Handle empty tables in dependency chain"
    },
    {
        "name": "Intelligent Table Preparation",
        "agent": "synthetic_data_agent",
        "query": """Prepare the financial transactions database for testing:
1. List all financial tables from schema
2. For each table, check its status:
   - If NOT_FOUND: create from schema
   - If EMPTY: generate seed data (5 records)
   - If OK: skip (has data)
3. After all tables are seeded, generate synthetic data:
   - 20 extra ACCOUNTS
   - 50 CARD_TRANSACTIONS
4. Show final summary""",
        "expected_tools": ["list_available_schemas", "load_schema_from_file", "get_table_data_status", "create_table_from_schema", "generate_seed_data_prompt", "insert_seed_data", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "Intelligent table preparation with status-based actions"
    },
    {
        "name": "Seed Data Quality Check",
        "agent": "synthetic_data_agent",
        "query": """For the BANK_TRANSACTIONS table:
1. Check if it exists and has data
2. If empty, analyze its schema and relationships
3. Generate a detailed seed data prompt that includes:
   - Column constraints
   - Foreign key references
   - Sample values
4. Use the prompt to generate 10 realistic seed records
5. Validate and insert the seed data
6. Generate 100 synthetic records based on the seed
7. Provide a summary including data quality metrics""",
        "expected_tools": ["check_table_exists", "get_table_schema_for_synth", "get_table_relationships", "generate_seed_data_prompt", "insert_seed_data", "generate_synthetic_data", "insert_synthetic_data", "get_generation_summary"],
        "description": "High-quality seed data generation with validation"
    },
    {
        "name": "Full Database Bootstrap",
        "agent": "synthetic_data_agent",
        "query": """Bootstrap a complete test database from scratch:
1. Drop all existing SYNTH_* tables
2. Load the financial_transactions.json schema
3. Create all tables in dependency order
4. For each table in order:
   a. Check status (should be EMPTY after creation)
   b. Generate seed data prompt
   c. Insert 5-10 seed records
   d. Generate 50+ synthetic records
5. Verify all tables have data
6. Show comprehensive summary with row counts""",
        "expected_tools": ["drop_all_synth_tables", "list_available_schemas", "load_schema_from_file", "create_tables_with_dependencies", "get_table_data_status", "generate_seed_data_prompt", "insert_seed_data", "generate_synthetic_data", "insert_synthetic_data", "list_synth_tables", "get_generation_summary"],
        "description": "Complete database bootstrap with seeding"
    },
]


def print_examples():
    """Print all examples in a formatted way."""

    all_categories = [
        ("🟢 SIMPLE EXAMPLES", SIMPLE_EXAMPLES),
        ("🟡 MEDIUM EXAMPLES", MEDIUM_EXAMPLES),
        ("🔴 COMPLEX EXAMPLES", COMPLEX_EXAMPLES),
        ("⚠️ EDGE CASES", EDGE_CASES),
        ("💬 CONVERSATIONAL", CONVERSATIONAL_EXAMPLES),
        ("🗄️ SQL SIMPLE", SQL_SIMPLE_EXAMPLES),
        ("🗄️ SQL MEDIUM", SQL_MEDIUM_EXAMPLES),
        ("🗄️ SQL COMPLEX", SQL_COMPLEX_EXAMPLES),
        ("🏷️ NAME MATCHING SIMPLE", NAME_MATCHING_SIMPLE_EXAMPLES),
        ("🏷️ NAME MATCHING MEDIUM", NAME_MATCHING_MEDIUM_EXAMPLES),
        ("🏷️ NAME MATCHING COMPLEX", NAME_MATCHING_COMPLEX_EXAMPLES),
        ("💼 WEALTH SIMPLE", WEALTH_SIMPLE_EXAMPLES),
        ("💼 WEALTH MEDIUM", WEALTH_MEDIUM_EXAMPLES),
        ("💼 WEALTH COMPLEX", WEALTH_COMPLEX_EXAMPLES),
        ("🤖 MULTI-DATA-AGENT SIMPLE", MULTI_DATA_AGENT_SIMPLE_EXAMPLES),
        ("🤖 MULTI-DATA-AGENT MEDIUM", MULTI_DATA_AGENT_MEDIUM_EXAMPLES),
        ("🤖 MULTI-DATA-AGENT COMPLEX", MULTI_DATA_AGENT_COMPLEX_EXAMPLES),
        ("✏️ MODIFIED QUERY SIMPLE", MODIFIED_QUERY_SIMPLE_EXAMPLES),
        ("✏️ MODIFIED QUERY MEDIUM", MODIFIED_QUERY_MEDIUM_EXAMPLES),
        ("✏️ MODIFIED QUERY COMPLEX", MODIFIED_QUERY_COMPLEX_EXAMPLES),
        ("❓ FOLLOWUP QUESTION SIMPLE", FOLLOWUP_QUESTION_SIMPLE_EXAMPLES),
        ("❓ FOLLOWUP QUESTION MEDIUM", FOLLOWUP_QUESTION_MEDIUM_EXAMPLES),
        ("❓ FOLLOWUP QUESTION COMPLEX", FOLLOWUP_QUESTION_COMPLEX_EXAMPLES),
        ("🆕 NEW QUERY EXAMPLES", NEW_QUERY_EXAMPLES),
        ("🧪 SYNTHETIC DATA SIMPLE", SYNTH_SIMPLE_EXAMPLES),
        ("🧪 SYNTHETIC DATA MEDIUM", SYNTH_MEDIUM_EXAMPLES),
        ("🧪 SYNTHETIC DATA COMPLEX", SYNTH_COMPLEX_EXAMPLES),
        ("📁 SYNTH SCHEMA SIMPLE", SYNTH_SCHEMA_SIMPLE_EXAMPLES),
        ("📁 SYNTH SCHEMA MEDIUM", SYNTH_SCHEMA_MEDIUM_EXAMPLES),
        ("📁 SYNTH SCHEMA COMPLEX", SYNTH_SCHEMA_COMPLEX_EXAMPLES),
        ("🌱 SYNTH SEED SIMPLE", SYNTH_SEED_SIMPLE_EXAMPLES),
        ("🌱 SYNTH SEED MEDIUM", SYNTH_SEED_MEDIUM_EXAMPLES),
        ("🌱 SYNTH SEED COMPLEX", SYNTH_SEED_COMPLEX_EXAMPLES),
    ]
    
    for category_name, examples in all_categories:
        print(f"\n{'='*80}")
        print(f"{category_name}")
        print('='*80)
        
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. {example['name']}")
            print(f"   Agent: {example['agent']}")
            print(f"   Tools: {', '.join(example['expected_tools'])}")
            print(f"   Description: {example['description']}")
            print(f"   Query: {example['query'][:100]}{'...' if len(example['query']) > 100 else ''}")


def get_example_by_name(name: str) -> dict:
    """Get a specific example by name."""
    all_examples = (
        SIMPLE_EXAMPLES +
        MEDIUM_EXAMPLES +
        COMPLEX_EXAMPLES +
        EDGE_CASES +
        CONVERSATIONAL_EXAMPLES +
        SQL_SIMPLE_EXAMPLES +
        SQL_MEDIUM_EXAMPLES +
        SQL_COMPLEX_EXAMPLES +
        NAME_MATCHING_SIMPLE_EXAMPLES +
        NAME_MATCHING_MEDIUM_EXAMPLES +
        NAME_MATCHING_COMPLEX_EXAMPLES +
        WEALTH_SIMPLE_EXAMPLES +
        WEALTH_MEDIUM_EXAMPLES +
        WEALTH_COMPLEX_EXAMPLES +
        MULTI_DATA_AGENT_SIMPLE_EXAMPLES +
        MULTI_DATA_AGENT_MEDIUM_EXAMPLES +
        MULTI_DATA_AGENT_COMPLEX_EXAMPLES +
        FOLLOWUP_SIMPLE_EXAMPLES +
        FOLLOWUP_MEDIUM_EXAMPLES +
        FOLLOWUP_COMPLEX_EXAMPLES +
        NEW_QUERY_EXAMPLES +
        SYNTH_SIMPLE_EXAMPLES +
        SYNTH_MEDIUM_EXAMPLES +
        SYNTH_COMPLEX_EXAMPLES +
        SYNTH_SCHEMA_SIMPLE_EXAMPLES +
        SYNTH_SCHEMA_MEDIUM_EXAMPLES +
        SYNTH_SCHEMA_COMPLEX_EXAMPLES +
        SYNTH_SEED_SIMPLE_EXAMPLES +
        SYNTH_SEED_MEDIUM_EXAMPLES +
        SYNTH_SEED_COMPLEX_EXAMPLES
    )

    for example in all_examples:
        if example['name'].lower() == name.lower():
            return example
    return None


def get_examples_by_agent(agent_name: str) -> list:
    """Get all examples for a specific agent."""
    all_examples = (
        SIMPLE_EXAMPLES +
        MEDIUM_EXAMPLES +
        COMPLEX_EXAMPLES +
        EDGE_CASES +
        CONVERSATIONAL_EXAMPLES +
        SQL_SIMPLE_EXAMPLES +
        SQL_MEDIUM_EXAMPLES +
        SQL_COMPLEX_EXAMPLES +
        NAME_MATCHING_SIMPLE_EXAMPLES +
        NAME_MATCHING_MEDIUM_EXAMPLES +
        NAME_MATCHING_COMPLEX_EXAMPLES +
        WEALTH_SIMPLE_EXAMPLES +
        WEALTH_MEDIUM_EXAMPLES +
        WEALTH_COMPLEX_EXAMPLES +
        MULTI_DATA_AGENT_SIMPLE_EXAMPLES +
        MULTI_DATA_AGENT_MEDIUM_EXAMPLES +
        MULTI_DATA_AGENT_COMPLEX_EXAMPLES +
        FOLLOWUP_SIMPLE_EXAMPLES +
        FOLLOWUP_MEDIUM_EXAMPLES +
        FOLLOWUP_COMPLEX_EXAMPLES +
        NEW_QUERY_EXAMPLES +
        SYNTH_SIMPLE_EXAMPLES +
        SYNTH_MEDIUM_EXAMPLES +
        SYNTH_COMPLEX_EXAMPLES +
        SYNTH_SCHEMA_SIMPLE_EXAMPLES +
        SYNTH_SCHEMA_MEDIUM_EXAMPLES +
        SYNTH_SCHEMA_COMPLEX_EXAMPLES +
        SYNTH_SEED_SIMPLE_EXAMPLES +
        SYNTH_SEED_MEDIUM_EXAMPLES +
        SYNTH_SEED_COMPLEX_EXAMPLES
    )

    return [ex for ex in all_examples if ex['agent'] == agent_name]


def get_examples_by_tool(tool_name: str) -> list:
    """Get all examples that use a specific tool."""
    all_examples = (
        SIMPLE_EXAMPLES +
        MEDIUM_EXAMPLES +
        COMPLEX_EXAMPLES +
        EDGE_CASES +
        CONVERSATIONAL_EXAMPLES +
        SQL_SIMPLE_EXAMPLES +
        SQL_MEDIUM_EXAMPLES +
        SQL_COMPLEX_EXAMPLES +
        NAME_MATCHING_SIMPLE_EXAMPLES +
        NAME_MATCHING_MEDIUM_EXAMPLES +
        NAME_MATCHING_COMPLEX_EXAMPLES +
        WEALTH_SIMPLE_EXAMPLES +
        WEALTH_MEDIUM_EXAMPLES +
        WEALTH_COMPLEX_EXAMPLES +
        MULTI_DATA_AGENT_SIMPLE_EXAMPLES +
        MULTI_DATA_AGENT_MEDIUM_EXAMPLES +
        MULTI_DATA_AGENT_COMPLEX_EXAMPLES +
        FOLLOWUP_SIMPLE_EXAMPLES +
        FOLLOWUP_MEDIUM_EXAMPLES +
        FOLLOWUP_COMPLEX_EXAMPLES +
        NEW_QUERY_EXAMPLES +
        SYNTH_SIMPLE_EXAMPLES +
        SYNTH_MEDIUM_EXAMPLES +
        SYNTH_COMPLEX_EXAMPLES +
        SYNTH_SCHEMA_SIMPLE_EXAMPLES +
        SYNTH_SCHEMA_MEDIUM_EXAMPLES +
        SYNTH_SCHEMA_COMPLEX_EXAMPLES +
        SYNTH_SEED_SIMPLE_EXAMPLES +
        SYNTH_SEED_MEDIUM_EXAMPLES +
        SYNTH_SEED_COMPLEX_EXAMPLES
    )

    return [ex for ex in all_examples if tool_name in ex['expected_tools']]


def get_followup_conversation(conversation_id: str) -> list:
    """Get all examples in a follow-up conversation sequence, ordered by sequence number."""
    all_followup = (
        FOLLOWUP_SIMPLE_EXAMPLES +
        FOLLOWUP_MEDIUM_EXAMPLES +
        FOLLOWUP_COMPLEX_EXAMPLES
    )

    conversation = [ex for ex in all_followup if ex.get('conversation_id') == conversation_id]
    return sorted(conversation, key=lambda x: x.get('sequence', 0))


if __name__ == "__main__":
    print_examples()
    
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("""
1. Run the Streamlit app: streamlit run app.py
2. Select an agent from the sidebar
3. Copy any query from above and paste it into the chat
4. Watch the ReAct reasoning process in action!

Or run automated tests:
    python run_tests.py
    """)
