"""
Example Queries Page for Streamlit UI
=====================================

This module provides the example queries tab for the Streamlit application.
"""

import streamlit as st


def render_example_queries():
    """Render the example queries section in Streamlit."""
    
    st.header("📝 Example Queries")
    st.markdown("Click any example to copy it, then paste in the chat!")
    
    # Simple Examples
    with st.expander("🟢 Simple Examples (Single Tool)", expanded=True):
        simple_examples = [
            ("Calculator", "What is 15% of 850?"),
            ("Unit Conversion", "Convert 98.6 degrees Fahrenheit to Celsius"),
            ("Current Time", "What is today's date and time?"),
            ("Text Analysis", "Analyze this text: 'The quick brown fox jumps over the lazy dog.'"),
            ("Create Task", "Create a high priority task called 'Review report'"),
            ("Sort List", "Sort alphabetically: banana, apple, cherry, date"),
            ("Knowledge Search", "What is Python programming language?"),
        ]
        
        for name, query in simple_examples:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**{name}**")
            with col2:
                st.code(query, language=None)
    
    # Medium Examples
    with st.expander("🟡 Medium Examples (Multiple Tools)", expanded=False):
        medium_examples = [
            ("Trip Cost Calculator", 
             "I'm traveling 150 miles. If my car gets 30 miles per gallon and gas costs $3.50 per gallon, how much will the trip cost? Also convert the distance to kilometers."),
            
            ("Date Planning", 
             "Today is the start date. Calculate the end date if a project takes 45 days, and how many days until December 31, 2025."),
            
            ("Task Workflow", 
             "Create three tasks: 'Design mockups' (high), 'Write docs' (medium), 'Code review' (low). Then list all tasks."),
            
            ("Recipe Scaling", 
             "A recipe for 4 people needs 2.5 cups of flour. I'm cooking for 7 people. How many cups do I need? Also convert to milliliters."),
            
            ("Bill Splitting", 
             "Split a $247.50 restaurant bill between 5 friends with a 20% tip. How much does each person owe?"),
        ]
        
        for name, query in medium_examples:
            st.markdown(f"**{name}**")
            st.code(query, language=None)
            st.markdown("---")
    
    # Complex Examples
    with st.expander("🔴 Complex Examples (Multi-Step Reasoning)", expanded=False):
        complex_examples = [
            ("Project Planning Suite", """Help me plan a project:
1. Get today's date as the start date
2. Calculate the end date if the project takes 90 days
3. Create tasks: 'Requirements' (high), 'Development' (high), 'Testing' (medium)
4. List all tasks
5. Search for information about 'Python'"""),
            
            ("Conference Planning", """Plan a tech conference:
1. Conference is on 2025-06-15. Days from today until then?
2. Registration deadline: 30 days before. What date?
3. Create tasks: 'Book venue' (high), 'Send invites' (high), 'Catering' (medium)
4. Budget: Venue $5000 + Catering $3000 + Marketing $2000"""),
            
            ("Investment Analysis", """Analyze this investment:
1. Initial: $50,000
2. Calculate 7% annual return after 5 years (P * 1.07^5)
3. Calculate total gain
4. What percentage gain?
5. Convert final amount to EUR (multiply by 0.92)"""),
            
            ("Fitness Progress", """Track my fitness:
1. I ran 5.5 miles. Convert to km
2. Time: 48 minutes. Calculate pace (min/mile)
3. Calculate speed in mph
4. Convert speed to km/h
5. Weight: 180 lbs. Convert to kg
6. Create task 'Log fitness' (medium)"""),
            
            ("Business Metrics", """Calculate business metrics:
1. Parse: '{"sales": [12000, 15000, 18000, 14000, 20000]}'
2. Calculate total sales
3. Calculate average monthly sales
4. Growth from first to last month: ((20000-12000)/12000)*100
5. Create task 'Prepare Q2 forecast' (high)"""),
        ]
        
        for name, query in complex_examples:
            st.markdown(f"**{name}**")
            st.code(query, language=None)
            st.markdown("---")
    
    # Agent-Specific Examples
    with st.expander("🤖 Agent-Specific Examples", expanded=False):
        st.markdown("### Math Specialist")
        st.code("Calculate compound interest: $10,000 at 5% for 10 years. Use: Principal * (1 + 0.05)^10")
        st.code("Convert 26.2 miles to km, then calculate how many meters that is")
        
        st.markdown("### Task Manager")
        st.code("Create a task list for a product launch: Research, Design, Development, Testing, Launch. Set priorities appropriately.")
        st.code("List all pending tasks, then update the first one to 'in_progress'")
        
        st.markdown("### Data Analyst")
        st.code("""Parse this JSON and analyze: '{"users": ["Alice", "Bob", "Charlie"], "scores": [95, 87, 92]}'
Calculate the average score and sort users alphabetically.""")
        
        st.markdown("### Researcher")  
        st.code("Search for information about LangGraph, ReAct pattern, and AI agents. Summarize the key concepts.")
    
    # Tips Section
    st.markdown("---")
    st.subheader("💡 Tips for Better Results")
    
    tips = [
        "**Be specific**: Instead of 'calculate something', say 'calculate 15% of 200'",
        "**Multi-step tasks**: Number your steps for complex queries",
        "**Use the right agent**: Math Specialist for calculations, Task Manager for todos",
        "**Watch the trace**: Expand 'View ReAct Execution Trace' to see the reasoning",
        "**Chain operations**: Ask for conversions + calculations in one query",
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")


def get_quick_examples():
    """Return a list of quick example queries for the sidebar."""
    return [
        "What is 25% of 480?",
        "Convert 100 km to miles",
        "What's today's date?",
        "Create a task: Review code (high priority)",
        "Sort: zebra, apple, mango, banana",
        "Days between 2025-01-01 and 2025-12-31?",
    ]


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Example Queries", layout="wide")
    render_example_queries()
