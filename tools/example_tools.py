"""
Example Tools - Concrete implementations of tools for the agent.
Each tool represents a specific capability the agent can use.
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import json
import math
from datetime import datetime, timedelta
import random

from core.tools_base import tool_registry


# ============================================
# MATH TOOLS
# ============================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use this for any arithmetic calculations.
    
    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16) * 3"
    
    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation with limited functions
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

tool_registry.register(calculator, "math")


@tool  
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units of measurement.
    
    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., 'km', 'miles', 'celsius', 'fahrenheit')
        to_unit: Target unit
    
    Returns:
        The converted value with units
    """
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("kg", "pounds"): lambda x: x * 2.20462,
        ("pounds", "kg"): lambda x: x / 2.20462,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x / 3.28084,
        ("liters", "gallons"): lambda x: x * 0.264172,
        ("gallons", "liters"): lambda x: x / 0.264172,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported"

tool_registry.register(unit_converter, "math")


# ============================================
# DATE/TIME TOOLS
# ============================================

@tool
def get_current_datetime() -> str:
    """
    Get the current date and time.
    Use this when you need to know the current date or time.
    
    Returns:
        Current date and time information
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"

tool_registry.register(get_current_datetime, "datetime")


@tool
def calculate_date_difference(date1: str, date2: str) -> str:
    """
    Calculate the difference between two dates.
    
    Args:
        date1: First date in YYYY-MM-DD format
        date2: Second date in YYYY-MM-DD format
    
    Returns:
        The difference in days between the two dates
    """
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return f"Difference between {date1} and {date2}: {diff} days"
    except ValueError as e:
        return f"Error parsing dates: {str(e)}. Use YYYY-MM-DD format."

tool_registry.register(calculate_date_difference, "datetime")


@tool
def add_days_to_date(date: str, days: int) -> str:
    """
    Add or subtract days from a date.
    
    Args:
        date: Starting date in YYYY-MM-DD format
        days: Number of days to add (negative to subtract)
    
    Returns:
        The resulting date
    """
    try:
        d = datetime.strptime(date, "%Y-%m-%d")
        result = d + timedelta(days=days)
        return f"{date} + {days} days = {result.strftime('%Y-%m-%d')} ({result.strftime('%A')})"
    except ValueError as e:
        return f"Error: {str(e)}"

tool_registry.register(add_days_to_date, "datetime")


# ============================================
# TEXT PROCESSING TOOLS
# ============================================

@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and provide statistics.
    
    Args:
        text: The text to analyze
    
    Returns:
        Statistics about the text including word count, character count, etc.
    """
    words = text.split()
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    analysis = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "average_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "average_sentence_length": len(words) / len(sentences) if sentences else 0
    }
    
    return json.dumps(analysis, indent=2)

tool_registry.register(text_analyzer, "text")


@tool
def text_transformer(text: str, operation: str) -> str:
    """
    Transform text using various operations.
    
    Args:
        text: The text to transform
        operation: The operation to perform (uppercase, lowercase, title, reverse, remove_spaces)
    
    Returns:
        The transformed text
    """
    operations = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "title": str.title,
        "reverse": lambda x: x[::-1],
        "remove_spaces": lambda x: x.replace(" ", ""),
    }
    
    if operation.lower() in operations:
        return operations[operation.lower()](text)
    return f"Unknown operation: {operation}. Available: {list(operations.keys())}"

tool_registry.register(text_transformer, "text")


# ============================================
# DATA/LIST TOOLS
# ============================================

@tool
def list_operations(items: str, operation: str) -> str:
    """
    Perform operations on a comma-separated list of items.
    
    Args:
        items: Comma-separated list of items
        operation: Operation to perform (sort, reverse, unique, count, shuffle)
    
    Returns:
        Result of the operation
    """
    item_list = [item.strip() for item in items.split(",")]
    
    if operation == "sort":
        return ", ".join(sorted(item_list))
    elif operation == "reverse":
        return ", ".join(reversed(item_list))
    elif operation == "unique":
        return ", ".join(dict.fromkeys(item_list))
    elif operation == "count":
        return str(len(item_list))
    elif operation == "shuffle":
        random.shuffle(item_list)
        return ", ".join(item_list)
    else:
        return f"Unknown operation. Available: sort, reverse, unique, count, shuffle"

tool_registry.register(list_operations, "data")


@tool
def json_parser(json_string: str, key_path: str = "") -> str:
    """
    Parse JSON and optionally extract a value by key path.
    
    Args:
        json_string: A JSON string to parse
        key_path: Optional dot-notation path to extract (e.g., "user.name")
    
    Returns:
        Parsed JSON or extracted value
    """
    try:
        data = json.loads(json_string)
        
        if key_path:
            keys = key_path.split(".")
            result = data
            for key in keys:
                if isinstance(result, dict):
                    result = result.get(key, f"Key '{key}' not found")
                elif isinstance(result, list) and key.isdigit():
                    result = result[int(key)]
                else:
                    return f"Cannot navigate to '{key}'"
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"

tool_registry.register(json_parser, "data")


# ============================================
# TASK MANAGEMENT TOOLS (Simulated)
# ============================================

# In-memory task store for demonstration
_task_store: Dict[str, Dict[str, Any]] = {}


@tool
def create_task(title: str, description: str = "", priority: str = "medium") -> str:
    """
    Create a new task in the task management system.
    
    Args:
        title: Title of the task
        description: Detailed description of the task
        priority: Priority level (low, medium, high)
    
    Returns:
        Confirmation with task ID
    """
    task_id = f"TASK-{len(_task_store) + 1:04d}"
    _task_store[task_id] = {
        "id": task_id,
        "title": title,
        "description": description,
        "priority": priority,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    return f"Task created successfully: {task_id} - '{title}' (Priority: {priority})"

tool_registry.register(create_task, "tasks")


@tool
def list_tasks(status: str = "all") -> str:
    """
    List all tasks or filter by status.
    
    Args:
        status: Filter by status (all, pending, in_progress, completed)
    
    Returns:
        List of tasks
    """
    if not _task_store:
        return "No tasks found."
    
    tasks = list(_task_store.values())
    if status != "all":
        tasks = [t for t in tasks if t["status"] == status]
    
    if not tasks:
        return f"No tasks with status '{status}'"
    
    result = []
    for task in tasks:
        result.append(f"- [{task['id']}] {task['title']} (Priority: {task['priority']}, Status: {task['status']})")
    
    return "\n".join(result)

tool_registry.register(list_tasks, "tasks")


@tool
def update_task_status(task_id: str, new_status: str) -> str:
    """
    Update the status of a task.
    
    Args:
        task_id: The ID of the task to update
        new_status: New status (pending, in_progress, completed)
    
    Returns:
        Confirmation of update
    """
    if task_id not in _task_store:
        return f"Task {task_id} not found"
    
    valid_statuses = ["pending", "in_progress", "completed"]
    if new_status not in valid_statuses:
        return f"Invalid status. Use: {valid_statuses}"
    
    _task_store[task_id]["status"] = new_status
    return f"Task {task_id} status updated to '{new_status}'"

tool_registry.register(update_task_status, "tasks")


# ============================================
# KNOWLEDGE/SEARCH TOOLS (Simulated)
# ============================================

@tool
def knowledge_base_search(query: str) -> str:
    """
    Search a simulated knowledge base for information.
    
    Args:
        query: Search query
    
    Returns:
        Relevant information from the knowledge base
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for its readability and versatility. It's widely used in web development, data science, AI, and automation.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with cyclic computational capabilities.",
        "react": "ReAct (Reasoning and Acting) is an AI agent paradigm that interleaves reasoning steps with actions, allowing for more deliberate and adaptable behavior.",
        "agent": "An AI agent is an autonomous system that can perceive its environment, reason about it, and take actions to achieve goals.",
        "streamlit": "Streamlit is an open-source Python library for creating web applications for machine learning and data science projects with minimal code."
    }
    
    query_lower = query.lower()
    results = []
    for key, value in knowledge.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            results.append(f"**{key.title()}**: {value}")
    
    if results:
        return "\n\n".join(results)
    return f"No results found for '{query}'. Try searching for: {', '.join(knowledge.keys())}"

tool_registry.register(knowledge_base_search, "knowledge")


def get_all_tools():
    """Get all registered tools as a list."""
    # Import SQL tools to ensure they're registered
    from tools import sql_tools
    # Import name matching tools to ensure they're registered
    from tools import name_matching_tools
    # Import RAG tools to ensure they're registered
    from tools import rag_tools
    return tool_registry.get_all_tools()
