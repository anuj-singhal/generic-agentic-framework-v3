"""
Quick Test Script for Data Agent LLM-Powered Tools
===================================================

This script tests the data_agent with LLM-powered tools:
- get_relevant_schema (LLM-based schema analysis)
- generate_sql (LLM-based SQL generation)
- validate_sql (LLM-based validation)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import FrameworkConfig, ModelConfig
from agents.agent_definitions import create_agent


def test_data_agent(api_key: str):
    """Test the data_agent with a simple query."""

    print("="*70)
    print("TESTING DATA_AGENT WITH LLM-POWERED TOOLS")
    print("="*70)

    # Create configuration
    config = FrameworkConfig(
        model=ModelConfig(
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key
        ),
        max_iterations=12
    )

    # Create the data_agent
    print("\n[1/3] Creating data_agent...")
    orchestrator = create_agent("data_agent", config)

    if orchestrator is None:
        print("ERROR: Could not create data_agent")
        return

    print("[OK] Data agent created successfully")
    print(f"     Tools available: {len(orchestrator.tools)}")

    # Test with a simple query
    test_query = "Show me all clients in the database"

    print(f"\n[2/3] Running test query: '{test_query}'")
    print("     This will test:")
    print("     - get_relevant_schema (LLM analyzes which tables are needed)")
    print("     - generate_sql (LLM generates the SQL query)")
    print("     - validate_sql (LLM validates syntax, schema, semantics)")
    print("     - execute_sql (Runs the validated query)")

    print("\n[3/3] Executing query...")
    print("-"*70)

    result = orchestrator.run(test_query)

    # Extract final answer
    messages = result.get("messages", [])
    final_answer = ""

    for msg in reversed(messages):
        if type(msg).__name__ == "AIMessage" and msg.content:
            final_answer = msg.content
            break

    print("\nFINAL ANSWER:")
    print("-"*70)
    print(final_answer)
    print("-"*70)

    # Show statistics
    iterations = result.get("iteration_count", 0)
    error = result.get("error")

    print(f"\nSTATISTICS:")
    print(f"  Iterations: {iterations}")
    print(f"  Status: {'ERROR' if error else 'SUCCESS'}")
    if error:
        print(f"  Error: {error}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return result


if __name__ == "__main__":
    # Get API key from environment or prompt
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("       Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Check if database exists
    if not os.path.exists("agent_ddb.db"):
        print("ERROR: Database 'agent_ddb.db' not found")
        print("       Run 'python init_duckdb.py' first to create the database")
        sys.exit(1)

    test_data_agent(api_key)
