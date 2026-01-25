"""
Test script for Agent1 memory integration.
This tests that short-term and long-term memory are properly passed to Agent1.
"""

import os
import sys

# Ensure API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

from core.multi_agent_orchestrator import MultiAgentDataOrchestrator, query_cache
from core.config import FrameworkConfig, ModelConfig

def test_memory_flow():
    """Test the memory flow through Agent1."""

    # Create config
    config = FrameworkConfig(
        model=ModelConfig(
            model_name="gpt-4o-mini",
            temperature=0.1,
            api_key=os.environ.get("OPENAI_API_KEY")
        ),
        max_iterations=10
    )

    orchestrator = MultiAgentDataOrchestrator(config=config)

    # Clear any existing cache
    query_cache.clear()

    print("=" * 60)
    print("TEST 1: New query with NO memory")
    print("=" * 60)

    # Test 1: No memory, no cache
    result1 = orchestrator.run(
        query="Show me all clients",
        short_term_memory=None,
        long_term_memory=None
    )

    print(f"Intent: Should be NEW_DATA_QUERY")
    print(f"Generated SQL: {result1.get('generated_sql', 'None')[:100] if result1.get('generated_sql') else 'None'}...")
    print(f"Has cache after: {query_cache.has_cache()}")

    # Extract intent from traces
    for trace in result1.get("agent_traces", []):
        if trace.get("agent_id") == "agent1":
            print(f"Agent1 Details: {trace.get('details')}")

    print("\n" + "=" * 60)
    print("TEST 2: Follow-up question WITH cache (no memory yet)")
    print("=" * 60)

    # Test 2: Has cache, no memory
    result2 = orchestrator.run(
        query="How many clients are there?",
        short_term_memory=None,
        long_term_memory=None
    )

    print(f"Intent: Should be FOLLOWUP_QUESTION")
    for trace in result2.get("agent_traces", []):
        if trace.get("agent_id") == "agent1":
            print(f"Agent1 Details: {trace.get('details')}")

    print("\n" + "=" * 60)
    print("TEST 3: General question WITH cache")
    print("=" * 60)

    # Test 3: Has cache, general question
    result3 = orchestrator.run(
        query="What is a risk profile?",
        short_term_memory=None,
        long_term_memory=None
    )

    print(f"Intent: Should be GENERAL_QUESTION")
    print(f"Cache preserved: {query_cache.has_cache()}")
    for trace in result3.get("agent_traces", []):
        if trace.get("agent_id") == "agent1":
            print(f"Agent1 Details: {trace.get('details')}")

    print("\n" + "=" * 60)
    print("TEST 4: New query WITH short-term memory")
    print("=" * 60)

    # Build simulated short-term memory from previous interactions
    short_term_memory = """--- Conversation 1 (Agent: data_agent, Intent: NEW_DATA_QUERY) ---
User Query: Show me all clients
Generated SQL: SELECT * FROM CLIENTS
Query Results Preview: client_id | client_name | risk_profile...
Assistant Response: Here are all the clients in the database...

--- Conversation 2 (Agent: data_agent, Intent: FOLLOWUP_QUESTION) ---
User Query: How many clients are there?
Assistant Response: There are 10 clients in the database.

--- Conversation 3 (Agent: data_agent, Intent: GENERAL_QUESTION) ---
User Query: What is a risk profile?
Assistant Response: A risk profile is a measure of an investor's tolerance for risk...
"""

    # Clear cache to test memory-only scenario
    query_cache.clear()

    result4 = orchestrator.run(
        query="Show me all portfolios",
        short_term_memory=short_term_memory,
        long_term_memory=None
    )

    print(f"Intent: Should be NEW_DATA_QUERY")
    print(f"Generated SQL: {result4.get('generated_sql', 'None')[:100] if result4.get('generated_sql') else 'None'}...")
    for trace in result4.get("agent_traces", []):
        if trace.get("agent_id") == "agent1":
            details = trace.get('details', {})
            print(f"Agent1 Details:")
            print(f"  - has_memory: {details.get('has_memory')}")
            print(f"  - short_term_memory_chars: {details.get('short_term_memory_chars')}")
            print(f"  - scenario: {details.get('scenario')}")
            print(f"  - intent: {details.get('intent')}")

    print("\n" + "=" * 60)
    print("TEST 5: Modified query using memory (no cache)")
    print("=" * 60)

    # Clear cache to test memory-based modification
    query_cache.clear()

    # Memory now includes the portfolios query
    short_term_memory_updated = short_term_memory + """
--- Conversation 4 (Agent: data_agent, Intent: NEW_DATA_QUERY) ---
User Query: Show me all portfolios
Generated SQL: SELECT * FROM PORTFOLIOS
Query Results Preview: portfolio_id | portfolio_name | base_currency...
Assistant Response: Here are all the portfolios in the database...
"""

    result5 = orchestrator.run(
        query="Go back to the clients query and filter by high risk only",
        short_term_memory=short_term_memory_updated,
        long_term_memory=None
    )

    print(f"Intent: Should be MODIFIED_QUERY (detected from memory)")
    for trace in result5.get("agent_traces", []):
        if trace.get("agent_id") == "agent1":
            details = trace.get('details', {})
            print(f"Agent1 Details:")
            print(f"  - has_memory: {details.get('has_memory')}")
            print(f"  - short_term_memory_chars: {details.get('short_term_memory_chars')}")
            print(f"  - scenario: {details.get('scenario')}")
            print(f"  - intent: {details.get('intent')}")
            print(f"  - memory_influenced: {details.get('memory_influenced')}")

    print("\n" + "=" * 60)
    print("TEST 6: Query with both short-term and long-term memory")
    print("=" * 60)

    long_term_memory = """--- Summary 1 (Conversations 1-5) ---
The user explored client data, asking about all clients, client counts, and risk profiles.
They also queried portfolio information. Key SQL queries included:
- SELECT * FROM CLIENTS
- SELECT * FROM PORTFOLIOS
The user seems interested in wealth management data analysis.
"""

    result6 = orchestrator.run(
        query="Tell me more about risk profiles like you explained earlier",
        short_term_memory=short_term_memory_updated,
        long_term_memory=long_term_memory
    )

    print(f"Intent: Should be GENERAL_QUESTION (memory-influenced)")
    for trace in result6.get("agent_traces", []):
        if trace.get("agent_id") == "agent1":
            details = trace.get('details', {})
            print(f"Agent1 Details:")
            print(f"  - has_memory: {details.get('has_memory')}")
            print(f"  - short_term_memory_chars: {details.get('short_term_memory_chars')}")
            print(f"  - long_term_memory_chars: {details.get('long_term_memory_chars')}")
            print(f"  - scenario: {details.get('scenario')}")
            print(f"  - intent: {details.get('intent')}")
            print(f"  - memory_influenced: {details.get('memory_influenced')}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_flow()
