"""
Test script for Multi-Agent Data System
========================================

Tests the multi-agent workflow for SQL query generation with validation.
"""

import os
import sys

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_schema_tools():
    """Test the schema extraction tools."""
    print("\n" + "="*70)
    print("TESTING SCHEMA EXTRACTION TOOLS")
    print("="*70)

    from tools.multi_data_agent_tools import (
        get_table_descriptions,
        extract_table_schema,
        get_database_relationships,
        analyze_query_requirements
    )

    # Test 1: Get table descriptions
    print("\n1. Testing get_table_descriptions()...")
    result = get_table_descriptions.invoke({})
    print(result[:500] + "..." if len(result) > 500 else result)

    # Test 2: Extract schema for CLIENTS table
    print("\n2. Testing extract_table_schema('CLIENTS')...")
    result = extract_table_schema.invoke({"table_name": "CLIENTS"})
    print(result[:500] + "..." if len(result) > 500 else result)

    # Test 3: Get relationships
    print("\n3. Testing get_database_relationships()...")
    result = get_database_relationships.invoke({})
    print(result[:500] + "..." if len(result) > 500 else result)

    # Test 4: Analyze query requirements
    print("\n4. Testing analyze_query_requirements()...")
    test_queries = [
        "Show all clients",
        "Show me clients with their portfolios",
        "Calculate total portfolio value for each client with transaction history"
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = analyze_query_requirements.invoke({"nl_query": query})
        print(result)

    print("\nSchema tools test completed!")


def test_multi_agent_orchestrator():
    """Test the multi-agent orchestrator."""
    print("\n" + "="*70)
    print("TESTING MULTI-AGENT ORCHESTRATOR")
    print("="*70)

    from core.multi_agent_orchestrator import MultiAgentDataOrchestrator

    orchestrator = MultiAgentDataOrchestrator()

    # Test queries
    test_queries = [
        # Simple query
        "Show me all clients in the database",

        # Medium complexity
        "Show me clients along with their portfolio names",

        # General question (should be answered directly)
        "What is SQL?",
    ]

    for query in test_queries:
        print(f"\n{'-'*50}")
        print(f"Query: {query}")
        print("-"*50)

        try:
            result = orchestrator.run(query)

            print(f"\nIs Data Query: {result.get('is_data_query', 'N/A')}")
            print(f"Query Complexity: {result.get('query_complexity', 'N/A')}")
            print(f"Validation Confidence: {result.get('overall_confidence', 'N/A')}")

            if result.get('generated_sql'):
                print(f"\nGenerated SQL:\n{result['generated_sql']}")

            if result.get('final_answer'):
                print(f"\nFinal Answer:\n{result['final_answer'][:500]}...")
            elif result.get('general_answer'):
                print(f"\nGeneral Answer:\n{result['general_answer'][:500]}...")

            if result.get('error'):
                print(f"\nError: {result['error']}")

        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nMulti-agent orchestrator test completed!")


def test_agent_via_factory():
    """Test the multi_data_agent through the AgentFactory."""
    print("\n" + "="*70)
    print("TESTING MULTI_DATA_AGENT VIA AGENTFACTORY")
    print("="*70)

    from agents.agent_definitions import AgentFactory, get_available_agents

    # List available agents
    print("\nAvailable agents:")
    agents = get_available_agents()
    for name, desc in agents.items():
        print(f"  - {name}: {desc[:60]}...")

    # Check if multi_data_agent is registered
    print("\n\nChecking multi_data_agent registration...")
    definition = AgentFactory.get_agent_definition("multi_data_agent")
    if definition:
        print(f"Name: {definition.name}")
        print(f"Description: {definition.description[:100]}...")
        print(f"Tool Categories: {definition.tool_categories}")
        print(f"Max Iterations: {definition.max_iterations}")

        # Get tools
        tools = definition.get_tools()
        print(f"\nAvailable tools ({len(tools)} total):")
        for tool in tools[:10]:  # Show first 10
            print(f"  - {tool.name}")
        if len(tools) > 10:
            print(f"  ... and {len(tools) - 10} more")
    else:
        print("ERROR: multi_data_agent not found!")

    print("\nAgent factory test completed!")


def test_run_multi_agent_query_tool():
    """Test the run_multi_agent_query tool directly."""
    print("\n" + "="*70)
    print("TESTING run_multi_agent_query TOOL")
    print("="*70)

    from tools.multi_data_agent_tools import run_multi_agent_query

    query = "Show me all clients"
    print(f"\nQuery: {query}")

    try:
        result = run_multi_agent_query.invoke({"query": query})
        print(f"\nResult:\n{result}")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nrun_multi_agent_query tool test completed!")


def test_workflow_info():
    """Test getting workflow information."""
    print("\n" + "="*70)
    print("TESTING WORKFLOW INFO")
    print("="*70)

    from core.multi_agent_orchestrator import MultiAgentDataOrchestrator

    orchestrator = MultiAgentDataOrchestrator()
    info = orchestrator.get_workflow_info()

    print(f"\nWorkflow Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Max Retries: {info['max_retries']}")
    print(f"Validation Threshold: {info['validation_threshold']}")

    print("\nAgents in workflow:")
    for agent in info['agents']:
        print(f"  - {agent['id']}: {agent['name']} - {agent['description']}")

    print("\nWorkflow info test completed!")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("MULTI-DATA-AGENT TEST SUITE")
    print("="*70)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set. LLM tests will fail.")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'")

    try:
        # Test 1: Schema tools (no LLM needed)
        test_schema_tools()

        # Test 2: Agent factory (no LLM needed)
        test_agent_via_factory()

        # Test 3: Workflow info (no LLM needed)
        test_workflow_info()

        # Test 4: Multi-agent orchestrator (requires LLM)
        if os.environ.get("OPENAI_API_KEY"):
            test_multi_agent_orchestrator()
        else:
            print("\nSkipping orchestrator test (no API key)")

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
