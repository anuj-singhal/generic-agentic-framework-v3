"""
Test Script for Complex Query Generation Improvements
=====================================================

Tests the improved multi_data_agent with 4 complex query examples to verify
that CTE generation uses only valid columns from the schema.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

from core.multi_agent_orchestrator import MultiAgentDataOrchestrator

# Define the 4 complex test queries
COMPLEX_QUERIES = [
    {
        "name": "Example 1: Comprehensive Client Summary",
        "query": """Create a comprehensive summary for each client showing:
- Client details (name, country, risk profile, KYC status)
- Number of portfolios owned
- Total assets across all portfolios
- Total portfolio value (sum of all holdings value)
- Most recent transaction date
- Most frequently held asset type
Order by total portfolio value descending"""
    },
    {
        "name": "Example 2: Transaction Pattern Analysis",
        "query": """Analyze transaction patterns:
1. Total transaction volume by month (count of transactions)
2. Total value traded by month (sum of quantity * price)
3. Buy vs Sell ratio for each month
Show results for the most recent 6 months of data."""
    },
    {
        "name": "Example 3: Asset Allocation Report",
        "query": """Create a comprehensive asset allocation report showing:
For client 'Anuj Singhal':
- Asset type breakdown (percentage of total portfolio value by asset type)
- Top 5 holdings by value
- Total portfolio value
- Number of different assets held"""
    },
    {
        "name": "Example 4: Risk Profile Distribution",
        "query": """Analyze the distribution of clients by risk profile and show:
1. Risk profile category
2. Number of clients in each category
3. Number of portfolios managed by clients in each category
4. Average portfolio value per risk profile
Include KYC approved clients only. Order by risk profile."""
    }
]


def run_test(query_info: dict, orchestrator: MultiAgentDataOrchestrator) -> dict:
    """Run a single test query and return results."""
    print(f"\n{'='*70}")
    print(f"TEST: {query_info['name']}")
    print(f"{'='*70}")
    print(f"\nQuery: {query_info['query'][:200]}...")
    print(f"\n{'-'*70}")

    try:
        result = orchestrator.run(query_info['query'])

        # Extract key information
        generated_sql = result.get("generated_sql", "")
        overall_confidence = result.get("overall_confidence", 0)
        validation_results = result.get("validation_results", {})
        final_answer = result.get("final_answer", "")
        error = result.get("error")
        retry_count = result.get("retry_count", 0)

        # Check for schema validation issues
        schema_validation = validation_results.get("schema", {})
        missing_columns = schema_validation.get("columns_missing", [])
        programmatic_check = schema_validation.get("programmatic_check", {})
        prog_missing = programmatic_check.get("missing_columns", [])

        print(f"\n[RESULTS]:")
        print(f"   Confidence: {overall_confidence:.0%}")
        print(f"   Retries: {retry_count}")
        print(f"   Has Error: {bool(error)}")
        print(f"   Missing Columns (LLM): {missing_columns}")
        print(f"   Missing Columns (Programmatic): {prog_missing}")

        print(f"\n[GENERATED SQL]:")
        print(f"   {'-'*60}")
        for line in generated_sql.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*60}")

        if final_answer:
            print(f"\n[OK] FINAL ANSWER (first 500 chars):")
            print(f"   {final_answer[:500]}...")

        if error:
            print(f"\n[ERROR]: {error}")

        return {
            "name": query_info["name"],
            "success": overall_confidence >= 0.75 and not error,
            "confidence": overall_confidence,
            "retries": retry_count,
            "missing_columns": missing_columns + prog_missing,
            "generated_sql": generated_sql,
            "error": error
        }

    except Exception as e:
        print(f"\n[EXCEPTION]: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "name": query_info["name"],
            "success": False,
            "confidence": 0,
            "retries": 0,
            "missing_columns": [],
            "generated_sql": "",
            "error": str(e)
        }


def main():
    """Run all complex query tests."""
    print("=" * 70)
    print("COMPLEX QUERY GENERATION TEST SUITE")
    print("Testing improved multi_data_agent with strict schema validation")
    print("=" * 70)

    # Initialize orchestrator
    print("\nInitializing MultiAgentDataOrchestrator...")
    orchestrator = MultiAgentDataOrchestrator()

    results = []

    for query_info in COMPLEX_QUERIES:
        # Clear cache between tests
        orchestrator.clear_query_cache()

        result = run_test(query_info, orchestrator)
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")
    print()

    for r in results:
        status = "[PASS]" if r["success"] else "[FAIL]"
        print(f"{status} | {r['name']}")
        print(f"       Confidence: {r['confidence']:.0%}, Retries: {r['retries']}")
        if r["missing_columns"]:
            print(f"       Missing columns: {r['missing_columns']}")
        if r["error"]:
            print(f"       Error: {r['error'][:100]}")
        print()

    # Overall assessment
    print("=" * 70)
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED! Complex query generation is working correctly.")
    elif passed >= total * 0.75:
        print("[WARNING] MOSTLY PASSING. Some queries need attention.")
    else:
        print("[FAILED] SIGNIFICANT ISSUES. Review the generated SQL for invalid columns.")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
