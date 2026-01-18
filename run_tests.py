"""
Automated Test Runner for the Agentic AI Framework
===================================================

This script runs the test examples and validates the agent responses.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --category simple  # Run only simple tests
    python run_tests.py --agent math_specialist  # Run tests for specific agent
    python run_tests.py --interactive      # Interactive mode
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_examples import (
    SIMPLE_EXAMPLES,
    MEDIUM_EXAMPLES,
    COMPLEX_EXAMPLES,
    EDGE_CASES,
    CONVERSATIONAL_EXAMPLES,
    SQL_SIMPLE_EXAMPLES,
    SQL_MEDIUM_EXAMPLES,
    SQL_COMPLEX_EXAMPLES,
    NAME_MATCHING_SIMPLE_EXAMPLES,
    NAME_MATCHING_MEDIUM_EXAMPLES,
    NAME_MATCHING_COMPLEX_EXAMPLES,
    WEALTH_SIMPLE_EXAMPLES,
    WEALTH_MEDIUM_EXAMPLES,
    WEALTH_COMPLEX_EXAMPLES,
    get_examples_by_agent,
    get_examples_by_tool
)


@dataclass
class TestResult:
    """Result of a single test execution."""
    name: str
    agent: str
    query: str
    success: bool
    response: str
    error: Optional[str]
    execution_time: float
    tools_used: List[str]
    iterations: int


class TestRunner:
    """Automated test runner for the agentic framework."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.results: List[TestResult] = []
        
        if not self.api_key:
            print("[WARN] OPENAI_API_KEY not set. Tests will fail.")
            print("   Set it with: export OPENAI_API_KEY='your-key'")
    
    def setup(self):
        """Initialize the framework components."""
        from core.config import FrameworkConfig, ModelConfig
        from agents.agent_definitions import create_agent

        self.config = FrameworkConfig(
            model=ModelConfig(
                model_name="gpt-4o-mini",
                temperature=0.7,
                api_key=self.api_key
            ),
            max_iterations=10
        )

        # Import all tools to register them (imports duckdb, sql, name_matching tools)
        from tools import get_all_tools, get_sql_tools, get_name_matching_tools, get_all_duckdb_tools
        self.tools = get_all_tools()

        # Count all registered tools
        from core.tools_base import tool_registry
        all_tools = tool_registry.get_all_tools()
        print(f"[OK] Framework initialized with {len(all_tools)} tools")
    
    def run_single_test(self, example: Dict[str, Any]) -> TestResult:
        """Run a single test example."""
        from agents.agent_definitions import create_agent
        from core.config import FrameworkConfig, ModelConfig
        # Ensure all tools are registered
        from tools import get_all_tools, get_sql_tools, get_name_matching_tools, get_all_duckdb_tools

        name = example['name']
        agent_name = example['agent']
        query = example['query']
        
        print(f"\n{'='*60}")
        print(f"[TEST] Running: {name}")
        print(f"   Agent: {agent_name}")
        print(f"   Query: {query[:80]}{'...' if len(query) > 80 else ''}")
        
        start_time = time.time()
        
        try:
            # Create agent with config
            config = FrameworkConfig(
                model=ModelConfig(
                    model_name="gpt-4o-mini",
                    temperature=0.7,
                    api_key=self.api_key
                ),
                max_iterations=10
            )
            
            orchestrator = create_agent(agent_name, config)
            
            if orchestrator is None:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            # Run the agent
            result = orchestrator.run(query)
            
            execution_time = time.time() - start_time
            
            # Extract response
            messages = result.get("messages", [])
            response = ""
            tools_used = []
            
            for msg in messages:
                msg_type = type(msg).__name__
                if msg_type == "AIMessage":
                    if msg.content:
                        response = msg.content
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append(tc['name'])
            
            iterations = result.get("iteration_count", 0)
            error = result.get("error")
            
            success = error is None and len(response) > 0
            
            test_result = TestResult(
                name=name,
                agent=agent_name,
                query=query,
                success=success,
                response=response[:500] if response else "",
                error=error,
                execution_time=execution_time,
                tools_used=list(set(tools_used)),
                iterations=iterations
            )
            
            # Print result
            if success:
                print(f"   [PASS] ({execution_time:.2f}s, {iterations} iterations)")
                print(f"   Tools used: {', '.join(test_result.tools_used) or 'None'}")
                print(f"   Response: {response[:100]}...")
            else:
                print(f"   [FAIL] ({execution_time:.2f}s)")
                print(f"   Error: {error}")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   [ERROR] {str(e)}")
            
            return TestResult(
                name=name,
                agent=agent_name,
                query=query,
                success=False,
                response="",
                error=str(e),
                execution_time=execution_time,
                tools_used=[],
                iterations=0
            )
    
    def run_category(self, category: str) -> List[TestResult]:
        """Run all tests in a category."""
        categories = {
            'simple': SIMPLE_EXAMPLES,
            'medium': MEDIUM_EXAMPLES,
            'complex': COMPLEX_EXAMPLES,
            'edge': EDGE_CASES,
            'conversational': CONVERSATIONAL_EXAMPLES,
            'sql_simple': SQL_SIMPLE_EXAMPLES,
            'sql_medium': SQL_MEDIUM_EXAMPLES,
            'sql_complex': SQL_COMPLEX_EXAMPLES,
            'sql': SQL_SIMPLE_EXAMPLES + SQL_MEDIUM_EXAMPLES + SQL_COMPLEX_EXAMPLES,
            'name_simple': NAME_MATCHING_SIMPLE_EXAMPLES,
            'name_medium': NAME_MATCHING_MEDIUM_EXAMPLES,
            'name_complex': NAME_MATCHING_COMPLEX_EXAMPLES,
            'name_matching': NAME_MATCHING_SIMPLE_EXAMPLES + NAME_MATCHING_MEDIUM_EXAMPLES + NAME_MATCHING_COMPLEX_EXAMPLES,
            'wealth_simple': WEALTH_SIMPLE_EXAMPLES,
            'wealth_medium': WEALTH_MEDIUM_EXAMPLES,
            'wealth_complex': WEALTH_COMPLEX_EXAMPLES,
            'wealth': WEALTH_SIMPLE_EXAMPLES + WEALTH_MEDIUM_EXAMPLES + WEALTH_COMPLEX_EXAMPLES,
        }
        
        if category not in categories:
            print(f"[ERROR] Unknown category: {category}")
            print(f"   Available: {', '.join(categories.keys())}")
            return []
        
        examples = categories[category]
        print(f"\n{'='*60}")
        print(f"📁 Running {category.upper()} tests ({len(examples)} examples)")
        print('='*60)
        
        results = []
        for example in examples:
            result = self.run_single_test(example)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_all(self) -> List[TestResult]:
        """Run all test categories."""
        all_results = []

        for category in ['simple', 'medium', 'complex', 'edge', 'conversational',
                         'sql_simple', 'sql_medium', 'sql_complex',
                         'name_simple', 'name_medium', 'name_complex',
                         'wealth_simple', 'wealth_medium', 'wealth_complex']:
            results = self.run_category(category)
            all_results.extend(results)

        return all_results
    
    def run_for_agent(self, agent_name: str) -> List[TestResult]:
        """Run all tests for a specific agent."""
        examples = get_examples_by_agent(agent_name)
        
        if not examples:
            print(f"[ERROR] No examples found for agent: {agent_name}")
            return []
        
        print(f"\n{'='*60}")
        print(f"🤖 Running tests for agent: {agent_name} ({len(examples)} examples)")
        print('='*60)
        
        results = []
        for example in examples:
            result = self.run_single_test(example)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def print_summary(self):
        """Print test summary."""
        if not self.results:
            print("\n[WARN] No tests were run.")
            return
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        
        avg_time = sum(r.execution_time for r in self.results) / total
        total_time = sum(r.execution_time for r in self.results)
        
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print('='*60)
        print(f"   Total Tests:  {total}")
        print(f"   ✅ Passed:    {passed} ({passed/total*100:.1f}%)")
        print(f"   ❌ Failed:    {failed} ({failed/total*100:.1f}%)")
        print(f"   ⏱️  Avg Time:  {avg_time:.2f}s")
        print(f"   ⏱️  Total:    {total_time:.2f}s")
        
        if failed > 0:
            print(f"\n{'─'*60}")
            print("Failed Tests:")
            for r in self.results:
                if not r.success:
                    print(f"   • {r.name}: {r.error}")
        
        # Tool usage stats
        all_tools = []
        for r in self.results:
            all_tools.extend(r.tools_used)
        
        if all_tools:
            tool_counts = {}
            for tool in all_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            print(f"\n{'─'*60}")
            print("Tool Usage:")
            for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"   • {tool}: {count} times")
    
    def export_results(self, filename: str = "test_results.json"):
        """Export results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "results": [
                {
                    "name": r.name,
                    "agent": r.agent,
                    "query": r.query,
                    "success": r.success,
                    "response": r.response,
                    "error": r.error,
                    "execution_time": r.execution_time,
                    "tools_used": r.tools_used,
                    "iterations": r.iterations
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📄 Results exported to {filename}")


def interactive_mode(runner: TestRunner):
    """Run tests in interactive mode."""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE TEST MODE")
    print("="*60)
    print("""
Commands:
  simple, medium, complex, edge, conversational - Run category
  agent <name>  - Run tests for specific agent
  all           - Run all tests
  list          - List all examples
  summary       - Show current results summary
  export        - Export results to JSON
  quit          - Exit
    """)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd in ['quit', 'exit', 'q']:
                break
            elif cmd in ['simple', 'medium', 'complex', 'edge', 'conversational']:
                runner.run_category(cmd)
            elif cmd.startswith('agent '):
                agent_name = cmd.split(' ', 1)[1]
                runner.run_for_agent(agent_name)
            elif cmd == 'all':
                runner.run_all()
            elif cmd == 'list':
                from test_examples import print_examples
                print_examples()
            elif cmd == 'summary':
                runner.print_summary()
            elif cmd == 'export':
                runner.export_results()
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Run agentic framework tests")
    parser.add_argument('--category', '-c',
                        choices=['simple', 'medium', 'complex', 'edge', 'conversational',
                                 'sql', 'sql_simple', 'sql_medium', 'sql_complex',
                                 'name_matching', 'name_simple', 'name_medium', 'name_complex',
                                 'wealth', 'wealth_simple', 'wealth_medium', 'wealth_complex', 'all'],
                        help='Test category to run')
    parser.add_argument('--agent', '-a', help='Run tests for specific agent')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--export', '-e', action='store_true', help='Export results to JSON')
    parser.add_argument('--list', '-l', action='store_true', help='List all examples')
    
    args = parser.parse_args()
    
    if args.list:
        from test_examples import print_examples
        print_examples()
        return
    
    runner = TestRunner()
    runner.setup()
    
    if args.interactive:
        interactive_mode(runner)
    elif args.agent:
        runner.run_for_agent(args.agent)
        runner.print_summary()
    elif args.category:
        if args.category == 'all':
            runner.run_all()
        else:
            runner.run_category(args.category)
        runner.print_summary()
    else:
        # Default: run simple examples
        print("\n💡 Tip: Use --help to see all options")
        print("    Running SIMPLE examples by default...\n")
        runner.run_category('simple')
        runner.print_summary()
    
    if args.export:
        runner.export_results()


if __name__ == "__main__":
    main()
