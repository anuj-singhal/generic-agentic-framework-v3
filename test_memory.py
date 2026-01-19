"""
Test file for Short-term and Long-term Memory functionality.
Tests memory across general_assistant, name_matcher, and data_agent.

Usage:
    python test_memory.py

This script simulates conversations to test:
1. Short-term memory (last 5 conversations with sliding window)
2. Long-term memory (automatic summarization every 5 conversations)
3. Memory persistence across agent switches
4. Follow-up questions using memory context
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import FrameworkConfig, ModelConfig
from agents.agent_definitions import create_agent, get_available_agents
from core.token_counter import get_token_counter
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Management (simplified version for testing)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """Manages short-term and long-term memory for testing."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
        self.token_counter = get_token_counter(model_name)

        # Short-term memory (last 5 conversations)
        self.short_term_memory: List[Dict] = []
        self.short_term_tokens = 0

        # Long-term memory (summaries)
        self.long_term_memory: List[Dict] = []
        self.long_term_tokens = 0

        # Conversation counter
        self.total_conversation_count = 0

    def get_next_threshold(self) -> int:
        """Get the next summarization threshold."""
        return ((self.total_conversation_count // 5) + 1) * 5

    def calculate_short_term_tokens(self) -> int:
        """Calculate tokens in short-term memory."""
        total = 0
        for conv in self.short_term_memory:
            total += self.token_counter.count_text(conv.get("query", ""))
            total += self.token_counter.count_text(conv.get("response", ""))
        return total

    def calculate_long_term_tokens(self) -> int:
        """Calculate tokens in long-term memory."""
        total = 0
        for mem in self.long_term_memory:
            total += self.token_counter.count_text(mem.get("summary", ""))
        return total

    def summarize_conversations(self):
        """Summarize short-term memory and store in long-term memory."""
        if not self.short_term_memory:
            return

        print("\n" + "="*60)
        print("🔄 TRIGGERING LONG-TERM MEMORY SUMMARIZATION")
        print("="*60)

        # Build conversation text
        conversations_text = []
        for i, conv in enumerate(self.short_term_memory, 1):
            conversations_text.append(f"Conversation {i} (Agent: {conv.get('agent', 'unknown')}):")
            conversations_text.append(f"User: {conv.get('query', '')}")
            conversations_text.append(f"Assistant: {conv.get('response', '')[:300]}...")
            conversations_text.append("")

        conversations_str = "\n".join(conversations_text)

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes conversations. Create a concise summary capturing key topics, questions, and important information."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these {len(self.short_term_memory)} conversations:\n\n{conversations_str}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            summary = response.choices[0].message.content

            # Store in long-term memory
            start_conv = self.total_conversation_count - len(self.short_term_memory) + 1
            end_conv = self.total_conversation_count

            long_term_entry = {
                "summary": summary,
                "conversation_range": f"{start_conv}-{end_conv}",
                "tokens": self.token_counter.count_text(summary),
                "timestamp": datetime.now().isoformat()
            }

            self.long_term_memory.append(long_term_entry)
            self.long_term_tokens = self.calculate_long_term_tokens()

            print(f"✅ Created summary for conversations {start_conv}-{end_conv}")
            print(f"📝 Summary: {summary[:200]}...")
            print(f"🔢 Long-term memory now has {len(self.long_term_memory)} summaries")

            # Clear short-term memory after summarization
            self.short_term_memory = []
            self.short_term_tokens = 0

        except Exception as e:
            print(f"❌ Error summarizing: {e}")

    def store_conversation(self, query: str, response: str, agent: str):
        """Store a conversation in memory."""
        self.total_conversation_count += 1

        conversation = {
            "query": query,
            "response": response,
            "agent": agent,
            "timestamp": datetime.now().isoformat()
        }

        self.short_term_memory.append(conversation)

        # Check if we need to summarize (every 5 conversations)
        if self.total_conversation_count > 0 and self.total_conversation_count % 5 == 0:
            self.summarize_conversations()

        # Sliding window - keep max 5 in short-term
        while len(self.short_term_memory) > 5:
            self.short_term_memory.pop(0)

        self.short_term_tokens = self.calculate_short_term_tokens()

    def build_memory_context(self) -> str:
        """Build context string from both memories."""
        memory_parts = []

        # Long-term memory first
        if self.long_term_memory:
            memory_parts.append("[LONG-TERM MEMORY - Historical summaries]\n")
            for i, mem in enumerate(self.long_term_memory, 1):
                memory_parts.append(f"--- Summary {i} (Conversations {mem.get('conversation_range')}) ---")
                memory_parts.append(mem.get("summary", ""))
                memory_parts.append("")
            memory_parts.append("[END OF LONG-TERM MEMORY]\n")

        # Short-term memory
        if self.short_term_memory:
            memory_parts.append("[SHORT-TERM MEMORY - Recent conversations]\n")
            for i, conv in enumerate(self.short_term_memory, 1):
                memory_parts.append(f"--- Conversation {i} (Agent: {conv.get('agent')}) ---")
                memory_parts.append(f"User: {conv.get('query')}")
                response = conv.get('response', '')[:500]
                memory_parts.append(f"Assistant: {response}...")
                memory_parts.append("")
            memory_parts.append("[END OF SHORT-TERM MEMORY]\n")

        return "\n".join(memory_parts) if memory_parts else ""

    def print_status(self):
        """Print current memory status."""
        print("\n" + "-"*60)
        print("📊 MEMORY STATUS")
        print("-"*60)
        print(f"Total Conversations: {self.total_conversation_count}/{self.get_next_threshold()}")
        print(f"Short-term Memory: {len(self.short_term_memory)} conversations ({self.short_term_tokens:,} tokens)")
        print(f"Long-term Memory: {len(self.long_term_memory)} summaries ({self.long_term_tokens:,} tokens)")
        print("-"*60)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

# Test conversations for different agents
TEST_CONVERSATIONS = {
    "general_assistant": [
        # Conversation 1
        {"query": "What is the capital of France?", "expected_topic": "geography"},
        # Conversation 2 - Follow-up
        {"query": "What about Germany?", "expected_topic": "geography follow-up"},
        # Conversation 3
        {"query": "Calculate 25 * 4 + 100", "expected_topic": "math"},
        # Conversation 4 - Follow-up
        {"query": "Now divide that result by 5", "expected_topic": "math follow-up"},
        # Conversation 5
        {"query": "What is today's date?", "expected_topic": "datetime"},
    ],

    "data_agent": [
        # Conversation 6
        {"query": "Show me all tables in the database", "expected_topic": "schema exploration"},
        # Conversation 7
        {"query": "How many clients are in the database?", "expected_topic": "client count"},
        # Conversation 8 - Follow-up
        {"query": "Show me their names and risk profiles", "expected_topic": "client details follow-up"},
        # Conversation 9
        {"query": "What is the total portfolio value?", "expected_topic": "portfolio value"},
        # Conversation 10 - Follow-up (this triggers summarization)
        {"query": "Which client has the highest portfolio value?", "expected_topic": "top client"},
    ],

    "general_assistant_2": [
        # Conversation 11 - Tests long-term memory
        {"query": "Earlier we discussed some capitals. Can you remind me what we talked about?", "expected_topic": "memory recall"},
        # Conversation 12
        {"query": "What programming languages are best for data science?", "expected_topic": "programming"},
        # Conversation 13 - Follow-up
        {"query": "Which one would you recommend for beginners?", "expected_topic": "programming follow-up"},
        # Conversation 14
        {"query": "How do I install Python?", "expected_topic": "installation"},
        # Conversation 15 - Follow-up (triggers another summarization)
        {"query": "What IDE should I use with it?", "expected_topic": "IDE recommendation"},
    ],
}


def run_conversation(agent_name: str, query: str, memory_manager: MemoryManager, config: FrameworkConfig) -> str:
    """Run a single conversation with an agent."""
    print(f"\n{'='*60}")
    print(f"🤖 Agent: {agent_name}")
    print(f"❓ Query: {query}")
    print("="*60)

    # Build memory context
    memory_context = memory_manager.build_memory_context()

    if memory_context:
        enhanced_query = f"{memory_context}\n[CURRENT QUESTION]\n{query}"
        print("📚 Memory context injected into query")
    else:
        enhanced_query = query
        print("📭 No memory context available")

    # Create and run agent
    try:
        orchestrator = create_agent(agent_name, config)
        if orchestrator is None:
            return f"Agent '{agent_name}' not found"

        # Run the agent
        result = orchestrator.invoke(enhanced_query)

        # Extract final answer
        messages = result.get("messages", [])
        response = ""
        for msg in reversed(messages):
            if type(msg).__name__ == "AIMessage" and msg.content:
                response = msg.content
                break

        if not response:
            response = "No response generated"

        print(f"\n✅ Response: {response[:500]}{'...' if len(response) > 500 else ''}")

        return response

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


def run_memory_tests():
    """Run all memory tests."""
    print("\n" + "="*80)
    print("🧪 MEMORY SYSTEM TEST")
    print("Testing Short-term and Long-term Memory across Multiple Agents")
    print("="*80)

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize memory manager
    memory_manager = MemoryManager(api_key=api_key)

    # Create config
    config = FrameworkConfig(
        model=ModelConfig(
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key
        ),
        max_iterations=10
    )

    # Print available agents
    print("\n📋 Available Agents:")
    for name, desc in get_available_agents().items():
        print(f"  - {name}: {desc[:50]}...")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: General Assistant (Conversations 1-5)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("📌 PHASE 1: General Assistant (Conversations 1-5)")
    print("Testing basic memory and follow-up questions")
    print("="*80)

    for i, conv in enumerate(TEST_CONVERSATIONS["general_assistant"], 1):
        print(f"\n🔢 Conversation {memory_manager.total_conversation_count + 1}")
        response = run_conversation("general_assistant", conv["query"], memory_manager, config)
        memory_manager.store_conversation(conv["query"], response, "general_assistant")
        memory_manager.print_status()

        # Pause for user to see output
        input("\nPress Enter to continue...")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Data Agent (Conversations 6-10)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("📌 PHASE 2: Data Agent (Conversations 6-10)")
    print("Testing memory across agent switch + triggering first summarization")
    print("="*80)

    for i, conv in enumerate(TEST_CONVERSATIONS["data_agent"], 1):
        print(f"\n🔢 Conversation {memory_manager.total_conversation_count + 1}")
        response = run_conversation("data_agent", conv["query"], memory_manager, config)
        memory_manager.store_conversation(conv["query"], response, "data_agent")
        memory_manager.print_status()

        input("\nPress Enter to continue...")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: General Assistant Again (Conversations 11-15)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("📌 PHASE 3: General Assistant (Conversations 11-15)")
    print("Testing long-term memory recall + second summarization")
    print("="*80)

    for i, conv in enumerate(TEST_CONVERSATIONS["general_assistant_2"], 1):
        print(f"\n🔢 Conversation {memory_manager.total_conversation_count + 1}")
        response = run_conversation("general_assistant", conv["query"], memory_manager, config)
        memory_manager.store_conversation(conv["query"], response, "general_assistant")
        memory_manager.print_status()

        input("\nPress Enter to continue...")

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("📊 FINAL MEMORY STATUS")
    print("="*80)
    memory_manager.print_status()

    print("\n📚 Long-term Memory Contents:")
    for i, mem in enumerate(memory_manager.long_term_memory, 1):
        print(f"\n  Summary {i} (Conversations {mem['conversation_range']}):")
        print(f"  {mem['summary'][:300]}...")

    print("\n🧠 Short-term Memory Contents:")
    for i, conv in enumerate(memory_manager.short_term_memory, 1):
        print(f"\n  Conversation {i} (Agent: {conv['agent']}):")
        print(f"  Q: {conv['query'][:100]}...")
        print(f"  A: {conv['response'][:100]}...")

    print("\n" + "="*80)
    print("✅ MEMORY TEST COMPLETE")
    print("="*80)


def run_quick_test():
    """Run a quick test with fewer conversations."""
    print("\n" + "="*80)
    print("🧪 QUICK MEMORY TEST")
    print("="*80)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY not set")
        return

    memory_manager = MemoryManager(api_key=api_key)

    config = FrameworkConfig(
        model=ModelConfig(model_name="gpt-4o-mini", temperature=0.7, api_key=api_key),
        max_iterations=10
    )

    quick_tests = [
        ("general_assistant", "What is 2 + 2?"),
        ("general_assistant", "Now multiply that by 10"),
        ("general_assistant", "What is the square root of that?"),
        ("data_agent", "Show me all tables"),
        ("data_agent", "How many clients are there?"),
        ("general_assistant", "Earlier we did some math. What was the final result?"),
    ]

    for agent, query in quick_tests:
        print(f"\n🔢 Conversation {memory_manager.total_conversation_count + 1}")
        response = run_conversation(agent, query, memory_manager, config)
        memory_manager.store_conversation(query, response, agent)
        memory_manager.print_status()

    print("\n✅ Quick test complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test memory functionality")
    parser.add_argument("--quick", action="store_true", help="Run quick test (6 conversations)")
    parser.add_argument("--full", action="store_true", help="Run full test (15 conversations)")

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.full:
        run_memory_tests()
    else:
        print("Memory System Test")
        print("-" * 40)
        print("Usage:")
        print("  python test_memory.py --quick    # Quick test (6 conversations)")
        print("  python test_memory.py --full     # Full test (15 conversations)")
        print()
        print("Make sure OPENAI_API_KEY is set in your environment.")
