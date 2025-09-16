#!/usr/bin/env python3
"""
Test script for the updated User Memory Agent with React Pattern
"""

import os
import sys
import logging
from agent import UserMemoryAgent, UserMemoryConfig
from config import Config, MemoryMode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_memory_operations():
    """Test basic memory operations with React pattern"""
    print("\n" + "="*60)
    print("TEST: Basic Memory Operations with React Pattern")
    print("="*60)
    
    # Check API key
    if not Config.MOONSHOT_API_KEY:
        print("❌ Error: MOONSHOT_API_KEY not set in environment")
        print("Please set it in .env file or export it")
        return False
    
    try:
        # Create agent with React pattern
        config = UserMemoryConfig(
            enable_memory_updates=True,
            enable_conversation_history=True,
            enable_memory_search=True,
            memory_mode=MemoryMode.NOTES,
            save_trajectory=True
        )
        
        agent = UserMemoryAgent(
            user_id="test_user",
            config=config,
            verbose=True
        )
        
        print(f"✅ Agent initialized successfully")
        print(f"   Model: {agent.model}")
        print(f"   Memory Mode: {config.memory_mode.value}")
        print(f"   Session: {agent.session_id}")
        
        # Test 1: Add a memory
        print("\n📝 Test 1: Adding a memory")
        print("-"*40)
        result = agent.execute_task(
            "Please remember that I'm a Python developer who loves machine learning and uses PyTorch.",
            max_iterations=10
        )
        
        if result.get('success'):
            print("✅ Memory added successfully")
            print(f"   Tool calls: {len(result.get('tool_calls', []))}")
        else:
            print("❌ Failed to add memory")
            return False
        
        # Test 2: Query memories
        print("\n📝 Test 2: Querying memories")
        print("-"*40)
        result = agent.execute_task(
            "What do you remember about my programming preferences?",
            max_iterations=10
        )
        
        if result.get('success') and result.get('final_answer'):
            print("✅ Memory query successful")
            print(f"   Response: {result['final_answer'][:200]}...")
        else:
            print("❌ Failed to query memories")
            return False
        
        # Test 3: Search memories
        print("\n📝 Test 3: Searching memories")
        print("-"*40)
        result = agent.execute_task(
            "Search my memories for information about machine learning",
            max_iterations=10
        )
        
        if result.get('success'):
            print("✅ Memory search successful")
            
            # Check tool calls
            search_calls = [tc for tc in result.get('tool_calls', []) 
                          if tc.tool_name == 'search_memories']
            if search_calls:
                print(f"   Found {len(search_calls)} search operation(s)")
        else:
            print("❌ Failed to search memories")
            return False
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.error("Test error", exc_info=True)
        return False


def test_tool_calling_pattern():
    """Test that the agent follows the React tool-calling pattern"""
    print("\n" + "="*60)
    print("TEST: React Tool-Calling Pattern")
    print("="*60)
    
    if not Config.MOONSHOT_API_KEY:
        print("❌ Error: MOONSHOT_API_KEY not set")
        return False
    
    try:
        config = UserMemoryConfig(
            enable_memory_updates=True,
            enable_memory_search=True,
            memory_mode=MemoryMode.NOTES
        )
        
        agent = UserMemoryAgent(
            user_id="test_react_user",
            config=config,
            verbose=False  # Less verbose for this test
        )
        
        # Execute a task that should trigger tool calls
        result = agent.execute_task(
            "First read my current memories, then add that I work at OpenAI, then search for work-related information.",
            max_iterations=15
        )
        
        if not result.get('success'):
            print("❌ Task execution failed")
            return False
        
        # Analyze tool calls
        tool_calls = result.get('tool_calls', [])
        print(f"\n📊 Tool Call Analysis:")
        print(f"   Total tool calls: {len(tool_calls)}")
        
        tool_sequence = [tc.tool_name for tc in tool_calls]
        print(f"   Tool sequence: {' -> '.join(tool_sequence)}")
        
        # Check for expected tools
        expected_tools = {'read_memories', 'add_memory', 'search_memories'}
        used_tools = set(tool_sequence)
        
        if expected_tools.issubset(used_tools):
            print("✅ All expected tools were called")
        else:
            missing = expected_tools - used_tools
            print(f"⚠️ Missing tools: {missing}")
        
        # Check for errors
        errors = [tc for tc in tool_calls if tc.error]
        if errors:
            print(f"⚠️ {len(errors)} tool calls had errors")
            for err in errors:
                print(f"   - {err.tool_name}: {err.error}")
        else:
            print("✅ No tool errors")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n🧪 TESTING USER MEMORY AGENT WITH REACT PATTERN")
    print("=" * 60)
    
    # Check configuration
    if not Config.validate():
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    # Create directories
    Config.create_directories()
    
    # Run tests
    tests_passed = 0
    tests_total = 2
    
    if test_basic_memory_operations():
        tests_passed += 1
    
    if test_tool_calling_pattern():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 TEST SUMMARY: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("✅ All tests passed! The React pattern implementation is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()

