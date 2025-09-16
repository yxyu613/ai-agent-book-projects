#!/usr/bin/env python3
"""
Test script for User Memory System
"""

import os
import sys
import json
import tempfile
import shutil
from dotenv import load_dotenv
from agent import UserMemoryAgent
from config import Config, MemoryMode
from memory_manager import NotesMemoryManager, JSONMemoryManager, MemoryNote
from conversation_history import ConversationHistory, ConversationTurn
from locomo_benchmark import LOCOMOBenchmark

# Load environment variables
load_dotenv()


def test_notes_memory_manager():
    """Test notes-based memory manager"""
    print("\n" + "="*60)
    print("TEST: Notes Memory Manager")
    print("="*60)
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        Config.MEMORY_STORAGE_DIR = temp_dir
        
        # Create manager
        manager = NotesMemoryManager("test_user")
        
        # Test adding notes
        note_id1 = manager.add_memory(
            content="User prefers Python",
            session_id="session-001",
            tags=["preference", "language"]
        )
        print(f"✅ Added note 1: {note_id1}")
        
        note_id2 = manager.add_memory(
            content="User works at TechCorp",
            session_id="session-001",
            tags=["work", "company"]
        )
        print(f"✅ Added note 2: {note_id2}")
        
        # Test context string
        context = manager.get_context_string()
        assert "Python" in context
        assert "TechCorp" in context
        print("✅ Context string contains expected content")
        
        # Test search
        results = manager.search_memories("Python")
        assert len(results) == 1
        assert results[0].content == "User prefers Python"
        print("✅ Search found expected note")
        
        # Test update
        manager.update_memory(
            memory_id=note_id1,
            content="User prefers Python and JavaScript",
            session_id="session-002",
            tags=["preference", "language", "updated"]
        )
        context = manager.get_context_string()
        assert "JavaScript" in context
        print("✅ Note updated successfully")
        
        # Test delete
        manager.delete_memory(note_id2)
        context = manager.get_context_string()
        assert "TechCorp" not in context
        print("✅ Note deleted successfully")
        
        # Test persistence
        manager.save_memory()
        manager2 = NotesMemoryManager("test_user")
        context2 = manager2.get_context_string()
        assert "JavaScript" in context2
        print("✅ Memory persisted and loaded correctly")
    
    print("✅ All notes memory tests passed!")
    return True


def test_json_memory_manager():
    """Test JSON cards memory manager"""
    print("\n" + "="*60)
    print("TEST: JSON Memory Manager")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        Config.MEMORY_STORAGE_DIR = temp_dir
        
        # Create manager
        manager = JSONMemoryManager("test_user")
        
        # Test adding cards
        card_id1 = manager.add_memory(
            content={
                'category': 'personal',
                'subcategory': 'info',
                'key': 'name',
                'value': 'Alice'
            },
            session_id="session-001"
        )
        print(f"✅ Added card 1: {card_id1}")
        
        card_id2 = manager.add_memory(
            content={
                'category': 'preferences',
                'subcategory': 'tech',
                'key': 'language',
                'value': 'Python'
            },
            session_id="session-001"
        )
        print(f"✅ Added card 2: {card_id2}")
        
        # Test context string
        context = manager.get_context_string()
        assert "Alice" in context
        assert "Python" in context
        print("✅ Context contains expected values")
        
        # Test search
        results = manager.search_memories("Alice")
        assert len(results) > 0
        print("✅ Search found expected cards")
        
        # Test update
        manager.update_memory(
            memory_id=card_id1,
            content={'value': 'Alice Smith'},
            session_id="session-002"
        )
        context = manager.get_context_string()
        assert "Alice Smith" in context
        print("✅ Card updated successfully")
        
        # Test delete
        manager.delete_memory(card_id2)
        context = manager.get_context_string()
        assert "Python" not in context
        print("✅ Card deleted successfully")
        
        # Test persistence
        manager.save_memory()
        manager2 = JSONMemoryManager("test_user")
        context2 = manager2.get_context_string()
        assert "Alice Smith" in context2
        print("✅ Memory persisted and loaded correctly")
    
    print("✅ All JSON memory tests passed!")
    return True


def test_conversation_history():
    """Test conversation history management"""
    print("\n" + "="*60)
    print("TEST: Conversation History")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        Config.CONVERSATION_HISTORY_DIR = temp_dir
        
        # Create history manager
        history = ConversationHistory("test_user")
        
        # Add turns
        history.add_turn(
            session_id="session-001",
            user_message="Hello, I'm Bob",
            assistant_message="Nice to meet you, Bob!"
        )
        print("✅ Added conversation turn 1")
        
        history.add_turn(
            session_id="session-001",
            user_message="What's the weather?",
            assistant_message="I don't have access to real-time weather data."
        )
        print("✅ Added conversation turn 2")
        
        # Test retrieval
        recent = history.get_recent_turns(limit=1)
        assert len(recent) == 1
        assert "weather" in recent[0].user_message
        print("✅ Retrieved recent turns")
        
        # Test session retrieval
        session_turns = history.get_session_turns("session-001")
        assert len(session_turns) == 2
        print("✅ Retrieved session turns")
        
        # Test search
        results = history.search_history("Bob")
        assert len(results) > 0
        assert "Bob" in results[0].user_message or "Bob" in results[0].assistant_message
        print("✅ Search found expected conversations")
        
        # Test persistence
        history.save_history()
        history2 = ConversationHistory("test_user")
        assert len(history2.conversations) == 2
        print("✅ History persisted and loaded correctly")
    
    print("✅ All conversation history tests passed!")
    return True


def test_agent_basic():
    """Test basic agent functionality"""
    print("\n" + "="*60)
    print("TEST: Agent Basic Functions")
    print("="*60)
    
    # Check API key
    if not Config.MOONSHOT_API_KEY:
        print("⚠️  Skipping agent test - MOONSHOT_API_KEY not set")
        return None
    
    with tempfile.TemporaryDirectory() as temp_dir:
        Config.MEMORY_STORAGE_DIR = temp_dir
        Config.CONVERSATION_HISTORY_DIR = temp_dir
        
        try:
            # Create agent
            agent = UserMemoryAgent(
                user_id="test_user",
                memory_mode=MemoryMode.NOTES,
                enable_streaming=False
            )
            print("✅ Agent created successfully")
            
            # Start session
            session_id = agent.start_session()
            assert session_id.startswith("session-")
            print(f"✅ Session started: {session_id}")
            
            # Test chat (simple math to avoid rate limits)
            response = agent.chat("What is 2 + 2?")
            assert "4" in response or "four" in response.lower()
            print("✅ Basic chat working")
            
            # Test memory summary
            summary = agent.get_memory_summary()
            assert summary is not None
            print("✅ Memory summary retrieved")
            
            print("✅ All agent tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Agent test failed: {e}")
            return False


def test_locomo_sample():
    """Test LOCOMO benchmark with a sample test"""
    print("\n" + "="*60)
    print("TEST: LOCOMO Benchmark Sample")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        Config.LOCOMO_DATASET_PATH = temp_dir
        Config.LOCOMO_OUTPUT_DIR = temp_dir
        
        # Create benchmark
        benchmark = LOCOMOBenchmark()
        
        # Check if test cases loaded
        assert len(benchmark.test_cases) > 0
        print(f"✅ Loaded {len(benchmark.test_cases)} test cases")
        
        # Get first test case
        test_case = benchmark.test_cases[0]
        print(f"✅ Sample test: {test_case.get('test_id')}")
        
        print("✅ LOCOMO benchmark initialized successfully!")
        return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("USER MEMORY SYSTEM - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test memory managers
    results.append(("Notes Memory Manager", test_notes_memory_manager()))
    results.append(("JSON Memory Manager", test_json_memory_manager()))
    
    # Test conversation history
    results.append(("Conversation History", test_conversation_history()))
    
    # Test LOCOMO
    results.append(("LOCOMO Benchmark", test_locomo_sample()))
    
    # Test agent (only if API key available)
    agent_result = test_agent_basic()
    if agent_result is not None:
        results.append(("Agent Basic Functions", agent_result))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        if result is None:
            status = "⚠️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed or skipped")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
