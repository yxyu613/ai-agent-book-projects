#!/usr/bin/env python3
"""
Test script for the separated architecture
Demonstrates conversation and background memory processing
"""

import time
import os
import sys
from conversational_agent import ConversationalAgent, ConversationConfig
from background_memory_processor import BackgroundMemoryProcessor, MemoryProcessorConfig
from config import Config, MemoryMode
from memory_manager import create_memory_manager


def test_separated_flow():
    """Test the separated conversation and memory flow"""
    
    # Check API key
    if not Config.MOONSHOT_API_KEY:
        print("❌ Please set MOONSHOT_API_KEY environment variable")
        return
    
    user_id = "test_separated_user"
    memory_mode = MemoryMode.NOTES
    
    print("="*60)
    print("TESTING SEPARATED ARCHITECTURE")
    print("="*60)
    
    # Step 1: Initialize conversational agent
    print("\n1️⃣ Initializing Conversational Agent...")
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True,
        temperature=0.7
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=Config.MOONSHOT_API_KEY,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=True
    )
    print("✅ Conversational agent ready")
    
    # Step 2: Initialize background processor
    print("\n2️⃣ Initializing Background Memory Processor...")
    proc_config = MemoryProcessorConfig(
        process_interval=20,  # Process every 20 seconds for testing
        min_conversation_turns=2,
        context_window=10,
        update_threshold=0.6,
        enable_auto_processing=False  # Manual control for testing
    )
    
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        api_key=Config.MOONSHOT_API_KEY,
        config=proc_config,
        memory_mode=memory_mode,
        verbose=True
    )
    print("✅ Memory processor ready")
    
    # Step 3: Have a conversation
    print("\n3️⃣ Starting Conversation...")
    print("-"*40)
    
    test_messages = [
        "Hi! I'm Bob and I work as a data scientist at DataCorp.",
        "I specialize in machine learning and deep learning. My favorite frameworks are PyTorch and TensorFlow.",
        "I'm currently working on a computer vision project for autonomous vehicles.",
        "Can you summarize what you know about me so far?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Turn {i}]")
        print(f"👤 User: {message}")
        
        response = agent.chat(message)
        print(f"🤖 Assistant: {response[:200]}..." if len(response) > 200 else f"🤖 Assistant: {response}")
        
        time.sleep(1)  # Small delay between messages
    
    # Step 4: Check memory before processing
    print("\n4️⃣ Current Memory State (Before Processing):")
    print("-"*40)
    memory_manager = create_memory_manager(user_id, memory_mode)
    current_memory = memory_manager.get_context_string()
    print(current_memory if current_memory else "No memories yet")
    
    # Step 5: Process conversation for memory updates
    print("\n5️⃣ Processing Conversation for Memory Updates...")
    print("-"*40)
    
    # Get conversation context
    conversation_context = agent.get_conversation_context()
    print(f"📊 Analyzing {len(conversation_context)} messages...")
    
    # Analyze for updates
    updates = processor.analyze_conversation(conversation_context)
    print(f"🔍 Found {len(updates)} potential memory updates:")
    
    for update in updates:
        print(f"\n  • Action: {update.action}")
        if update.content:
            print(f"    Content: {update.content[:100]}...")
        if update.reason:
            print(f"    Reason: {update.reason}")
        print(f"    Confidence: {update.confidence:.2f}")
        if update.tags:
            print(f"    Tags: {', '.join(update.tags)}")
    
    # Apply updates
    if updates:
        print("\n📝 Applying memory updates...")
        results = processor.apply_memory_updates(updates)
        print(f"✅ Results: {results['added']} added, {results['updated']} updated, {results['deleted']} deleted")
    else:
        print("\nℹ️ No updates to apply")
    
    # Step 6: Check memory after processing
    print("\n6️⃣ Updated Memory State (After Processing):")
    print("-"*40)
    memory_manager = create_memory_manager(user_id, memory_mode)
    updated_memory = memory_manager.get_context_string()
    print(updated_memory if updated_memory else "No memories")
    
    # Step 7: Test memory recall in new session
    print("\n7️⃣ Testing Memory Recall in New Session...")
    print("-"*40)
    
    agent.reset_session()
    print("🔄 Started new conversation session")
    
    test_recall_messages = [
        "What do you know about my work?",
        "What are my areas of expertise?"
    ]
    
    for message in test_recall_messages:
        print(f"\n👤 User: {message}")
        response = agent.chat(message)
        print(f"🤖 Assistant: {response}")
    
    # Step 8: Test background processing
    print("\n8️⃣ Testing Background Processing Mode...")
    print("-"*40)
    
    processor.start_background_processing()
    print("🔄 Background processing started")
    
    # Have more conversation
    print("\n💬 Having more conversation while background processes...")
    more_messages = [
        "I forgot to mention, I also have experience with cloud platforms like AWS and GCP.",
        "I enjoy hiking and photography in my free time."
    ]
    
    for message in more_messages:
        print(f"\n👤 User: {message}")
        response = agent.chat(message)
        print(f"🤖 Assistant: {response[:150]}...")
    
    print(f"\n⏳ Waiting {proc_config.process_interval} seconds for background processing...")
    time.sleep(proc_config.process_interval + 2)
    
    # Check final memory state
    print("\n9️⃣ Final Memory State:")
    print("-"*40)
    memory_manager = create_memory_manager(user_id, memory_mode)
    final_memory = memory_manager.get_context_string()
    print(final_memory if final_memory else "No memories")
    
    # Stop background processing
    processor.stop_background_processing()
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("="*60)


def test_manual_vs_background():
    """Compare manual vs background memory processing"""
    
    if not Config.MOONSHOT_API_KEY:
        print("❌ Please set MOONSHOT_API_KEY environment variable")
        return
    
    print("="*60)
    print("COMPARING MANUAL VS BACKGROUND PROCESSING")
    print("="*60)
    
    # Test manual processing
    print("\n📋 MANUAL PROCESSING TEST")
    print("-"*40)
    
    user_id = "manual_test_user"
    agent = ConversationalAgent(user_id=user_id, memory_mode=MemoryMode.NOTES)
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        memory_mode=MemoryMode.NOTES,
        config=MemoryProcessorConfig(enable_auto_processing=False)
    )
    
    # Have conversation
    messages = [
        "I'm Sarah and I work at FinTech Solutions.",
        "I'm a financial analyst specializing in risk assessment."
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        response = agent.chat(msg)
    
    # Manual processing
    print("\nManually processing...")
    results = processor.process_recent_conversations()
    print(f"Results: {results}")
    
    # Test background processing
    print("\n\n🔄 BACKGROUND PROCESSING TEST")
    print("-"*40)
    
    user_id = "background_test_user"
    agent = ConversationalAgent(user_id=user_id, memory_mode=MemoryMode.NOTES)
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        memory_mode=MemoryMode.NOTES,
        config=MemoryProcessorConfig(
            process_interval=10,
            enable_auto_processing=True
        )
    )
    
    processor.start_background_processing()
    
    # Have conversation
    for msg in messages:
        print(f"User: {msg}")
        response = agent.chat(msg)
    
    print("\nWaiting for background processing (10s)...")
    time.sleep(12)
    
    # Check memory
    memory_manager = create_memory_manager(user_id, MemoryMode.NOTES)
    print(f"Memory after background processing: {memory_manager.get_context_string()}")
    
    processor.stop_background_processing()
    
    print("\n✅ Comparison test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test separated architecture")
    parser.add_argument(
        "--test",
        choices=["flow", "comparison", "all"],
        default="flow",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    if args.test == "flow":
        test_separated_flow()
    elif args.test == "comparison":
        test_manual_vs_background()
    else:  # all
        test_separated_flow()
        print("\n" + "="*60 + "\n")
        test_manual_vs_background()
