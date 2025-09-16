#!/usr/bin/env python3
"""
Main entry point for User Memory System with Separated Architecture
Conversational agent handles dialogue, background processor handles memory
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from conversational_agent import ConversationalAgent, ConversationConfig
from background_memory_processor import BackgroundMemoryProcessor, MemoryProcessorConfig
from config import Config, MemoryMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_result(result: dict):
    """Print formatted result"""
    if result.get('success'):
        print("\n✅ Task completed successfully!")
        if result.get('final_answer'):
            print("\n📝 Final Answer:")
            print("-"*40)
            print(result['final_answer'])
    else:
        print("\n❌ Task failed!")
        if result.get('error'):
            print(f"Error: {result['error']}")
    
    print(f"\n📊 Statistics:")
    print(f"  - Iterations: {result.get('iterations', 0)}")
    print(f"  - Tool calls: {len(result.get('tool_calls', []))}")
    
    if result.get('trajectory_file'):
        print(f"\n💾 Trajectory saved to: {result['trajectory_file']}")
    
    # Show tool call summary
    if result.get('tool_calls'):
        print(f"\n🔧 Tool Call Summary:")
        tool_summary = {}
        for call in result['tool_calls']:
            tool_name = call.tool_name
            if tool_name not in tool_summary:
                tool_summary[tool_name] = {
                    'count': 0,
                    'success': 0,
                    'failed': 0
                }
            tool_summary[tool_name]['count'] += 1
            if call.error:
                tool_summary[tool_name]['failed'] += 1
            else:
                tool_summary[tool_name]['success'] += 1
        
        for tool_name, stats in tool_summary.items():
            print(f"  - {tool_name}: {stats['count']} calls "
                  f"({stats['success']} success, {stats['failed']} failed)")
    
    # Show memory state
    if result.get('memory_state'):
        print(f"\n💭 Memory State:")
        print("-"*40)
        memory_preview = result['memory_state'][:500]
        if len(result['memory_state']) > 500:
            memory_preview += "..."
        print(memory_preview)


def execute_single_conversation(user_id: str, message: str, memory_mode: MemoryMode = MemoryMode.NOTES, verbose: bool = False):
    """Have a single conversation with the agent"""
    api_key = Config.MOONSHOT_API_KEY
    if not api_key:
        print("❌ Error: Please set MOONSHOT_API_KEY environment variable")
        print("   export MOONSHOT_API_KEY='your-api-key-here'")
        return None
    
    config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        config=config,
        memory_mode=memory_mode,
        verbose=verbose
    )
    
    print(f"\n💬 User ({user_id}): {message}")
    response = agent.chat(message)
    print(f"🤖 Assistant: {response}")
    return response


def interactive_mode(user_id: str, memory_mode: MemoryMode = MemoryMode.NOTES, 
                    enable_background_processing: bool = True,
                    conversation_interval: int = 1):
    """Run the agent in interactive mode with separated architecture"""
    print_section(f"Interactive Mode - Conversational Agent (User: {user_id})")
    
    api_key = Config.MOONSHOT_API_KEY
    if not api_key:
        print("❌ Error: Please set MOONSHOT_API_KEY environment variable")
        print("   export MOONSHOT_API_KEY='your-api-key-here'")
        return
    
    # Initialize conversational agent
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=False
    )
    
    # Initialize and start background memory processor if enabled
    memory_processor = None
    if enable_background_processing:
        proc_config = MemoryProcessorConfig(
            conversation_interval=conversation_interval,
            min_conversation_turns=1,
            context_window=10,
            update_threshold=0.7,
            enable_auto_processing=True,
            output_operations=True
        )
        
        memory_processor = BackgroundMemoryProcessor(
            user_id=user_id,
            api_key=api_key,
            config=proc_config,
            memory_mode=memory_mode,
            verbose=False
        )
        
        memory_processor.start_background_processing()
        print(f"\n🧠 Background memory processing enabled (every {conversation_interval} conversation{'s' if conversation_interval > 1 else ''})")
    
    print("\n✅ Conversational agent initialized")
    print(f"📦 Memory Mode: {memory_mode.value}")
    print(f"🆔 Session: {agent.get_session_id()}")
    print(f"🔄 Background Processing: {'Enabled' if enable_background_processing else 'Disabled'}")
    if enable_background_processing:
        print(f"📊 Processing Trigger: Every {conversation_interval} conversation{'s' if conversation_interval > 1 else ''}")
    print("\nAvailable commands:")
    print("  'memory'  - Show current memory state")
    print("  'process' - Manually trigger memory processing")
    print("  'reset'   - Start new conversation session")
    print("  'quit'    - Exit interactive mode")
    print("\nOr enter any message to chat.")
    
    conversation_count = 0
    
    while True:
        try:
            print("\n" + "-"*60)
            user_input = input("You > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                if memory_processor:
                    print("\n⏳ Processing final memory updates...")
                    results = memory_processor.process_recent_conversations()
                    print(f"📝 Final processing: {results}")
                    memory_processor.stop_background_processing()
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'memory':
                print("\n💭 Current Memory State:")
                print("-"*40)
                print(agent.memory_manager.get_context_string())
                
            elif user_input.lower() == 'process':
                if memory_processor:
                    print("\n🔄 Manually triggering memory processing...")
                    results = memory_processor.process_recent_conversations()
                    
                    # Display operations
                    operations = results.get('operations', [])
                    if operations:
                        print(f"\n📝 Memory Operations ({len(operations)} total):")
                        for i, op in enumerate(operations, 1):
                            icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
                            print(f"{i}. {icon} {op['action'].upper()}: {op.get('content', op.get('memory_id', 'N/A'))}")
                            if op.get('confidence'):
                                print(f"   Confidence: {op['confidence']:.2%}")
                    else:
                        print("ℹ️ No memory updates needed")
                    
                    summary = results.get('summary', {})
                    print(f"\nSummary: {summary.get('added', 0)} added, {summary.get('updated', 0)} updated, {summary.get('deleted', 0)} deleted")
                else:
                    print("❌ Background processing not enabled")
                    
            elif user_input.lower() == 'reset':
                agent.reset_session()
                print("✅ Started new conversation session")
                conversation_count = 0
                
            else:
                # Have a conversation
                response = agent.chat(user_input)
                print(f"\n🤖 Assistant: {response}")
                conversation_count += 1
                
                # Increment conversation counter in processor
                if memory_processor:
                    memory_processor.increment_conversation_count()
                    
                    # Check if processing will trigger
                    if memory_processor.should_process():
                        print(f"\n[Memory processing triggered after {conversation_interval} conversation{'s' if conversation_interval > 1 else ''}]")
                        # Give a moment for background thread to process
                        time.sleep(2)
                    elif conversation_interval > 1:
                        conversations_until_process = conversation_interval - (conversation_count % conversation_interval)
                        if conversations_until_process < conversation_interval:
                            print(f"\n[Memory processing in {conversations_until_process} more conversation{'s' if conversations_until_process > 1 else ''}]")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
    
    # Cleanup
    if memory_processor:
        memory_processor.stop_background_processing()


def demo_memory_system():
    """Demonstrate the separated memory system architecture"""
    print_section("Demo: Separated Memory Architecture")
    
    api_key = Config.MOONSHOT_API_KEY
    if not api_key:
        print("❌ Please set MOONSHOT_API_KEY environment variable")
        return
    
    # Create test user
    user_id = "demo_user"
    memory_mode = MemoryMode.NOTES
    
    # Initialize conversational agent
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # Initialize background processor
    proc_config = MemoryProcessorConfig(
        conversation_interval=2,  # Process every 2 conversations for demo
        min_conversation_turns=1,
        update_threshold=0.6,
        output_operations=True
    )
    
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        api_key=api_key,
        config=proc_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # Session 1: Have conversations
    print("\n📝 Session 1: Having conversations")
    print("-"*40)
    
    messages = [
        "Hi! My name is Alice and I work as a product manager at TechCorp.",
        "I prefer Python for scripting and use VS Code as my IDE. I also like dark themes.",
        "I'm currently working on a new mobile app project for our company."
    ]
    
    for message in messages:
        print(f"\n👤 User: {message}")
        response = agent.chat(message)
        print(f"🤖 Assistant: {response[:200]}..." if len(response) > 200 else f"🤖 Assistant: {response}")
        time.sleep(1)  # Brief pause between messages
    
    # Process memories
    print("\n\n🔄 Processing conversation for memory updates...")
    print("-"*40)
    
    # Increment conversation count to trigger processing
    for _ in range(len(messages)):
        processor.increment_conversation_count()
    
    # Process conversations
    results = processor.process_recent_conversations()
    
    # Display operations
    operations = results.get('operations', [])
    if operations:
        print(f"\n📝 Memory Operations ({len(operations)} total):")
        for i, op in enumerate(operations, 1):
            icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
            print(f"{i}. {icon} {op['action'].upper()}: {op.get('content', op.get('memory_id', 'N/A'))}")
            print(f"   Confidence: {op.get('confidence', 0):.2%}")
    else:
        print("ℹ️ No memory updates needed")
    
    summary = results.get('summary', {})
    print(f"\n✅ Summary: {summary.get('added', 0)} added, {summary.get('updated', 0)} updated, {summary.get('deleted', 0)} deleted")
    
    # Start new session to test memory persistence
    print("\n\n📝 Session 2: Testing memory persistence")
    print("-"*40)
    
    agent.reset_session()
    
    test_message = "What do you know about me and my work?"
    print(f"\n👤 User: {test_message}")
    response = agent.chat(test_message)
    print(f"🤖 Assistant: {response}")
    
    # Show final memory state
    print("\n\n💭 Final Memory State:")
    print("-"*40)
    print(agent.memory_manager.get_context_string())


def run_benchmark():
    """Run a simple benchmark of the separated architecture"""
    print_section("Benchmarking Separated Memory Architecture")
    
    api_key = Config.MOONSHOT_API_KEY
    if not api_key:
        print("❌ Please set MOONSHOT_API_KEY environment variable")
        return
    
    test_cases = [
        ("user1", ["I love hiking and photography", "I also enjoy reading sci-fi books"]),
        ("user2", ["I'm a software engineer", "I work with React and Node.js"]),
    ]
    
    results = []
    
    for user_id, messages in test_cases:
        print(f"\n📝 Testing User: {user_id}")
        
        # Create agent and processor
        agent = ConversationalAgent(
            user_id=user_id,
            api_key=api_key,
            memory_mode=MemoryMode.NOTES,
            verbose=False
        )
        
        processor = BackgroundMemoryProcessor(
            user_id=user_id,
            api_key=api_key,
            memory_mode=MemoryMode.NOTES,
            verbose=False
        )
        
        # Have conversations
        for msg in messages:
            print(f"  💬 {msg}")
            response = agent.chat(msg)
        
        # Process memories
        conversation_context = agent.get_conversation_context()
        updates = processor.analyze_conversation(conversation_context)
        process_results = processor.apply_memory_updates(updates)
        
        # Test memory recall
        test_response = agent.chat("What do you know about me?")
        
        results.append({
            'user_id': user_id,
            'messages_sent': len(messages),
            'memory_updates': len(updates),
            'memories_added': process_results['added'],
            'response_includes_memory': any(
                keyword in test_response.lower() 
                for keyword in ['hiking', 'photography', 'reading', 'engineer', 'react', 'node']
            )
        })
    
    print("\n📊 Benchmark Results:")
    print("-"*60)
    for r in results:
        status = "✅" if r['response_includes_memory'] else "⚠️"
        print(f"{status} User: {r['user_id']}, Messages: {r['messages_sent']}, "
              f"Updates: {r['memory_updates']}, Added: {r['memories_added']}, "
              f"Memory Recall: {'Yes' if r['response_includes_memory'] else 'No'}")


def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description="User Memory Agent with React Pattern - Following system-hint architecture"
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "interactive", "demo", "benchmark"],
        default="interactive",
        help="Execution mode (default: interactive)"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="default_user",
        help="User ID for memory system (default: default_user)"
    )
    
    parser.add_argument(
        "--message",
        type=str,
        help="Message to send (for single mode)"
    )
    
    parser.add_argument(
        "--background-processing",
        type=bool,
        default=True,
        help="Enable background memory processing (default: True)"
    )
    
    parser.add_argument(
        "--conversation-interval",
        type=int,
        default=1,
        help="Process memory after N conversations (default: 1 - every conversation)"
    )
    
    parser.add_argument(
        "--memory-mode",
        choices=["notes", "json_cards"],
        default="notes",
        help="Memory mode (default: notes)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    if not Config.validate():
        sys.exit(1)
    
    # Create necessary directories
    Config.create_directories()
    
    # Configure based on command-line flags
    memory_mode = MemoryMode.NOTES if args.memory_mode == "notes" else MemoryMode.JSON_CARDS
    
    print("\n" + "🧠"*40)
    print("  USER MEMORY SYSTEM - SEPARATED ARCHITECTURE")
    print("🧠"*40)
    
    if args.mode == "single":
        if not args.message:
            print("❌ Error: --message required for single mode")
            print("Example: python main.py --mode single --user alice --message 'My name is Alice'")
            sys.exit(1)
        
        response = execute_single_conversation(args.user, args.message, memory_mode, verbose=args.verbose)
        if response:
            print(f"\n✅ Conversation completed")
    
    elif args.mode == "demo":
        demo_memory_system()
    
    elif args.mode == "benchmark":
        run_benchmark()
    
    else:  # interactive mode
        interactive_mode(
            user_id=args.user,
            memory_mode=memory_mode,
            enable_background_processing=args.background_processing,
            conversation_interval=args.conversation_interval
        )
    
    print("\n👋 Thank you for using User Memory Agent!")


if __name__ == "__main__":
    main()