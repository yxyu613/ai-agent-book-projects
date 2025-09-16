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

# Add evaluation framework support
# We load it dynamically only when needed to avoid import conflicts
EVALUATION_AVAILABLE = False
UserMemoryEvaluationFramework = None
TestCase = None

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


def run_evaluation_mode(user_id: str, memory_mode: MemoryMode, verbose: bool = False):
    """Run evaluation mode using the evaluation framework"""
    
    # Import the evaluation framework with proper module isolation
    from pathlib import Path
    
    eval_framework_path = Path(__file__).parent.parent / "user-memory-evaluation"
    
    try:
        # Save the current modules to avoid conflicts
        saved_modules = {}
        conflicting_modules = ['config', 'models', 'evaluator', 'framework']
        
        # Temporarily remove conflicting modules from sys.modules
        for module_name in conflicting_modules:
            if module_name in sys.modules:
                saved_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]
        
        # Temporarily add evaluation framework path with highest priority
        original_path = sys.path.copy()
        sys.path.insert(0, str(eval_framework_path))
        
        # Import evaluation framework modules
        import config as eval_config
        import models as eval_models  
        import evaluator as eval_evaluator
        import framework as eval_framework
        
        # Get the class we need
        framework_class = eval_framework.UserMemoryEvaluationFramework
        
        # Restore original path
        sys.path = original_path
        
        # Remove evaluation modules from sys.modules to avoid future conflicts
        for module_name in conflicting_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Restore original modules
        for module_name, module in saved_modules.items():
            sys.modules[module_name] = module
            
    except Exception as e:
        # Restore on error
        sys.path = original_path if 'original_path' in locals() else sys.path
        for module_name, module in saved_modules.items():
            sys.modules[module_name] = module
            
        print(f"❌ Error: Could not load evaluation framework: {e}")
        print("Please ensure user-memory-evaluation is properly installed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print_section("Evaluation Mode - Test Case Based Evaluation")
    
    # Initialize evaluation framework
    framework = framework_class()
    
    if not framework.test_suite:
        print("❌ Error: No test cases loaded")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(framework.test_suite.test_cases)} test cases")
    
    # Initialize agents without incorrect parameters
    # ConversationConfig is a dataclass and doesn't take parameters in __init__
    conv_config = ConversationConfig()
    conv_config.enable_memory_context = True
    conv_config.enable_conversation_history = True
    
    mem_config = MemoryProcessorConfig()
    mem_config.verbose = verbose
    mem_config.memory_mode = memory_mode
    
    # Initialize agents with correct parameters
    agent = ConversationalAgent(
        user_id=user_id,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=verbose
    )
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        config=mem_config
    )
    
    while True:
        print("\n" + "-"*60)
        print("Options:")
        print("1. List test cases by category")
        print("2. Run a specific test case")
        print("3. View current memory state")
        print("4. Clear memory and start fresh")
        print("5. Exit evaluation mode")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # List test cases
            print("\n📋 Available Test Cases:")
            framework.display_test_case_summary(show_full_titles=True, by_category=True)
            
        elif choice == "2":
            # Run a specific test case
            test_id = input("\nEnter test case ID: ").strip()
            test_case = framework.get_test_case(test_id)
            
            if not test_case:
                print(f"❌ Test case '{test_id}' not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running Test Case: {test_case.title}")
            print(f"Category: {test_case.category}")
            print("="*60)
            
            # Clear existing memory and conversation history before test
            print("\n🧹 Clearing previous memory and conversation history...")
            
            # Clear memory
            if hasattr(agent.memory_manager, 'clear_all_memories'):
                agent.memory_manager.clear_all_memories()
            if hasattr(processor.memory_manager, 'clear_all_memories'):
                processor.memory_manager.clear_all_memories()
            
            # Clear conversation history
            if agent.conversation_history:
                agent.conversation_history.conversations = []
                agent.conversation_history.save_history()
                print(f"  🧹 Cleared conversation history for user {user_id}")
            
            # Reset agent conversation
            agent.conversation = []
            agent._init_system_prompt()
            
            # Process conversation histories
            print(f"\n📚 Processing {len(test_case.conversation_histories)} conversation histories...")
            
            # Build conversation contexts from test case histories
            conversation_contexts = []
            
            for i, history in enumerate(test_case.conversation_histories, 1):
                print(f"\nConversation {i}/{len(test_case.conversation_histories)}: {history.conversation_id}")
                
                # Build conversation context for this history
                conversation = []
                
                # Process each message in the conversation
                for msg in history.messages:
                    if msg.role.value == "user":
                        conversation.append({"role": "user", "content": msg.content})
                    elif msg.role.value == "assistant":
                        conversation.append({"role": "assistant", "content": msg.content})
                
                conversation_contexts.append(conversation)
                
                # Also add to the agent's conversation history for context
                # This is needed for the agent to have context when answering the question
                if agent.conversation_history and hasattr(agent.conversation_history, 'add_turn'):
                    # Add pairs of user/assistant messages
                    user_msg = None
                    for msg in history.messages:
                        if msg.role.value == "user":
                            user_msg = msg.content
                        elif msg.role.value == "assistant" and user_msg:
                            agent.conversation_history.add_turn(
                                session_id=f"eval_{history.conversation_id}",
                                user_message=user_msg,
                                assistant_message=msg.content
                            )
                            user_msg = None
            
            # Process all conversations through the memory processor
            if conversation_contexts:
                print(f"\n💾 Processing memory for all conversations...")
                try:
                    results = processor.process_conversation_batch(conversation_contexts)
                    
                    # Summarize results
                    total_added = sum(r.get('summary', {}).get('added', 0) for r in results)
                    total_updated = sum(r.get('summary', {}).get('updated', 0) for r in results)
                    total_deleted = sum(r.get('summary', {}).get('deleted', 0) for r in results)
                    
                    print(f"  ✅ Memory processing complete:")
                    print(f"     - Added: {total_added} memories")
                    print(f"     - Updated: {total_updated} memories")
                    print(f"     - Deleted: {total_deleted} memories")
                except Exception as e:
                    print(f"  ⚠️ Memory processing error: {e}")
            
            # Now answer the user question
            print(f"\n{'='*60}")
            print("USER QUESTION:")
            print("-"*60)
            print(test_case.user_question)
            print("="*60)
            
            # Get agent response
            print("\n🤔 Generating response...")
            response = agent.chat(test_case.user_question)
            
            print("\n📝 Agent Response:")
            print("-"*60)
            print(response)
            print("-"*60)
            
            # Evaluate the response
            print("\n⚖️ Evaluating response...")
            result = framework.submit_and_evaluate(test_id, response)
            
            if result:
                # Display evaluation result
                is_passed = result.passed if result.passed is not None else result.reward >= 0.6
                status = "✅ PASSED" if is_passed else "❌ FAILED"
                
                print(f"\n{'='*60}")
                print("EVALUATION RESULT:")
                print("-"*60)
                print(f"Status: {status}")
                print(f"Reward Score: {result.reward:.3f}/1.000")
                
                if result.reasoning:
                    print(f"\nReasoning:")
                    print(result.reasoning)
                
                if result.suggestions:
                    print(f"\nSuggestions:")
                    print(result.suggestions)
                print("="*60)
            else:
                print("❌ Evaluation failed")
            
            # Clear conversation history for next test
            agent.conversation_history = []
            
        elif choice == "3":
            # View current memory
            print("\n📄 Current Memory State:")
            print("-"*60)
            memories = processor.get_current_memories()
            if memories:
                for mem in memories:
                    print(f"  • {mem}")
            else:
                print("  (No memories stored)")
                
        elif choice == "4":
            # Clear memory
            if input("\n⚠️ Are you sure you want to clear all memory? (yes/no): ").lower() == "yes":
                # Clear memory using the new method
                if hasattr(agent.memory_manager, 'clear_all_memories'):
                    agent.memory_manager.clear_all_memories()
                if hasattr(processor.memory_manager, 'clear_all_memories'):
                    processor.memory_manager.clear_all_memories()
                
                # Clear conversation history
                if agent.conversation_history:
                    agent.conversation_history.conversations = []
                    agent.conversation_history.save_history()
                
                # Reset agent conversation
                agent.conversation = []
                agent._init_system_prompt()
                
                print("✅ Memory and conversation history cleared")
                
        elif choice == "5":
            print("\nExiting evaluation mode...")
            break
        else:
            print(f"❌ Invalid choice: {choice}")


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
        choices=["single", "interactive", "demo", "benchmark", "evaluation"],
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
    
    elif args.mode == "evaluation":
        run_evaluation_mode(args.user, memory_mode, args.verbose)
    
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