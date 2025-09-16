#!/usr/bin/env python3
"""
Integrated User Memory Agent with Evaluation Framework
This script properly handles the evaluation workflow without import conflicts
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import user-memory components (these are in the current directory)
from agent import UserMemoryAgent, UserMemoryConfig
from config import Config, MemoryMode
from memory_manager import create_memory_manager


class IntegratedEvaluationAgent:
    """Agent that integrates with the evaluation framework without import conflicts"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the integrated evaluation agent"""
        self.api_key = api_key or Config.MOONSHOT_API_KEY
        if not self.api_key:
            raise ValueError("API key required. Set MOONSHOT_API_KEY environment variable.")
        
        self.eval_framework_path = Path(__file__).parent.parent / "user-memory-evaluation"
        self.results = {}
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get test cases from the evaluation framework using subprocess"""
        script = """
import sys
import json
import os
from pathlib import Path
import io
import contextlib

# Capture all output except our final JSON
captured_output = io.StringIO()

# Monkey-patch the console to avoid rich output
import rich.console
original_console = rich.console.Console
rich.console.Console = lambda *args, **kwargs: type('FakeConsole', (), {
    'print': lambda self, *a, **k: None,
    '__getattr__': lambda self, name: lambda *a, **k: None
})()

# Redirect stdout temporarily
old_stdout = sys.stdout
sys.stdout = captured_output

try:
    from framework import UserMemoryEvaluationFramework
    
    framework = UserMemoryEvaluationFramework()
    test_cases = []
    
    for tc in framework.list_test_cases():
        test_cases.append({
            'test_id': tc.test_id,
            'category': tc.category,
            'title': tc.title,
            'description': tc.description,
            'num_conversations': len(tc.conversation_histories),
            'user_question': tc.user_question
        })
    
    # Restore stdout for JSON output
    sys.stdout = old_stdout
    print(json.dumps(test_cases))
    
except Exception as e:
    # Restore stdout and output empty list on error
    sys.stdout = old_stdout
    print(json.dumps([]))
finally:
    # Restore console
    rich.console.Console = original_console
"""
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.eval_framework_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error getting test cases: {result.stderr}")
            return []
        
        if not result.stdout:
            print(f"Warning: No output from subprocess. stderr: {result.stderr}")
            return []
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"stdout: {result.stdout[:500]}")
            print(f"stderr: {result.stderr[:500]}")
            return []
    
    def get_test_case_details(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed test case including conversation histories"""
        script = f"""
import sys
import json
from pathlib import Path

from framework import UserMemoryEvaluationFramework
from models import MessageRole

framework = UserMemoryEvaluationFramework()
tc = framework.get_test_case("{test_id}")

if not tc:
    print(json.dumps(None))
else:
    details = {{
        'test_id': tc.test_id,
        'category': tc.category,
        'title': tc.title,
        'description': tc.description,
        'user_question': tc.user_question,
        'expected_behavior': tc.expected_behavior,
        'conversation_histories': []
    }}
    
    for conv in tc.conversation_histories:
        conv_data = {{
            'conversation_id': conv.conversation_id,
            'timestamp': conv.timestamp,
            'messages': []
        }}
        
        for msg in conv.messages:
            conv_data['messages'].append({{
                'role': msg.role.value,
                'content': msg.content
            }})
        
        if conv.metadata:
            conv_data['metadata'] = conv.metadata
        
        details['conversation_histories'].append(conv_data)
    
    print(json.dumps(details))
"""
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.eval_framework_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error getting test case details: {result.stderr}")
            return None
        
        return json.loads(result.stdout)
    
    def evaluate_response(self, test_id: str, agent_response: str, extracted_memory: str = "") -> Dict[str, Any]:
        """Evaluate the agent's response using the evaluation framework"""
        # Create temporary files for passing data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'test_id': test_id,
                'agent_response': agent_response,
                'extracted_memory': extracted_memory
            }, f)
            temp_file = f.name
        
        script = f"""
import sys
import json
from pathlib import Path

from framework import UserMemoryEvaluationFramework

# Load input data
with open("{temp_file}", 'r') as f:
    data = json.load(f)

framework = UserMemoryEvaluationFramework()
result = framework.submit_and_evaluate(
    data['test_id'],
    data['agent_response'],
    data['extracted_memory'] if data['extracted_memory'] else None
)

if result:
    output = {{
        'test_id': result.test_id,
        'passed': result.passed,
        'score': result.score,
        'reasoning': result.reasoning,
        'required_info_found': result.required_info_found,
        'suggestions': result.suggestions
    }}
    print(json.dumps(output))
else:
    print(json.dumps(None))
"""
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.eval_framework_path,
            capture_output=True,
            text=True,
            env={**os.environ, 'MOONSHOT_API_KEY': self.api_key}
        )
        
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass
        
        if result.returncode != 0:
            print(f"Error evaluating response: {result.stderr}")
            return {'error': result.stderr}
        
        return json.loads(result.stdout) if result.stdout else {'error': 'No evaluation result'}
    
    def process_test_case(self, test_id: str) -> Dict[str, Any]:
        """Process a complete test case: load, process histories, answer, evaluate"""
        print(f"\n{'='*60}")
        print(f"Processing Test Case: {test_id}")
        print('='*60)
        
        # Get test case details
        test_case = self.get_test_case_details(test_id)
        if not test_case:
            return {'error': f'Test case {test_id} not found'}
        
        print(f"\nTitle: {test_case['title']}")
        print(f"Category: {test_case['category']}")
        print(f"Conversations: {len(test_case['conversation_histories'])}")
        
        # Initialize agent for this test
        user_id = f"test_{test_id}"
        config = UserMemoryConfig(
            enable_memory_updates=True,
            enable_conversation_history=True,
            enable_memory_search=True,
            memory_mode=MemoryMode.NOTES,
            save_trajectory=False
        )
        
        agent = UserMemoryAgent(
            user_id=user_id,
            api_key=self.api_key,
            config=config,
            verbose=False
        )
        
        # Process conversation histories
        print("\nProcessing conversation histories...")
        total_messages = 0
        
        for conv_idx, conversation in enumerate(test_case['conversation_histories'], 1):
            conv_id = conversation['conversation_id']
            num_messages = len(conversation['messages'])
            print(f"  Conversation {conv_idx}: {conv_id} ({num_messages} messages)")
            
            # Process user messages to extract information
            for msg_idx, message in enumerate(conversation['messages']):
                total_messages += 1
                
                if message['role'] == 'user':
                    # Process every 10th user message to save API calls
                    if msg_idx % 10 == 0:
                        content = message['content'][:500]  # Limit content length
                        agent.execute_task(
                            f"Process and remember this information: {content}",
                            max_iterations=3
                        )
                
                # Show progress
                if total_messages % 50 == 0:
                    print(f"    Processed {total_messages} messages...")
        
        print(f"  Total messages processed: {total_messages}")
        
        # Get extracted memory
        extracted_memory = agent.memory_manager.get_context_string()
        print(f"\nExtracted memory length: {len(extracted_memory)} characters")
        
        # Answer the user question
        user_question = test_case['user_question']
        print(f"\nUser Question: {user_question}")
        
        result = agent.execute_task(user_question, max_iterations=10)
        agent_answer = result.get('final_answer', 'Unable to generate answer')
        
        print(f"\nAgent Answer: {agent_answer[:200]}..." if len(agent_answer) > 200 else f"\nAgent Answer: {agent_answer}")
        
        # Evaluate the response
        print("\nEvaluating response...")
        evaluation = self.evaluate_response(test_id, agent_answer, extracted_memory)
        
        if 'error' not in evaluation:
            status = "✓ PASSED" if evaluation.get('passed') else "✗ FAILED"
            score = evaluation.get('score', 0.0)
            print(f"\nEvaluation Result: {status} (Score: {score:.2f})")
            
            if evaluation.get('reasoning'):
                print(f"Reasoning: {evaluation['reasoning'][:200]}...")
        else:
            print(f"\nEvaluation Error: {evaluation['error']}")
        
        # Clean up agent
        agent.reset()
        
        return {
            'test_id': test_id,
            'agent_answer': agent_answer,
            'extracted_memory': extracted_memory,
            'evaluation': evaluation
        }
    
    def interactive_session(self):
        """Run an interactive evaluation session"""
        print("\n" + "🧠"*30)
        print("  USER MEMORY AGENT - EVALUATION MODE")
        print("🧠"*30)
        print("\nThis mode integrates the User Memory Agent with the Evaluation Framework")
        print("for structured testing using predefined test cases.\n")
        
        # Get available test cases
        test_cases = self.get_test_cases()
        if not test_cases:
            print("Error: Could not load test cases from evaluation framework")
            return
        
        print(f"Loaded {len(test_cases)} test cases")
        
        # Group by category
        categories = {}
        for tc in test_cases:
            cat = tc['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tc)
        
        while True:
            print("\n" + "-"*60)
            print("Options:")
            print("1. List test cases by category")
            print("2. Run a specific test case")
            print("3. Run all tests in a category")
            print("4. View results summary")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                for cat in sorted(categories.keys()):
                    print(f"\n{cat.upper()}: {len(categories[cat])} test cases")
                    for tc in categories[cat][:3]:
                        print(f"  - {tc['test_id']}: {tc['title'][:50]}...")
                    if len(categories[cat]) > 3:
                        print(f"  ... and {len(categories[cat])-3} more")
            
            elif choice == "2":
                test_id = input("\nEnter test case ID: ").strip()
                if any(tc['test_id'] == test_id for tc in test_cases):
                    result = self.process_test_case(test_id)
                    self.results[test_id] = result
                else:
                    print(f"Test case '{test_id}' not found")
            
            elif choice == "3":
                print("\nAvailable categories:", ', '.join(sorted(categories.keys())))
                cat = input("Enter category: ").strip()
                
                if cat in categories:
                    print(f"\nRunning {len(categories[cat])} tests in {cat}...")
                    for tc in categories[cat]:
                        result = self.process_test_case(tc['test_id'])
                        self.results[tc['test_id']] = result
                else:
                    print(f"Category '{cat}' not found")
            
            elif choice == "4":
                if not self.results:
                    print("\nNo results yet. Run some tests first!")
                else:
                    print(f"\n{'='*60}")
                    print("RESULTS SUMMARY")
                    print('='*60)
                    
                    passed = sum(1 for r in self.results.values() 
                               if 'evaluation' in r and r['evaluation'].get('passed'))
                    total = len(self.results)
                    
                    print(f"Total Tests Run: {total}")
                    print(f"Passed: {passed}")
                    print(f"Failed: {total - passed}")
                    print(f"Pass Rate: {100*passed/total:.1f}%" if total > 0 else "N/A")
                    
                    print("\nIndividual Results:")
                    for test_id, result in self.results.items():
                        if 'evaluation' in result and 'error' not in result['evaluation']:
                            status = "✓" if result['evaluation'].get('passed') else "✗"
                            score = result['evaluation'].get('score', 0.0)
                            print(f"  {status} {test_id}: {score:.2f}")
                        else:
                            print(f"  ? {test_id}: Error or incomplete")
            
            elif choice == "5":
                print("\nExiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated User Memory Agent with Evaluation Framework"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "single", "batch"],
        default="interactive",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--test-id",
        type=str,
        help="Test case ID (for single mode)"
    )
    
    parser.add_argument(
        "--category",
        choices=["layer1", "layer2", "layer3"],
        help="Category (for batch mode)"
    )
    
    args = parser.parse_args()
    
    try:
        agent = IntegratedEvaluationAgent()
        
        if args.mode == "interactive":
            agent.interactive_session()
        
        elif args.mode == "single":
            if not args.test_id:
                print("Error: --test-id required for single mode")
                sys.exit(1)
            
            result = agent.process_test_case(args.test_id)
            if result and 'evaluation' in result:
                if 'error' not in result['evaluation']:
                    passed = result['evaluation'].get('passed')
                    score = result['evaluation'].get('score', 0.0)
                    print(f"\nFinal Result: {'PASSED' if passed else 'FAILED'} (Score: {score:.2f})")
        
        elif args.mode == "batch":
            test_cases = agent.get_test_cases()
            
            if args.category:
                test_cases = [tc for tc in test_cases if tc['category'] == args.category]
            
            print(f"Running {len(test_cases)} tests...")
            results = []
            
            for tc in test_cases:
                result = agent.process_test_case(tc['test_id'])
                results.append(result)
            
            # Summary
            passed = sum(1 for r in results 
                        if 'evaluation' in r and r['evaluation'].get('passed'))
            total = len(results)
            
            print(f"\n{'='*60}")
            print(f"Batch Test Complete")
            print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
