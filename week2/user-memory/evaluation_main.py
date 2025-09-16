#!/usr/bin/env python3
"""
User Memory Agent integrated with Evaluation Framework
Allows testing the agent with structured test cases from the evaluation framework
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import user-memory components first (from current directory)
from agent import UserMemoryAgent, UserMemoryConfig
from config import Config, MemoryMode
from memory_manager import create_memory_manager

# Then add evaluation framework to path and import
eval_path = Path(__file__).parent.parent / "user-memory-evaluation"
sys.path.insert(0, str(eval_path))

# Import evaluation framework (avoid config conflict)
import framework as eval_framework
import models as eval_models

# Alias the imports for easier use
UserMemoryEvaluationFramework = eval_framework.UserMemoryEvaluationFramework
TestCase = eval_models.TestCase
ConversationHistory = eval_models.ConversationHistory
ConversationMessage = eval_models.ConversationMessage
MessageRole = eval_models.MessageRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class UserMemoryEvaluationAgent:
    """Agent that integrates with the evaluation framework for structured testing"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the evaluation agent
        
        Args:
            api_key: API key for Kimi/Moonshot (defaults to env)
        """
        self.api_key = api_key or Config.MOONSHOT_API_KEY
        if not self.api_key:
            raise ValueError("API key required. Set MOONSHOT_API_KEY environment variable.")
        
        # Initialize evaluation framework
        test_cases_dir = Path(__file__).parent.parent / "user-memory-evaluation" / "test_cases"
        self.framework = UserMemoryEvaluationFramework(test_cases_dir=test_cases_dir)
        
        # Storage for results
        self.results = {}
        self.current_test_case = None
        self.current_agent = None
        self.extracted_memory = None
    
    def list_test_cases(self, category: Optional[str] = None):
        """Display available test cases"""
        test_cases = self.framework.list_test_cases(category)
        
        table = Table(title="Available Test Cases", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Test ID", style="magenta")
        table.add_column("Title", style="green")
        table.add_column("Conversations", justify="center")
        
        for test_case in test_cases:
            table.add_row(
                test_case.category,
                test_case.test_id,
                test_case.title[:50] + "..." if len(test_case.title) > 50 else test_case.title,
                str(len(test_case.conversation_histories))
            )
        
        console.print(table)
        return test_cases
    
    def process_conversation_histories(self, test_case: TestCase) -> str:
        """
        Process conversation histories to extract and summarize memories
        
        Args:
            test_case: The test case containing conversation histories
            
        Returns:
            Summarized memory string
        """
        console.print("\n[cyan]Processing conversation histories...[/cyan]")
        
        # Create a temporary user ID for this test
        user_id = f"test_{test_case.test_id}"
        
        # Initialize agent with memory features
        config = UserMemoryConfig(
            enable_memory_updates=True,
            enable_conversation_history=True,
            enable_memory_search=True,
            memory_mode=MemoryMode.NOTES,
            save_trajectory=False
        )
        
        self.current_agent = UserMemoryAgent(
            user_id=user_id,
            api_key=self.api_key,
            config=config,
            verbose=False
        )
        
        # Process each conversation history
        total_messages = 0
        for conv_idx, conversation in enumerate(test_case.conversation_histories, 1):
            console.print(f"\n[yellow]Processing Conversation {conv_idx}: {conversation.conversation_id}[/yellow]")
            
            if conversation.metadata:
                business = conversation.metadata.get('business', 'N/A')
                console.print(f"  Business: {business}")
            
            console.print(f"  Messages: {len(conversation.messages)}")
            
            # Process each message in the conversation
            for msg_idx, message in enumerate(conversation.messages):
                total_messages += 1
                
                if message.role == MessageRole.USER:
                    # Process user message to extract and store memories
                    # The agent should extract important information from user messages
                    result = self.current_agent.execute_task(
                        f"Process and remember this information from the user: {message.content}",
                        max_iterations=5
                    )
                    
                    if msg_idx % 10 == 0:  # Progress indicator
                        console.print(f"    Processed {msg_idx}/{len(conversation.messages)} messages...")
        
        console.print(f"\n[green]✓ Processed {total_messages} total messages across {len(test_case.conversation_histories)} conversations[/green]")
        
        # Get the extracted memory
        memory_context = self.current_agent.memory_manager.get_context_string()
        self.extracted_memory = memory_context
        
        console.print("\n[cyan]Extracted Memory Summary:[/cyan]")
        if len(memory_context) > 500:
            console.print(memory_context[:500] + "...")
        else:
            console.print(memory_context)
        
        return memory_context
    
    def answer_question(self, test_case: TestCase, memory_context: str) -> str:
        """
        Answer the user question based on the extracted memory
        
        Args:
            test_case: The test case containing the user question
            memory_context: The extracted memory context
            
        Returns:
            Agent's answer to the question
        """
        console.print(f"\n[cyan]Answering User Question:[/cyan]")
        console.print(f"[yellow]Question:[/yellow] {test_case.user_question}")
        
        if not self.current_agent:
            # Recreate agent if needed
            user_id = f"test_{test_case.test_id}"
            config = UserMemoryConfig(
                enable_memory_updates=False,  # Don't update memory while answering
                enable_conversation_history=True,
                enable_memory_search=True,
                memory_mode=MemoryMode.NOTES,
                save_trajectory=False
            )
            
            self.current_agent = UserMemoryAgent(
                user_id=user_id,
                api_key=self.api_key,
                config=config,
                verbose=False
            )
        
        # Answer the question using the memory context
        result = self.current_agent.execute_task(
            test_case.user_question,
            max_iterations=10
        )
        
        answer = result.get('final_answer', '')
        
        console.print(f"\n[green]Agent's Answer:[/green]")
        console.print(answer)
        
        return answer
    
    def run_test_case(self, test_id: str) -> Dict[str, Any]:
        """
        Run a complete test case: process histories -> answer question -> evaluate
        
        Args:
            test_id: The test case ID to run
            
        Returns:
            Dictionary with test results
        """
        # Get the test case
        test_case = self.framework.get_test_case(test_id)
        if not test_case:
            console.print(f"[red]Test case {test_id} not found![/red]")
            return None
        
        self.current_test_case = test_case
        
        # Display test case info
        console.print(Panel(
            f"[bold cyan]Test Case:[/bold cyan] {test_case.title}\n"
            f"[bold cyan]Category:[/bold cyan] {test_case.category}\n"
            f"[bold cyan]Description:[/bold cyan] {test_case.description}\n"
            f"[bold cyan]Conversations:[/bold cyan] {len(test_case.conversation_histories)}\n"
            f"[bold cyan]Expected Behavior:[/bold cyan] {test_case.expected_behavior}",
            title=f"Test: {test_id}",
            expand=False
        ))
        
        try:
            # Step 1: Process conversation histories
            console.print("\n" + "="*60)
            console.print("[bold]Step 1: Processing Conversation Histories[/bold]")
            console.print("="*60)
            memory_context = self.process_conversation_histories(test_case)
            
            # Step 2: Answer the user question
            console.print("\n" + "="*60)
            console.print("[bold]Step 2: Answering User Question[/bold]")
            console.print("="*60)
            agent_answer = self.answer_question(test_case, memory_context)
            
            # Step 3: Evaluate the response
            console.print("\n" + "="*60)
            console.print("[bold]Step 3: Evaluating Response[/bold]")
            console.print("="*60)
            evaluation_result = self.framework.submit_and_evaluate(
                test_id,
                agent_answer,
                self.extracted_memory
            )
            
            # Display results
            if evaluation_result:
                self.results[test_id] = evaluation_result
                self._display_evaluation_result(evaluation_result)
            
            return {
                'test_id': test_id,
                'agent_answer': agent_answer,
                'extracted_memory': self.extracted_memory,
                'evaluation_result': evaluation_result
            }
            
        except Exception as e:
            console.print(f"[red]Error running test case: {str(e)}[/red]")
            logger.error(f"Error in test case {test_id}: {e}", exc_info=True)
            return None
        
        finally:
            # Clean up agent state
            if self.current_agent:
                self.current_agent.reset()
    
    def _display_evaluation_result(self, result):
        """Display evaluation result in a formatted way"""
        status_color = "green" if result.passed else "red"
        status_text = "✓ PASSED" if result.passed else "✗ FAILED"
        
        console.print(f"\n[{status_color}]{status_text}[/{status_color}]")
        console.print(f"[bold]Score:[/bold] {result.score:.2f}/1.00")
        console.print(f"[bold]Reasoning:[/bold] {result.reasoning}")
        
        if result.required_info_found:
            console.print("\n[bold]Required Information:[/bold]")
            for info, found in result.required_info_found.items():
                check = "✓" if found else "✗"
                color = "green" if found else "red"
                console.print(f"  [{color}]{check}[/{color}] {info}")
        
        if result.suggestions:
            console.print(f"\n[yellow]Suggestions:[/yellow] {result.suggestions}")
    
    def interactive_session(self):
        """Run an interactive evaluation session"""
        console.print("[bold cyan]User Memory Agent - Evaluation Mode[/bold cyan]")
        console.print("="*60)
        console.print("This mode integrates with the evaluation framework for structured testing.\n")
        
        while True:
            console.print("\n[bold]Options:[/bold]")
            console.print("1. List all test cases")
            console.print("2. List test cases by category")
            console.print("3. Run single test case")
            console.print("4. View test case details")
            console.print("5. View results summary")
            console.print("6. Generate evaluation report")
            console.print("7. Exit")
            
            choice = Prompt.ask("Select an option", choices=["1","2","3","4","5","6","7"])
            
            if choice == "1":
                self.list_test_cases()
            
            elif choice == "2":
                category = Prompt.ask(
                    "Select category",
                    choices=["layer1", "layer2", "layer3"]
                )
                self.list_test_cases(category)
            
            elif choice == "3":
                test_id = Prompt.ask("Enter test case ID to run")
                console.print(f"\n[cyan]Running test case: {test_id}[/cyan]")
                result = self.run_test_case(test_id)
                if result:
                    console.print(f"\n[green]Test case {test_id} completed![/green]")
            
            elif choice == "4":
                test_id = Prompt.ask("Enter test case ID to view")
                self.framework.display_test_case_detail(test_id)
            
            elif choice == "5":
                if not self.results:
                    console.print("[yellow]No results available yet. Run some test cases first.[/yellow]")
                else:
                    console.print(f"\n[bold]Results Summary:[/bold]")
                    passed = sum(1 for r in self.results.values() if r.passed)
                    total = len(self.results)
                    console.print(f"Total Tests Run: {total}")
                    console.print(f"Passed: {passed}")
                    console.print(f"Failed: {total - passed}")
                    console.print(f"Pass Rate: {100*passed/total:.1f}%")
                    
                    for test_id, result in self.results.items():
                        status = "✓" if result.passed else "✗"
                        color = "green" if result.passed else "red"
                        console.print(f"  [{color}]{status}[/{color}] {test_id} (Score: {result.score:.2f})")
            
            elif choice == "6":
                if not self.results:
                    console.print("[yellow]No results to report. Run some test cases first.[/yellow]")
                else:
                    output_file = Prompt.ask(
                        "Enter output file path",
                        default="user_memory_evaluation_report.txt"
                    )
                    report = self.framework.generate_report(self.results, output_file)
                    console.print(f"[green]Report saved to {output_file}[/green]")
            
            elif choice == "7":
                if Confirm.ask("Are you sure you want to exit?"):
                    break


def run_single_test(test_id: str, api_key: Optional[str] = None):
    """Run a single test case directly"""
    try:
        agent = UserMemoryEvaluationAgent(api_key)
        result = agent.run_test_case(test_id)
        return result
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None


def run_batch_tests(category: Optional[str] = None, api_key: Optional[str] = None):
    """Run batch tests for a category"""
    try:
        agent = UserMemoryEvaluationAgent(api_key)
        test_cases = agent.framework.list_test_cases(category)
        
        console.print(f"[cyan]Running {len(test_cases)} test cases...[/cyan]")
        
        results = []
        for test_case in test_cases:
            console.print(f"\n[yellow]Running: {test_case.test_id}[/yellow]")
            result = agent.run_test_case(test_case.test_id)
            if result:
                results.append(result)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold]Batch Test Summary[/bold]")
        console.print("="*60)
        
        passed = sum(1 for r in results if r['evaluation_result'] and r['evaluation_result'].passed)
        total = len(results)
        console.print(f"Total: {total}")
        console.print(f"Passed: {passed}")
        console.print(f"Failed: {total - passed}")
        console.print(f"Pass Rate: {100*passed/total:.1f}%")
        
        return results
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return []


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="User Memory Agent with Evaluation Framework Integration"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "single", "batch"],
        default="interactive",
        help="Execution mode (default: interactive)"
    )
    
    parser.add_argument(
        "--test-id",
        type=str,
        help="Test case ID (for single mode)"
    )
    
    parser.add_argument(
        "--category",
        choices=["layer1", "layer2", "layer3"],
        help="Category filter (for batch mode)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Moonshot API key (defaults to MOONSHOT_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    if not Config.validate():
        sys.exit(1)
    
    # Create necessary directories
    Config.create_directories()
    
    console.print("\n" + "🧠"*30)
    console.print("  USER MEMORY AGENT - EVALUATION MODE")
    console.print("🧠"*30 + "\n")
    
    if args.mode == "single":
        if not args.test_id:
            console.print("[red]Error: --test-id required for single mode[/red]")
            sys.exit(1)
        
        result = run_single_test(args.test_id, args.api_key)
        if result:
            console.print("[green]Test completed successfully![/green]")
    
    elif args.mode == "batch":
        results = run_batch_tests(args.category, args.api_key)
        if results:
            console.print(f"[green]Batch testing completed: {len(results)} tests[/green]")
    
    else:  # interactive mode
        try:
            agent = UserMemoryEvaluationAgent(args.api_key)
            agent.interactive_session()
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            logger.error("Fatal error", exc_info=True)
    
    console.print("\n[cyan]Thank you for using User Memory Agent - Evaluation Mode![/cyan]")


if __name__ == "__main__":
    main()
