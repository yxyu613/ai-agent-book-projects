"""
Main entry point for Context-Aware Agent
"""

import os
import sys
import argparse
import logging
from agent import ContextAwareAgent, ContextMode
from ablation_tests import AblationTestSuite
import json
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_task(api_key: str, task: str, context_mode: str = "full", provider: str = "siliconflow", model: str = None):
    """
    Run a single task with the agent
    
    Args:
        api_key: API key for the LLM provider
        task: Task description
        context_mode: Context mode to use
        provider: LLM provider to use
        model: Optional model override
    """
    # Parse context mode
    mode_map = {
        "full": ContextMode.FULL,
        "no_history": ContextMode.NO_HISTORY,
        "no_reasoning": ContextMode.NO_REASONING,
        "no_tool_calls": ContextMode.NO_TOOL_CALLS,
        "no_tool_results": ContextMode.NO_TOOL_RESULTS
    }
    
    if context_mode not in mode_map:
        logger.error(f"Invalid context mode: {context_mode}")
        logger.info(f"Valid modes: {', '.join(mode_map.keys())}")
        return
    
    # Create agent
    agent = ContextAwareAgent(api_key, mode_map[context_mode], provider=provider, model=model)
    
    logger.info(f"Running task with context mode: {context_mode}")
    logger.info(f"Task: {task[:100]}...")
    
    # Execute task
    result = agent.execute_task(task)
    
    # Print results
    print("\n" + "="*60)
    print("TASK EXECUTION RESULT")
    print("="*60)
    print(f"Context Mode: {context_mode}")
    print(f"Success: {result.get('success', False)}")
    print(f"Iterations: {result.get('iterations', 0)}")
    print(f"Tool Calls: {len(result['trajectory'].tool_calls)}")
    
    if result.get('final_answer'):
        print(f"\nFinal Answer:")
        print("-"*40)
        print(result['final_answer'])
    
    if result.get('error'):
        print(f"\nError: {result['error']}")
    
    # Save detailed results
    output_file = f"task_result_{context_mode}.json"
    with open(output_file, 'w') as f:
        # Convert trajectory to serializable format
        serializable_result = {
            "success": result.get("success", False),
            "iterations": result.get("iterations", 0),
            "final_answer": result.get("final_answer"),
            "error": result.get("error"),
            "context_mode": context_mode,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "timestamp": tc.timestamp
                }
                for tc in result["trajectory"].tool_calls
            ],
            "reasoning_steps": result["trajectory"].reasoning_steps
        }
        json.dump(serializable_result, f, indent=2)
    
    logger.info(f"Detailed results saved to {output_file}")


def ensure_sample_pdfs():
    """
    Ensure sample PDFs exist, create them if they don't
    
    Returns:
        bool: True if PDFs are available
    """
    pdf_dir = Path("test_pdfs")
    sample_pdf = pdf_dir / "simple_expense_report.pdf"
    
    if not pdf_dir.exists() or not sample_pdf.exists():
        print("\n📚 Sample PDFs not found. Creating them...")
        try:
            # Run the PDF creation script
            result = subprocess.run(
                [sys.executable, "create_sample_pdf.py"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✅ Sample PDFs created successfully in test_pdfs/")
                return True
            else:
                print(f"⚠️ Could not create PDFs: {result.stderr}")
                return False
        except Exception as e:
            print(f"⚠️ Error creating PDFs: {str(e)}")
            return False
    return True


def get_sample_tasks():
    """
    Get sample tasks for testing
    
    Returns:
        list: List of sample task dictionaries
    """
    # Check if we're running locally or need to use online PDFs
    local_pdfs = Path("test_pdfs").exists()
    
    if local_pdfs:
        pdf_path = "file://" + str(Path.cwd() / "test_pdfs" / "simple_expense_report.pdf")
        pdf_note = "Using local PDF"
    else:
        # Use a publicly available PDF for testing
        pdf_path = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_note = "Using online sample PDF"
    
    return [
        {
            "name": "Simple Currency Conversion",
            "description": "Basic currency conversion with calculations",
            "task": """Convert $5,000 USD to EUR, GBP, and JPY. 
Then calculate the average of all three amounts when converted back to USD.
Finally, calculate what 15% of the average would be."""
        },
        {
            "name": "Multi-Currency Budget Analysis",
            "description": "Analyze expenses across multiple currencies",
            "task": """A company has these Q1 2024 expenses:
- US Office: $1,500,000 USD
- UK Office: £900,000 GBP
- Japan Office: ¥200,000,000 JPY
- EU Office: €1,100,000 EUR

Convert all to USD, calculate the total, and determine what percentage 
each office represents of the total budget."""
        },
        {
            "name": "PDF Financial Analysis",
            "description": f"Parse PDF and perform analysis ({pdf_note})",
            "task": f"""Analyze the financial document at: {pdf_path}

Extract any monetary values mentioned in the document.
If there are expenses in multiple currencies, convert them all to USD.
Calculate the total and provide a summary of the financial data found."""
        },
        {
            "name": "Investment Growth Calculation",
            "description": "Complex financial calculation with currency conversion",
            "task": """I have €50,000 EUR to invest. 
1. Convert this to USD
2. Calculate the value after 3 years with 7% annual compound interest
3. Convert the final amount back to EUR, GBP, and JPY
4. Calculate the total gain in EUR and as a percentage"""
        },
        {
            "name": "Comprehensive Financial Report",
            "description": "Complete analysis requiring all tools",
            "task": f"""Perform a comprehensive analysis:
1. Parse this PDF: {pdf_path}
2. Assume the company has additional expenses of:
   - Marketing: $250,000 USD
   - R&D: €180,000 EUR
   - Operations: ¥50,000,000 JPY
3. Convert all amounts to USD
4. Calculate the total expenses
5. If we need to reduce costs by 12%, how much would each department's budget be?
6. What would the new total be in EUR?"""
        }
    ]


def run_ablation_study(api_key: str, provider: str = "siliconflow", model: str = None):
    """
    Run the full ablation study
    
    Args:
        api_key: API key for the LLM provider
        provider: LLM provider to use
        model: Optional model override
    """
    logger.info(f"Starting ablation study with provider: {provider}")
    
    # Create test suite
    test_suite = AblationTestSuite(api_key, provider=provider, model=model)
    
    # Run study
    results = test_suite.run_ablation_study()
    
    # Analyze and report
    analysis = test_suite.analyze_results(results)
    test_suite.print_results_table(results)
    
    try:
        test_suite.visualize_results(results)
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    
    # Generate report
    report = test_suite.generate_report(results, analysis)
    with open("ablation_study_report.md", "w") as f:
        f.write(report)
    
    logger.info("Ablation study complete!")


def interactive_mode(api_key: str, provider: str = "siliconflow", model: str = None):
    """
    Run the agent in interactive mode
    
    Args:
        api_key: API key for the LLM provider
        provider: LLM provider to use
        model: Optional model override
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Context-Aware Agent")
    print(f"Provider: {provider.upper()} | Model: {model or 'default'}")
    print("="*60)
    print("Available commands:")
    print("  - Type your task/question")
    print("  - 'samples' to see sample tasks")
    print("  - 'sample <number>' to run a sample task")
    print("  - 'mode <mode_name>' to change context mode")
    print("  - 'modes' to list available context modes")
    print("  - 'create_pdfs' to create sample PDFs")
    print("  - 'reset' to reset agent trajectory")
    print("  - 'help' for command help")
    print("  - 'quit' or 'exit' to exit")
    print("="*60)
    
    # Check for sample PDFs
    ensure_sample_pdfs()
    
    # Get sample tasks
    sample_tasks = get_sample_tasks()
    
    print(f"\n💡 TIP: Try 'samples' to see {len(sample_tasks)} pre-defined tasks")
    print("         or 'sample 3' to test PDF parsing capabilities\n")
    
    # Initialize agent with full context
    current_mode = ContextMode.FULL
    agent = ContextAwareAgent(api_key, current_mode, provider=provider, model=model)
    
    mode_map = {
        "full": ContextMode.FULL,
        "no_history": ContextMode.NO_HISTORY,
        "no_reasoning": ContextMode.NO_REASONING,
        "no_tool_calls": ContextMode.NO_TOOL_CALLS,
        "no_tool_results": ContextMode.NO_TOOL_RESULTS
    }
    
    while True:
        try:
            user_input = input(f"\n[{current_mode.value}]> ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n📖 HELP - Available Commands:")
                print("-"*40)
                print("samples         - Show all sample tasks")
                print("sample <n>      - Run sample task n (e.g., 'sample 1')")
                print("mode <name>     - Switch context mode")
                print("modes           - List all context modes")
                print("create_pdfs     - Create sample PDF files")
                print("reset           - Reset agent trajectory")
                print("help            - Show this help")
                print("quit/exit       - Exit the program")
                print("-"*40)
                print("\nContext Modes explain what happens when parts are removed:")
                print("  full           - Complete agent with all capabilities")
                print("  no_history     - Agent can't see its previous actions")
                print("  no_reasoning   - Agent doesn't plan before acting")
                print("  no_tool_calls  - Agent can't use any tools")
                print("  no_tool_results - Agent can't see tool outputs")
            
            elif user_input.lower() == 'samples':
                print("\n📋 SAMPLE TASKS:")
                print("="*60)
                for i, sample in enumerate(sample_tasks, 1):
                    print(f"\n{i}. {sample['name']}")
                    print(f"   {sample['description']}")
                    print(f"   Use: 'sample {i}' to run this task")
                print("\n" + "="*60)
            
            elif user_input.lower().startswith('sample '):
                try:
                    sample_num = int(user_input.split()[1])
                    if 1 <= sample_num <= len(sample_tasks):
                        sample = sample_tasks[sample_num - 1]
                        print(f"\n🚀 Running Sample: {sample['name']}")
                        print(f"Description: {sample['description']}")
                        print("-"*40)
                        print("Task:")
                        print(sample['task'])
                        print("-"*40)
                        
                        # Execute the sample task
                        print("\nProcessing...")
                        result = agent.execute_task(sample['task'])
                        
                        print(f"\n{'='*40}")
                        print(f"Success: {result.get('success', False)}")
                        print(f"Iterations: {result.get('iterations', 0)}")
                        print(f"Tool Calls: {len(result['trajectory'].tool_calls)}")
                        
                        if result.get('final_answer'):
                            print(f"\nAnswer:")
                            print(result['final_answer'])
                        
                        if result.get('error'):
                            print(f"\nError: {result['error']}")
                    else:
                        print(f"Invalid sample number. Choose 1-{len(sample_tasks)}")
                except (ValueError, IndexError):
                    print(f"Invalid sample number. Use 'sample 1' through 'sample {len(sample_tasks)}'")
            
            elif user_input.lower() == 'create_pdfs':
                print("\n📄 Creating sample PDFs...")
                try:
                    result = subprocess.run(
                        [sys.executable, "create_sample_pdf.py"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        print("✅ Sample PDFs created successfully in test_pdfs/")
                        # Update sample tasks with new PDF paths
                        sample_tasks = get_sample_tasks()
                    else:
                        print(f"⚠️ Could not create PDFs: {result.stderr}")
                except Exception as e:
                    print(f"⚠️ Error creating PDFs: {str(e)}")
            
            elif user_input.lower() == 'modes':
                print("\n🔧 Available context modes:")
                for mode in mode_map.keys():
                    print(f"  - {mode}")
            
            elif user_input.lower().startswith('mode '):
                new_mode = user_input[5:].strip()
                if new_mode in mode_map:
                    current_mode = mode_map[new_mode]
                    agent = ContextAwareAgent(api_key, current_mode, provider=provider, model=model)
                    print(f"✅ Switched to context mode: {current_mode.value}")
                    if current_mode != ContextMode.FULL:
                        print(f"⚠️ Warning: This mode intentionally disables certain features for testing")
                else:
                    print(f"❌ Invalid mode. Available: {', '.join(mode_map.keys())}")
            
            elif user_input.lower() == 'reset':
                agent.reset()
                print("✅ Agent trajectory reset.")
            
            elif user_input:
                # Execute task
                print("\nProcessing...")
                result = agent.execute_task(user_input)
                
                print(f"\n{'='*40}")
                print(f"Success: {result.get('success', False)}")
                print(f"Iterations: {result.get('iterations', 0)}")
                print(f"Tool Calls: {len(result['trajectory'].tool_calls)}")
                
                if result.get('final_answer'):
                    print(f"\nAnswer:")
                    print(result['final_answer'])
                
                if result.get('error'):
                    print(f"\nError: {result['error']}")
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Context-Aware AI Agent")
    parser.add_argument(
        "--mode",
        choices=["single", "ablation", "interactive"],
        default="interactive",
        help="Execution mode"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to execute (for single mode)"
    )
    parser.add_argument(
        "--context-mode",
        choices=["full", "no_history", "no_reasoning", "no_tool_calls", "no_tool_results"],
        default="full",
        help="Context mode for single task execution"
    )
    parser.add_argument(
        "--provider",
        choices=["siliconflow", "doubao"],
        default="doubao",
        help="LLM provider to use (default: doubao)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (optional, uses provider default if not specified)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the LLM provider (or set SILICONFLOW_API_KEY/ARK_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key based on provider
    if args.api_key:
        api_key = args.api_key
    elif args.provider == "doubao":
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            logger.error("Please provide API key via --api-key or ARK_API_KEY environment variable")
            sys.exit(1)
    elif args.provider == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            logger.error("Please provide API key via --api-key or SILICONFLOW_API_KEY environment variable")
            sys.exit(1)
    else:
        logger.error(f"Unknown provider: {args.provider}")
        sys.exit(1)
    
    # Log provider info
    logger.info(f"Using provider: {args.provider}, model: {args.model or 'default'}")
    
    # Execute based on mode
    if args.mode == "single":
        if not args.task:
            # Prompt user to select a sample task
            print("\n" + "="*60)
            print("SINGLE TASK MODE - No task provided")
            print("="*60)
            
            # Ensure PDFs exist
            ensure_sample_pdfs()
            
            # Get and display sample tasks
            sample_tasks = get_sample_tasks()
            print("\n📋 Available sample tasks:")
            for i, sample in enumerate(sample_tasks, 1):
                print(f"\n{i}. {sample['name']}")
                print(f"   {sample['description']}")
            
            print("\n" + "="*60)
            try:
                choice = input("\nSelect a task number (1-{}) or 'q' to quit: ".format(len(sample_tasks))).strip()
                if choice.lower() == 'q':
                    sys.exit(0)
                
                task_num = int(choice)
                if 1 <= task_num <= len(sample_tasks):
                    selected_task = sample_tasks[task_num - 1]
                    print(f"\n✅ Selected: {selected_task['name']}")
                    print("\nTask details:")
                    print("-"*40)
                    print(selected_task['task'])
                    print("-"*40)
                    
                    confirm = input("\nRun this task? (y/n): ").strip().lower()
                    if confirm == 'y':
                        run_single_task(api_key, selected_task['task'], args.context_mode, 
                                      provider=args.provider, model=args.model)
                    else:
                        print("Task cancelled.")
                else:
                    print(f"Invalid selection. Please choose 1-{len(sample_tasks)}")
                    sys.exit(1)
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                sys.exit(0)
        else:
            run_single_task(api_key, args.task, args.context_mode, 
                          provider=args.provider, model=args.model)
    
    elif args.mode == "ablation":
        run_ablation_study(api_key, provider=args.provider, model=args.model)
    
    else:  # interactive
        interactive_mode(api_key, provider=args.provider, model=args.model)


if __name__ == "__main__":
    main()
