"""
Ablation Test Cases for Context-Aware Agent
Demonstrates the critical importance of different context components in agent behavior
"""

import os
import json
import logging
from typing import Dict, Any, List
from agent import ContextAwareAgent, ContextMode
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AblationTestSuite:
    """Test suite for exploring context importance through ablation studies"""
    
    def __init__(self, api_key: str, provider: str = "siliconflow", model: str = None):
        """
        Initialize test suite
        
        Args:
            api_key: API key for the LLM provider
            provider: LLM provider to use
            model: Optional model override
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.test_results = []
        
    def create_complex_financial_task(self) -> str:
        """
        Create a complex task that requires multiple tool calls and reasoning
        
        Returns:
            Task description
        """
        return """Analyze the financial report from this PDF: https://www.berkshirehathaway.com/qtrly/1stqtr23.pdf

Please complete the following analysis:
1. Extract the total revenue figures from Q1 2023
2. Convert the revenue from USD to EUR, GBP, and JPY
3. Calculate the following metrics:
   - Average revenue across the three converted currencies
   - Percentage difference between highest and lowest converted values
   - If the company maintains a 15% profit margin, what would be the profit in each currency?

Provide a comprehensive financial summary with all calculations shown."""
    
    def create_multinational_budget_task(self) -> str:
        """
        Create a task requiring multiple currency conversions and calculations
        
        Returns:
            Task description  
        """
        return """A multinational company has the following Q1 2024 expenses documented in this report:
https://github.com/adobe/pdf-services-node-sdk-samples/raw/refs/heads/master/resources/extractPDFInput.pdf

Tasks to complete:
1. Parse the PDF and extract all monetary values mentioned
2. The company operates in 5 regions with expenses in different currencies:
   - US Office: $2,500,000 USD
   - UK Office: £1,800,000 GBP  
   - Japan Office: ¥380,000,000 JPY
   - EU Office: €2,100,000 EUR
   - Singapore Office: $3,200,000 SGD
3. Convert all expenses to USD for consolidation
4. Calculate:
   - Total global expenses in USD
   - Average expense per region
   - What percentage each region represents of total expenses
   - If we apply a 8% cost reduction uniformly, what would be the new expense for each region in their local currency?

Present a detailed financial analysis with all conversions and calculations."""
    
    def run_single_test(self, task: str, context_mode: ContextMode, test_name: str) -> Dict[str, Any]:
        """
        Run a single ablation test
        
        Args:
            task: Task to execute
            context_mode: Context mode to test
            test_name: Name of the test
            
        Returns:
            Test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"Context mode: {context_mode.value}")
        logger.info(f"{'='*60}")
        
        agent = ContextAwareAgent(self.api_key, context_mode, provider=self.provider, model=self.model)
        
        start_time = time.time()
        result = agent.execute_task(task)
        execution_time = time.time() - start_time
        
        # Analyze the result
        test_result = {
            "test_name": test_name,
            "context_mode": context_mode.value,
            "execution_time": round(execution_time, 2),
            "iterations": result.get("iterations", 0),
            "num_tool_calls": len(result["trajectory"].tool_calls),
            "success": result.get("success", False),
            "has_final_answer": result.get("final_answer") is not None,
            "error": result.get("error"),
            "reasoning_steps": len(result["trajectory"].reasoning_steps),
            "final_answer_preview": (result.get("final_answer", "")[:200] + "...") if result.get("final_answer") else None
        }
        
        # Log summary
        logger.info(f"Test completed in {test_result['execution_time']}s")
        logger.info(f"Success: {test_result['success']}")
        logger.info(f"Tool calls made: {test_result['num_tool_calls']}")
        logger.info(f"Iterations: {test_result['iterations']}")
        
        if test_result["error"]:
            logger.error(f"Error occurred: {test_result['error']}")
        
        return test_result
    
    def run_ablation_study(self) -> List[Dict[str, Any]]:
        """
        Run complete ablation study across all context modes
        
        Returns:
            List of test results
        """
        # Use the complex financial task for all tests
        task = self.create_multinational_budget_task()
        
        # Define test configurations
        test_configs = [
            (ContextMode.FULL, "Baseline - Full Context"),
            (ContextMode.NO_HISTORY, "Ablation 1 - No Historical Tool Calls"),
            (ContextMode.NO_REASONING, "Ablation 2 - No Reasoning Process"),
            (ContextMode.NO_TOOL_CALLS, "Ablation 3 - No Tool Call Commands"),
            (ContextMode.NO_TOOL_RESULTS, "Ablation 4 - No Tool Call Results")
        ]
        
        results = []
        for context_mode, test_name in test_configs:
            try:
                result = self.run_single_test(task, context_mode, test_name)
                results.append(result)
                self.test_results.append(result)
                
                # Add delay between tests to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to run test {test_name}: {str(e)}")
                results.append({
                    "test_name": test_name,
                    "context_mode": context_mode.value,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze ablation study results
        
        Args:
            results: List of test results
            
        Returns:
            Analysis summary
        """
        analysis = {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.get("success", False)),
            "context_mode_impact": {}
        }
        
        # Analyze impact of each ablation
        baseline = next((r for r in results if r["context_mode"] == "full"), None)
        
        if baseline:
            for result in results:
                if result["context_mode"] != "full":
                    mode_analysis = {
                        "success_maintained": result.get("success", False),
                        "execution_time_delta": result.get("execution_time", 0) - baseline.get("execution_time", 0),
                        "iteration_delta": result.get("iterations", 0) - baseline.get("iterations", 0),
                        "tool_call_delta": result.get("num_tool_calls", 0) - baseline.get("num_tool_calls", 0),
                        "failure_reason": None
                    }
                    
                    # Identify failure reasons
                    if not result.get("success", False):
                        if result["context_mode"] == "no_tool_calls":
                            mode_analysis["failure_reason"] = "Cannot execute tools without tool call capability"
                        elif result["context_mode"] == "no_tool_results":
                            mode_analysis["failure_reason"] = "Cannot make informed decisions without tool results"
                        elif result["context_mode"] == "no_history":
                            mode_analysis["failure_reason"] = "May repeat actions or lose track of progress"
                        elif result["context_mode"] == "no_reasoning":
                            mode_analysis["failure_reason"] = "Lacks planning and strategic thinking"
                    
                    analysis["context_mode_impact"][result["context_mode"]] = mode_analysis
        
        return analysis
    
    def print_results_table(self, results: List[Dict[str, Any]]):
        """
        Print results in a formatted table
        
        Args:
            results: List of test results
        """
        # Prepare data for tabulation
        table_data = []
        for result in results:
            table_data.append([
                result["test_name"],
                result["context_mode"],
                "✓" if result.get("success", False) else "✗",
                f"{result.get('execution_time', 0)}s",
                result.get("iterations", 0),
                result.get("num_tool_calls", 0),
                result.get("reasoning_steps", 0),
                "Yes" if result.get("has_final_answer", False) else "No"
            ])
        
        headers = ["Test Name", "Context Mode", "Success", "Time", "Iterations", "Tool Calls", "Reasoning Steps", "Final Answer"]
        
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def visualize_results(self, results: List[Dict[str, Any]]):
        """
        Create visualizations of ablation study results
        
        Args:
            results: List of test results
        """
        # Extract data for visualization
        modes = [r["context_mode"] for r in results]
        iterations = [r.get("iterations", 0) for r in results]
        tool_calls = [r.get("num_tool_calls", 0) for r in results]
        exec_times = [r.get("execution_time", 0) for r in results]
        success = [1 if r.get("success", False) else 0 for r in results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Ablation Study: Impact of Context Components", fontsize=16)
        
        # Plot 1: Success Rate
        axes[0, 0].bar(modes, success, color=['green' if s else 'red' for s in success])
        axes[0, 0].set_title("Task Success by Context Mode")
        axes[0, 0].set_ylabel("Success (1) / Failure (0)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Iterations Required
        axes[0, 1].bar(modes, iterations, color='blue')
        axes[0, 1].set_title("Iterations Required")
        axes[0, 1].set_ylabel("Number of Iterations")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Tool Calls Made
        axes[1, 0].bar(modes, tool_calls, color='orange')
        axes[1, 0].set_title("Tool Calls Made")
        axes[1, 0].set_ylabel("Number of Tool Calls")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Execution Time
        axes[1, 1].bar(modes, exec_times, color='purple')
        axes[1, 1].set_title("Execution Time")
        axes[1, 1].set_ylabel("Time (seconds)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("ablation_study_results.png", dpi=150, bbox_inches='tight')
        logger.info("Visualization saved as 'ablation_study_results.png'")
    
    def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report of the ablation study
        
        Args:
            results: List of test results
            analysis: Analysis summary
            
        Returns:
            Report text
        """
        report = """
# Context Ablation Study Report

## Executive Summary
This ablation study explores the critical importance of different context components in AI agent behavior.

## Test Configuration
- **Model**: Qwen/Qwen3-235B-A22B-Thinking-2507 via SiliconFlow API
- **Task**: Complex financial analysis requiring PDF parsing, currency conversion, and calculations
- **Context Modes Tested**: 5 (Full, No History, No Reasoning, No Tool Calls, No Tool Results)

## Key Findings

### 1. Complete Lack of Historical Tool Calls (NO_HISTORY)
**Impact**: Agent loses track of previous actions and may repeat operations unnecessarily.
- **Behavior**: Agent cannot reference past tool executions, leading to redundant API calls
- **Performance**: {no_history_perf}
- **Critical for**: Multi-step tasks requiring sequential dependencies

### 2. Lack of Reasoning Process (NO_REASONING)
**Impact**: Agent operates without strategic planning or step-by-step thinking.
- **Behavior**: Direct execution without planning leads to inefficient or incorrect solutions
- **Performance**: {no_reasoning_perf}
- **Critical for**: Complex tasks requiring logical decomposition

### 3. Lack of Tool Call Commands (NO_TOOL_CALLS)
**Impact**: Agent cannot execute any external tools, rendering it unable to complete the task.
- **Behavior**: Complete failure - agent can only describe what it would do, not actually do it
- **Performance**: {no_tool_calls_perf}
- **Critical for**: Any task requiring external data or computation

### 4. Lack of Tool Call Results (NO_TOOL_RESULTS)
**Impact**: Agent operates blind to the outcomes of its actions.
- **Behavior**: Cannot adapt based on results, leading to incorrect conclusions
- **Performance**: {no_tool_results_perf}
- **Critical for**: Tasks requiring iterative refinement or result validation

## Statistical Summary
- **Total Tests Run**: {total_tests}
- **Successful Completions**: {successful_tests}
- **Average Execution Time (Full Context)**: {avg_exec_time}s
- **Average Tool Calls (Full Context)**: {avg_tool_calls}

## Conclusion
The ablation study clearly demonstrates that each context component plays a vital role:
1. **Tool calls** are fundamental - without them, the agent cannot interact with the world
2. **Tool results** provide essential feedback for decision-making
3. **Reasoning** enables strategic planning and efficient execution
4. **History** prevents redundancy and maintains task coherence

## Recommendations
- Always maintain complete context for production agents
- Consider context windowing rather than removal for memory optimization
- Implement fallback mechanisms when context components are unavailable
"""
        
        # Fill in performance metrics
        def get_perf_string(mode):
            mode_result = next((r for r in results if r["context_mode"] == mode), None)
            if mode_result:
                return f"{'SUCCESS' if mode_result.get('success', False) else 'FAILURE'} - {mode_result.get('iterations', 0)} iterations, {mode_result.get('execution_time', 0)}s"
            return "N/A"
        
        # Calculate baseline metrics
        baseline = next((r for r in results if r["context_mode"] == "full"), None)
        avg_exec_time = baseline.get("execution_time", 0) if baseline else 0
        avg_tool_calls = baseline.get("num_tool_calls", 0) if baseline else 0
        
        report = report.format(
            no_history_perf=get_perf_string("no_history"),
            no_reasoning_perf=get_perf_string("no_reasoning"),
            no_tool_calls_perf=get_perf_string("no_tool_calls"),
            no_tool_results_perf=get_perf_string("no_tool_results"),
            total_tests=analysis["total_tests"],
            successful_tests=analysis["successful_tests"],
            avg_exec_time=avg_exec_time,
            avg_tool_calls=avg_tool_calls
        )
        
        return report


def main():
    """Main function to run ablation studies"""
    # Get API key from environment
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        logger.error("Please set SILICONFLOW_API_KEY environment variable")
        return
    
    # Initialize test suite
    test_suite = AblationTestSuite(api_key)
    
    # Run ablation study
    logger.info("Starting ablation study...")
    results = test_suite.run_ablation_study()
    
    # Analyze results
    analysis = test_suite.analyze_results(results)
    
    # Print results table
    test_suite.print_results_table(results)
    
    # Generate visualizations
    try:
        test_suite.visualize_results(results)
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    
    # Generate and save report
    report = test_suite.generate_report(results, analysis)
    with open("ablation_study_report.md", "w") as f:
        f.write(report)
    logger.info("Report saved as 'ablation_study_report.md'")
    
    # Save raw results
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Raw results saved as 'ablation_results.json'")
    
    # Print analysis summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Success Rate: {analysis['successful_tests']}/{analysis['total_tests']}")
    print("\nContext Mode Impacts:")
    for mode, impact in analysis["context_mode_impact"].items():
        print(f"\n{mode.upper()}:")
        print(f"  - Success Maintained: {impact['success_maintained']}")
        print(f"  - Execution Time Delta: {impact['execution_time_delta']:.2f}s")
        print(f"  - Iteration Delta: {impact['iteration_delta']}")
        print(f"  - Tool Call Delta: {impact['tool_call_delta']}")
        if impact['failure_reason']:
            print(f"  - Failure Reason: {impact['failure_reason']}")


if __name__ == "__main__":
    main()
