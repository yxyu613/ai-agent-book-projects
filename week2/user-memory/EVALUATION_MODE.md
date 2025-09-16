# User Memory Agent - Evaluation Mode

This document describes the evaluation mode of the User Memory Agent, which integrates with the user-memory-evaluation framework for structured testing.

## Overview

The evaluation mode allows the User Memory Agent to be tested using predefined test cases from the evaluation framework. Instead of accepting random user inputs, the agent:

1. Loads structured test cases from the evaluation framework
2. Processes conversation histories to build memory
3. Answers questions based on the extracted memory
4. Receives automated evaluation of its performance

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env to add your MOONSHOT_API_KEY
```

## Usage

### Interactive Mode

The default mode provides an interactive menu for selecting and running test cases:

```bash
python evaluation_main.py
```

This will present you with options to:
- List available test cases
- Run individual test cases
- View test case details
- Generate evaluation reports
- See results summary

### Single Test Mode

Run a specific test case by ID:

```bash
python evaluation_main.py --mode single --test-id layer1_01_bank_account
```

### Batch Test Mode

Run all test cases in a category:

```bash
# Run all Layer 1 tests
python evaluation_main.py --mode batch --category layer1

# Run all Layer 2 tests
python evaluation_main.py --mode batch --category layer2

# Run all tests (no category filter)
python evaluation_main.py --mode batch
```

## Test Case Structure

Test cases are organized in three layers:

### Layer 1: Basic Recall & Direct Retrieval
- Simple factual information retrieval
- Single conversation history
- Direct questions about stored information

### Layer 2: Contextual Reasoning & Disambiguation
- Multiple conversation histories
- Requires disambiguation between similar items
- Tests contextual understanding

### Layer 3: Cross-Session Synthesis & Proactive Assistance
- Complex multi-conversation scenarios
- Requires synthesis across multiple sessions
- Tests proactive assistance capabilities

## Workflow

The evaluation workflow for each test case:

1. **Load Test Case**: Retrieve test case from evaluation framework
2. **Process Histories**: Extract and summarize information from conversation histories
3. **Build Memory**: Store extracted information in agent's memory system
4. **Answer Question**: Use memory to answer the user's question
5. **Evaluate**: Submit answer to evaluation framework for scoring

## Evaluation Metrics

Each test is evaluated on:
- **Pass/Fail**: Binary success indicator
- **Score**: Numerical score from 0.0 to 1.0
- **Required Information**: Checklist of required information pieces
- **Reasoning**: Detailed explanation of the evaluation

## Memory Processing

The agent processes conversation histories by:

1. Creating a temporary user profile for the test
2. Iterating through each conversation message
3. Extracting important information from user messages
4. Building a memory context using the configured memory mode (NOTES or JSON_CARDS)
5. Using the memory context to answer questions

## Configuration

The evaluation mode uses the same configuration as the main agent:

```python
UserMemoryConfig(
    enable_memory_updates=True,      # Allow memory updates during processing
    enable_conversation_history=True, # Track conversation history
    enable_memory_search=True,        # Enable memory search capabilities
    memory_mode=MemoryMode.NOTES,    # Memory format (NOTES or JSON_CARDS)
    save_trajectory=False             # Don't save trajectories in eval mode
)
```

## Output and Reports

### Console Output
- Real-time progress during test execution
- Color-coded pass/fail indicators
- Detailed evaluation reasoning
- Score breakdowns

### Evaluation Reports
Generated reports include:
- Overall pass rate and average score
- Category-wise performance breakdown
- Individual test results with reasoning
- Suggestions for improvement

## Integration with CI/CD

The evaluation mode can be integrated into CI/CD pipelines:

```bash
# Run specific category and save report
python evaluation_main.py --mode batch --category layer1 > layer1_results.txt

# Check exit code for pass/fail
echo $?  # 0 if all tests passed, 1 if any failed
```

## Debugging

For debugging, you can:
1. Run individual test cases to isolate issues
2. View extracted memory to understand what the agent learned
3. Check evaluation reasoning to understand why tests failed
4. Enable verbose mode in the agent for detailed logging

## Examples

### Example 1: Running a Simple Test

```bash
$ python evaluation_main.py --mode single --test-id layer1_01_bank_account

Test Case: Bank Account Setup
Processing conversation histories...
  Processed 90 messages
Extracted Memory Summary:
  User's checking account: 4429853327
  Savings account: 9987234561
  ...
User Question: What's my checking account number?
Agent's Answer: Your checking account number is 4429853327.
Evaluation: ✓ PASSED (Score: 1.00)
```

### Example 2: Batch Testing Layer 2

```bash
$ python evaluation_main.py --mode batch --category layer2

Running 20 test cases...
[1/20] layer2_01_multiple_vehicles
[2/20] layer2_02_multiple_properties
...
Batch Test Summary:
Total: 20
Passed: 18
Failed: 2
Pass Rate: 90.0%
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure MOONSHOT_API_KEY is set in environment or .env file
2. **Import Error**: Make sure you're in the correct directory and dependencies are installed
3. **Test Case Not Found**: Verify test case ID matches exactly (case-sensitive)
4. **Memory Processing Timeout**: Some test cases with long histories may take time to process

### Performance Tips

1. Start with Layer 1 tests to verify basic functionality
2. Use single test mode for debugging specific failures
3. Monitor API usage when running batch tests
4. Consider implementing caching for repeated test runs

## Contributing

To add new test capabilities:

1. Extend the memory processing logic in `process_conversation_histories()`
2. Enhance answer generation in `answer_question()`
3. Add custom evaluation metrics if needed
4. Submit results to the evaluation framework maintainers

## Related Documentation

- [Main README](README.md) - General agent documentation
- [User Memory Evaluation Framework](../user-memory-evaluation/README.md) - Test framework details
- [Implementation Notes](IMPLEMENTATION_NOTES.md) - Technical implementation details
