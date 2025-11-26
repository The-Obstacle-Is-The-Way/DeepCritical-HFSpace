"""Demo script to test Modal code execution integration.

Run with: uv run python examples/modal_demo/test_code_execution.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.code_execution import CodeExecutionError, get_code_executor


def test_basic_execution():
    """Test basic code execution."""
    print("\n=== Test 1: Basic Execution ===")
    executor = get_code_executor()

    code = """
print("Hello from Modal sandbox!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""

    result = executor.execute(code)
    print(f"Success: {result['success']}")
    print(f"Stdout:\n{result['stdout']}")
    if result["stderr"]:
        print(f"Stderr:\n{result['stderr']}")


def test_scientific_computing():
    """Test scientific computing libraries."""
    print("\n=== Test 2: Scientific Computing ===")
    executor = get_code_executor()

    code = """
import pandas as pd
import numpy as np

# Create sample data
data = {
    'drug': ['DrugA', 'DrugB', 'DrugC'],
    'efficacy': [0.75, 0.82, 0.68],
    'sample_size': [100, 150, 120]
}

df = pd.DataFrame(data)

# Calculate weighted average
weighted_avg = np.average(df['efficacy'], weights=df['sample_size'])

print(f"Drugs tested: {len(df)}")
print(f"Weighted average efficacy: {weighted_avg:.3f}")
print("\\nDataFrame:")
print(df.to_string())
"""

    result = executor.execute(code)
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['stdout']}")


def test_statistical_analysis():
    """Test statistical analysis."""
    print("\n=== Test 3: Statistical Analysis ===")
    executor = get_code_executor()

    code = """
import numpy as np
from scipy import stats

# Simulate two treatment groups
np.random.seed(42)
control_group = np.random.normal(100, 15, 50)
treatment_group = np.random.normal(110, 15, 50)

# Perform t-test
t_stat, p_value = stats.ttest_ind(treatment_group, control_group)

print(f"Control mean: {np.mean(control_group):.2f}")
print(f"Treatment mean: {np.mean(treatment_group):.2f}")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant difference")
else:
    print("Result: No significant difference")
"""

    result = executor.execute(code)
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['stdout']}")


def test_with_return_value():
    """Test execute_with_return method."""
    print("\n=== Test 4: Return Value ===")
    executor = get_code_executor()

    code = """
import numpy as np

# Calculate something
data = np.array([1, 2, 3, 4, 5])
result = {
    'mean': float(np.mean(data)),
    'std': float(np.std(data)),
    'sum': int(np.sum(data))
}
"""

    try:
        result = executor.execute_with_return(code)
        print(f"Returned result: {result}")
        print(f"Mean: {result['mean']}")
        print(f"Std: {result['std']}")
        print(f"Sum: {result['sum']}")
    except CodeExecutionError as e:
        print(f"Error: {e}")


def test_error_handling():
    """Test error handling."""
    print("\n=== Test 5: Error Handling ===")
    executor = get_code_executor()

    code = """
# This will fail
x = 1 / 0
"""

    result = executor.execute(code)
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Modal Code Execution Demo")
    print("=" * 60)

    tests = [
        test_basic_execution,
        test_scientific_computing,
        test_statistical_analysis,
        test_with_return_value,
        test_error_handling,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
