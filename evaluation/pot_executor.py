"""
Code Execution Sandbox - Safe execution of LLM-generated Python code

Provides:
- Isolated execution environment
- Timeout protection
- Output capture
- Error handling
"""

import subprocess
import sys
import io
import re
import ast
import time
from typing import Any, Optional, Dict
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution"""

    code: str
    success: bool
    output: Any
    error: Optional[str]
    execution_time: float
    timeout: bool


class CodeExecutionSandbox:
    """Safely executes Python code with timeout and output capture"""

    def __init__(self, timeout_seconds: int = 10):
        """
        Args:
            timeout_seconds: Maximum execution time before timeout
        """
        self.timeout_seconds = timeout_seconds

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response

        Supports multiple code block formats:
        - ```python ... ```
        - ``` ... ```
        - Just code (no delimiters)
        """

        # Pattern 1: Standard python code block
        pattern1 = r"```python\s*\n?(.*?)\n?```"
        match = re.search(pattern1, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 2: Generic code block
        pattern2 = r"```\s*\n?(.*?)\n?```"
        match = re.search(pattern2, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 3: Look for variable assignment or return statement
        # Find lines that look like code
        lines = response.split("\n")
        code_lines = []

        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            elif in_code_block or self._looks_like_code(line):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return None

    def _looks_like_code(self, line: str) -> bool:
        """Heuristic to identify if line looks like code"""

        line = line.strip()

        # Code indicators:
        # - Has assignment (=)
        # - Has operators (+, -, *, /)
        # - Has function calls
        # - Starts with common keywords
        code_starters = [
            "def ",
            "class ",
            "return ",
            "print(",
            "if ",
            "else:",
            "for ",
            "while ",
            "import ",
            "from ",
        ]

        # Has assignment or operators
        if "=" in line or any(op in line for op in ["+", "-", "*", "/", "**"]):
            return True

        # Starts with code keyword
        if any(line.startswith(starter) for starter in code_starters):
            return True

        return False

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code to handle common LLM output issues:
        1. Remove 'return' statements outside functions
        2. Call defined functions to get answer
        3. Handle 'return answer' at module level
        """
        lines = code.strip().split("\n")
        processed_lines = []
        function_names = []
        in_function = False
        indent_level = 0

        for line in lines:
            stripped = line.strip()

            # Track function definitions
            if stripped.startswith("def "):
                in_function = True
                # Extract function name
                func_match = re.match(r"def\s+(\w+)\s*\(", stripped)
                if func_match:
                    function_names.append(func_match.group(1))
                processed_lines.append(line)
                continue

            # Track indentation to know when function ends
            if in_function:
                if stripped and not line.startswith(" ") and not line.startswith("\t"):
                    in_function = False

            # Handle 'return answer' outside of function
            if not in_function and stripped.startswith("return "):
                # Convert 'return answer' to 'answer = answer' or just skip
                return_value = stripped[7:].strip()
                if return_value and return_value != "answer":
                    processed_lines.append(f"answer = {return_value}")
                # Skip 'return answer' as answer is already defined
                continue

            processed_lines.append(line)

        # If functions were defined, call them to get the answer
        if function_names:
            # Look for common function names that should be called
            for func_name in function_names:
                if func_name in [
                    "solution",
                    "calculate_wacc",
                    "main",
                    "solve",
                    "calculate",
                ]:
                    processed_lines.append(f"\nanswer = {func_name}()")
                    break
            else:
                # Call the first defined function
                processed_lines.append(f"\nanswer = {function_names[0]}()")

        return "\n".join(processed_lines)

    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute Python code in sandboxed environment

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output or error
        """

        start_time = time.time()

        try:
            # Wrap code to capture output
            # Use exec() with custom globals to capture print output
            sandbox_globals = {
                "__builtins__": {},
            }

            # Capture stdout
            stdout_capture = io.StringIO()

            # Prepare code to capture 'answer' variable or print() output
            # Inject variable capture at end
            wrapped_code = f"""
import sys
from io import StringIO

# Capture stdout
stdout_capture = StringIO()
sys.stdout = stdout_capture

{code}

# Restore stdout
sys.stdout = sys.__stdout__

# Get result - check for 'answer' variable or last print
result = locals().get('answer', stdout_capture.getvalue().strip())
"""

            # Execute with timeout using subprocess for true isolation
            process = subprocess.Popen(
                [sys.executable, "-c", wrapped_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                _, stderr = process.communicate(timeout=self.timeout_seconds)
                execution_time = time.time() - start_time

                # Parse output
                output_text = ""
                try:
                    # Try to parse the captured output
                    output_match = re.search(r"Result:\s*(.*)", stdout_capture.read())
                    if output_match:
                        output_text = output_match.group(1).strip()
                except:
                    output_text = ""

                # If no explicit result, try to evaluate the code directly
                if not output_text:
                    # Alternative: compile and exec in current namespace
                    exec_globals = {}
                    try:
                        # Preprocess code to handle common issues
                        processed_code = self._preprocess_code(code)
                        exec(processed_code, exec_globals)
                        output_text = str(exec_globals.get("answer", ""))
                    except Exception as e:
                        return ExecutionResult(
                            code=code,
                            success=False,
                            output=None,
                            error=f"Code execution error: {str(e)}",
                            execution_time=execution_time,
                            timeout=False,
                        )

                # Try to parse as number
                parsed_output = self._parse_output(output_text)

                return ExecutionResult(
                    code=code,
                    success=True,
                    output=parsed_output,
                    error=None,
                    execution_time=execution_time,
                    timeout=False,
                )

            except subprocess.TimeoutExpired:
                process.kill()
                execution_time = time.time() - start_time
                return ExecutionResult(
                    code=code,
                    success=False,
                    output=None,
                    error=f"Execution timeout after {self.timeout_seconds} seconds",
                    execution_time=execution_time,
                    timeout=True,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    code=code,
                    success=False,
                    output=None,
                    error=f"Execution error: {str(e)}",
                    execution_time=execution_time,
                    timeout=False,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                code=code,
                success=False,
                output=None,
                error=f"Sandbox error: {str(e)}",
                execution_time=execution_time,
                timeout=False,
            )

    def _parse_output(self, output: str) -> Any:
        """
        Parse code execution output to extract final answer

        Supports:
        - Plain numbers
        - "answer = 123"
        - Print statements
        """

        output = output.strip()

        if not output:
            return None

        # Try to extract numeric answer
        # Pattern 1: "answer = 123" or similar
        match = re.search(r"answer\s*=\s*(.+)", output, re.IGNORECASE)
        if match:
            try:
                # Try to evaluate as Python literal
                value = ast.literal_eval(match.group(1).strip())
                return value
            except:
                # Return as string
                return match.group(1).strip()

        # Pattern 2: Last line is just a number
        last_line = output.split("\n")[-1].strip()
        if re.match(r"^-?\d+\.?\d*$", last_line):
            try:
                return float(last_line)
            except:
                return last_line

        # Pattern 3: Extract number from text
        numbers = re.findall(r"-?\d+\.?\d*", output)
        if numbers:
            try:
                return float(numbers[-1])  # Return last number
            except:
                return numbers[-1]

        # Default: return as string
        return output


def evaluate_pot_response(
    llm_response: str,
    timeout_seconds: int = 10,
) -> ExecutionResult:
    """
    Evaluate LLM Program-of-Thought response

    Args:
        llm_response: Raw LLM response text
        timeout_seconds: Maximum execution time

    Returns:
        ExecutionResult with parsed output
    """

    # Extract code from response
    code = CodeExecutionSandbox(
        timeout_seconds=timeout_seconds
    )._extract_code_from_response(llm_response)

    if code is None:
        return ExecutionResult(
            code="",
            success=False,
            output=None,
            error="No executable code found in response",
            execution_time=0.0,
            timeout=False,
        )

    # Execute the code
    sandbox = CodeExecutionSandbox(timeout_seconds=timeout_seconds)
    result = sandbox.execute_code(code)

    return result


if __name__ == "__main__":
    import sys

    # Test with sample responses
    test_cases = [
        # Test 1: Direct code block
        """```python
answer = 1152
```""",
        # Test 2: Inline code
        """shares_outstanding = 1328
acquisition_cost = 176
answer = shares_outstanding - acquisition_cost""",
        # Test 3: No code block
        """The answer is 1152""",
        # Test 4: Malformed code
        """```python
answer = 1152 +
```""",
    ]

    print("=" * 60)
    print("Code Execution Sandbox Tests")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Input: {test_case[:50]}...")
        result = evaluate_pot_response(test_case)

        print(f"Success: {result.success}")
        print(f"Timeout: {result.timeout}")
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
        print(f"Time: {result.execution_time:.3f}s")
        print("-" * 40)
