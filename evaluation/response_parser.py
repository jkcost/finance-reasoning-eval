"""
ResponseParser & TraceExtractor - Parse LLM outputs

Extracts final answers and reasoning steps from structured LLM outputs.
"""

import re
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ParsedStep:
    """Parsed reasoning step"""

    step_number: int
    step_type: str  # variable_definition, calculation, conditional, return_result
    description: str
    code_snippet: str


@dataclass
class ParsedResponse:
    """Complete parsed response"""

    final_answer: Any  # int, float, str, or None
    final_answer_type: Optional[str]  # numeric, text, choice
    reasoning_steps: List[ParsedStep]
    parse_status: str  # "success", "partial", "failed"
    parse_message: str
    # For POT: code execution result
    code_execution_success: bool = False
    code_execution_output: Any = None
    code_execution_error: Optional[str] = None


class ResponseParser:
    """Parses LLM responses into structured format"""

    def __init__(self):
        pass

    def parse_openai_response(self, response_data: Dict[str, Any]) -> ParsedResponse:
        """Parse OpenAI response"""

        # Handle nested response structure from model_runner
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        api_response = response_data.get("response", response_data)

        # Extract content from OpenAI response structure
        choices = api_response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        else:
            content = response_data.get("content", "")

        # Preprocess: replace double braces with single braces
        content = content.replace("{{", "{").replace("}}", "}")

        # Try to extract JSON using multiple strategies
        parsed_json = self._extract_json_from_content(content)

        if parsed_json:
            return ParsedResponse(
                final_answer=parsed_json.get("final_answer"),
                final_answer_type="numeric"
                if isinstance(parsed_json.get("final_answer"), (int, float))
                else "text",
                reasoning_steps=self._parse_steps_json(
                    parsed_json.get("reasoning_steps", [])
                ),
                parse_status="success",
                parse_message="Successfully parsed structured JSON",
            )

        # Fallback 1: Extract final_answer using regex
        final_answer = self._extract_final_answer(content)
        if final_answer is not None:
            return ParsedResponse(
                final_answer=final_answer,
                final_answer_type="numeric"
                if self._is_numeric(final_answer)
                else "text",
                reasoning_steps=[],
                parse_status="partial",
                parse_message="Extracted final_answer from content",
            )

        # Fallback 2: Look for ANSWER: prefix
        lines = content.strip().split("\n")
        for line in reversed(lines):
            if line.strip().upper().startswith("ANSWER:"):
                final_answer = line.strip()[7:].strip()
                return ParsedResponse(
                    final_answer=final_answer,
                    final_answer_type="numeric"
                    if self._is_numeric(final_answer)
                    else "text",
                    reasoning_steps=[],
                    parse_status="partial",
                    parse_message="Extracted ANSWER from last line (fallback mode)",
                )

        # Last resort: return raw content
        return ParsedResponse(
            final_answer=content.strip()[:100] if content else None,
            final_answer_type="text",
            reasoning_steps=[],
            parse_status="partial",
            parse_message="No structured format found, using raw text",
        )

    def _extract_json_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from content using multiple strategies"""

        # Strategy 1: Find JSON block with "final_answer"
        patterns = [
            r'\{[^{}]*"final_answer"[^{}]*\}',  # Simple single-level
            r'\{[\s\S]*?"final_answer"[\s\S]*?\}(?=\s*$|\s*```)',  # Until end or code block
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        # Strategy 2: Find largest JSON-like block
        brace_count = 0
        start_idx = None
        for i, char in enumerate(content):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_str = content[start_idx : i + 1]
                    try:
                        parsed = json.loads(json_str)
                        if "final_answer" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        return None

    def _extract_final_answer(self, content: str) -> Optional[Any]:
        """Extract final_answer value using regex"""

        # Pattern: "final_answer": value (number or string)
        patterns = [
            r'"final_answer"\s*:\s*(\d+\.?\d*)',  # Number
            r'"final_answer"\s*:\s*"([^"]*)"',  # Quoted string
            r'"final_answer"\s*:\s*([^,}\s]+)',  # Unquoted value
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                # Try to convert to number
                try:
                    if "." in value:
                        return float(value)
                    return int(value)
                except ValueError:
                    return value

        return None

    def parse_anthropic_response(self, response_data: Dict[str, Any]) -> ParsedResponse:
        """Parse Anthropic response"""

        # Anthropic response structure: { 'response': { 'content': [ {'type': 'text', 'text': '...'}] } }
        api_response = response_data.get("response", {})
        content_block = api_response.get("content", [{}])

        # Extract text from content block
        content = ""
        if content_block and len(content_block) > 0:
            content = content_block[0].get("text", "")

        # Remove double curly braces that sometimes appear in LLM responses
        content = content.replace("{{", "{").replace("}}", "}")

        # Use shared JSON extraction logic
        parsed_json = self._extract_json_from_content(content)

        if parsed_json:
            return ParsedResponse(
                final_answer=parsed_json.get("final_answer"),
                final_answer_type="numeric"
                if isinstance(parsed_json.get("final_answer"), (int, float))
                else "text",
                reasoning_steps=self._parse_steps_json(
                    parsed_json.get("reasoning_steps", [])
                ),
                parse_status="success",
                parse_message="Successfully parsed structured JSON",
            )

        # Fallback 1: Extract final_answer using regex
        final_answer = self._extract_final_answer(content)
        if final_answer is not None:
            return ParsedResponse(
                final_answer=final_answer,
                final_answer_type="numeric"
                if self._is_numeric(final_answer)
                else "text",
                reasoning_steps=[],
                parse_status="partial",
                parse_message="Extracted final_answer from content",
            )
        else:
            # No JSON found, try parsing from text block
            return self._parse_from_text_block(content)

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric"""

        if value is None:
            return False

        # Try to convert to float
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _parse_from_text_block(self, content: str) -> ParsedResponse:
        """Parse reasoning steps from unstructured text block"""

        steps = []
        lines = content.split("\n")
        step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to identify step type
            step_type = None
            description = ""

            # Variable definition: x = value
            if re.match(r"^[\w\s]+\s*\s*=\s*[\d.]+", line):
                step_type = "variable_definition"
                description = f"Define variable: {line}"
            # Calculation: operations with operators
            elif re.search(r"[+\-*/]", line):
                step_type = "calculation"
                description = f"Calculate: {line}"
            # Conditional: if/elif statements
            elif re.match(r"^(if|elif)\s", line):
                step_type = "conditional"
                description = f"Conditional: {line}"
            # Return: return/answer statements
            elif re.match(r"^(return|answer)\s", line):
                step_type = "return_result"
                description = f"Final result: {line}"
            else:
                # Any other line is part of reasoning
                if re.match(r"^[\d\s-]+\.\s", line):
                    step_type = "variable_definition"
                    description = f"Step {step_num + 1}: {line}"
                    step_num += 1
                else:
                    step_num += 1

            steps.append(
                ParsedStep(
                    step_number=step_num,
                    step_type=step_type,
                    description=description,
                    code_snippet=line,
                )
            )

        return ParsedResponse(
            final_answer=None,
            final_answer_type="text",
            reasoning_steps=steps,
            parse_status="partial",
            parse_message=f"Parsed {len(steps)} reasoning steps from unstructured text",
        )

    def parse_pot_response(
        self, response_data: Dict[str, Any], code_executor=None
    ) -> ParsedResponse:
        """Parse Program-of-Thought response

        Args:
            response_data: Raw API response
            code_executor: Optional code executor

        Returns:
            ParsedResponse with final answer from executed code
        """

        # Extract raw response text
        raw_text = response_data.get("content", "")

        # Import code executor if not provided
        if code_executor is None:
            from pot_executor import evaluate_pot_response

            code_executor = evaluate_pot_response

        # Execute the code and get result
        exec_result = code_executor.evaluate_pot_response(
            raw_text,
            timeout_seconds=10,
        )

        if not exec_result.success:
            # Code execution failed
            return ParsedResponse(
                final_answer=None,
                final_answer_type="text",
                reasoning_steps=[],
                parse_status="failed",
                parse_message=exec_result.error,
                code_execution_success=False,
                code_execution_output=None,
                code_execution_error=exec_result.error,
            )

        # Code execution succeeded, extract final answer
        return ParsedResponse(
            final_answer=exec_result.output,
            final_answer_type="numeric"
            if isinstance(exec_result.output, (int, float))
            else "text",
            reasoning_steps=[],
            parse_status="success",
            parse_message="Code executed successfully",
            code_execution_success=True,
            code_execution_output=exec_result.output,
            code_execution_error=None,
        )

    def _parse_steps_json(
        self, steps_data: Optional[List[Dict[str, Any]]]
    ) -> List[ParsedStep]:
        """Parse steps from structured JSON format"""

        if not steps_data:
            return []

        parsed_steps = []
        for step_data in steps_data:
            if not isinstance(step_data, dict):
                continue

            parsed_step = ParsedStep(
                step_number=step_data.get("step_number", 0),
                step_type=step_data.get("step_type", "variable_definition"),
                description=step_data.get("description", ""),
                code_snippet=step_data.get("code_snippet", ""),
            )
            parsed_steps.append(parsed_step)

        return parsed_steps

    def parse_response(
        self,
        response_data: Dict[str, Any],
        provider_type: str,  # "openai" or "anthropic"
        is_pot: bool = False,  # Program-of-Thought evaluation
        code_executor=None,
    ) -> ParsedResponse:
        """Parse response based on provider type and evaluation mode"""

        if is_pot:
            # POT evaluation - execute code
            from pot_executor import evaluate_pot_response

            return self.parse_pot_response(response_data, evaluate_pot_response)
        elif provider_type == "openai":
            return self.parse_openai_response(response_data)
        elif provider_type == "anthropic":
            return self.parse_anthropic_response(response_data)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ResponseParser & TraceExtractor")
    print("=" * 60)

    # Test parsing
    parser = ResponseParser()

    # Test OpenAI format
    test_response_openai = {
        "content": """ANSWER: 42
The answer is 42 because..."""
    }

    result_openai = parser.parse_response(
        test_response_openai,
        provider_type="openai",
    )
    print(f"\nTest 1 - OpenAI Response:")
    print(f"  Final Answer: {result_openai.final_answer}")
    print(f"  Answer Type: {result_openai.final_answer_type}")
    print(f"  Steps Parsed: {len(result_openai.reasoning_steps)}")
    print(f"  Parse Status: {result_openai.parse_status}")

    # Test Anthropic format
    test_response_anthropic = {
        "content": """{
  "final_answer": 42,
  "reasoning_steps": [
    {
      "step_number": 1,
      "step_type": "variable_definition",
      "description": "Define shares_outstanding",
      "code_snippet": "shares_outstanding = 1328"
    },
    {
      "step_number": 2,
      "step_type": "variable_definition",
      "description": "Define acquisition_cost",
      "code_snippet": "acquisition_cost = 176"
    },
    {
      "step_number": 3,
      "step_type": "calculation",
      "description": "Calculate: shares_outstanding - acquisition_cost",
      "code_snippet": "answer = shares_outstanding - acquisition_cost"
    },
    {
      "step_number": 4,
      "step_type": "return_result",
      "description": "Final answer: 1152",
      "code_snippet": "return 1152"
    }
  ]
}"""
    }

    result_anthropic = parser.parse_response(
        test_response_anthropic,
        provider_type="anthropic",
    )
    print(f"\nTest 2 - Anthropic Response:")
    print(f"  Final Answer: {result_anthropic.final_answer}")
    print(f"  Answer Type: {result_anthropic.final_answer_type}")
    print(f"  Steps Parsed: {len(result_anthropic.reasoning_steps)}")
    print(f"  Parse Status: {result_anthropic.parse_status}")
