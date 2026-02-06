"""
PromptBuilder - Create structured prompts for LLM evaluation

Enforces consistent output format to enable step tracking and metrics calculation.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Prompt template for different evaluation strategies"""

    name: str
    description: str
    system_message: str
    user_message_template: str


# Prompt templates designed to produce structured output
PROMPT_TEMPLATES = {
    "default": PromptTemplate(
        name="Default (Chain-of-Thought)",
        description="Structured reasoning with step-by-step breakdown",
        system_message="""You are a financial reasoning assistant. Answer question step-by-step, showing your work clearly.

For each step:
1. State what you're calculating or looking up
2. Show numbers/formulas you're using
3. Explain why you're doing this step
4. Give intermediate results
5. Finally, provide your final answer in the format: ANSWER: [number]

IMPORTANT: Your final answer must be the last line of your response, formatted exactly as: ANSWER: [number]""",
        user_message_template="""Question: {question}

Context:
{context}

{python_solution}

Instructions:
Use the Python solution as a reference. Break down your reasoning into clear steps.
Output your answer and reasoning in following JSON format:

{{
  "final_answer": [your computed answer],
  "reasoning_steps": [
    {{
      "step_number": 1,
      "step_type": "variable_definition|calculation|conditional|return_result",
      "description": "[what you did in this step]",
      "code_snippet": "[the actual code or formula]"
    }}
  ]
}}""",
    ),
    "cot_rag": PromptTemplate(
        name="Chain-of-Thought with RAG",
        description="COT reasoning with function retrieval from financial library",
        system_message="""You are a financial reasoning assistant with access to a library of financial functions.

Available Functions:
{functions_instructions}

For each step:
1. State what you're calculating or looking up
2. If relevant financial function is available, use it
3. Show your reasoning and calculations clearly
4. Reference any functions you use by name
5. Provide final answer in specified format

IMPORTANT: The available functions are meant to HELP you. Use them when appropriate to the problem.
""",
        user_message_template="""Question: {question}

Context:
{context}

{python_solution}

Available Financial Functions:
{functions}

Instructions:
1. Review the available financial functions above
2. Use relevant functions to help solve the problem
3. Reference any function you use by name (e.g., "Using NPV calculation function")
4. Show your reasoning step-by-step
5. If a function doesn't perfectly match your needs, adapt it or solve manually
6. Output your final answer and reasoning in following JSON format:

{{
  "final_answer": [your computed answer],
  "reasoning_steps": [
    {{
      "step_number": 1,
      "step_type": "variable_definition|calculation|conditional|return_result",
      "description": "[what you did in this step]",
      "code_snippet": "[the actual code or formula, note if you used a function: function_name]"
    }}
  ]
}}""",
    ),
    "pot_rag": PromptTemplate(
        name="Program-of-Thought with RAG",
        description="POT reasoning with function retrieval from financial library",
        system_message="""You are a financial reasoning assistant with access to a library of financial functions.

Available Functions:
{functions_instructions}

Your task is to write Python code to solve the financial problem:
1. Define all necessary variables
2. Use relevant financial functions when applicable (by function name)
3. Perform calculations step-by-step
4. Include comments explaining each step
5. The final answer should be stored in a variable called 'answer'
6. Use the final statement: return answer

Requirements:
- Only use standard Python libraries (math, statistics if needed)
- For complex calculations, prefer using available financial functions over manual implementation
- Clear variable names matching problem context
- Proper handling of numeric operations
- Include type annotations where helpful
- Comment which financial function you're using (e.g., "# Using NPV calculation function")

Example format:
```python
# Define variables
shares_outstanding = 1328

# Use financial function if available
# from financial_functions import calculate_npv
# npv = calculate_npv(...)

# Calculate answer
answer = shares_outstanding - acquisition_cost

# Return final result
return answer
```

IMPORTANT: Output ONLY as Python code, no explanations outside code block.""",
        user_message_template="""Question: {question}

Context:
{context}

{python_solution}

Available Financial Functions:
{functions}

Instructions:
Write a Python program to solve this financial problem:
1. Define a variable called 'answer' that contains the final computed value
2. Use relevant financial functions when applicable
3. For each function, reference it by name (e.g., "Using calculate_npv function")
4. Show your reasoning step-by-step in code comments
5. Use 'return answer' as the last statement

Output your complete Python program in a code block```python
...```""",
    ),
    "pot": PromptTemplate(
        name="Program-of-Thought (POT)",
        description="Generate Python code to solve the financial problem",
        system_message="""You are a financial reasoning assistant with strong Python programming skills.

Generate a Python program to solve the given financial problem:
1. Define all necessary variables
2. Perform calculations step-by-step
3. Include comments explaining each step
4. The final answer should be stored in a variable called 'answer'
5. Use the final statement: return answer

Requirements:
- Only use standard Python libraries (math, statistics if needed)
- No external dependencies or imports
- Clear variable names matching the problem context
- Proper handling of numeric operations
- Include type annotations where helpful

Example format:
```python
# Define variables
shares_outstanding = 1328  # shares at end of 2012
acquisition_cost = 176  # cost of Smith International acquisition

# Calculate answer
answer = shares_outstanding - acquisition_cost

# Return final result
return answer
```

IMPORTANT: Output ONLY the Python code, no explanations outside code block.""",
        user_message_template="""Question: {question}

Context:
{context}

{python_solution}

Instructions:
Generate a Python program to solve this financial problem.
Define a variable called 'answer' that contains the final computed value.
Use 'return answer' as the last statement.

Output your complete Python program in a code block```python
...```""",
    ),
    "finance_reasoning_compliant": PromptTemplate(
        name="FinanceReasoning Compliant",
        description="Matches paper methodology with explicit structured output",
        system_message="""You are a financial reasoning assistant. Answer the question using the provided context and Python solution.

Follow this structured output format:

{{
  "final_answer": [number or string],
  "reasoning_steps": [
    {{
      "step_number": 1,
      "step_type": "[variable_definition|calculation|conditional|return_result]",
      "description": "...",
      "code_snippet": "..."
    }},
    ...
  ]
}}

Requirements:
- final_answer: The computed answer (same format as ground_truth)
- reasoning_steps: Array of step objects with step_number, step_type, description, code_snippet
- step_type must be one of: variable_definition, calculation, conditional, return_result
- step_number: Sequential integer starting from 1

Example:
{{
  "final_answer": 1152,
  "reasoning_steps": [
    {{
      "step_number": 1,
      "step_type": "variable_definition",
      "description": "Define shares_outstanding",
      "code_snippet": "shares_outstanding = 1328"
    }},
    {{
      "step_number": 2,
      "step_type": "variable_definition",
      "description": "Define acquisition_cost",
      "code_snippet": "acquisition_cost = 176"
    }},
    {{
      "step_number": 3,
      "step_type": "calculation",
      "description": "Calculate shares after acquisition",
      "code_snippet": "answer = shares_outstanding - acquisition_cost"
    }},
    {{
      "step_number": 4,
      "step_type": "return_result",
      "description": "Final answer",
      "code_snippet": "return answer"
    }}
  ]
}}""",
        user_message_template="""Question: {question}

Context:
{context}

{python_solution}

Instructions:
Use the Python solution as a reference. Break down your reasoning into clear steps.
Output your answer and reasoning in the following JSON format:

{{
  "final_answer": [your computed answer],
  "reasoning_steps": [
    {{
      "step_number": 1,
      "step_type": "variable_definition|calculation|conditional|return_result",
      "description": "[what you did in this step]",
      "code_snippet": "[the actual code or formula]"
    }}
  ]
}}""",
    ),
    "cot_only": PromptTemplate(
        name="Chain-of-Thought Only",
        description="Pure chain-of-thought without structured JSON",
        system_message="""Think step-by-step and show your work.""",
        user_message_template="""Question: {question}

Context:
{context}

{python_solution}

Think step-by-step. Show intermediate calculations.
Final answer format: ANSWER: [number]""",
    ),
}


class PromptBuilder:
    """Builds prompts from templates"""

    def __init__(self, template_name: str = "finance_reasoning_compliant"):
        """Initialize with a specific template"""

        if template_name not in PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(PROMPT_TEMPLATES.keys())}"
            )

        self.template = PROMPT_TEMPLATES[template_name]
        self.template_name = template_name

    def build_prompt(
        self,
        question: str,
        context: str,
        python_solution: Optional[str] = None,
        functions: Optional[str] = None,
        template_type: Optional[str] = None,
    ) -> str:
        """Build prompt from template

        Args:
            question: The financial question to answer
            context: Problem context (tables, data, etc.)
            python_solution: Reference Python solution
            functions: Retrieved financial functions (for RAG templates)
            template_type: Override template type for this build

        Returns:
            Complete prompt string
        """

        template = self.template

        # Allow template override
        if template_type and template_type != self.template_name:
            if template_type in PROMPT_TEMPLATES:
                template = PROMPT_TEMPLATES[template_type]
            else:
                raise ValueError(f"Unknown template type: {template_type}")

        # Prepare format arguments
        format_args = {
            "question": question,
            "context": context,
            "python_solution": python_solution if python_solution else "Not provided",
        }

        # Add functions for RAG templates
        if functions is not None:
            format_args["functions"] = functions
            format_args["functions_instructions"] = (
                "You have access to a library of financial functions. Use them when relevant to solve the problem."
            )
        else:
            # Provide empty strings for non-RAG templates
            format_args["functions"] = ""
            format_args["functions_instructions"] = ""

        # Replace placeholders in user message
        try:
            user_message = template.user_message_template.format(**format_args)
        except KeyError as e:
            # Handle missing placeholders gracefully
            print(f"[WARN] Missing placeholder in template: {e}")
            user_message = template.user_message_template.format(
                question=question,
                context=context,
                python_solution=python_solution if python_solution else "Not provided",
            )

        # Replace placeholders in system message
        try:
            system_message = template.system_message.format(**format_args)
        except KeyError:
            # System message might not have all placeholders
            system_message = template.system_message

        # Combine system and user messages
        if system_message:
            prompt = f"{system_message}\n\n{user_message}"
        else:
            prompt = user_message

        return prompt

    def get_template_names(self) -> List[str]:
        """Get available prompt templates"""
        return list(PROMPT_TEMPLATES.keys())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        template_name = sys.argv[1]
        builder = PromptBuilder(template_name)

        # Example: Show available templates
        if template_name == "list":
            templates = builder.get_template_names()
            print("Available prompt templates:")
            for name in templates:
                print(f"  - {name}")
                template = PROMPT_TEMPLATES[name]
                print(f"    Description: {template.description}")
        else:
            # Build example prompt
            example_question = "what would 2012 shares outstanding in millions have been without acquisition of smith international? Answer to nearest integer."

            example_context = "schlumberger limited and subsidiaries..."

            example_solution = """
shares_outstanding = 1328
acquisition_cost = 176
answer = shares_outstanding - acquisition_cost
return answer
"""

            prompt = builder.build_prompt(
                question=example_question,
                context=example_context,
                python_solution=example_solution,
            )

            print("=" * 60)
            print(f"Prompt Template: {builder.template_name}")
            print("=" * 60)
            print()
            print(prompt)
            print()
            print("=" * 60)
    else:
        print("Usage: python prompt_builder.py [template_name|list]")
        print()
        print("Available templates:")
        for name in PROMPT_TEMPLATES:
            print(f"  - {name}")
