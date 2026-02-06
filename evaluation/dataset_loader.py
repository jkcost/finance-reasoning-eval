"""
DatasetLoader - Load FinanceReasoning datasets

Supports loading from:
1. hard.json (original dataset)
2. financial_reasoning_traced.json (with reasoning traces)
3. hard_transformed.json (transformed dataset)
4. hard_transformed.json (legacy)

Normalizes all formats into a unified Example structure.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Example:
    """Normalized evaluation example from FinanceReasoning dataset"""

    # Core fields (required, no defaults)
    id: str
    question: str
    context: str
    ground_truth_final: Any  # int, float, str, or choice
    question_id: str
    level: str  # easy, medium, hard
    source: str

    # Core fields (optional, with defaults)
    ground_truth_steps: Optional[List[Dict[str, Any]]] = (
        None  # Python solution steps if available
    )

    # Metadata (optional, with defaults)
    difficulty: Optional[float] = None

    # Trace data (if available)
    reasoning_trace: Optional[Dict[str, Any]] = None

    # Additional fields from dataset (optional, with defaults)
    inventory: Optional[Dict[str, Any]] = None  # years, entities, numbers
    statistics: Optional[Dict[str, Any]] = None  # operator statistics
    python_solution: Optional[str] = None  # Original Python solution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "ground_truth_final": self.ground_truth_final,
            "ground_truth_steps": self.ground_truth_steps,
            "question_id": self.question_id,
            "level": self.level,
            "source": self.source,
            "difficulty": self.difficulty,
            "reasoning_trace": self.reasoning_trace,
            "inventory": self.inventory,
            "statistics": self.statistics,
            "python_solution": self.python_solution,
        }


class DatasetLoader:
    """Loads FinanceReasoning datasets from various sources"""

    def __init__(self, data_root: Optional[Path] = None):
        # Default data_root to data/financereasoning/
        if data_root is None:
            data_root = Path(__file__).parent.parent / "data" / "financereasoning"

        self.data_root = data_root
        self.raw_dir = data_root / "raw" / "FinanceReasoning"
        self.transformed_dir = data_root / "transformed"
        self.traced_dir = data_root / "traced"

    def load_dataset(
        self, dataset_name: str, split: Optional[str] = None
    ) -> List[Example]:
        """Load dataset by name and split (e.g., easy, medium, hard)"""
        if dataset_name != "financereasoning":
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Load all splits if no split specified, otherwise load specific split
        if split:
            if split not in ["easy", "medium", "hard"]:
                raise ValueError(
                    f"Invalid split: {split}. Must be 'easy', 'medium', or 'hard'"
                )
            return self._load_split(split)
        else:
            # Load all splits combined
            examples = []
            for split_name in ["easy", "medium", "hard"]:
                examples.extend(self._load_split(split_name))
            return examples

    def _load_split(self, split: str) -> List[Example]:
        """Load a specific split (easy, medium, or hard)"""
        split_file = self.raw_dir / f"{split}.json"

        if not split_file.exists():
            print(f"[WARN] Split file not found: {split_file}")
            return []

        data = self._load_json(split_file)

        examples = []
        for item in data:
            question_id = item.get("question_id", "")

            # Map to Example structure
            example = Example(
                id=question_id,
                question=item.get("question", ""),
                context=item.get("context", ""),
                ground_truth_final=item.get("ground_truth"),
                question_id=question_id,
                level=split,  # Use split as level
                source=item.get("source", "FinanceReasoning"),
                difficulty=item.get("difficulty"),
                inventory=item.get("inventory"),
                statistics=item.get("statistics"),
                python_solution=item.get("python_solution"),
            )

            examples.append(example)

        print(f"[OK] Loaded {len(examples)} examples from {split}.json")
        return examples

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _merge_trace(
        self, example: Example, trace_data: Optional[Dict[str, Any]]
    ) -> Example:
        """Merge reasoning trace data into example"""
        if trace_data:
            example.reasoning_trace = trace_data
        return example

    def load_hard(
        self, limit: Optional[int] = None, level: Optional[str] = None
    ) -> List[Example]:
        """Load hard.json dataset"""

        hard_file = self.raw_dir / "hard.json"
        data = self._load_json(hard_file)

        examples = []
        for item in data:
            question_id = item.get("question_id", "")
            level = item.get("level", "unknown")

            # Apply filters
            if level and level != level:
                continue
            if limit and len(examples) >= limit:
                break

            # Map to Example structure
            example = Example(
                id=question_id,
                question=item.get("question", ""),
                context=item.get("context", ""),
                ground_truth_final=item.get("ground_truth"),
                question_id=question_id,
                level=level,
                source=item.get("source", ""),
                difficulty=item.get("difficulty"),
                inventory=item.get("inventory"),
                statistics=item.get("statistics"),
                python_solution=item.get("python_solution"),
            )

            examples.append(example)

        print(
            f"[OK] Loaded {len(examples)} examples from hard.json (level={level or 'all'}, limit={limit or 'none'})"
        )
        return examples

    def load_traced(
        self, limit: Optional[int] = None, level: Optional[str] = None
    ) -> List[Example]:
        """Load financial_reasoning_traced.json with reasoning traces"""

        trace_file = self.traced_dir / "financial_reasoning_traced.json"

        if not trace_file.exists():
            print(f"[WARN] Trace file not found: {trace_file}")
            return []

        data = self._load_json(trace_file)

        examples = []
        for item in data.get("items", []):
            trace_id = item.get("trace_id", "")
            question_id = trace_id.replace(".trace", "")

            # Apply filters
            if level and item.get("level", "unknown") != level:
                continue
            if limit and len(examples) >= limit:
                break

            example = Example(
                id=trace_id,
                question=item.get("question", ""),
                context=item.get("context", ""),
                ground_truth_final=item.get("ground_truth"),
                question_id=question_id,
                level=item.get("level", "unknown"),
                source="FinanceReasoning-traced",
                reasoning_trace=item.get("reasoning_trace"),
            )

            examples.append(example)

        print(
            f"[OK] Loaded {len(examples)} traced examples from financial_reasoning_traced.json (level={level or 'all'}, limit={limit or 'none'})"
        )
        return examples

    def load_transformed(self, limit: Optional[int] = None) -> List[Example]:
        """Load hard_transformed.json dataset"""

        transformed_file = self.transformed_dir / "hard_transformed.json"

        if not transformed_file.exists():
            print(f"[WARN] Transformed file not found: {transformed_file}")
            return []

        data = self._load_json(transformed_file)

        examples = []
        for item in data:
            question_id = item.get("question_id", "")

            example = Example(
                id=question_id,
                question=item.get("question", ""),
                context=item.get("context", ""),
                ground_truth_final=item.get("ground_truth"),
                question_id=question_id,
                level=item.get("level", "unknown"),
                source="hard_transformed",
            )

            examples.append(example)

        print(
            f"[OK] Loaded {len(examples)} transformed examples (limit={limit or 'none'})"
        )
        return examples

    def load_combined(
        self,
        hard_limit: Optional[int] = None,
        hard_level: Optional[str] = None,
        traced_limit: Optional[int] = None,
        traced_level: Optional[str] = None,
    ) -> List[Example]:
        """Load and combine multiple datasets"""

        examples = []

        # Load from hard.json (load by default if no specific limit provided)
        if hard_limit is not None or hard_level is not None:
            examples.extend(self.load_hard(limit=hard_limit, level=hard_level))
        else:
            # Load all hard.json by default
            examples.extend(self.load_hard())

        # Load from traced
        if traced_limit is not None or traced_level is not None:
            examples.extend(self.load_traced(limit=traced_limit, level=traced_level))

        # Deduplicate by ID
        seen_ids = set()
        unique_examples = []
        for example in examples:
            if example.id not in seen_ids:
                seen_ids.add(example.id)
                unique_examples.append(example)

        print(
            f"[OK] Loaded {len(unique_examples)} unique examples (hard: {len(examples) - len(unique_examples)}, traced: {len(unique_examples) - len(examples)})"
        )
        return unique_examples

    def get_stats(self, examples: List[Example]) -> Dict[str, Any]:
        """Get statistics about loaded dataset"""

        level_counts = {"easy": 0, "medium": 0, "hard": 0}
        source_counts = {}

        for example in examples:
            level = example.level
            if level in level_counts:
                level_counts[level] += 1

            source = example.source
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "total": len(examples),
            "by_level": level_counts,
            "by_source": source_counts,
        }


if __name__ == "__main__":
    import sys

    # data_root points to data/financereasoning/
    # __file__ is evaluation/dataset_loader.py, parent is root, data_root is data/financereasoning
    data_root = Path(__file__).parent.parent / "data" / "financereasoning"

    if len(sys.argv) > 1:
        dataset_type = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        level = sys.argv[3] if len(sys.argv) > 3 else None

        loader = DatasetLoader(data_root)

        if dataset_type == "hard":
            examples = loader.load_hard(limit=limit, level=level)
        elif dataset_type == "traced":
            examples = loader.load_traced(limit=limit, level=level)
        elif dataset_type == "transformed":
            examples = loader.load_transformed(limit=limit)
        elif dataset_type == "combined":
            examples = loader.load_combined(
                hard_limit=limit,
                hard_level=level,
            )
        else:
            print(f"[ERROR] Unknown dataset type: {dataset_type}")
            print(
                "Usage: python dataset_loader.py [hard|traced|transformed|combined] [limit] [level]"
            )
            sys.exit(1)

        stats = loader.get_stats(examples)

        print(f"\n" + "=" * 60)
        print(f"Dataset Statistics:")
        print("=" * 60)
        print(f"  Total: {stats['total']}")
        print(f"  By Level:")
        for level, count in stats["by_level"].items():
            print(f"    {level:10}: {count}")
        print(f"  By Source:")
        for source, count in stats["by_source"].items():
            print(f"    {source:20}: {count}")
        print("=" * 60)
