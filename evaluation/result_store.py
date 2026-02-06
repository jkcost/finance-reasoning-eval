"""
ResultStore - SQLite persistence and caching for evaluation results

Stores:
1. Evaluation results (all metrics per example/model)
2. Usage statistics (tokens, costs, latency)
3. Caches responses to avoid redundant API calls
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class CachedResponse:
    """Cached LLM response"""

    response_id: str
    model_id: str
    example_id: str
    prompt_hash: str  # MD5 hash of prompt to detect duplicates
    created_at: str  # ISO timestamp
    response_time_seconds: float
    parsed_response: str
    parsed_data: Dict[str, Any]


@dataclass
class EvaluatedExample:
    """Complete evaluation for one example"""

    example_id: str
    model_results: Dict[str, "EvaluationMetrics"]  # model_id -> metrics
    created_at: str
    total_cost_usd: float
    total_tokens: int


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for one example"""

    final_answer_correct: bool
    final_answer_matches_type: bool
    step_completeness: float  # Percentage of ground truth steps covered
    step_order_correct: bool
    reasoning_similarity: float  # Jaccard index (0-1.0)
    has_hallucination: bool
    hallucination_rate: float  # Extra steps rate
    overall_reasoning_score: float


class ResultStore:
    """Persistent storage for evaluation results with caching"""

    def __init__(self, db_path: Path = None):
        # Default db_path to results/evaluations.db
        if db_path is None:
            db_path = Path(__file__).parent.parent / "results" / "evaluations.db"

        self.db_path = db_path
        self.results_dir = self.db_path.parent
        self.conn = None

    def _get_connection(self):
        """Get SQLite connection with row factory"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def _init_db(self):
        """Initialize database schema"""
        cursor = self._get_connection().cursor()

        # Tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                total_cost_usd REAL NOT NULL DEFAULT 0.0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL NOT NULL DEFAULT 0.0,
                dataset_version TEXT,
                num_examples INTEGER,
                num_models INTEGER,
                models TEXT,
                config TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id TEXT NOT NULL,
                example_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                final_answer_correct BOOLEAN NOT NULL,
                final_answer_matches_type BOOLEAN NOT NULL,
                step_completeness REAL NOT NULL,
                step_order_correct BOOLEAN NOT NULL,
                reasoning_similarity REAL NOT NULL,
                has_hallucination BOOLEAN NOT NULL,
                hallucination_rate REAL NOT NULL,
                overall_reasoning_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (eval_id) REFERENCES evaluations(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cached_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                response_time_seconds REAL,
                parsed_response TEXT,
                ttl_hours INTEGER NOT NULL DEFAULT 24
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cached_example ON cached_responses(example_id, model_id, prompt_hash, created_at)
        """)

        self._get_connection().commit()

    def _hash_prompt(self, prompt: str) -> str:
        """Create MD5 hash of prompt for caching"""
        return hashlib.md5(prompt.encode()).hexdigest()

    async def get_cached_response(
        self,
        example_id: str,
        model_id: str,
        prompt: str,
        ttl_hours: int = 24,
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""

        prompt_hash = self._hash_prompt(prompt)
        cursor = self._get_connection().cursor()

        # Check for non-expired cache entry
        cutoff_time = datetime.now().timestamp() - (ttl_hours * 3600)

        cursor.execute(
            """
            SELECT parsed_response, response_time_seconds
            FROM cached_responses
            WHERE example_id = ?
              AND model_id = ?
              AND prompt_hash = ?
              AND created_at > ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (example_id, model_id, prompt_hash, cutoff_time),
        )

        result = cursor.fetchone()

        if result:
            return {
                "parsed_response": result[0],
                "response_time_seconds": result[1],
            }

        return None

    async def cache_response(
        self,
        example_id: str,
        model_id: str,
        prompt: str,
        parsed_response: str,
        response_time_seconds: float,
    ):
        """Cache a response to avoid redundant API calls"""

        prompt_hash = self._hash_prompt(prompt)
        created_at = datetime.now().isoformat()

        cursor = self._get_connection().cursor()

        cursor.execute(
            """
            INSERT INTO cached_responses
            (example_id, model_id, prompt_hash, created_at, response_time_seconds, parsed_response)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                example_id,
                model_id,
                prompt_hash,
                created_at,
                response_time_seconds,
                parsed_response,
            ),
        )

        self._get_connection().commit()

        print(f"  ✓ Cached response for {model_id}")

    def create_evaluation(
        self,
        dataset_version: str,
        num_examples: int,
        num_models: int,
        models: str,
        config: str,
    ) -> str:
        """Create new evaluation record"""

        eval_id = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        cursor = self._get_connection().cursor()

        cursor.execute(
            """
            INSERT INTO evaluations
            (id, created_at, dataset_version, num_examples, num_models, models, config, total_cost_usd, total_tokens, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                eval_id,
                datetime.now().isoformat(),
                dataset_version,
                num_examples,
                num_models,
                models,
                config,
                0,
                0,
                0,
            ),
        )

        self._get_connection().commit()

        return eval_id

    def save_result(
        self,
        eval_id: str,
        model_id: str,
        example_id: str,
        metrics: EvaluationMetrics,
    ):
        """Save individual evaluation result"""

        cursor = self._get_connection().cursor()

        cursor.execute(
            """
            INSERT INTO evaluation_results
            (eval_id, example_id, model_id, final_answer_correct, final_answer_matches_type,
             step_completeness, step_order_correct, reasoning_similarity,
             has_hallucination, hallucination_rate, overall_reasoning_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                eval_id,
                example_id,
                model_id,
                metrics.final_answer_correct,
                metrics.final_answer_matches_type,
                metrics.step_completeness,
                metrics.step_order_correct,
                metrics.reasoning_similarity,
                metrics.has_hallucination,
                metrics.hallucination_rate,
                metrics.overall_reasoning_score,
                datetime.now().isoformat(),
            ),
        )

        self._get_connection().commit()

    def get_evaluation(
        self,
        eval_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get complete evaluation by ID"""

        cursor = self._get_connection().cursor()

        cursor.execute(
            """
            SELECT id, created_at, total_cost_usd, total_tokens, duration_seconds, dataset_version, num_examples, num_models, models, config
            FROM evaluations
            WHERE id = ?
        """,
            (eval_id,),
        )

        result = cursor.fetchone()

        if not result:
            return None

        # Get model results
        cursor.execute(
            """
            SELECT model_id, final_answer_correct, final_answer_matches_type,
                   step_completeness, step_order_correct, reasoning_similarity,
                   has_hallucination, hallucination_rate, overall_reasoning_score
            FROM evaluation_results
            WHERE eval_id = ?
            ORDER BY model_id
        """,
            (eval_id,),
        )

        model_results = {}
        for row in cursor.fetchall():
            model_results[row[0]] = EvaluationMetrics(
                final_answer_correct=row[1],
                final_answer_matches_type=row[2],
                step_completeness=row[3],
                step_order_correct=row[4],
                reasoning_similarity=row[5],
                has_hallucination=row[6],
                hallucination_rate=row[7],
                overall_reasoning_score=row[8],
            )

        return {
            "example_id": result[0],
            "model_results": model_results,
            "created_at": result[1],
            "total_cost_usd": result[2],
            "total_tokens": result[3],
            "duration_seconds": result[4],
        }

    def get_all_evaluations(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent evaluations"""

        cursor = self._get_connection().cursor()

        cursor.execute(
            """
            SELECT id, created_at, total_cost_usd, total_tokens, duration_seconds, dataset_version, num_examples
            FROM evaluations
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        return [
            {
                "id": row[0],
                "created_at": row[1],
                "total_cost_usd": row[2],
                "total_tokens": row[3],
                "duration_seconds": row[4],
                "dataset_version": row[5],
                "num_examples": row[6],
            }
            for row in cursor.fetchall()
        ]

    def cleanup_expired_cache(self, ttl_hours: int = 24):
        """Remove expired cache entries"""
        cutoff_time = datetime.now().timestamp() - (ttl_hours * 3600)

        cursor = self._get_connection().cursor()

        cursor.execute(
            """
            DELETE FROM cached_responses
            WHERE created_at < ?
        """,
            (cutoff_time,),
        )

        deleted = cursor.rowcount

        if deleted > 0:
            print(f"  ✓ Cleaned up {deleted} expired cache entries")

    def save_example_result(
        self, experiment_name: str, example_id: str, model_results: Dict[str, Any]
    ):
        """Save results for a single example"""
        # For simplicity, store as JSON file for now
        results_dir = self.results_dir / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_dir / f"{example_id}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {"example_id": example_id, "model_results": model_results},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def generate_summary(self, experiment_name: str) -> Path:
        """Generate summary report for experiment"""
        results_dir = self.results_dir / experiment_name
        summary_file = results_dir / "summary.json"

        # Collect all results
        all_results = []
        total_cost = 0.0
        total_tokens = 0

        for result_file in results_dir.glob("*.json"):
            if result_file.name == "summary.json":
                continue

            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append(data)

        # Calculate statistics
        summary = {
            "experiment_name": experiment_name,
            "total_examples": len(all_results),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "results": all_results,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary_file


if __name__ == "__main__":
    import sys

    db_path = Path(__file__).parent / "results" / "evaluations.db"

    store = ResultStore(db_path)

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        store._init_db()
        print(f"✓ Database initialized: {db_path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        store.cleanup_expired_cache(ttl_hours=24)
        print(f"✓ Cache cleanup complete")
    elif len(sys.argv) > 1 and sys.argv[1] == "get-eval":
        eval_id = sys.argv[2]
        evaluation = store.get_evaluation(eval_id)

        if evaluation:
            print(f"✓ Found evaluation: {eval_id}")
            print(f"    Created: {evaluation['created_at']}")
            print(f"    Models: {evaluation['model_results']}")
            print(f"    Total Cost: ${evaluation['total_cost_usd']:.2f}")
            print(f"    Total Tokens: {evaluation['total_tokens']}")
            print(f"    Duration: {evaluation['duration_seconds']:.1f}s")
        else:
            print(f"✗ Evaluation {eval_id} not found")
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        evaluations = store.get_all_evaluations(limit=10)

        if evaluations:
            print(f"\n{'=' * 60}")
            print("Recent Evaluations:")
            print("=" * 60)
            for i, eval in enumerate(evaluations, 1):
                print(f"\n{i}. {eval['id']}")
                print(f"    Created: {eval['created_at']}")
                print(f"    Cost: ${eval['total_cost_usd']:.2f}")
                print(f"    Tokens: {eval['total_tokens']}")
                print(f"    Duration: {eval['duration_seconds']:.1f}s")
                print(
                    f"    Dataset: {eval['dataset_version']} ({eval['num_examples']} examples)"
                )
        else:
            print("No evaluations found")
    else:
        print("Usage: python result_store.py [init|cleanup|get-eval <id>|list]")
