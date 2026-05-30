"""Compare baseline RAG and agentic RAG on the evaluation set."""

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.trajectory import (
    DEFAULT_AGENT_EVAL_MARKDOWN_PATH,
    DEFAULT_AGENT_EVAL_OUTPUT_PATH,
    DEFAULT_AGENT_EVAL_PARTIAL_OUTPUT_PATH,
    DEFAULT_TRAJECTORY_METRICS,
    run_trajectory_evaluation,
)


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval_questions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline RAG against the agentic RAG workflow."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to evaluation questions JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AGENT_EVAL_OUTPUT_PATH,
        help="CSV path for per-question comparison results.",
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=DEFAULT_AGENT_EVAL_PARTIAL_OUTPUT_PATH,
        help="Checkpoint CSV path for resumable partial results.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_AGENT_EVAL_MARKDOWN_PATH,
        help="Markdown summary table path. Use 'none' to skip.",
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default="hybrid",
        help="Retriever strategy to use for both systems. Defaults to hybrid.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation questions for smoke tests.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_TRAJECTORY_METRICS),
        help="Comma-separated answer metrics to judge. Defaults to answer_relevancy,faithfulness.",
    )
    parser.add_argument(
        "--skip-answer-scoring",
        action="store_true",
        help="Only run trajectory/latency metrics; do not call the answer judge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_path = (
        None
        if str(args.markdown_output).strip().lower() == "none"
        else args.markdown_output
    )
    metrics = _parse_metrics(args.metrics)
    result = run_trajectory_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        partial_output_path=args.partial_output,
        markdown_path=markdown_path,
        retriever_name=args.retriever,
        metrics=metrics,
        limit=args.limit,
        score_answers=not args.skip_answer_scoring,
    )

    print(f"\nSaved detailed results: {result.output_path}")
    print(f"Saved checkpoint results: {result.partial_output_path}")
    if result.markdown_path is not None:
        print(f"Saved markdown summary: {result.markdown_path}")
    print("\nSummary:")
    print(result.summary_df.to_string(index=False))


def _parse_metrics(raw_metrics: str) -> tuple[str, ...]:
    metrics = tuple(
        metric.strip()
        for metric in raw_metrics.split(",")
        if metric.strip()
    )
    if not metrics:
        raise ValueError("At least one metric must be provided.")
    return metrics


if __name__ == "__main__":
    main()
