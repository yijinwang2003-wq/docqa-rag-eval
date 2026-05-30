"""Run the single-agent RAG workflow from the command line."""

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the agentic RAG workflow.")
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default="dense",
        help="Retriever strategy to use. Defaults to dense.",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Print the agent trajectory after the answer.",
    )
    parser.add_argument("question", nargs="+", help="Question to ask the agent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()
    result = run_agent(question, retriever_name=args.retriever)

    print("\nAnswer:\n")
    print(result.get("answer", "").strip())

    print("\nConfidence:")
    print(f"{result.get('confidence', 0.0):.2f}")
    if result.get("fallback"):
        print(f"Fallback: {result.get('fallback_reason', 'low_confidence')}")

    print("\nSources:")
    sources = result.get("sources", [])
    if sources:
        for source in sources:
            print(f"- {source}")
    else:
        print("- No sources retrieved. Run scripts/ingest_docs.py first.")

    if args.trajectory:
        print("\nTrajectory:")
        for index, step in enumerate(result.get("trajectory", []), start=1):
            print(f"{index}. {step['step']}: {step['output']}")


if __name__ == "__main__":
    main()
