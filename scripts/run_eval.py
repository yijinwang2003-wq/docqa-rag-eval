"""Run evaluation across RAG configurations."""

from pathlib import Path
import argparse
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval_questions.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and answer quality."
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
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for per-question evaluation results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation questions for debugging.",
    )
    parser.add_argument(
        "--configs",
        default="all",
        help=(
            "Comma-separated configs to run, such as fixed+dense or "
            "semantic+hybrid. Use all to run every default config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from src.evaluation import DEFAULT_CONFIGS, DEFAULT_METRICS, run_evaluation
    except ModuleNotFoundError:
        _reexec_with_project_venv()
        raise

    configurations = _parse_config_specs(args.configs, DEFAULT_CONFIGS)

    print("Configurations:")
    for chunking, retriever in configurations:
        print(f"- {chunking} + {retriever}")

    _, summary_df = run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        configurations=configurations,
        metrics=DEFAULT_METRICS,
        limit=args.limit,
    )

    print(f"\nSaved results: {args.output}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


def _parse_config_specs(
    config_specs: str,
    default_configs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if config_specs.strip().lower() == "all":
        return default_configs

    available_configs = {
        f"{chunking}+{retriever}": (chunking, retriever)
        for chunking, retriever in default_configs
    }
    selected_configs = []

    for raw_spec in config_specs.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        if spec not in available_configs:
            valid_specs = ", ".join(sorted(available_configs))
            raise ValueError(f"Unsupported config '{spec}'. Valid configs: {valid_specs}")
        selected_configs.append(available_configs[spec])

    if not selected_configs:
        raise ValueError("At least one config must be selected.")

    return tuple(selected_configs)


def _reexec_with_project_venv() -> None:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable) == venv_python:
        return
    if os.environ.get("DOCQA_NO_VENV_REEXEC"):
        return

    os.environ["DOCQA_NO_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


if __name__ == "__main__":
    main()
