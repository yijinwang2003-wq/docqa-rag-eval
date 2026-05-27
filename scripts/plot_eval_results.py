"""Plot grouped RAGAS metric means from evaluation results."""

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import DEFAULT_METRICS, DEFAULT_OUTPUT_PATH  # noqa: E402


DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "eval_metrics.png"
CONFIG_ORDER = (
    "fixed+dense",
    "fixed+hybrid",
    "semantic+dense",
    "semantic+hybrid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped bar chart from RAGAS evaluation results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to outputs/eval_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path for the generated PNG chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pd = _import_pandas()
    plt = _import_pyplot()

    results_df = pd.read_csv(args.input)
    metric_columns = [metric for metric in DEFAULT_METRICS if metric in results_df]
    if not metric_columns:
        raise ValueError(
            f"No metric columns found in {args.input}. Expected: {DEFAULT_METRICS}"
        )

    summary_df = (
        results_df.groupby("configuration", as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .set_index("configuration")
        .reindex(CONFIG_ORDER)
        .dropna(how="all")
    )
    if summary_df.empty:
        raise ValueError(f"No recognized configurations found in {args.input}.")

    ax = summary_df.plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Mean metric score")
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS evaluation metrics by RAG configuration")
    ax.legend(title="Metric")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(f"Saved chart: {args.output}")


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "The pandas package is required for plotting. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return pd


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "The matplotlib package is required for plotting. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return plt


if __name__ == "__main__":
    main()
