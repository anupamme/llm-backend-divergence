"""Multi-backend orchestrator CLI."""

from __future__ import annotations

import gc
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from divergence.backends import (
    Backend,
    LlamaCppQ4KMBackend,
    LlamaCppQ8Backend,
    MlxFp16Backend,
    MlxQ4Backend,
    MockBackend,
    TorchMpsBackend,
)
from divergence.evals import EvalItem, load_canary_set, load_gsm8k, load_mmlu_subset
from divergence.runner import RunConfig, RunSummary, run_eval

app = typer.Typer(help="LLM Backend Divergence Evaluation Framework")
console = Console()

BACKEND_REGISTRY: dict[str, type[Backend]] = {
    "mlx-fp16": MlxFp16Backend,
    "mlx-q4": MlxQ4Backend,
    "llamacpp-q8": LlamaCppQ8Backend,
    "llamacpp-q4km": LlamaCppQ4KMBackend,
    "torch-mps": TorchMpsBackend,
    "mock": MockBackend,
}

DEFAULT_BACKENDS = "mlx-fp16,mlx-q4,llamacpp-q8,llamacpp-q4km,torch-mps"

DATASET_REGISTRY: dict[str, Callable[[], list[EvalItem]]] = {
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu_subset,
    "canary": load_canary_set,
}

COOLDOWN_SECONDS = 5


def _default_output_path() -> str:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return f"results/run-{ts}.db"


def _estimate_runtime(
    backend: Backend,
    total_items: int,
    max_tokens: int,
) -> float:
    """Estimate total runtime in seconds based on a warmup prompt."""
    result = backend.generate(
        "Hello, how are you?",
        max_tokens=max_tokens,
        temperature=0.0,
        seed=0,
    )
    per_item_s = result.total_latency_ms / 1000.0
    return per_item_s * total_items


def _print_summary_table(summaries: list[tuple[str, str, RunSummary]]) -> None:
    """Print a rich table of run summaries."""
    table = Table(title="Run Summary")
    table.add_column("Backend", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Completed", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Wall Time (s)", justify="right")
    table.add_column("TTFT p50 (ms)", justify="right")
    table.add_column("TTFT p95 (ms)", justify="right")
    table.add_column("Latency p50 (ms)", justify="right")
    table.add_column("Latency p95 (ms)", justify="right")

    for backend_name, dataset_name, summary in summaries:
        table.add_row(
            backend_name,
            dataset_name,
            str(summary.completed),
            str(summary.errors),
            f"{summary.total_wall_time_s:.1f}",
            f"{summary.ttft_p50_ms:.1f}",
            f"{summary.ttft_p95_ms:.1f}",
            f"{summary.latency_p50_ms:.1f}",
            f"{summary.latency_p95_ms:.1f}",
        )

    console.print(table)


@app.command()
def run(
    backends: str = typer.Option(
        DEFAULT_BACKENDS,
        help="Comma-separated backend names to run.",
    ),
    datasets: str = typer.Option(
        "gsm8k,mmlu,canary",
        help="Comma-separated dataset names to evaluate.",
    ),
    output: str | None = typer.Option(
        None,
        help="Output SQLite database path. Default: results/run-<timestamp>.db",
    ),
    max_tokens: int = typer.Option(256, help="Maximum tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    seed: int = typer.Option(42, help="Random seed."),
    model_id: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="Model ID to load."),
    resume: bool = typer.Option(False, help="Resume incomplete runs."),
    no_chat_template: bool = typer.Option(
        False,
        help="Skip chat template formatting (pass raw prompts to backends).",
    ),
) -> None:
    """Run backends across datasets and persist results to SQLite."""
    output_path = output or _default_output_path()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    backend_names = [b.strip() for b in backends.split(",")]
    dataset_names = [d.strip() for d in datasets.split(",")]

    for name in backend_names:
        if name not in BACKEND_REGISTRY:
            console.print(f"[red]Unknown backend: {name}[/red]")
            raise typer.Exit(1)
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            console.print(f"[red]Unknown dataset: {name}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Loading {len(dataset_names)} dataset(s)...[/bold]")
    loaded_datasets: dict[str, list[EvalItem]] = {}
    for name in dataset_names:
        loaded_datasets[name] = DATASET_REGISTRY[name]()
        console.print(f"  {name}: {len(loaded_datasets[name])} items")

    total_items = sum(len(ds) for ds in loaded_datasets.values()) * len(backend_names)
    console.print(
        f"\n[bold]Plan:[/bold] {len(backend_names)} backend(s) x "
        f"{len(dataset_names)} dataset(s) = {total_items} total evaluations"
    )

    # Warmup and estimate
    first_backend_cls = BACKEND_REGISTRY[backend_names[0]]
    first_backend = first_backend_cls()
    try:
        first_backend.load(model_id)
        estimated_s = _estimate_runtime(first_backend, total_items, max_tokens)
        console.print(
            f"[bold]Estimated runtime:[/bold] {estimated_s / 60:.1f} minutes "
            f"(based on warmup TTFT)"
        )
        first_backend.unload()
    except Exception as e:
        console.print(f"[yellow]Warmup failed: {e}. Skipping estimate.[/yellow]")
        try:
            first_backend.unload()
        except Exception:
            pass

    gc.collect()
    time.sleep(1)

    summaries: list[tuple[str, str, RunSummary]] = []

    for i, backend_name in enumerate(backend_names):
        console.print(
            f"\n[bold cyan]Backend {i + 1}/{len(backend_names)}: "
            f"{backend_name}[/bold cyan]"
        )
        backend_cls = BACKEND_REGISTRY[backend_name]
        backend = backend_cls()

        try:
            backend.load(model_id)
        except Exception as e:
            console.print(f"[red]Failed to load {backend_name}: {e}. Skipping.[/red]")
            continue

        for dataset_name in dataset_names:
            dataset = loaded_datasets[dataset_name]
            config = RunConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                output_db_path=output_path,
                resume=resume,
                dataset_name=dataset_name,
                apply_chat_template=not no_chat_template,
                model_id=model_id,
            )

            console.print(
                f"  Running {backend_name} on {dataset_name} ({len(dataset)} items)..."
            )
            summary = run_eval(backend, dataset, config)
            summaries.append((backend_name, dataset_name, summary))
            console.print(
                f"  Done: {summary.completed} completed, "
                f"{summary.errors} errors, "
                f"{summary.total_wall_time_s:.1f}s"
            )

        backend.unload()
        gc.collect()

        if i < len(backend_names) - 1:
            console.print(f"  Cooling down {COOLDOWN_SECONDS}s before next backend...")
            time.sleep(COOLDOWN_SECONDS)

    console.print()
    _print_summary_table(summaries)
    console.print(f"\n[bold green]Results saved to:[/bold green] {output_path}")


@app.command("list-backends")
def list_backends() -> None:
    """List available backends."""
    table = Table(title="Available Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Class")

    for name, cls in BACKEND_REGISTRY.items():
        table.add_row(name, cls.__name__)

    console.print(table)


@app.command("list-datasets")
def list_datasets() -> None:
    """List available evaluation datasets."""
    table = Table(title="Available Datasets")
    table.add_column("Name", style="magenta")
    table.add_column("Loader")

    for name, loader in DATASET_REGISTRY.items():
        table.add_row(name, loader.__name__)

    console.print(table)


@app.command()
def summarize(
    db_path: str = typer.Argument(help="Path to the results SQLite database."),
) -> None:
    """Print a summary table of results from a database."""
    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    runs = conn.execute(
        "SELECT run_id, backend_name, dataset_name FROM runs"
    ).fetchall()

    if not runs:
        console.print("[yellow]No runs found in database.[/yellow]")
        conn.close()
        return

    table = Table(title=f"Results from {db_path}")
    table.add_column("Backend", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Items", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")

    for run_row in runs:
        run_id = run_row["run_id"]
        stats = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN finish_reason = 'error' THEN 1 ELSE 0 END) as errors, "
            "AVG(ttft_ms) as avg_ttft, "
            "AVG(total_latency_ms) as avg_latency "
            "FROM inference_results WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        if stats and stats["total"] > 0:
            table.add_row(
                run_row["backend_name"],
                run_row["dataset_name"],
                str(stats["total"]),
                str(stats["errors"] or 0),
                f"{stats['avg_ttft'] or 0:.1f}",
                f"{stats['avg_latency'] or 0:.1f}",
            )

    console.print(table)
    conn.close()


if __name__ == "__main__":
    app()
