from data_preparation import prepare_data
from model_development import train_and_evaluate_models, store_results
import sqlite3
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_creditsafe():
    console.print(Panel.fit(
        "[bold blue]Starting CreditSafe: A Python-Driven Credit Risk Assessment Model[/bold blue]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Preparing data...", total=None)
        console.print("\n[bold green]Step 1: Data Preparation[/bold green]")
        prepare_data()

        progress.add_task(
            description="Developing and evaluating models...", total=None)
        console.print(
            "\n[bold green]Step 2: Model Development and Evaluation[/bold green]")
        results = train_and_evaluate_models()

        progress.add_task(description="Storing results...", total=None)
        console.print("\n[bold green]Step 3: Storing Results[/bold green]")
        store_results(results, 'credit_risk.db')

    console.print("\n[bold green]Step 4: Displaying Results[/bold green]")
    display_results()

    console.print(
        "\n[bold blue]CreditSafe project completed successfully![/bold blue]")


def display_results():
    with sqlite3.connect('credit_risk.db') as conn:
        results = pd.read_sql_query("SELECT * FROM model_results", conn)

    table = Table(title="Model Evaluation Results")
    table.add_column("Model", style="cyan")
    table.add_column("CV Score", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Precision", style="yellow")
    table.add_column("Recall", style="blue")
    table.add_column("F1 Score", style="red")
    table.add_column("AUC", style="purple")

    for _, row in results.iterrows():
        table.add_row(
            row['model'],
            f"{row['cv_score']:.4f}",
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1_score']:.4f}",
            f"{row['auc']:.4f}"
        )

    console.print(table)


if __name__ == "__main__":
    run_creditsafe()
