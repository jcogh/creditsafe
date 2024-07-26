import warnings
from data_preparation import prepare_data
from model_development import load_data, train_and_evaluate_models, store_results, display_feature_importance
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_creditsafe():
    console.print(Panel.fit(
        "[bold blue]Starting CreditSafe: A Python-Driven Credit Risk Assessment Model[/bold blue]"))

    console.print("\n[bold green]Step 1: Data Preparation[/bold green]")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    console.print(
        "\n[bold green]Step 2: Model Development and Evaluation[/bold green]")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    console.print("\n[bold green]Step 3: Storing Results[/bold green]")
    store_results(results, 'credit_risk.db')

    console.print("\n[bold green]Step 4: Displaying Results[/bold green]")
    for name, model, cv_score, evaluation in results:
        console.print(f"[cyan]{name}:[/cyan]")
        console.print(f"  CV Score: {cv_score:.4f}")
        console.print(f"  Test Set Performance:")
        for metric, value in evaluation.items():
            console.print(f"    {metric}: {value:.4f}")
        console.print("")

    console.print("\n[bold green]Step 5: Feature Importance[/bold green]")
    best_model = max(results, key=lambda x: x[3]['auc'])[1]
    console.print(f"Best model: {best_model.best_estimator_}")
    display_feature_importance(
        best_model.best_estimator_.named_steps['classifier'], X_train.columns)

    console.print(
        "\n[bold blue]CreditSafe project completed successfully![/bold blue]")


if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    try:
        run_creditsafe()
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
        console.print("[yellow]Traceback:[/yellow]")
        import traceback
        console.print(traceback.format_exc())
