import contextlib
import io
import sys
from pathlib import Path

import click
import logfire
import typer
from rich.console import Console

from timecopilot.agent import TimeCopilot as TimeCopilotAgent

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


class TimeCopilot:
    def __init__(self):
        self.console = Console()

    @contextlib.contextmanager
    def _capture_prints_static(self):
        """Capture print statements and format them nicely."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Process captured output
            stdout_content = stdout_capture.getvalue().strip()
            stderr_content = stderr_capture.getvalue().strip()

            if stdout_content:
                # Format as subdued info
                for line in stdout_content.split("\n"):
                    if line.strip():
                        self.console.print(f"[dim]  → {line}[/dim]")

            if stderr_content:
                # Format as subdued warning
                for line in stderr_content.split("\n"):
                    if line.strip():
                        self.console.print(f"[dim yellow]  ⚠ {line}[/dim yellow]")

    def forecast(
        self,
        path: str | Path,
        llm: str = "openai:gpt-4o-mini",
        freq: str | None = None,
        h: int | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        retries: int = 3,
    ):
        with (
            self.console.status(
                "[bold blue]TimeCopilot is navigating through time...[/bold blue]"
            ),
            self._capture_prints_static(),
        ):
            forecasting_agent = TimeCopilotAgent(llm=llm, retries=retries)
            result = forecasting_agent.analyze(
                df=path,
                freq=freq,
                h=h,
                seasonality=seasonality,
                query=query,
            )

        result.output.prettify(
            self.console,
            features_df=result.features_df,
            eval_df=result.eval_df,
            fcst_df=result.fcst_df,
        )


app = typer.Typer(
    name="timecopilot",
    help="TimeCopilot - Your GenAI Forecasting Agent",
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    llm: str = typer.Option(
        "openai:gpt-4o-mini", "--llm", "-l", help="LLM to use for the agent"
    ),
):
    """
    TimeCopilot - Your GenAI Forecasting Agent

    Just run 'timecopilot' to start chatting with your AI forecasting companion!
    Talk naturally about your data and get intelligent predictions and insights.
    """
    if ctx.invoked_subcommand is None:
        llm_explicit = ctx.get_parameter_source("llm") == click.core.ParameterSource.COMMANDLINE
        from timecopilot._tui import TimeCopilotApp

        tui_app = TimeCopilotApp(llm=llm, llm_explicit=llm_explicit)
        tui_app.run()


@app.command("forecast")
def forecast_command(
    path: str = typer.Argument(..., help="Path to CSV file or URL"),
    llm: str = typer.Option(
        "openai:gpt-4o-mini", "--llm", "-l", help="LLM to use for forecasting"
    ),
    freq: str = typer.Option(None, "--freq", "-f", help="Data frequency"),
    h: int = typer.Option(None, "--horizon", "-h", help="Forecast horizon"),
    seasonality: int = typer.Option(None, "--seasonality", "-s", help="Seasonality"),
    query: str = typer.Option(None, "--query", "-q", help="Additional query"),
    retries: int = typer.Option(3, "--retries", "-r", help="Number of retries"),
):
    """Generate forecast (legacy one-shot mode)."""
    tc = TimeCopilot()
    tc.forecast(
        path=path,
        llm=llm,
        freq=freq,
        h=h,
        seasonality=seasonality,
        query=query,
        retries=retries,
    )


def main():
    app()


if __name__ == "__main__":
    main()
