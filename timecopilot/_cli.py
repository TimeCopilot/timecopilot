import asyncio
import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

import click
import logfire
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

from timecopilot.agent import AsyncTimeCopilot
from timecopilot.agent import TimeCopilot as TimeCopilotAgent

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

CONFIG_DIR = Path(__file__).resolve().parent

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GOOGLE_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_PROJECT",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "CO_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok": "GROK_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "huggingface": "HF_TOKEN",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "moonshotai": "MOONSHOTAI_API_KEY",
    "heroku": "HEROKU_INFERENCE_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "ollama": "OLLAMA_API_KEY",
    "github": "GITHUB_API_KEY",
}

MODEL_ENV_VAR = "TIMECOPILOT_MODEL"


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


class InteractiveChat:
    """Simplified interactive chat for TimeCopilot."""

    DEFAULT_PLACEHOLDER = "Let's forecast! Enter a file path or URL"
    FOLLOWUP_PLACEHOLDER = "Try: 'show me the plot', 'explain this', or try a different model"

    def __init__(self, llm: str = "openai:gpt-4o-mini", llm_explicit: bool = False):
        self.llm = llm
        self.llm_explicit = llm_explicit
        self.agent: AsyncTimeCopilot | None = None
        self.console = Console()
        self._placeholder = self.DEFAULT_PLACEHOLDER

    @contextlib.contextmanager
    def _capture_prints(self):
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

    def _print_welcome(self):
        """Print welcome message and instructions."""
        welcome_text = """
# 👋 Hi there! I'm TimeCopilot, the GenAI Forecasting Agent!

I'm here to help you understand your data and predict the future. Just talk to me 
naturally - no complex commands needed! I can seamlessly work with different models 
and explain why forecasts look the way they do, not just give you numbers.

## 💭 **Natural conversation examples:**
- "I have sales data at /path/to/sales.csv, forecast the next 6 months"
- "Can you spot any weird patterns in my server data?"
- "Show me a plot of the forecast you just created"
- "Why does this forecast look different from the previous one?"
- "What should I expect for user engagement next month?"
- "How confident are you about next week's predictions?"
- "Compare Chronos and TimesFM models for my dataset"

## 🎯 **Try saying:**
- "Forecast this dataset: /path/to/sales_data.csv"
- "Analyze anomalies in s3://bucket/server-metrics.csv"
- "What will my website traffic look like next month?"
- "Show me a plot of the forecast vs actual values"
- "Which forecasting model gives the most accurate results?"

The same workflow can be used for monitoring as well as forecasting.
Ready to dive into your data? Just tell me what you'd like to explore! 🚀

## ⌨️  **Commands:**
| Command | Description |
|---------|-------------|
| help or ? | Show this welcome message |
| llm | Switch to a different model |
| back | Go back to the previous step |
| exit, quit, or bye | Exit TimeCopilot |
        """

        panel = Panel(
            Markdown(welcome_text),
            title="[bold white]Welcome to TimeCopilot[/bold white]",
            border_style="white",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _save_model(self, model_name: str) -> None:
        """Persist selected model to .env."""
        env_path = str(CONFIG_DIR / ".env")
        dotenv.set_key(env_path, MODEL_ENV_VAR, model_name)

    def _has_valid_saved_config(self, saved_model: str) -> bool:
        """Check if saved model + API key are both present."""
        provider = saved_model.split(":")[0]
        env_var = PROVIDER_API_KEY_ENV.get(provider)
        if env_var is None:
            return True  # unknown provider, let pydantic-ai handle it
        return bool(os.environ.get(env_var))

    def _print_welcome_back(self) -> None:
        """Short greeting for returning users."""
        welcome_text = f"""\
# 👋 Welcome back!

**Current model:** {self.llm}

## ⌨️  **Commands:**
| Command | Description |
|---------|-------------|
| help or ? | Show full help |
| llm | Switch to a different model |
| exit, quit, or bye | Exit TimeCopilot |"""

        panel = Panel(
            Markdown(welcome_text),
            title="[bold white]TimeCopilot[/bold white]",
            border_style="white",
            padding=(1, 2),
        )
        self.console.print(panel)

    async def _run_onboarding(self) -> None:
        """Interactive model selection + API key setup with back support."""
        while True:
            session = PromptSession()
            answer = await session.prompt_async(
                HTML("<b><ansiblue>Would you like to select an LLM? (yes/no)</ansiblue></b> "),
                placeholder=HTML("<ansigray>(default: yes) </ansigray><blink>█</blink>"),
                cursor=CursorShape.BEAM,
                style=Style.from_dict({"blink": "blink"}),
            )
            if answer.strip().lower() not in ("n", "no"):
                selected = await self._select_model()
                if selected is None:
                    # User typed 'back' at provider selection
                    continue
                self.llm = selected
                if await self._ensure_api_key():
                    self._save_model(self.llm)
                    break
                # User typed 'back' at API key — re-ask from the top
                self.llm = "openai:gpt-4o-mini"
                continue
            else:
                if await self._ensure_api_key():
                    self._save_model(self.llm)
                    break
                # User typed 'back' at API key — re-ask from the top
                continue

    async def _select_model(self) -> str | None:
        """Two-step interactive model selection: provider then model."""
        from pydantic_ai.models import KnownModelName
        from typing_inspection.introspection import get_literal_values

        default_model = "openai:gpt-4o-mini"

        # Get all models with a provider prefix
        all_models = sorted(
            n
            for n in get_literal_values(KnownModelName.__value__)
            if ":" in str(n)
        )

        # Group by provider
        providers: dict[str, list[str]] = defaultdict(list)
        for model in all_models:
            provider = str(model).split(":")[0]
            providers[provider].append(str(model))

        # Sort providers, putting common ones first
        priority = [
            "openai", "anthropic", "google-gla",
            "google-vertex", "groq", "mistral",
        ]
        sorted_providers = [p for p in priority if p in providers]
        sorted_providers += sorted(p for p in providers if p not in priority)

        while True:
            # Step 1: Pick a provider
            self.console.print()
            table = Table(
                title="[bold blue]Select a provider[/bold blue]",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="bold", width=4)
            table.add_column("Provider", style="green")
            table.add_column("Models", style="dim")

            for i, provider in enumerate(sorted_providers, 1):
                count = len(providers[provider])
                table.add_row(str(i), provider, f"{count} models")

            self.console.print(table)

            default_provider_idx = sorted_providers.index("openai") + 1
            session = PromptSession()
            choice = await session.prompt_async(
                HTML("<b><ansiblue>Select a provider</ansiblue></b>: "),
                placeholder=HTML(f"<ansigray>enter a number, name, or 'back' (default: {default_provider_idx}) </ansigray><blink>█</blink>"),
                cursor=CursorShape.BEAM,
                style=Style.from_dict({"blink": "blink"}),
            )
            if not choice.strip():
                choice = str(default_provider_idx)
            if choice.strip().lower() in ("exit", "quit", "bye"):
                raise SystemExit(0)
            if choice.strip().lower() == "back":
                return None
            selected_provider: str | None = None
            try:
                provider_idx = int(choice) - 1
                if not (0 <= provider_idx < len(sorted_providers)):
                    raise ValueError
                selected_provider = sorted_providers[provider_idx]
            except ValueError:
                # Try matching by name
                query = choice.strip().lower()
                exact = [p for p in sorted_providers if p == query]
                if exact:
                    selected_provider = exact[0]
                else:
                    partial = [p for p in sorted_providers if query in p]
                    if len(partial) == 1:
                        selected_provider = partial[0]
                    elif len(partial) > 1:
                        self.console.print(
                            f"[yellow]Ambiguous match: {', '.join(partial)}. "
                            f"Please pick a number.[/yellow]"
                        )
                        continue
                    else:
                        self.console.print(
                            f"[yellow]No provider matching '{choice.strip()}'. "
                            f"Please try again.[/yellow]"
                        )
                        continue

            # Step 2: Pick a model from that provider
            models = providers[selected_provider]
            self.console.print()
            table = Table(
                title=f"[bold blue]{selected_provider} Models[/bold blue]",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="bold", width=4)
            table.add_column("Model", style="green")

            for i, model in enumerate(models, 1):
                # Show model name without provider prefix for readability
                display = model.split(":", 1)[1]
                table.add_row(str(i), display)

            self.console.print(table)

            # Find default model index if it belongs to this provider
            default_idx = "1"
            if default_model in models:
                default_idx = str(models.index(default_model) + 1)

            session = PromptSession()
            choice = await session.prompt_async(
                HTML("<b><ansiblue>Select a model</ansiblue></b>: "),
                placeholder=HTML(f"<ansigray>enter a number, name, or 'back' (default: {default_idx}) </ansigray><blink>█</blink>"),
                cursor=CursorShape.BEAM,
                style=Style.from_dict({"blink": "blink"}),
            )
            if not choice.strip():
                choice = default_idx
            if choice.strip().lower() in ("exit", "quit", "bye"):
                raise SystemExit(0)
            if choice.strip().lower() == "back":
                continue
            try:
                model_idx = int(choice) - 1
                if not (0 <= model_idx < len(models)):
                    raise ValueError
                return models[model_idx]
            except ValueError:
                # Try matching by name (with or without provider prefix)
                query = choice.strip().lower()
                exact = [m for m in models if m.lower() == query or m.split(":", 1)[1].lower() == query]
                if exact:
                    return exact[0]
                partial = [m for m in models if query in m.lower()]
                if len(partial) == 1:
                    return partial[0]
                if len(partial) > 1:
                    self.console.print(
                        f"[yellow]Ambiguous match:[/yellow]"
                    )
                    for i, m in enumerate(partial, 1):
                        self.console.print(f"  [dim]{i}.[/dim] {m.split(':', 1)[1]}")
                    self.console.print("[yellow]Please pick a number.[/yellow]")
                    fallback = models[0]
                    return fallback
                fallback = models[0]
                self.console.print(
                    f"[yellow]No model matching '{choice.strip()}'. "
                    f"Using: {fallback}[/yellow]"
                )
                return fallback

    async def _ensure_api_key(self) -> bool:
        """Check the API key env var for the selected provider; prompt and save to .env if missing.
        Returns False if the user wants to go back to model selection."""
        provider = self.llm.split(":")[0]
        env_var = PROVIDER_API_KEY_ENV.get(provider)
        if env_var is None:
            return True  # unknown provider, let pydantic-ai handle it
        if os.environ.get(env_var):
            self.console.print(f"[dim]✓ {env_var} is already set[/dim]")
            return True
        # Prompt the user until a valid key is provided
        self.console.print(
            f"\n[bold yellow]⚠ {env_var} is not set.[/bold yellow]"
        )
        while True:
            session = PromptSession()
            key = await session.prompt_async(
                HTML(f"<b><ansiblue>Enter your {env_var}</ansiblue></b>: "),
                placeholder=HTML(f"<ansigray>paste your API key, or 'back' </ansigray><blink>█</blink>"),
                is_password=True,
                cursor=CursorShape.BEAM,
                style=Style.from_dict({"blink": "blink"}),
            )
            stripped = key.strip().lower()
            if stripped in ("exit", "quit", "bye"):
                raise SystemExit(0)
            if stripped == "back":
                return False
            if stripped in ("help", "?"):
                self.console.print(
                    f"[dim]Set the [bold]{env_var}[/bold] environment variable or "
                    f"enter your API key here to save it to timecopilot/.env[/dim]"
                )
                continue
            if key.strip():
                os.environ[env_var] = key.strip()
                # Persist to timecopilot/.env so it's available next time
                env_path = CONFIG_DIR / ".env"
                dotenv.set_key(str(env_path), env_var, key.strip())
                self.console.print(
                    f"[green]✓ {env_var} saved to timecopilot/.env and set for this session[/green]"
                )
                return True
            self.console.print(
                f"[yellow]No key provided. Please enter your {env_var} to continue.[/yellow]"
            )

    def _extract_file_path(self, user_input: str) -> str | None:
        """Extract file path from user input."""
        words = user_input.split()
        for word in words:
            if (
                word.endswith(".csv")
                or word.endswith(".parquet")
                or word.startswith("http")
                or "/" in word
                or "\\" in word
            ):
                return word
        return None

    async def _handle_command(self, user_input: str) -> bool:
        """Handle user command. Returns False to exit."""
        user_input_lower = user_input.strip().lower()

        # Handle exit commands
        if user_input_lower in ["exit", "quit", "bye"]:
            return False

        # Handle help
        if user_input_lower in ["help", "?"]:
            self._print_welcome()
            return True

        # Handle LLM switch
        if user_input_lower == "llm":
            selected = await self._select_model()
            if selected is None:
                return True
            self.llm = selected
            await self._ensure_api_key()
            self._save_model(self.llm)
            self.agent = None  # reset agent so it uses the new model
            self.console.print(
                f"\n[bold green]Switched to model:[/bold green] [cyan]{self.llm}[/cyan]"
            )
            self._placeholder = self.DEFAULT_PLACEHOLDER
            return True

        # Check if we have an agent and can query
        if self.agent and self.agent.is_queryable():
            # Agent is ready for follow-up queries
            try:
                with (
                    Status(
                        "[bold blue]TimeCopilot is thinking...[/bold blue]",
                        console=self.console,
                    ),
                    self._capture_prints(),
                ):
                    result = await self.agent.query(user_input)

                # Display result
                response_panel = Panel(
                    result.output,
                    title="[bold cyan]TimeCopilot Response[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
                self.console.print(response_panel)

            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            # Handle initial data loading
            file_path = self._extract_file_path(user_input)

            if file_path:
                try:
                    # Create agent if needed
                    if not self.agent:
                        self.agent = AsyncTimeCopilot(llm=self.llm)

                    # Run analysis
                    with (
                        Status(
                            "[bold blue]Loading data and analyzing...[/bold blue]",
                            console=self.console,
                        ),
                        self._capture_prints(),
                    ):
                        result = await self.agent.analyze(
                            df=file_path, query=user_input
                        )

                    # Display conversational summary
                    selected_model = getattr(result.output, "selected_model", "Unknown")
                    horizon = (
                        len(getattr(result, "fcst_df", []))
                        if hasattr(result, "fcst_df")
                        else 0
                    )
                    beats_naive = getattr(
                        result.output, "is_better_than_seasonal_naive", False
                    )
                    performance_msg = (
                        "performing well" if beats_naive else "needs improvement"
                    )

                    self.console.print(
                        "\n[bold green]Great! I've completed the analysis.[/bold green]"
                    )
                    self.console.print(
                        f"[cyan]Selected Model:[/cyan] {selected_model} "
                        f"({performance_msg})"
                    )

                    if horizon > 0:
                        self.console.print(
                            f"[cyan]Forecast:[/cyan] Generated {horizon} future periods"
                        )

                    # Check for anomalies
                    if (
                        hasattr(result, "anomalies_df")
                        and result.anomalies_df is not None
                    ):
                        anomaly_cols = [
                            col
                            for col in result.anomalies_df.columns
                            if col.endswith("-anomaly")
                        ]
                        total_anomalies = sum(
                            result.anomalies_df[col].sum() for col in anomaly_cols
                        )
                        if total_anomalies > 0:
                            total_points = len(result.anomalies_df)
                            anomaly_rate = (total_anomalies / total_points) * 100
                            self.console.print(
                                f"[red]Found {total_anomalies} anomalies "
                                f"({anomaly_rate:.1f}%)[/red]"
                            )
                            self.console.print(
                                "[dim yellow]So the same workflow can be used for "
                                "monitoring as well as forecasting.[/dim yellow]"
                            )
                    # User response
                    user_response = result.output.user_query_response
                    if user_response:
                        response_panel = Panel(
                            user_response,
                            title="[bold cyan]TimeCopilot Response[/bold cyan]",
                            border_style="cyan",
                            padding=(1, 2),
                        )
                        self.console.print(response_panel)

                    self._placeholder = self.FOLLOWUP_PLACEHOLDER

                except Exception as e:
                    self.console.print(f"[bold red]Error loading data:[/bold red] {e}")
            else:
                self.console.print(
                    "[bold yellow]Please provide a file path or URL to get started."
                    "[/bold yellow]"
                )
                self.console.print("[dim]Example: forecast /path/to/data.csv[/dim]")

        return True

    async def run(self):
        """Run the interactive chat session."""
        # Load saved config from timecopilot/.env
        dotenv.load_dotenv(CONFIG_DIR / ".env")

        saved_model = os.environ.get(MODEL_ENV_VAR)

        if self.llm_explicit:
            # --llm was passed explicitly: skip model picker, ensure API key, save
            self._print_welcome()
            await self._ensure_api_key()
            self._save_model(self.llm)
        elif saved_model and self._has_valid_saved_config(saved_model):
            # Returning user with valid saved config
            self.llm = saved_model
            self._print_welcome_back()
        else:
            # First time or invalid saved config: full onboarding
            self._print_welcome()
            await self._run_onboarding()

        self.console.print(
            f"\n[bold green]Using model:[/bold green] [cyan]{self.llm}[/cyan]"
        )

        try:
            while True:
                try:
                    self.console.print()
                    session = PromptSession()
                    user_input = await session.prompt_async(
                        HTML("<b><ansiblue>TimeCopilot</ansiblue></b>: "),
                        placeholder=HTML(f"<ansigray>{self._placeholder} </ansigray><blink>█</blink>"),
                        cursor=CursorShape.BEAM,
                        style=Style.from_dict({"blink": "blink"}),
                    )
                except KeyboardInterrupt:
                    self.console.print(
                        "\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]"
                    )
                    continue

                if not user_input.strip():
                    continue

                should_continue = await self._handle_command(user_input)
                if not should_continue:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.console.print(
                "\n[bold blue]👋 Thanks for using TimeCopilot! "
                "See you next time![/bold blue]"
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
        chat = InteractiveChat(llm=llm, llm_explicit=llm_explicit)
        asyncio.run(chat.run())


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
