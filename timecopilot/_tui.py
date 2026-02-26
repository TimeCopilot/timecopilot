"""Bloomberg Terminal-style TUI for TimeCopilot."""

from __future__ import annotations

import _posixsubprocess
import asyncio
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import pandas as pd
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

from timecopilot.agent import ForecastAgentOutput
from timecopilot.llm_config import (
    MatchStatus,
    extract_file_path,
    get_provider_env_var,
    get_provider_for_model,
    has_valid_saved_config,
    list_providers,
    load_env,
    load_saved_model,
    match_model,
    match_provider,
    save_api_key,
    save_model,
)

# ---------------------------------------------------------------------------
# Subprocess safety for Textual's _PrintCapture (fileno() == -1).
# Two patches, each solving a different symptom:
#
# 1. _posixsubprocess.fork_exec — prevents "bad value(s) in fds_to_keep"
#    crash by filtering negative fds out of the keep-set before it reaches C.
#    Catches ALL callers: subprocess, multiprocessing, logfire, etc.
#
# 2. subprocess.Popen.__init__ — prevents TUI freeze by redirecting child
#    stdin/stdout/stderr to DEVNULL when in Textual capture mode.  Without
#    this, child processes inherit the real terminal fd and may write escape
#    sequences that corrupt Textual's display.
# ---------------------------------------------------------------------------

# --- Patch 1: fork_exec fd filter ---

_orig_fork_exec = _posixsubprocess.fork_exec


def _safe_fork_exec(*args):
    args = list(args)
    args[3] = tuple(fd for fd in args[3] if fd >= 0)
    return _orig_fork_exec(*args)


_posixsubprocess.fork_exec = _safe_fork_exec


# --- Patch 2: redirect child stdio to DEVNULL in Textual mode ---

_orig_popen_init = subprocess.Popen.__init__


def _safe_popen_init(self, *args, **kwargs):
    # Detect Textual capture mode (fileno() returns -1).
    try:
        in_textual = sys.stdout.fileno() < 0
    except (OSError, AttributeError, ValueError):
        in_textual = True

    if in_textual:
        capture = kwargs.get("capture_output", False)
        for name in ("stdin", "stdout", "stderr"):
            val = kwargs.get(name)
            if val is None:
                # Don't override stdout/stderr when capture_output sets PIPE
                if capture and name in ("stdout", "stderr"):
                    continue
                kwargs[name] = subprocess.DEVNULL
            elif not isinstance(val, int):
                # Explicit file-like with bad fd → replace with DEVNULL
                try:
                    if val.fileno() < 0:
                        kwargs[name] = subprocess.DEVNULL
                except (OSError, AttributeError, ValueError):
                    kwargs[name] = subprocess.DEVNULL

    _orig_popen_init(self, *args, **kwargs)


subprocess.Popen.__init__ = _safe_popen_init

# ---------------------------------------------------------------------------

logger = logging.getLogger("timecopilot.tui")

TIME_RANGES = {
    "3M": 3,
    "6M": 6,
    "1Y": 12,
    "2Y": 24,
    "5Y": 60,
    "All": 0,
}


class TimeCopilotApp(App):
    """Bloomberg Terminal-style TUI for TimeCopilot."""

    CSS_PATH = "_tui.tcss"
    TITLE = "TIMECOPILOT TERMINAL"

    BINDINGS = [
        Binding("ctrl+q", "quit", "^Q Quit", show=True, priority=True),
        Binding("f1", "switch_tab('chat')", "F1 Chat", show=True, priority=True),
        Binding("f2", "switch_tab('chart')", "F2 Chart", show=True, priority=True),
        Binding("f3", "switch_tab('analysis')", "F3 Analysis", show=True, priority=True),
        Binding("f4", "switch_tab('forecast')", "F4 Forecast", show=True, priority=True),
        Binding("f5", "switch_tab('settings')", "F5 Settings", show=True, priority=True),
    ]

    active_range: reactive[str] = reactive("All")
    is_busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        llm: str = "openai:gpt-4o-mini",
        llm_explicit: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.llm_explicit = llm_explicit
        self._agent_state: dict | None = None
        self.df: pd.DataFrame | None = None
        self.fcst_df: pd.DataFrame | None = None
        self.eval_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None
        self.anomalies_df: pd.DataFrame | None = None
        self._onboarding_state: str = "init"
        self._providers: list[str] = []
        self._provider_models: dict[str, list[str]] = {}
        self._selected_provider: str | None = None
        self._last_result_output = None
        self.has_analysis: bool = False
        self._dirty_tabs: set[str] = set()
        self._settings_state: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="tabs"):
            # --- Chat Tab ---
            with TabPane("Chat", id="chat"):
                with Vertical(id="chat-container"):
                    yield RichLog(
                        id="chat-log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                    )
                    yield LoadingIndicator(id="loading")
                    yield Input(
                        placeholder="Let's forecast! Enter a file path or URL",
                        id="chat-input",
                    )
            # --- Chart Tab (all pre-composed) ---
            with TabPane("Chart", id="chart"):
                with Vertical(id="chart-wrapper"):
                    with Horizontal(id="time-range-bar"):
                        for label in TIME_RANGES:
                            yield Button(label, id=f"range-{label}")
                    yield Static(
                        "Run an analysis in the Chat tab to see charts here.",
                        id="chart-empty",
                    )
                    yield PlotextPlot(id="chart-plot")
            # --- Analysis Tab (all pre-composed) ---
            with TabPane("Analysis", id="analysis"):
                yield Static(
                    "Run an analysis in the Chat tab to see results here.",
                    id="analysis-empty",
                )
                with Vertical(id="analysis-container"):
                    yield DataTable(id="features-table")
                    yield DataTable(id="models-table")
                    yield Static("", id="feature-analysis")
                    yield Static("", id="model-comparison")
            # --- Forecast Tab (all pre-composed) ---
            with TabPane("Forecast", id="forecast"):
                yield Static(
                    "Run an analysis in the Chat tab to see forecast here.",
                    id="forecast-empty",
                )
                with Vertical(id="forecast-container"):
                    yield DataTable(id="forecast-table")
                    yield Static("", id="anomaly-summary")
                    yield Static("", id="forecast-analysis")
                    yield Static("", id="anomaly-analysis")
            # --- Settings Tab ---
            with TabPane("Settings", id="settings"):
                with Vertical(id="settings-container"):
                    yield Static("", id="settings-current-model")
                    yield Button("Change LLM", id="settings-change-llm")
                    yield RichLog(id="settings-log", highlight=True, markup=True, wrap=True)
                    yield Input(placeholder="Select an option", id="settings-input")
        yield Footer()

    def on_mount(self) -> None:
        logger.info("TUI mounted, starting onboarding")

        # Hide loading indicator
        self.query_one("#loading", LoadingIndicator).display = False

        # Hide data widgets, show empty placeholders
        self.query_one("#chart-plot", PlotextPlot).display = False
        self.query_one("#chart-empty").display = True
        self.query_one("#analysis-container").display = False
        self.query_one("#analysis-empty").display = True
        self.query_one("#forecast-container").display = False
        self.query_one("#forecast-empty").display = True

        # Setup pre-composed tables (add columns once)
        ft = self.query_one("#features-table", DataTable)
        ft.add_columns("Feature", "Value")
        mt = self.query_one("#models-table", DataTable)
        mt.add_columns("Model", "MASE Score")
        fct = self.query_one("#forecast-table", DataTable)
        fct.add_columns("Period", "Value")

        # Default time range button
        self.query_one("#range-All", Button).add_class("-active")

        # Settings tab: hide input until LLM change flow starts
        self.query_one("#settings-input", Input).display = False
        self._update_settings_model_label()

        load_env()
        self._run_onboarding_start()

    # --- Busy state ---

    def watch_is_busy(self, busy: bool) -> None:
        self.query_one("#loading", LoadingIndicator).display = busy
        self.query_one("#chat-input", Input).disabled = busy

    # --- Logging helper ---

    def _log_to_chat(self, msg: str) -> None:
        try:
            self.query_one("#chat-log", RichLog).write(msg)
        except Exception as e:
            logger.error("Failed to write to chat log: %s", e)

    # --- Onboarding ---

    def _run_onboarding_start(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        chat_input = self.query_one("#chat-input", Input)

        saved_model = load_saved_model()
        logger.info(
            "Onboarding: llm_explicit=%s saved_model=%s",
            self.llm_explicit,
            saved_model,
        )

        if self.llm_explicit:
            self._print_welcome(log)
            provider = get_provider_for_model(self.llm)
            env_var = get_provider_env_var(provider)
            if env_var and not os.environ.get(env_var):
                log.write(f"[bold yellow]API key {env_var} is not set.[/bold yellow]")
                log.write("Please enter your API key:")
                self._onboarding_state = "api_key"
                chat_input.placeholder = f"Enter your {env_var}"
            else:
                if env_var:
                    log.write(f"[dim]API key {env_var} is set[/dim]")
                save_model(self.llm)
                self._finish_onboarding(log, chat_input)
        elif saved_model and has_valid_saved_config(saved_model):
            self.llm = saved_model
            self._print_welcome_back(log)
            self._finish_onboarding(log, chat_input)
        else:
            self._print_welcome(log)
            log.write("")
            log.write("[bold blue]Would you like to select an LLM? (yes/no)[/bold blue]")
            self._onboarding_state = "select_llm"
            chat_input.placeholder = "yes or no (default: yes)"

    def _update_settings_model_label(self) -> None:
        self.query_one("#settings-current-model", Static).update(
            f"[bold]Current model:[/bold] [cyan]{self.llm}[/cyan]"
        )

    def _apply_model_change(self, log: RichLog) -> None:
        self.sub_title = f"model: {self.llm}"
        self._update_settings_model_label()
        log.write(f"\n[bold green]Using model:[/bold green] [cyan]{self.llm}[/cyan]")

    def _finish_onboarding(self, log: RichLog, chat_input: Input) -> None:
        self._onboarding_state = "ready"
        self._apply_model_change(log)
        log.write("")
        chat_input.placeholder = "Let's forecast! Enter a file path or URL"
        logger.info("Onboarding complete, model=%s", self.llm)

    def _print_welcome(self, log: RichLog) -> None:
        log.write("[bold white]Welcome to TimeCopilot[/bold white]")
        log.write("")
        log.write("I'm TimeCopilot, the GenAI Forecasting Agent!")
        log.write("I help you understand your data and predict the future.")
        log.write("")
        log.write("[cyan]Natural conversation examples:[/cyan]")
        log.write("  - 'Forecast this dataset: /path/to/sales.csv'")
        log.write("  - 'Can you spot any weird patterns in my server data?'")
        log.write("  - 'Show me a plot of the forecast'")
        log.write("  - 'Compare Chronos and TimesFM models'")
        log.write("")
        log.write("[dim]Commands: help | llm | exit[/dim]")
        log.write("[dim]Tabs: F1 Chat | F2 Chart | F3 Analysis | F4 Forecast | F5 Settings | Ctrl+Q Quit[/dim]")

    def _print_welcome_back(self, log: RichLog) -> None:
        log.write("[bold white]Welcome back![/bold white]")
        log.write(f"[cyan]Current model:[/cyan] {self.llm}")
        log.write("[dim]Commands: help | llm | exit[/dim]")
        log.write("[dim]Tabs: F1 Chat | F2 Chart | F3 Analysis | F4 Forecast | F5 Settings | Ctrl+Q Quit[/dim]")

    # --- Onboarding input handlers ---

    def _handle_select_llm(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() in ("n", "no"):
            provider = get_provider_for_model(self.llm)
            env_var = get_provider_env_var(provider)
            if env_var and not os.environ.get(env_var):
                log.write(f"\n[bold yellow]API key {env_var} is not set.[/bold yellow]")
                log.write("Please enter your API key:")
                self._onboarding_state = "api_key"
                chat_input.placeholder = f"Enter your {env_var}"
            else:
                if env_var:
                    log.write(f"[dim]API key {env_var} is set[/dim]")
                save_model(self.llm)
                self._finish_onboarding(log, chat_input)
        else:
            self._show_providers(log, chat_input)

    def _show_providers(self, log: RichLog, chat_input: Input) -> None:
        self._providers, self._provider_models = list_providers()

        log.write("\n[bold blue]Select a provider:[/bold blue]")
        for i, prov in enumerate(self._providers, 1):
            count = len(self._provider_models[prov])
            log.write(f"  [bold]{i:>2}.[/bold] [green]{prov}[/green] ({count} models)")

        self._onboarding_state = "provider"
        chat_input.placeholder = "Enter a number or provider name"

    def _handle_provider_select(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() == "back":
            log.write("\n[bold blue]Would you like to select an LLM? (yes/no)[/bold blue]")
            self._onboarding_state = "select_llm"
            chat_input.placeholder = "yes or no (default: yes)"
            return

        result = match_provider(value, self._providers)
        if result.status == MatchStatus.AMBIGUOUS:
            log.write(f"[yellow]Ambiguous: {', '.join(result.candidates)}[/yellow]")
            return
        if result.status == MatchStatus.NO_MATCH:
            log.write(f"[yellow]No provider matching '{value}'[/yellow]")
            return

        selected_provider = result.value

        self._selected_provider = selected_provider
        models = self._provider_models[selected_provider]
        log.write(f"\n[bold blue]{selected_provider} Models:[/bold blue]")
        for i, model in enumerate(models, 1):
            display = model.split(":", 1)[1]
            log.write(f"  [bold]{i:>2}.[/bold] [green]{display}[/green]")
        self._onboarding_state = "model"
        chat_input.placeholder = "Enter a number or model name"

    def _handle_model_select(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() == "back":
            self._show_providers(log, chat_input)
            return

        models = self._provider_models[self._selected_provider]
        result = match_model(value, models)
        if result.status == MatchStatus.NO_MATCH:
            log.write(f"[yellow]Using default: {result.value}[/yellow]")
        selected = result.value

        self.llm = selected
        log.write(f"\n[green]Selected: {self.llm}[/green]")

        provider = get_provider_for_model(self.llm)
        env_var = get_provider_env_var(provider)
        if env_var and not os.environ.get(env_var):
            log.write(f"\n[bold yellow]API key {env_var} is not set.[/bold yellow]")
            log.write("Please enter your API key:")
            self._onboarding_state = "api_key"
            chat_input.placeholder = f"Enter your {env_var} (or 'back')"
        else:
            if env_var:
                log.write(f"[dim]API key {env_var} is set[/dim]")
            save_model(self.llm)
            self._finish_onboarding(log, chat_input)

    def _handle_api_key(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() == "back":
            self._show_providers(log, chat_input)
            return

        provider = get_provider_for_model(self.llm)
        env_var = get_provider_env_var(provider) or ""

        if value.strip():
            save_api_key(provider, value.strip())
            log.write("[green]API key saved and set for this session[/green]")
            save_model(self.llm)
            self._finish_onboarding(log, chat_input)
        else:
            log.write(f"[yellow]No key provided. Please enter your {env_var}.[/yellow]")

    # --- Main input handler ---

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        event.input.clear()

        # --- Settings tab input routing ---
        if event.input.id == "settings-input":
            settings_log = self.query_one("#settings-log", RichLog)
            settings_input = self.query_one("#settings-input", Input)
            if self._settings_state == "provider":
                self._handle_settings_provider_select(value, settings_log, settings_input)
            elif self._settings_state == "model":
                self._handle_settings_model_select(value, settings_log, settings_input)
            elif self._settings_state == "api_key":
                self._handle_settings_api_key(value, settings_log, settings_input)
            return

        log = self.query_one("#chat-log", RichLog)
        chat_input = self.query_one("#chat-input", Input)

        if self._onboarding_state == "select_llm":
            self._handle_select_llm(value, log, chat_input)
            return
        if self._onboarding_state == "provider":
            self._handle_provider_select(value, log, chat_input)
            return
        if self._onboarding_state == "model":
            self._handle_model_select(value, log, chat_input)
            return
        if self._onboarding_state == "api_key":
            self._handle_api_key(value, log, chat_input)
            return

        if value.lower() in ("exit", "quit", "bye"):
            log.write("\n[bold blue]Thanks for using TimeCopilot! See you next time![/bold blue]")
            self.exit()
            return
        if value.lower() in ("help", "?"):
            self._print_welcome(log)
            return
        if value.lower() == "llm":
            self._show_providers(log, chat_input)
            return

        log.write(f"\n[bold white]You:[/bold white] {value}")
        logger.info("User input: %s", value)

        if self._agent_state is not None:
            self._run_query(value)
        else:
            file_path = extract_file_path(value)
            if file_path:
                self._run_analysis(file_path, value)
            else:
                log.write("\n[bold yellow]Please provide a file path or URL to get started.[/bold yellow]")
                log.write("[dim]Example: forecast /path/to/data.csv[/dim]")

    # --- Subprocess helpers ---

    async def _run_in_subprocess(self, request: dict) -> dict:
        """Spawn a worker subprocess and return the response dict.

        The child runs with stdin/stdout/stderr=DEVNULL so that library
        output (logfire, tqdm, matplotlib, Rich) cannot reach the TUI.
        """
        req_fd, req_path = tempfile.mkstemp(suffix=".pkl", prefix="tc_req_")
        resp_fd, resp_path = tempfile.mkstemp(suffix=".pkl", prefix="tc_resp_")
        try:
            os.close(resp_fd)
            with os.fdopen(req_fd, "wb") as f:
                pickle.dump(request, f, protocol=pickle.HIGHEST_PROTOCOL)

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "timecopilot._worker", req_path, resp_path,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await proc.wait()

            with open(resp_path, "rb") as f:
                response = pickle.load(f)

            if isinstance(response, dict) and "error" in response:
                raise RuntimeError(response["error"])
            return response
        finally:
            for p in (req_path, resp_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # --- Async workers (UI stays on event loop; heavy work in subprocess) ---

    @work(exclusive=True)
    async def _run_analysis(self, file_path: str, user_input: str) -> None:
        logger.info("Starting analysis: file=%s", file_path)

        self.is_busy = True
        self._log_to_chat("\n[bold blue]Loading data and analyzing...[/bold blue]")

        try:
            response = await self._run_in_subprocess({
                "action": "analyze",
                "llm": self.llm,
                "file_path": file_path,
                "query": user_input,
            })

            logger.info("Analysis completed successfully")

            # Store subprocess state for follow-up queries
            self._agent_state = {
                "df": response["df"],
                "freq": response["freq"],
                "h": response["h"],
                "seasonality": response["seasonality"],
                "fcst_df": response["fcst_df"],
                "eval_df": response["eval_df"],
                "features_df": response["features_df"],
                "anomalies_df": response["anomalies_df"],
                "eval_forecasters": response["eval_forecasters"],
                "conversation_history": response["conversation_history"],
            }

            # Store dataframes on self for tab rendering
            self.df = response["df"]
            self.fcst_df = response["fcst_df"]
            self.eval_df = response["eval_df"]
            self.features_df = response["features_df"]
            self.anomalies_df = response["anomalies_df"]

            # Reconstruct ForecastAgentOutput from dict
            output = ForecastAgentOutput(**response["output_dict"])
            self._last_result_output = output

            # Build summary lines
            summary: list[str] = []
            selected_model = output.selected_model
            beats_naive = output.is_better_than_seasonal_naive
            perf_msg = "performing well" if beats_naive else "needs improvement"

            summary.append("\n[bold green]Analysis complete![/bold green]")
            summary.append(f"[cyan]Selected Model:[/cyan] {selected_model} ({perf_msg})")

            if self.fcst_df is not None:
                summary.append(f"[cyan]Forecast:[/cyan] Generated {len(self.fcst_df)} future periods")

            if self.anomalies_df is not None:
                anomaly_cols = [c for c in self.anomalies_df.columns if c.endswith("-anomaly")]
                total_anomalies = sum(self.anomalies_df[c].sum() for c in anomaly_cols)
                if total_anomalies > 0:
                    total_points = len(self.anomalies_df)
                    rate = (total_anomalies / total_points) * 100
                    summary.append(f"[red]Found {total_anomalies} anomalies ({rate:.1f}%)[/red]")

            user_response = output.user_query_response
            if user_response:
                summary.append(f"\n[bold cyan]TimeCopilot:[/bold cyan]\n{user_response}")

            summary.append("\n[dim]Switch tabs: F2 Chart | F3 Analysis | F4 Forecast[/dim]")

            self._on_analysis_done(summary)

        except Exception as e:
            logger.exception("Analysis failed")
            self._log_to_chat(f"\n[bold red]Error:[/bold red] {e}")
        finally:
            self.is_busy = False

    @work(exclusive=True)
    async def _run_query(self, user_input: str) -> None:
        logger.info("Starting query: %s", user_input)

        self.is_busy = True
        self._log_to_chat("\n[bold blue]Thinking...[/bold blue]")

        try:
            response = await self._run_in_subprocess({
                "action": "query",
                "llm": self.llm,
                "query": user_input,
                "agent_state": self._agent_state,
            })

            logger.info("Query completed")

            # Update agent state (query may have triggered _maybe_rerun)
            self._agent_state = {
                "df": response["df"],
                "freq": response["freq"],
                "h": response["h"],
                "seasonality": response["seasonality"],
                "fcst_df": response["fcst_df"],
                "eval_df": response["eval_df"],
                "features_df": response["features_df"],
                "anomalies_df": response["anomalies_df"],
                "eval_forecasters": response["eval_forecasters"],
                "conversation_history": response["conversation_history"],
            }

            # Update TUI dataframes in case re-analysis happened
            self.df = response["df"]
            self.fcst_df = response["fcst_df"]
            self.eval_df = response["eval_df"]
            self.features_df = response["features_df"]
            self.anomalies_df = response["anomalies_df"]
            # Mark tabs dirty so they re-render if user switches
            self._dirty_tabs = {"chart", "analysis", "forecast"}

            self._log_to_chat(f"\n[bold cyan]TimeCopilot:[/bold cyan]\n{response['output']}")

        except Exception as e:
            logger.exception("Query failed")
            self._log_to_chat(f"\n[bold red]Error:[/bold red] {e}")
        finally:
            self.is_busy = False

    # --- Post-analysis UI update ---

    def _on_analysis_done(self, summary_lines: list[str]) -> None:
        """Single callback on main thread when analysis completes.

        Only writes to the chat log and sets flags — NO display toggling
        on other tabs (that happens lazily in on_tabbed_content_tab_activated).
        """
        logger.info("Analysis done callback — writing summary and setting flags")
        log = self.query_one("#chat-log", RichLog)
        for line in summary_lines:
            log.write(line)
        # Set plain flags — no reactive watchers, no display toggles
        self.has_analysis = True
        self._dirty_tabs = {"chart", "analysis", "forecast"}
        self.query_one("#chat-input", Input).placeholder = (
            "Ask a follow-up question or enter a new file path"
        )

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Populate a data tab lazily when the user switches to it."""
        tab_id = event.pane.id
        if tab_id == "settings":
            self._update_settings_model_label()
        if tab_id not in self._dirty_tabs:
            return
        logger.info("Lazy-populating tab: %s", tab_id)
        # Remove from dirty set before populating (prevent re-entry)
        self._dirty_tabs.discard(tab_id)
        try:
            if tab_id == "chart":
                self.query_one("#chart-empty").display = False
                self.query_one("#chart-plot", PlotextPlot).display = True
                self.call_later(self._populate_chart)
            elif tab_id == "analysis":
                self.query_one("#analysis-empty").display = False
                self.query_one("#analysis-container").display = True
                self.call_later(self._populate_analysis)
            elif tab_id == "forecast":
                self.query_one("#forecast-empty").display = False
                self.query_one("#forecast-container").display = True
                self.call_later(self._populate_forecast)
        except Exception:
            logger.exception("Failed to populate tab %s", tab_id)
            self._log_to_chat(
                f"[bold red]Error populating {tab_id} tab."
                " Check logs for details.[/bold red]"
            )

    def _populate_chart(self) -> None:
        self._render_chart()

    def _render_chart(self, plot_widget: PlotextPlot | None = None) -> None:
        if plot_widget is None:
            plot_widget = self.query_one("#chart-plot", PlotextPlot)

        if self.df is None:
            return

        plt = plot_widget.plt
        plt.clear_figure()
        plt.date_form("Y-m-d")
        plt.theme("dark")
        plt.title("Time Series Data")
        plt.xlabel("Date")
        plt.ylabel("Value")

        df = self.df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        months = TIME_RANGES.get(self.active_range, 0)
        if months > 0:
            cutoff = df["ds"].max() - pd.DateOffset(months=months)
            df = df[df["ds"] >= cutoff]

        colors = ["cyan", "green", "yellow", "magenta", "blue", "red"]

        for i, uid in enumerate(df["unique_id"].unique()[:10]):
            series = df[df["unique_id"] == uid].sort_values("ds")
            dates = series["ds"].dt.strftime("%Y-%m-%d").tolist()
            values = series["y"].tolist()
            plt.plot(
                dates, values, label=str(uid), color=colors[i % len(colors)]
            )

        if self.fcst_df is not None:
            fcst = self.fcst_df.copy()
            fcst["ds"] = pd.to_datetime(fcst["ds"])
            model_cols = [
                c
                for c in fcst.columns
                if c not in ("unique_id", "ds") and "-" not in c
            ]
            if model_cols:
                model_col = model_cols[0]
                for i, uid in enumerate(fcst["unique_id"].unique()[:10]):
                    series = fcst[fcst["unique_id"] == uid].sort_values("ds")
                    dates = series["ds"].dt.strftime("%Y-%m-%d").tolist()
                    values = series[model_col].tolist()
                    plt.plot(
                        dates,
                        values,
                        label=f"{uid} forecast",
                        color=colors[i % len(colors)],
                        marker="dot",
                    )

        if self.anomalies_df is not None:
            anom = self.anomalies_df.copy()
            anom["ds"] = pd.to_datetime(anom["ds"])
            for anom_col in [
                c for c in anom.columns if c.endswith("-anomaly")
            ]:
                anomalies = anom[anom[anom_col] == True]  # noqa: E712
                if len(anomalies) > 0:
                    dates = (
                        anomalies["ds"].dt.strftime("%Y-%m-%d").tolist()
                    )
                    values = anomalies["y"].tolist()
                    plt.scatter(
                        dates,
                        values,
                        label="anomalies",
                        color="red",
                        marker="x",
                    )

        plt.grid(True, True)
        plot_widget.refresh()

    def _populate_analysis(self) -> None:
        ft = self.query_one("#features-table", DataTable)
        ft.clear()
        if self.features_df is not None:
            for col in self.features_df.columns:
                val = self.features_df.iloc[0][col]
                if pd.notna(val):
                    ft.add_row(str(col), f"{float(val):.4f}")

        mt = self.query_one("#models-table", DataTable)
        mt.clear()
        if self.eval_df is not None:
            scores = []
            for col in self.eval_df.columns:
                if col != "metric" and pd.notna(self.eval_df[col].iloc[0]):
                    scores.append((col, float(self.eval_df[col].iloc[0])))
            scores.sort(key=lambda x: x[1])
            for model, score in scores:
                color = "green" if score < 1.0 else "red"
                mt.add_row(model, f"[{color}]{score:.4f}[/{color}]")

        if self._last_result_output is not None:
            out = self._last_result_output
            fa_text = getattr(out, "tsfeatures_analysis", "")
            self.query_one("#feature-analysis", Static).update(
                f"[bold cyan]Feature Analysis[/bold cyan]\n{fa_text}"
                if fa_text
                else ""
            )
            mc_text = getattr(out, "model_comparison", "")
            self.query_one("#model-comparison", Static).update(
                f"[bold yellow]Model Comparison[/bold yellow]\n{mc_text}"
                if mc_text
                else ""
            )

    def _populate_forecast(self) -> None:
        fct = self.query_one("#forecast-table", DataTable)
        fct.clear()
        if self.fcst_df is not None:
            model_cols = [
                c
                for c in self.fcst_df.columns
                if c not in ("unique_id", "ds") and "-" not in c
            ]
            if model_cols:
                model_col = model_cols[0]
                for _, row in self.fcst_df.iterrows():
                    period = (
                        row["ds"].strftime("%Y-%m-%d")
                        if hasattr(row["ds"], "strftime")
                        else str(row["ds"])
                    )
                    fct.add_row(period, f"{row[model_col]:.2f}")

        anom_widget = self.query_one("#anomaly-summary", Static)
        if self.anomalies_df is not None:
            anomaly_cols = [
                c
                for c in self.anomalies_df.columns
                if c.endswith("-anomaly")
            ]
            total_anomalies = sum(
                self.anomalies_df[c].sum() for c in anomaly_cols
            )
            total_points = len(self.anomalies_df)
            rate = (
                (total_anomalies / total_points) * 100
                if total_points > 0
                else 0
            )

            anom_text = (
                f"[bold red]Anomaly Detection Summary[/bold red]\n\n"
                f"Total data points: {total_points}\n"
                f"Anomalies found: [red]{int(total_anomalies)}[/red]\n"
                f"Anomaly rate: [red]{rate:.1f}%[/red]\n"
            )
            if total_anomalies > 0:
                for anom_col in anomaly_cols:
                    anomalies = self.anomalies_df[
                        self.anomalies_df[anom_col] == True  # noqa: E712
                    ]
                    if len(anomalies) > 0:
                        dates = (
                            anomalies["ds"]
                            .dt.strftime("%Y-%m-%d")
                            .head(5)
                            .tolist()
                        )
                        anom_text += f"\nAnomalous dates ({anom_col}):\n"
                        for d in dates:
                            anom_text += f"  [red]{d}[/red]\n"
                        remaining = len(anomalies) - 5
                        if remaining > 0:
                            anom_text += f"  ...and {remaining} more\n"
            anom_widget.update(anom_text)
        else:
            anom_widget.update("[dim]No anomaly data available[/dim]")

        if self._last_result_output is not None:
            out = self._last_result_output
            fa_text = getattr(out, "forecast_analysis", "")
            self.query_one("#forecast-analysis", Static).update(
                f"[bold magenta]Forecast Analysis[/bold magenta]\n{fa_text}"
                if fa_text
                else ""
            )
            aa_text = getattr(out, "anomaly_analysis", "")
            self.query_one("#anomaly-analysis", Static).update(
                f"[bold red]Anomaly Analysis[/bold red]\n{aa_text}"
                if aa_text
                else ""
            )

    # --- Settings tab handlers ---

    def _start_settings_llm_change(self) -> None:
        settings_log = self.query_one("#settings-log", RichLog)
        settings_input = self.query_one("#settings-input", Input)
        settings_log.clear()
        settings_input.display = True
        self._settings_state = "provider"
        saved = self._onboarding_state
        self._show_providers(settings_log, settings_input)
        self._onboarding_state = saved
        settings_input.focus()

    def _finish_settings_llm_change(self, log: RichLog) -> None:
        save_model(self.llm)
        self._apply_model_change(log)
        self._settings_state = None
        self.query_one("#settings-input", Input).display = False

    def _handle_settings_provider_select(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() == "back":
            self._settings_state = None
            log.clear()
            chat_input.display = False
            return
        saved = self._onboarding_state
        self._handle_provider_select(value, log, chat_input)
        if self._onboarding_state == "model":
            self._settings_state = "model"
        self._onboarding_state = saved

    def _handle_settings_model_select(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() == "back":
            self._settings_state = "provider"
            saved = self._onboarding_state
            self._show_providers(log, chat_input)
            self._onboarding_state = saved
            self._settings_state = "provider"
            return
        saved = self._onboarding_state
        self._handle_model_select(value, log, chat_input)
        if self._onboarding_state == "api_key":
            self._settings_state = "api_key"
        elif self._onboarding_state == "ready":
            self._finish_settings_llm_change(log)
        self._onboarding_state = saved

    def _handle_settings_api_key(self, value: str, log: RichLog, chat_input: Input) -> None:
        if value.lower() == "back":
            self._settings_state = "provider"
            saved = self._onboarding_state
            self._show_providers(log, chat_input)
            self._onboarding_state = saved
            self._settings_state = "provider"
            return
        saved = self._onboarding_state
        self._handle_api_key(value, log, chat_input)
        if self._onboarding_state == "ready":
            self._finish_settings_llm_change(log)
        self._onboarding_state = saved

    # --- Button and tab actions ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "settings-change-llm":
            self._start_settings_llm_change()
            return
        if button_id.startswith("range-"):
            range_label = button_id.replace("range-", "")
            if range_label in TIME_RANGES:
                for btn in self.query("#time-range-bar Button"):
                    btn.remove_class("-active")
                event.button.add_class("-active")
                self.active_range = range_label
                if self.has_analysis:
                    self._render_chart()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    def action_quit(self) -> None:
        self.exit()
