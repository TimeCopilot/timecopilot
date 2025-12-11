from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import Model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    series_length,
    stability,
    stl_features,
    unitroot_kpss,
    unitroot_pp,
)
from tsfeatures.tsfeatures import _get_feats

from .forecaster import Forecaster, TimeCopilotForecaster
from .models.prophet import Prophet
from .models.stats import (
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    DynamicOptimizedTheta,
    HistoricAverage,
    SeasonalNaive,
    Theta,
    ZeroModel,
)
from .utils.experiment_handler import ExperimentDataset, ExperimentDatasetParser

DEFAULT_MODELS: list[Forecaster] = [
    ADIDA(),
    AutoARIMA(),
    AutoCES(),
    AutoETS(),
    CrostonClassic(),
    DynamicOptimizedTheta(),
    HistoricAverage(),
    IMAPA(),
    SeasonalNaive(),
    Theta(),
    ZeroModel(),
    Prophet(),
]

TSFEATURES: dict[str, Callable] = {
    "acf_features": acf_features,
    "arch_stat": arch_stat,
    "crossing_points": crossing_points,
    "entropy": entropy,
    "flat_spots": flat_spots,
    "heterogeneity": heterogeneity,
    "holt_parameters": holt_parameters,
    "lumpiness": lumpiness,
    "nonlinearity": nonlinearity,
    "pacf_features": pacf_features,
    "stl_features": stl_features,
    "stability": stability,
    "hw_parameters": hw_parameters,
    "unitroot_kpss": unitroot_kpss,
    "unitroot_pp": unitroot_pp,
    "series_length": series_length,
    "hurst": hurst,
}


class ForecastAgentOutput(BaseModel):
    """The output of the forecasting agent."""

    tsfeatures_analysis: str = Field(
        description=(
            "Analysis of what the time series features reveal about the data "
            "and their implications for forecasting."
        )
    )
    selected_model: str = Field(
        description="The model that was selected for the forecast"
    )
    model_details: str = Field(
        description=(
            "Technical details about the selected model including its assumptions, "
            "strengths, and typical use cases."
        )
    )
    model_comparison: str = Field(
        description=(
            "Detailed comparison of model performances, explaining why certain "
            "models performed better or worse on this specific time series."
        )
    )
    is_better_than_seasonal_naive: bool = Field(
        description="Whether the selected model is better than the seasonal naive model"
    )
    reason_for_selection: str = Field(
        description="Explanation for why the selected model was chosen"
    )
    forecast_analysis: str = Field(
        description=(
            "Detailed interpretation of the forecast, including trends, patterns, "
            "and potential problems."
        )
    )
    anomaly_analysis: str = Field(
        description=(
            "Analysis of detected anomalies, their patterns, potential causes, "
            "and recommendations for handling them."
        )
    )
    user_query_response: str | None = Field(
        description=(
            "The response to the user's query, if any. "
            "If the user did not provide a query, this field will be None."
        )
    )

    def prettify(
        self,
        console: Console | None = None,
        features_df: pd.DataFrame | None = None,
        eval_df: pd.DataFrame | None = None,
        fcst_df: pd.DataFrame | None = None,
        anomalies_df: pd.DataFrame | None = None,
    ) -> None:
        """Pretty print the forecast results using rich formatting."""
        console = console or Console()

        # Create header with title and overview
        header = Panel(
            f"[bold cyan]{self.selected_model}[/bold cyan] forecast analysis\n"
            f"[{'green' if self.is_better_than_seasonal_naive else 'red'}]"
            f"{'Better' if self.is_better_than_seasonal_naive else 'Not better'} "
            "than Seasonal Naive[/"
            f"{'green' if self.is_better_than_seasonal_naive else 'red'}]",
            title="[bold blue]TimeCopilot Forecast[/bold blue]",
            style="blue",
        )

        # Time Series Analysis Section - check if features_df is available
        ts_features = Table(
            title="Time Series Features",
            show_header=True,
            title_style="bold cyan",
            header_style="bold magenta",
        )
        ts_features.add_column("Feature", style="cyan")
        ts_features.add_column("Value", style="magenta")

        # Use features_df if available (attached after forecast run)
        if features_df is not None:
            for feature_name, feature_value in features_df.iloc[0].items():
                if pd.notna(feature_value):
                    ts_features.add_row(feature_name, f"{float(feature_value):.3f}")
        else:
            # Fallback: show a note that detailed features are not available
            ts_features.add_row("Features", "Available in analysis text below")

        ts_analysis = Panel(
            f"{self.tsfeatures_analysis}",
            title="[bold cyan]Feature Analysis[/bold cyan]",
            style="blue",
        )

        # Model Selection Section
        model_details = Panel(
            f"[bold]Technical Details[/bold]\n{self.model_details}\n\n"
            f"[bold]Selection Rationale[/bold]\n{self.reason_for_selection}",
            title="[bold green]Model Information[/bold green]",
            style="green",
        )

        # Model Comparison Table - check if eval_df is available
        model_scores = Table(
            title="Model Performance", show_header=True, title_style="bold yellow"
        )
        model_scores.add_column("Model", style="yellow")
        model_scores.add_column("MASE", style="cyan", justify="right")

        # Use eval_df if available (attached after forecast run)
        if eval_df is not None:
            # Get the MASE scores from eval_df
            model_scores_data = []
            for col in eval_df.columns:
                if col != "metric" and pd.notna(eval_df[col].iloc[0]):
                    model_scores_data.append((col, float(eval_df[col].iloc[0])))

            # Sort by score (lower MASE is better)
            model_scores_data.sort(key=lambda x: x[1])
            for model, score in model_scores_data:
                model_scores.add_row(model, f"{score:.3f}")
        else:
            # Fallback: show a note that detailed scores are not available
            model_scores.add_row("Scores", "Available in analysis text below")

        model_analysis = Panel(
            self.model_comparison,
            title="[bold yellow]Performance Analysis[/bold yellow]",
            style="yellow",
        )

        # Forecast Results Section - check if fcst_df is available
        forecast_table = Table(
            title="Forecast Values", show_header=True, title_style="bold magenta"
        )
        forecast_table.add_column("Period", style="magenta")
        forecast_table.add_column("Value", style="cyan", justify="right")

        # Use fcst_df if available (attached after forecast run)
        if fcst_df is not None:
            # Show forecast values from fcst_df
            fcst_data = fcst_df.copy()
            if "ds" in fcst_data.columns and self.selected_model in fcst_data.columns:
                for _, row in fcst_data.iterrows():
                    period = (
                        row["ds"].strftime("%Y-%m-%d")
                        if hasattr(row["ds"], "strftime")
                        else str(row["ds"])
                    )
                    value = row[self.selected_model]
                    forecast_table.add_row(period, f"{value:.2f}")

                # Add note about number of periods if many
                if len(fcst_data) > 12:
                    forecast_table.caption = (
                        f"[dim]Showing all {len(fcst_data)} forecasted periods. "
                        "Use aggregation functions for summarized views.[/dim]"
                    )
            else:
                forecast_table.add_row("Forecast", "Available in analysis text below")
        else:
            # Fallback: show a note that detailed forecast is not available
            forecast_table.add_row("Forecast", "Available in analysis text below")

        forecast_analysis = Panel(
            self.forecast_analysis,
            title="[bold magenta]Forecast Analysis[/bold magenta]",
            style="magenta",
        )

        # Anomaly Detection Section
        anomaly_analysis = Panel(
            self.anomaly_analysis,
            title="[bold red]Anomaly Detection[/bold red]",
            style="red",
        )

        # Optional user response section
        user_response = None
        if self.user_query_response:
            user_response = Panel(
                self.user_query_response,
                title="[bold]Response to Query[/bold]",
                style="cyan",
            )

        # Print all sections with clear separation
        console.print("\n")
        console.print(header)

        console.print("\n[bold]1. Time Series Analysis[/bold]")
        console.print(ts_features)
        console.print(ts_analysis)

        console.print("\n[bold]2. Model Selection[/bold]")
        console.print(model_details)
        console.print(model_scores)
        console.print(model_analysis)

        console.print("\n[bold]3. Forecast Results[/bold]")
        console.print(forecast_table)
        console.print(forecast_analysis)

        console.print("\n[bold]4. Anomaly Detection[/bold]")
        console.print(anomaly_analysis)

        if user_response:
            console.print("\n[bold]5. Additional Information[/bold]")
            console.print(user_response)

        console.print("\n")


def _transform_time_series_to_text(df: pd.DataFrame) -> str:
    df_agg = df.groupby("unique_id").agg(list)
    output = (
        "these are the time series in json format where the key is the "
        "identifier of the time series and the values is also a json "
        "of two elements: "
        "the first element is the date column and the second element is the "
        "value column."
        f"{df_agg.to_json(orient='index')}"
    )
    return output


def _transform_features_to_text(features_df: pd.DataFrame) -> str:
    output = (
        "these are the time series features in json format where the key is "
        "the identifier of the time series and the values is also a json of "
        "feature names and their values."
        f"{features_df.to_json(orient='index')}"
    )
    return output


def _transform_eval_to_text(eval_df: pd.DataFrame, models: list[str]) -> str:
    output = ", ".join([f"{model}: {eval_df[model].iloc[0]}" for model in models])
    return output


def _transform_fcst_to_text(fcst_df: pd.DataFrame) -> str:
    df_agg = fcst_df.groupby("unique_id").agg(list)
    output = (
        "these are the forecasted values in json format where the key is the "
        "identifier of the time series and the values is also a json of two "
        "elements: the first element is the date column and the second "
        "element is the value column."
        f"{df_agg.to_json(orient='index')}"
    )
    return output


def _transform_anomalies_to_text(anomalies_df: pd.DataFrame) -> str:
    """Transform anomaly detection results to text for the agent."""
    # Get anomaly columns
    anomaly_cols = [col for col in anomalies_df.columns if col.endswith("-anomaly")]

    if not anomaly_cols:
        return "No anomaly detection results available."

    # Count anomalies per series
    anomaly_summary = {}
    for unique_id in anomalies_df["unique_id"].unique():
        series_data = anomalies_df[anomalies_df["unique_id"] == unique_id]
        series_summary = {}

        for anomaly_col in anomaly_cols:
            if anomaly_col in series_data.columns:
                anomaly_count = series_data[anomaly_col].sum()
                total_points = len(series_data)
                anomaly_rate = (
                    (anomaly_count / total_points) * 100 if total_points > 0 else 0
                )

                # Get timestamps of anomalies
                anomalies = series_data[series_data[anomaly_col]]
                anomaly_dates = (
                    anomalies["ds"].dt.strftime("%Y-%m-%d").tolist()
                    if len(anomalies) > 0
                    else []
                )

                series_summary[anomaly_col] = {
                    "count": int(anomaly_count),
                    "rate_percent": round(anomaly_rate, 2),
                    "dates": anomaly_dates[:10],  # Limit to first 10
                    "total_points": int(total_points),
                }

        anomaly_summary[unique_id] = series_summary

    output = (
        "these are the anomaly detection results in json format where the key is the "
        "identifier of the time series and the values contain anomaly statistics "
        "including count, rate, and timestamps of detected anomalies. "
        f"{anomaly_summary}"
    )
    return output


class TimeCopilot:
    """
    TimeCopilot: An AI agent for comprehensive time series analysis.

    Supports multiple analysis workflows:
    - Forecasting: Predict future values
    - Anomaly Detection: Identify outliers and unusual patterns
    - Visualization: Generate plots and charts
    - Combined: Multiple analysis types together
    """

    # Tool organization by workflow
    FORECASTING_TOOLS = ["tsfeatures_tool", "cross_validation_tool", "forecast_tool"]
    ANOMALY_TOOLS = ["tsfeatures_tool", "detect_anomalies_tool"]
    VISUALIZATION_TOOLS = ["plot_tool"]
    SHARED_TOOLS = ["tsfeatures_tool"]  # Used across multiple workflows

    def __init__(
        self,
        llm: str | Model,
        forecasters: list[Forecaster] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM to use.
            forecasters: A list of forecasters to use. If not provided,
                TimeCopilot will use the default forecasters.
            **kwargs: Additional keyword arguments to pass to the agent.
        """

        if forecasters is None:
            forecasters = DEFAULT_MODELS
        self.forecasters = {forecaster.alias: forecaster for forecaster in forecasters}
        if "SeasonalNaive" not in self.forecasters:
            self.forecasters["SeasonalNaive"] = SeasonalNaive()
        self.system_prompt = f"""
        You're a forecasting expert. You will be given a time series 
        as a list of numbers and your task is to determine the best model for it. 
        You have access to the following tools:

        1. tsfeatures_tool: Calculates time series features to help 
        with model selection.
        Available features are: {", ".join(TSFEATURES.keys())}

        2. cross_validation_tool: Performs cross-validation for one or more models.
        Takes a list of model names and returns their cross-validation results.
        Available models are: {", ".join(self.forecasters.keys())}

        3. forecast_tool: Generates forecasts using a selected model.
        Takes a model name and returns forecasted values.

        4. detect_anomalies_tool: Detects anomalies using the best performing model.
        Takes a model name and confidence level, returns anomaly detection results.

        You MUST complete all four steps and use all four tools in order:

        1. Time Series Feature Analysis (REQUIRED - use tsfeatures_tool):
           - ALWAYS call tsfeatures_tool first with a focused set of key features
           - Calculate time series features to identify characteristics (trend, 
                seasonality, stationarity, etc.)
           - Use these insights to guide efficient model selection
           - Focus on features that directly inform model choice

        2. Model Selection and Evaluation (REQUIRED - use cross_validation_tool):
           - ALWAYS call cross_validation_tool with multiple models
           - IMPORTANT: Check if user has requested specific models in their query
           - If user mentioned specific models (e.g., "try Chronos and ARIMA"), 
             PRIORITIZE those models in cross-validation
           - If no specific models mentioned, start with simple models that can 
             potentially beat seasonal naive
           - Select additional candidate models based on the time series 
                values and features
           - Document each model's technical details and assumptions
           - Explain why these models are suitable for the identified features
           - If initial models don't beat seasonal naive, try more complex models
           - Prioritize finding a model that outperforms seasonal naive benchmark
           - Balance model complexity with forecast accuracy

        3. Final Model Selection and Forecasting (REQUIRED - use forecast_tool):
           - Choose the best performing model based on the cross-validation results,
             specifically chose the one with the lowest MASE
           - If the user asks for a specific model, use that model
           - ALWAYS call forecast_tool with the selected model
           - Generate the forecast using just the selected model
           - Interpret trends and patterns in the forecast
           - Discuss reliability and potential uncertainties

        4. Anomaly Detection (REQUIRED - use detect_anomalies_tool):
           - ALWAYS call detect_anomalies_tool with the best performing model
           - Use the same model that was selected for forecasting
           - Apply appropriate confidence level (95% typical, 99% for strict detection)
           - Analyze detected anomalies and their patterns
           - Explain how anomalies relate to forecast reliability
           - Address any specific aspects from the user's prompt

        The evaluation will use MASE (Mean Absolute Scaled Error) by default.
        Use at least one cross-validation window for evaluation.
        The seasonality will be inferred from the date column.

        Your output must include:
        - Comprehensive feature analysis with clear implications
        - Detailed model comparison and selection rationale
        - Technical details of the selected model
        - Clear interpretation of cross-validation results
        - Analysis of the forecast and its implications
        - Comprehensive anomaly detection results and interpretation
        - Response to any user queries

        Focus on providing:
        - Clear connections between features and model choices
        - Technical accuracy with accessible explanations
        - Quantitative support for decisions
        - Practical implications of both forecasts and anomalies
        - Thorough responses to user concerns
        """

        if "model" in kwargs:
            raise ValueError(
                "model is not allowed to be passed as a keyword argument"
                "use `llm` instead"
            )
        self.llm = llm

        self.forecasting_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=ForecastAgentOutput,
            system_prompt=self.system_prompt,
            model=self.llm,
            **kwargs,
        )

        self.query_system_prompt = """
        You are a forecasting assistant. You have access to the following dataframes 
        from a previous analysis:
        - fcst_df: Forecasted values for each time series, including dates and 
          predicted values.
        - eval_df: Evaluation results for each model. The evaluation metric is always 
          MASE (Mean Absolute Scaled Error), as established in the main system prompt. 
          Each value in eval_df represents the MASE score for a model.
        - features_df: Extracted time series features for each series, such as trend, 
          seasonality, autocorrelation, and more.
        - anomalies_df: Anomaly detection results, including timestamps, actual values, 
          predictions, and anomaly flags.

        You also have access to a plot_tool that can generate visualizations:
        - plot_tool(plot_type="forecast"): Shows forecast vs actual values
        - plot_tool(plot_type="series"): Shows the raw time series data  
        - plot_tool(plot_type="anomalies"): Shows detected anomalies highlighted
        - plot_tool(plot_type="both"): Shows both forecasts and anomalies in subplots
        - plot_tool(plot_type="raw"): Alternative to "series" for raw data

        The plot tool automatically handles different environments (tmux, terminal, 
        GUI) and will save plots and try to display them using available viewers 
        (imgcat, catimg, system viewer, web browser).

        When the user asks a follow-up question, use these dataframes to provide 
        detailed, data-driven answers. Reference specific values, trends, or metrics 
        from the dataframes as needed. If the user asks about model performance, use 
        eval_df and explain that the metric is MASE. For questions about the forecast, 
        use fcst_df. For questions about the characteristics of the time series, use 
        features_df. For questions about anomalies, use anomalies_df.

        When users request plots, visualizations, or want to "see" something, use the 
        plot_tool with the appropriate plot_type. Common requests include:
        - "show me the plot", "plot the forecast" -> use plot_tool(plot_type="forecast")
        - "plot the series", "show the data" -> use plot_tool(plot_type="series")  
        - "plot the anomalies", "show anomalies" -> use plot_tool(plot_type="anomalies")
        - "show both", "plot everything" -> use plot_tool(plot_type="both")

        Always explain your reasoning and cite the relevant data when answering. If a 
        question cannot be answered with the available data, politely explain the 
        limitation.
        """

        self.query_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=str,
            system_prompt=self.query_system_prompt,
            model=self.llm,
            **kwargs,
        )

        self.dataset: ExperimentDataset
        self.fcst_df: pd.DataFrame
        self.eval_df: pd.DataFrame
        self.features_df: pd.DataFrame
        self.anomalies_df: pd.DataFrame
        self.eval_forecasters: list[str]

        # Cache for checking if parameters changed (for re-running workflow)
        self._last_forecast_params: dict = {}

        # Conversation history for maintaining context between queries
        self.conversation_history: list[dict] = []

        @self.query_agent.tool
        async def plot_tool(
            ctx: RunContext[ExperimentDataset],
            plot_type: str = "anomalies",
            models: list[str] | None = None,
        ) -> str:
            """Generate and display plots for the time series data and results."""
            try:
                import os
                import subprocess
                import sys

                import matplotlib
                import matplotlib.pyplot as plt

                from timecopilot.models.utils.forecaster import Forecaster

                # Configure matplotlib for different environments
                in_tmux = bool(os.environ.get("TMUX"))
                has_display = bool(os.environ.get("DISPLAY"))

                # Check if any terminal image viewers are available
                has_terminal_viewer = False
                terminal_viewers = ["imgcat", "catimg", "timg", "chafa"]
                for viewer in terminal_viewers:
                    try:
                        if (
                            subprocess.run(
                                ["which", viewer], capture_output=True
                            ).returncode
                            == 0
                        ):
                            has_terminal_viewer = True
                            break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue

                # Prefer terminal viewers if available, especially in tmux
                if in_tmux or has_terminal_viewer:
                    # Use terminal display - save file and display via terminal viewer
                    matplotlib.use("Agg")
                    save_and_display = True
                elif not has_display:
                    # No display available - save only
                    matplotlib.use("Agg")
                    save_and_display = True
                else:
                    # Normal environment without terminal viewer - use interactive
                    try:
                        matplotlib.use("TkAgg")
                        save_and_display = False
                    except ImportError:
                        try:
                            matplotlib.use("Qt5Agg")
                            save_and_display = False
                        except ImportError:
                            matplotlib.use("Agg")
                            save_and_display = True

                def try_display_plot(plot_file: str) -> str:
                    """Try different methods to display plot,
                    prioritizing terminal viewers."""

                    # Priority 1: Try terminal image viewers first (for tmux/terminal)
                    terminal_viewers = [
                        ("imgcat", [plot_file]),  # iTerm2
                        ("catimg", [plot_file]),  # Terminal image viewer
                        ("timg", [plot_file]),  # Terminal image viewer
                        ("chafa", [plot_file]),  # Terminal image viewer
                    ]

                    for viewer, cmd in terminal_viewers:
                        try:
                            if (
                                subprocess.run(
                                    ["which", viewer], capture_output=True
                                ).returncode
                                == 0
                            ):
                                subprocess.run([viewer] + cmd, check=True)
                                return (
                                    f"Plot saved as '{plot_file}' and "
                                    f"displayed with {viewer}"
                                )
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            continue

                    # Priority 2: Try system default (only if no terminal viewer worked)
                    try:
                        if sys.platform == "darwin":  # macOS
                            subprocess.run(
                                ["open", plot_file], check=True, capture_output=True
                            )
                            return (
                                f"Plot saved as '{plot_file}' and "
                                f"opened with system viewer"
                            )
                        elif sys.platform.startswith("linux"):
                            subprocess.run(
                                ["xdg-open", plot_file], check=True, capture_output=True
                            )
                            return (
                                f"Plot saved as '{plot_file}' and "
                                f"opened with system viewer"
                            )
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        pass

                    # Priority 3: Try web browser (as last resort)
                    try:
                        import webbrowser

                        webbrowser.open(f"file://{os.path.abspath(plot_file)}")
                        return f"Plot saved as '{plot_file}' and opened in web browser"
                    except Exception:
                        pass

                    # If nothing worked, just return the file location
                    return (
                        f"Plot saved as '{plot_file}'. "
                        "To view: 'open {plot_file}' (macOS) or install "
                        "imgcat/catimg for terminal viewing"
                    )

                # Determine what to plot based on available data and plot_type
                if plot_type == "series" or plot_type == "raw":
                    # Plot raw time series data
                    fig = Forecaster.plot(
                        df=ctx.deps.df,
                        forecasts_df=None,  # No forecasts, just raw data
                        engine="matplotlib",
                        max_ids=10,
                    )

                    if save_and_display:
                        plot_file = "timecopilot_series.png"
                        if fig is not None:
                            fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close(fig)
                        else:
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close()
                        return try_display_plot(plot_file)
                    else:
                        if fig is not None:
                            plt.show()
                        else:
                            plt.show()
                        return "Raw time series plot generated and displayed."

                elif plot_type == "anomalies" and hasattr(self, "anomalies_df"):
                    # Plot anomaly detection results
                    fig = Forecaster.plot(
                        df=ctx.deps.df,
                        forecasts_df=self.anomalies_df,
                        plot_anomalies=True,
                        engine="matplotlib",
                        max_ids=5,
                    )

                    if save_and_display:
                        plot_file = "timecopilot_anomalies.png"
                        if fig is not None:
                            fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close(fig)
                        else:
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close()
                        return try_display_plot(plot_file)
                    else:
                        if fig is not None:
                            plt.show()
                        else:
                            plt.show()
                        return "Anomaly plot generated and displayed."

                elif plot_type == "forecast" and hasattr(self, "fcst_df"):
                    # Plot forecast results
                    if models is None:
                        # Use all available models in forecast
                        model_cols = [
                            col
                            for col in self.fcst_df.columns
                            if col not in ["unique_id", "ds"] and "-" not in col
                        ]
                        models = model_cols

                    fig = Forecaster.plot(
                        df=ctx.deps.df,
                        forecasts_df=self.fcst_df,
                        models=models,
                        engine="matplotlib",
                        max_ids=5,
                    )

                    if save_and_display:
                        plot_file = "timecopilot_forecast.png"
                        if fig is not None:
                            fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close(fig)
                        else:
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close()
                        result_msg = try_display_plot(plot_file)
                        return f"{result_msg} (models: {', '.join(models)})"
                    else:
                        if fig is not None:
                            plt.show()
                        else:
                            plt.show()
                        return (
                            f"Forecast plot generated and displayed for models: "
                            f"{', '.join(models)}."
                        )

                elif plot_type == "both":
                    # Plot both forecasts and anomalies if available
                    if hasattr(self, "fcst_df") and hasattr(self, "anomalies_df"):
                        # Create subplots for both
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                        # Plot forecasts
                        if models is None:
                            model_cols = [
                                col
                                for col in self.fcst_df.columns
                                if col not in ["unique_id", "ds"] and "-" not in col
                            ]
                            models = model_cols

                        # Plot forecasts in first subplot
                        Forecaster.plot(
                            df=ctx.deps.df,
                            forecasts_df=self.fcst_df,
                            models=models,
                            engine="matplotlib",
                            max_ids=3,
                            ax=ax1,
                        )
                        ax1.set_title("Forecasts")

                        # Plot anomalies in second subplot
                        Forecaster.plot(
                            df=ctx.deps.df,
                            forecasts_df=self.anomalies_df,
                            plot_anomalies=True,
                            engine="matplotlib",
                            max_ids=3,
                            ax=ax2,
                        )
                        ax2.set_title("Anomaly Detection")

                        plt.tight_layout()

                        if save_and_display:
                            plot_file = "timecopilot_combined.png"
                            fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plt.close(fig)
                            return try_display_plot(plot_file)
                        else:
                            plt.show()
                            return (
                                "Combined forecast and anomaly plots "
                                "generated and displayed."
                            )
                    else:
                        return (
                            "Error: Need both forecast and anomaly data "
                            "for 'both' plot type."
                        )

                else:
                    # Determine what's available and suggest
                    available = []
                    if hasattr(self, "fcst_df"):
                        available.append("forecasts")
                    if hasattr(self, "anomalies_df"):
                        available.append("anomalies")

                    if available:
                        return (
                            f"Error: Cannot plot '{plot_type}'. "
                            f"Available data: {', '.join(available)}. "
                            "Try plot_type='series', 'forecast', 'anomalies', "
                            "or 'both'."
                        )
                    else:
                        return (
                            "No forecast or anomaly data available. "
                            "You can plot raw series with plot_type='series' "
                            "or run analysis first."
                        )

            except Exception as e:
                return f"Error generating plot: {str(e)}"

        @self.query_agent.system_prompt
        async def add_experiment_info(
            ctx: RunContext[ExperimentDataset],
        ) -> str:
            output = "\n".join(
                [
                    _transform_time_series_to_text(ctx.deps.df),
                    _transform_features_to_text(self.features_df),
                    _transform_eval_to_text(self.eval_df, self.eval_forecasters),
                    _transform_fcst_to_text(self.fcst_df),
                    _transform_anomalies_to_text(self.anomalies_df),
                ]
            )
            return output

        @self.forecasting_agent.system_prompt
        async def add_time_series(
            ctx: RunContext[ExperimentDataset],
        ) -> str:
            return _transform_time_series_to_text(ctx.deps.df)

        @self.forecasting_agent.tool
        async def tsfeatures_tool(
            ctx: RunContext[ExperimentDataset],
            features: list[str],
        ) -> str:
            callable_features = []
            for feature in features:
                if feature not in TSFEATURES:
                    raise ModelRetry(
                        f"Feature {feature} is not available. Available features are: "
                        f"{', '.join(TSFEATURES.keys())}"
                    )
                callable_features.append(TSFEATURES[feature])
            features_dfs = []
            for uid in ctx.deps.df["unique_id"].unique():
                features_df_uid = _get_feats(
                    index=uid,
                    ts=ctx.deps.df,
                    features=callable_features,
                    freq=ctx.deps.seasonality,
                )
                features_dfs.append(features_df_uid)
            features_df = pd.concat(features_dfs) if features_dfs else pd.DataFrame()
            features_df = features_df.rename_axis("unique_id")  # type: ignore
            self.features_df = features_df
            return _transform_features_to_text(features_df)

        @self.forecasting_agent.tool
        async def cross_validation_tool(
            ctx: RunContext[ExperimentDataset],
            models: list[str],
        ) -> str:
            callable_models = []
            for str_model in models:
                if str_model not in self.forecasters:
                    raise ModelRetry(
                        f"Model {str_model} is not available. Available models are: "
                        f"{', '.join(self.forecasters.keys())}"
                    )
                callable_models.append(self.forecasters[str_model])
            forecaster = TimeCopilotForecaster(models=callable_models)
            fcst_cv = forecaster.cross_validation(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            eval_df = ctx.deps.evaluate_forecast_df(
                forecast_df=fcst_cv,
                models=[model.alias for model in callable_models],
            )
            eval_df = eval_df.groupby(
                ["metric"],
                as_index=False,
            ).mean(numeric_only=True)
            self.eval_df = eval_df
            self.eval_forecasters = models
            return _transform_eval_to_text(eval_df, models)

        @self.forecasting_agent.tool
        async def forecast_tool(
            ctx: RunContext[ExperimentDataset],
            model: str,
        ) -> str:
            callable_model = self.forecasters[model]
            forecaster = TimeCopilotForecaster(models=[callable_model])
            fcst_df = forecaster.forecast(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            self.fcst_df = fcst_df
            return _transform_fcst_to_text(fcst_df)

        @self.forecasting_agent.tool
        async def detect_anomalies_tool(
            ctx: RunContext[ExperimentDataset],
            model: str,
            level: int = 95,
        ) -> str:
            """
            Detect anomalies in the time series using the specified model.

            Args:
                model: The model to use for anomaly detection
                level: Confidence level for anomaly detection (default: 95)
            """
            callable_model = self.forecasters[model]
            anomalies_df = callable_model.detect_anomalies(
                df=ctx.deps.df,
                freq=ctx.deps.freq,
                level=level,
            )
            self.anomalies_df = anomalies_df

            # Transform to text for the agent
            anomaly_count = anomalies_df[f"{model}-anomaly"].sum()
            total_points = len(anomalies_df)
            anomaly_rate = (
                (anomaly_count / total_points) * 100 if total_points > 0 else 0
            )

            output = (
                f"Anomaly detection completed using {model} model. "
                f"Found {anomaly_count} anomalies out of {total_points} data points "
                f"({anomaly_rate:.1f}% anomaly rate) at {level}% confidence level. "
                f"Anomalies are flagged in the '{model}-anomaly' column."
            )

            if anomaly_count > 0:
                # Add details about detected anomalies
                anomalies = anomalies_df[anomalies_df[f"{model}-anomaly"]]
                timestamps = list(anomalies["ds"].dt.strftime("%Y-%m-%d").head(10))
                output += f" Anomalies detected at timestamps: {timestamps}"
                if len(anomalies) > 10:
                    output += f" and {len(anomalies) - 10} more."

            return output

        @self.forecasting_agent.output_validator
        async def validate_best_model(
            ctx: RunContext[ExperimentDataset],
            output: ForecastAgentOutput,
        ) -> ForecastAgentOutput:
            if not output.is_better_than_seasonal_naive:
                raise ModelRetry(
                    "The selected model is not better than the seasonal naive model. "
                    "Please try again with a different model."
                    "The cross-validation results are: "
                    f"{output.model_comparison}"
                )
            return output

    def _should_rerun_workflow(self, h: int | None, freq: str | None) -> bool:
        """Check if parameters have changed and workflow should be re-run."""
        if not self._last_forecast_params:
            return True  # First run

        return (
            self._last_forecast_params.get("h") != h
            or self._last_forecast_params.get("freq") != freq
        )

    def _get_maybe_rerun_agent(self, query: str) -> tuple[Agent, str]:
        context_parts = []
        if hasattr(self, "conversation_history") and self.conversation_history:
            recent_context = self.conversation_history[-3:]  # Last 3 exchanges
            context_parts.append("Recent conversation context:")
            for msg in recent_context:
                context_parts.append(f"User: {msg.get('user', '')}")
                context_parts.append(f"Assistant: {msg.get('assistant', '')}")

        # Build current analysis context
        analysis_context = []
        if hasattr(self, "eval_df") and self.eval_df is not None:
            models_used = [col for col in self.eval_df.columns if col != "metric"]
            analysis_context.append(
                f"Current analysis used models: {', '.join(models_used)}"
            )

        if hasattr(self, "fcst_df") and self.fcst_df is not None:
            horizon = len(self.fcst_df)
            analysis_context.append(f"Current forecast horizon: {horizon} periods")

        if hasattr(self, "anomalies_df") and self.anomalies_df is not None:
            analysis_context.append("Anomaly detection was performed")

        # Create the decision prompt
        context_text = (
            "\n".join(context_parts) if context_parts else "No previous context"
        )
        analysis_text = (
            "\n".join(analysis_context) if analysis_context else "No previous analysis"
        )

        decision_prompt = f"""
        You are a time series analysis assistant. 
        You need to determine if a user's query 
        requires re-running the analysis workflow.

        CURRENT USER QUERY: "{query}"

        CONVERSATION CONTEXT:
        {context_text}

        CURRENT ANALYSIS STATE:
        {analysis_text}

        AVAILABLE ACTIONS:
        1. RERUN ANALYSIS: Generate new forecasts, try different models, 
        detect anomalies, change parameters
        2. QUERY EXISTING: Answer questions about existing results, show plots, 
        explain findings

        DECISION CRITERIA - RERUN ANALYSIS if the user wants to:
        - Try different models (e.g., "try Chronos", "use ARIMA instead", 
            "switch to TimesFM")
        - Change forecast parameters (e.g., "forecast next 12 months",
             "change horizon to 6")
        - Detect anomalies (e.g., "find anomalies", "detect outliers")
        - Compare models (e.g., "compare Chronos vs ARIMA", "which is better")
        - Load new data (e.g., "use this new dataset", "analyze different file")
        - Re-analyze with different approach (e.g., "analyze again", 
            "try different method")

        DO NOT RERUN if the user wants to:
        - Ask questions about existing results (e.g., "what does this mean", 
        "explain the forecast")
        - Show visualizations (e.g., "plot the results", "show me the chart")
        - Get explanations (e.g., "why did you choose this model",
             "what are the trends")
        - Request summaries (e.g., "summarize the findings", "what did you find")

        Respond with ONLY True or False.
        """

        decision_agent = Agent(
            model=self.llm,
            system_prompt=(
                "You are a decision-making assistant. Respond with only "
                "True or False based on the user's intent."
            ),
            output_type=bool,
        )
        return decision_agent, decision_prompt

    def _maybe_rerun(self, query: str) -> bool:
        if not query:
            return False

        decision_agent, decision_prompt = self._get_maybe_rerun_agent(query)
        result = decision_agent.run_sync(decision_prompt)
        return result.output

    def is_queryable(self) -> bool:
        """
        Check if the class is queryable.
        It needs to have `dataset`, `fcst_df`, `eval_df`, `features_df`,
        `anomalies_df` and `eval_forecasters`.
        """
        return all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in [
                "dataset",
                "fcst_df",
                "eval_df",
                "features_df",
                "anomalies_df",
                "eval_forecasters",
            ]
        )

    def analyze(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and anomaly analysis.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][timecopilot.agent.ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        query = f"User query: {query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(
            model=self.forecasting_agent.model,
        )
        self.dataset = experiment_dataset_parser.parse(
            df,
            freq,
            h,
            seasonality,
            query,
        )
        query = query if query else "complete system prompt"
        result = self.forecasting_agent.run_sync(
            user_prompt=query,
            deps=self.dataset,
        )
        result.fcst_df = getattr(self, "fcst_df", None)
        result.eval_df = getattr(self, "eval_df", None)
        result.features_df = getattr(self, "features_df", None)
        result.anomalies_df = getattr(self, "anomalies_df", None)
        return result

    def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and analysis.

        .. deprecated:: 0.1.0
            Use :meth:`analyze` instead. This method is kept for backwards
            compatibility.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][timecopilot.agent.
                ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        # Delegate to the new analyze method
        return self.analyze(df=df, h=h, freq=freq, seasonality=seasonality, query=query)

    def _maybe_raise_if_not_queryable(self):
        if not self.is_queryable():
            raise ValueError(
                "The class is not queryable. Please run analysis first using "
                "`analyze()` or `forecast()`."
            )

    def query(
        self,
        query: str,
    ) -> AgentRunResult[str]:
        # fmt: off
        """
        Ask a follow-up question about the analysis results with conversation history.

        This method enables chat-like, interactive querying after an analysis
        has been run. The agent will use the stored dataframes and maintain
        conversation history to provide contextual responses. It can answer
        questions about forecasts, anomalies, visualizations, and more.

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, anomaly detection, or visualizations.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., no analysis
                has been run and required dataframes are missing).

        Example:
            ```python
            import pandas as pd
            from timecopilot import TimeCopilot

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv") 
            tc = TimeCopilot(llm="openai:gpt-4o")
            tc.forecast(df, h=12, freq="MS")
            answer = tc.query("Which model performed best?")
            print(answer.output)
            ```
        Note:
            The class is not queryable until an analysis method has been called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()

        if self._maybe_rerun(query):
            self.analyze(df=self.dataset.df, query=query)

        # Build conversation context with history
        conversation_context = self._build_conversation_context(query)

        result = self.query_agent.run_sync(
            user_prompt=conversation_context,
            deps=self.dataset,
        )

        # Store the conversation in history
        self.conversation_history.append({"user": query, "assistant": result.output})

        return result

    def _build_conversation_context(self, current_query: str) -> str:
        """Build conversation context including history for better responses."""
        if not self.conversation_history:
            # No history, just return the current query
            return current_query

        # Build context with conversation history
        context_parts = ["Previous conversation:"]

        # Add recent conversation history (last 5 exchanges to avoid token limits)
        recent_history = self.conversation_history[-5:]
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")

        context_parts.append(f"\nCurrent question: {current_query}")

        return "\n".join(context_parts)

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []


class AsyncTimeCopilot(TimeCopilot):
    def __init__(self, **kwargs: Any):
        """
        Initialize an asynchronous TimeCopilot agent.

        Inherits from TimeCopilot and provides async methods for
        forecasting and querying.
        """
        super().__init__(**kwargs)

    async def _maybe_rerun(self, query: str) -> bool:  # type: ignore
        if not query:
            return False

        decision_agent, decision_prompt = self._get_maybe_rerun_agent(query)
        result = await decision_agent.run(decision_prompt)
        return result.output

    async def analyze(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[ForecastAgentOutput]:
        """
        Asynchronously analyze time series data with forecasting, anomaly detection,
        or visualization.

        This method can handle multiple types of analysis based on the query:
        - Forecasting: Generate predictions for future periods
        - Anomaly Detection: Identify outliers and unusual patterns
        - Visualization: Create plots and charts
        - Combined: Multiple analysis types together

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - You must always work with time series data with the columns
                  ds (date) and y (target value), if these are missing, attempt to
                  infer them from similar column names or, if unsure, request
                  clarification from the user.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments. Examples:
                - "forecast next 12 months"
                - "detect anomalies with 95% confidence"
                - "plot the time series data"
                - "forecast and detect anomalies"

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][timecopilot.agent.
                ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        query = f"User query: {query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(
            model=self.forecasting_agent.model,
        )
        self.dataset = await experiment_dataset_parser.parse_async(
            df,
            freq,
            h,
            seasonality,
            query,
        )
        result = await self.forecasting_agent.run(
            user_prompt=query,
            deps=self.dataset,
        )
        # Attach dataframes if they exist (depends on workflow)
        result.fcst_df = getattr(self, "fcst_df", None)
        result.eval_df = getattr(self, "eval_df", None)
        result.features_df = getattr(self, "features_df", None)
        result.anomalies_df = getattr(self, "anomalies_df", None)
        return result

    async def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[ForecastAgentOutput]:
        """
        Asynchronously generate forecast and analysis for the provided
        time series data.

        .. deprecated:: 0.1.0
            Use :meth:`analyze` instead. This method is kept for backwards
            compatibility.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - You must always work with time series data with the columns
                  ds (date) and y (target value), if these are missing, attempt to
                  infer them from similar column names or, if unsure, request
                  clarification from the user.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][timecopilot.agent.
                ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        # Delegate to the new analyze method
        return await self.analyze(
            df=df,
            h=h,
            freq=freq,
            seasonality=seasonality,
            query=query,
        )

    @asynccontextmanager
    async def query_stream(
        self,
        query: str,
    ) -> AsyncGenerator[AgentRunResult[str], None]:
        # fmt: off
        """
        Asynchronously stream the agent's answer to a follow-up question.

        This method enables chat-like, interactive querying after a forecast 
        has been run.
        The agent will use the stored dataframes and the original dataset 
        to answer the user's
        question, yielding results as they become available (streaming).

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, or time series features.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., forecast
                has not been run and required dataframes are missing).

        Example:
            ```python
            import asyncio

            import pandas as pd
            from timecopilot import AsyncTimeCopilot

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv") 

            async def example():
                tc = AsyncTimeCopilot(llm="openai:gpt-4o")
                await tc.forecast(df, h=12, freq="MS")
                async with tc.query_stream("Which model performed best?") as result:
                    async for text in result.stream(debounce_by=0.01):
                        print(text, end="", flush=True)
            
            asyncio.run(example())
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()
        if await self._maybe_rerun(query):
            await self.analyze(df=self.dataset.df, query=query)

        # Build conversation context with history
        conversation_context = self._build_conversation_context(query)

        async with self.query_agent.run_stream(
            user_prompt=conversation_context,
            deps=self.dataset,
        ) as result:
            # Store the conversation in history after streaming completes
            # Note: We'll store the final result when the stream is consumed
            yield result

            # Store conversation after streaming (this might not capture the full
            # response)
            # For streaming, we'll store what we can
            self.conversation_history.append(
                {"user": query, "assistant": "[Streaming response - see above]"}
            )

    async def query(
        self,
        query: str,
    ) -> AgentRunResult[str]:
        # fmt: off
        """
        Asynchronously ask a follow-up question about the forecast, 
        model evaluation, or time series features.

        This method enables chat-like, interactive querying after a forecast
        has been run. The agent will use the stored dataframes (`fcst_df`,
        `eval_df`, `features_df`) and the original dataset to answer the user's
        question in a data-driven manner. Typical queries include asking about
        the best model, forecasted values, or time series characteristics.

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, or time series features.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., forecast
                has not been run and required dataframes are missing).

        Example:
            ```python
            import asyncio

            import pandas as pd
            from timecopilot import AsyncTimeCopilot

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv") 

            async def example():
                tc = AsyncTimeCopilot(llm="openai:gpt-4o")
                await tc.forecast(df, h=12, freq="MS")
                answer = await tc.query("Which model performed best?")
                print(answer.output)

            asyncio.run(example())
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()
        if await self._maybe_rerun(query):
            await self.analyze(df=self.dataset.df, query=query)

        # Build conversation context with history
        conversation_context = self._build_conversation_context(query)

        result = await self.query_agent.run(
            user_prompt=conversation_context,
            deps=self.dataset,
        )

        # Store the conversation in history
        self.conversation_history.append({"user": query, "assistant": result.output})

        return result
