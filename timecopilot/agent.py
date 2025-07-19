from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.agent import AgentRunResult
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

from .forecaster import TimeCopilotForecaster
from .models.benchmarks import (
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
from .models.benchmarks.prophet import Prophet
from .models.foundational.timesfm import TimesFM
from .utils.experiment_handler import ExperimentDataset, ExperimentDatasetParser

MODELS = {
    "ADIDA": lambda max_length=None: ADIDA(max_length=max_length),
    "AutoARIMA": lambda max_length=None: AutoARIMA(max_length=max_length),
    "AutoCES": AutoCES,
    "AutoETS": AutoETS,
    "CrostonClassic": CrostonClassic,
    "DynamicOptimizedTheta": DynamicOptimizedTheta,
    "HistoricAverage": HistoricAverage,
    "IMAPA": IMAPA,
    "SeasonalNaive": lambda max_length=None: SeasonalNaive(max_length=max_length),
    "Theta": Theta,
    "ZeroModel": ZeroModel,
    "TimesFM": lambda max_length=None: TimesFM(max_length=max_length),
    "Prophet": Prophet,
}

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
    user_query_response: str | None = Field(
        description=(
            "The response to the user's query, if any. "
            "If the user did not provide a query, this field will be None."
        )
    )

    def prettify(self, console: Console | None = None) -> None:
        """Pretty print the forecast results using rich formatting."""
        console = console or Console()

        # Create header with title and overview
        header = Panel(
            f"[bold cyan]{self.selected_model}[/bold cyan] forecast analysis\n"
            f"[{'green' if self.is_better_than_seasonal_naive else 'red'}]"
            f"{'✓ Better' if self.is_better_than_seasonal_naive else '✗ Not better'} "
            "than Seasonal Naive[/"
            f"{'green' if self.is_better_than_seasonal_naive else 'red'}]",
            title="[bold blue]TimeCopilot Forecast[/bold blue]",
            style="blue",
        )

        # Time Series Analysis Section
        ts_features = Table(
            title="Time Series Features",
            show_header=True,
            title_style="bold cyan",
            header_style="bold magenta",
        )
        ts_features.add_column("Feature", style="cyan")
        ts_features.add_column("Value", style="magenta")

        # Group features by category for better organization
        for feature in self.tsfeatures_results:
            feature_name, feature_value = feature.split(":")
            ts_features.add_row(
                feature_name.strip(), f"{float(feature_value.strip()):.3f}"
            )

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

        # Model Comparison Table
        model_scores = Table(
            title="Model Performance", show_header=True, title_style="bold yellow"
        )
        model_scores.add_column("Model", style="yellow")
        model_scores.add_column("MASE", style="cyan", justify="right")

        # Sort models by performance
        cv_results = []
        for result in self.cross_validation_results:
            model, score = result.split(":")
            cv_results.append((model.strip(), float(score.strip())))

        cv_results.sort(key=lambda x: x[1])  # Sort by score
        for model, score in cv_results:  # type: ignore
            model_scores.add_row(model, f"{score:.2f}")

        model_analysis = Panel(
            self.model_comparison,
            title="[bold yellow]Performance Analysis[/bold yellow]",
            style="yellow",
        )

        # Forecast Results Section
        forecast_table = Table(
            title="Forecast Values", show_header=True, title_style="bold magenta"
        )
        forecast_table.add_column("Period", style="magenta")
        forecast_table.add_column("Value", style="cyan", justify="right")

        # Show all individual values with period indicators
        for fcst in self.forecast:
            period, value = fcst.split(":")
            forecast_table.add_row(f"{period}", f"{value}")

        # Add note about number of periods if many
        if len(self.forecast) > 12:
            forecast_table.caption = (
                f"[dim]Showing all {len(self.forecast)} forecasted periods. "
                "Use aggregation functions for summarized views.[/dim]"
            )

        forecast_analysis = Panel(
            self.forecast_analysis,
            title="[bold magenta]Forecast Analysis[/bold magenta]",
            style="magenta",
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

        if user_response:
            console.print("\n[bold]4. Additional Information[/bold]")
            console.print(user_response)

        console.print("\n")


class TimeCopilot:
    def __init__(
        self,
        llm: str,
        max_length: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM to use.
            max_length: Maximum number of observations to use from the
                end of each time series for training and inference. If None, all
                observations are used. This can significantly improve inference times
                for long time series by reducing the amount of data processed.
            **kwargs: Additional keyword arguments to pass to the agent.
        """
        self.max_length = max_length

        self.system_prompt = f"""
        You're a forecasting expert. You will be given a time series 
        as a list of numbers
        and your task is to determine the best forecasting model for that series. 
        You have access to the following tools:

        1. tsfeatures_tool: Calculates time series features to help 
        with model selection.
        Available features are: {", ".join(TSFEATURES.keys())}

        2. cross_validation_tool: Performs cross-validation for one or more models.
        Takes a list of model names and returns their cross-validation results.
        Available models are: {", ".join(MODELS.keys())}

        3. forecast_tool: Generates forecasts using a selected model.
        Takes a model name and returns forecasted values.

        Your task is to provide a comprehensive analysis following these steps:

        1. Time Series Feature Analysis:
           - Calculate a focused set of key time series features
           - Quickly identify the main characteristics (trend, seasonality, 
                stationarity, etc.)
           - Use these insights to guide efficient model selection
           - Avoid over-analysis - focus on features that directly inform model choice

        2. Model Selection and Evaluation:
           - Start with simple models that can potentially beat seasonal naive
           - Select additional candidate models based on the time series 
                values and features
           - Document each model's technical details and assumptions
           - Explain why these models are suitable for the identified features
           - Perform cross-validation to evaluate performance
           - Compare models quantitatively and qualitatively against seasonal naive
           - If initial models don't beat seasonal naive, try more complex models
           - Prioritize finding a model that outperforms seasonal naive benchmark
           - Balance model complexity with forecast accuracy

        3. Final Model Selection and Forecasting:
           - Choose the best performing model with clear justification
           - Generate the forecast using just the selected model
           - Interpret trends and patterns in the forecast
           - Discuss reliability and potential uncertainties
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
        - Response to any user queries

        Focus on providing:
        - Clear connections between features and model choices
        - Technical accuracy with accessible explanations
        - Quantitative support for decisions
        - Practical implications of the forecast
        - Thorough responses to user concerns
        """

        if "model" in kwargs:
            raise ValueError(
                "model is not allowed to be passed as a keyword argument"
                "use `llm` instead"
            )

        self.forecasting_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=ForecastAgentOutput,
            system_prompt=self.system_prompt,
            model=llm,
            **kwargs,
        )

        @self.forecasting_agent.system_prompt
        async def add_time_series(
            ctx: RunContext[ExperimentDataset],
        ) -> str:
            df_agg = ctx.deps.df.groupby("unique_id").agg(list)
            output = (
                "these are the time series in json format where the key is the "
                "identifier of the time series and the values is also a json "
                "of two elements: "
                "the first element is the date column and the second element is the "
                "value column."
                f"{df_agg.to_json(orient='index')}"
            )
            return output

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
            features_df: pd.DataFrame | None = None
            for uid in ctx.deps.df["unique_id"].unique():
                features_df_uid = _get_feats(
                    index=uid,
                    ts=ctx.deps.df,
                    features=callable_features,
                    freq=ctx.deps.seasonality,
                )
                if features_df is None:
                    features_df = features_df_uid
                else:
                    features_df = pd.concat([features_df, features_df_uid])
            features_df = features_df.rename_axis("unique_id")  # type: ignore
            self.features_df = features_df
            output = (
                "these are the time series features in json format where the key is "
                "the identifier of the time series and the values is also a json of "
                "feature names and their values."
                f"{features_df.to_json(orient='index')}"
            )
            return output

        @self.forecasting_agent.tool
        async def cross_validation_tool(
            ctx: RunContext[ExperimentDataset],
            models: list[str],
        ) -> str:
            callable_models = []
            for str_model in models:
                if str_model not in MODELS:
                    raise ModelRetry(
                        f"Model {str_model} is not available. Available models are: "
                        f"{', '.join(MODELS.keys())}"
                    )
                model_factory = MODELS[str_model]
                if callable(model_factory):
                    callable_models.append(model_factory(max_length=self.max_length))
                else:
                    callable_models.append(model_factory)
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
            return ", ".join(
                [
                    f"{model.alias}: {eval_df[model.alias].iloc[0]}"
                    for model in callable_models
                ]
            )

        @self.forecasting_agent.tool
        async def forecast_tool(
            ctx: RunContext[ExperimentDataset],
            model: str,
        ) -> str:
            model_factory = MODELS[model]
            if callable(model_factory):
                callable_model = model_factory(max_length=self.max_length)
            else:
                callable_model = model_factory
            forecaster = TimeCopilotForecaster(models=[callable_model])
            fcst_df = forecaster.forecast(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            df_agg = fcst_df.groupby("unique_id").agg(list)
            output = (
                "these are the forecasted values in json format where the key is the "
                "identifier of the time series and the values is also a json of two "
                "elements: the first element is the date column and the second "
                "element is the value column."
                f"{df_agg.to_json(orient='index')}"
            )
            self.fcst_df = fcst_df
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
                    "{output.cross_validation_results}"
                )
            return output

    def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        max_length: int | None = None,
    ) -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and analysis.

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
            max_length: Maximum number of observations to use from the
                end of each time series for training and inference. If None,
                uses the instance's default max_length, or all observations
                if no default is set. This can significantly improve inference
                times for long time series by reducing the amount of data processed.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][timecopilot.agent.ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        # Use the provided max_length or fall back to the instance's default
        if max_length is not None:
            self.max_length = max_length
        
        query = f"User query: {query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(
            model=self.forecasting_agent.model,
        )
        dataset = experiment_dataset_parser.parse(
            df,
            freq,
            h,
            seasonality,
            query,
        )
        result = self.forecasting_agent.run_sync(
            user_prompt=query,
            deps=dataset,
        )
        result.fcst_df = self.fcst_df
        result.eval_df = self.eval_df
        result.features_df = self.features_df
        return result
