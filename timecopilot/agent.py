from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
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

from .forecaster import Forecaster, TimeCopilotForecaster
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
from .utils.experiment_handler import ExperimentDataset, ExperimentDatasetParser

DEFAULT_MODELS = [
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
    ) -> None:
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


class TimeCopilot:
    def __init__(
        self,
        llm: str,
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
        as a list of numbers
        and your task is to determine the best forecasting model for that series. 
        You have access to the following tools:

        1. tsfeatures_tool: Calculates time series features to help 
        with model selection.
        Available features are: {", ".join(TSFEATURES.keys())}

        2. cross_validation_tool: Performs cross-validation for one or more models.
        Takes a list of model names and returns their cross-validation results.
        Available models are: {", ".join(self.forecasters.keys())}

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

        When the user asks a follow-up question, use these dataframes to provide 
        detailed, data-driven answers. Reference specific values, trends, or metrics 
        from the dataframes as needed. If the user asks about model performance, use 
        eval_df and explain that the metric is MASE. For questions about the forecast, 
        use fcst_df. For questions about the characteristics of the time series, use 
        features_df.

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
        self.eval_forecasters: list[str]

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

    def is_queryable(self) -> bool:
        """
        Check if the class is queryable.
        It needs to have `dataset`, `fcst_df`, `eval_df`, `features_df`
        and `eval_forecasters`.
        """
        return all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in [
                "dataset",
                "fcst_df",
                "eval_df",
                "features_df",
                "eval_forecasters",
            ]
        )

    def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
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
        result = self.forecasting_agent.run_sync(
            user_prompt=query,
            deps=self.dataset,
        )
        result.fcst_df = self.fcst_df
        result.eval_df = self.eval_df
        result.features_df = self.features_df
        return result

    def _maybe_raise_if_not_queryable(self):
        if not self.is_queryable():
            raise ValueError(
                "The class is not queryable. Please forecast first using `forecast`."
            )

    def query(
        self,
        query: str,
    ) -> AgentRunResult[str]:
        # fmt: off
        """
        Ask a follow-up question about the forecast, model evaluation, or time
        series features.

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
            tc = TimeCopilot(llm="openai:gpt-4o")
            tc.forecast(df, h=12)
            answer = tc.query("Which model performed best?")
            print(answer.output)
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()
        result = self.query_agent.run_sync(
            user_prompt=query,
            deps=self.dataset,
        )
        return result


class AsyncTimeCopilot(TimeCopilot):
    def __init__(self, **kwargs: Any):
        """
        Initialize an asynchronous TimeCopilot agent.

        Inherits from TimeCopilot and provides async methods for
        forecasting and querying.
        """
        super().__init__(**kwargs)

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

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - You must always work with time series data with the columns ds (date)
                  and y (target value), if these are missing, attempt to infer them
                  from similar column names or, if unsure, request clarification
                  from the user.
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
        result.fcst_df = self.fcst_df
        result.eval_df = self.eval_df
        result.features_df = self.features_df
        return result

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
            tc = TimeCopilotAsync(llm="openai:gpt-4o")
            await tc.forecast(df, h=12)
            async with tc.query_stream("Which model performed best?") as result:
                async for text in result.stream(debounce_by=0.01):
                    print(text, end="", flush=True)
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()
        async with self.query_agent.run_stream(
            user_prompt=query,
            deps=self.dataset,
        ) as result:
            yield result

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
            tc = TimeCopilotAsync(llm="openai:gpt-4o")
            await tc.forecast(df, h=12)
            answer = await tc.query("Which model performed best?")
            print(answer.output)
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()
        result = await self.query_agent.run(
            user_prompt=query,
            deps=self.dataset,
        )
        return result
