import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ... import TimeCopilotForecaster
from ..utils.forecaster import Forecaster, QuantileConverter
from ..utils.df_utils import to_pandas


class MedianEnsemble(Forecaster):
    def __init__(self, models: list[Forecaster], alias: str = "MedianEnsemble"):
        # fmt: off
        """
        Initialize a MedianEnsemble forecaster.

        This ensemble combines the forecasts of multiple models by taking the 
        median of their predictions for each time step and series. 
        For probabilistic forecasts (quantiles and levels),
        it uses isotonic regression to ensure monotonicity of the quantile outputs 
        across the ensemble. Optionally, you can set a custom alias 
        for the ensemble.

        Args:
            models (list[Forecaster]):
                List of instantiated forecaster models to be ensembled. Each model must
                implement the forecast method and have a unique alias.
            alias (str, optional):
                Name to use for the ensemble in output DataFrames and logs. Defaults to
                "MedianEnsemble".

        Example:
            ```python
            import pandas as pd
            from timecopilot.models.ensembles.median import MedianEnsemble
            from timecopilot.models.foundation.chronos import Chronos
            from timecopilot.models.stats import SeasonalNaive


            df = pd.read_csv(
                "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
                parse_dates=["ds"],
            )

            models = [
                Chronos(
                    repo_id="amazon/chronos-t5-tiny",
                    alias="Chronos-T5",
                ),
                Chronos(
                    repo_id="amazon/chronos-bolt-tiny",
                    alias="Chronos-Bolt",
                ),
                SeasonalNaive(),
            ]
            median_ensemble = MedianEnsemble(models=models)
            fcst_df = median_ensemble.forecast(
                df=df,
                h=12,
            )
            print(fcst_df)
            ```
        """
        # fmt: on
        self.tcf = TimeCopilotForecaster(models=models, fallback_model=None)
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values. If not provided, the frequency will be inferred
                from the data.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        df = to_pandas(df)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        _fcst_df = self.tcf._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=None,
            quantiles=qc.quantiles,
        )
        fcst_df = _fcst_df[["unique_id", "ds"]]
        model_cols = [model.alias for model in self.tcf.models]
        fcst_df[self.alias] = _fcst_df[model_cols].median(axis=1)
        if qc.quantiles is not None:
            qs = sorted(qc.quantiles)
            q_cols = []
            for q in qs:
                pct = int(q * 100)
                models_q_cols = [f"{col}-q-{pct}" for col in model_cols]
                q_col = f"{self.alias}-q-{pct}"
                fcst_df[q_col] = _fcst_df[models_q_cols].median(axis=1)
                q_cols.append(q_col)
            # enforce monotonicity
            ir = IsotonicRegression(increasing=True)

            def apply_isotonic(row):
                return ir.fit_transform(qs, row)

            # @Azul: this can be parallelized later
            vals_monotonic = fcst_df[q_cols].apply(
                apply_isotonic,
                axis=1,
                result_type="expand",
            )
            fcst_df[q_cols] = vals_monotonic
            if 0.5 in qc.quantiles:
                fcst_df[self.alias] = fcst_df[f"{self.alias}-q-50"].values
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
