from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from chronos import (
    BaseChronosPipeline,
    Chronos2Pipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
)
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset


class Chronos(Forecaster):
    """
    Chronos models are large pre-trained models for time series forecasting,
    supporting both probabilistic and point forecasts. See the
    [official repo](https://github.com/amazon-science/chronos-forecasting)
    for more details.
    """

    def __init__(
        self,
        repo_id: str = "amazon/chronos-t5-large",
        batch_size: int = 16,
        alias: str = "Chronos",
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local
                path to load the Chronos model from. Examples include
                "amazon/chronos-t5-tiny", "amazon/chronos-t5-large", or a
                local directory. Defaults to "amazon/chronos-t5-large". See
                the full list of available models at
                [Hugging Face](https://huggingface.co/collections/
                amazon/chronos-models-65f1791d630a8d57cb718444)
            batch_size (int, optional): Batch size to use for inference.
                Larger models may require smaller batch sizes due to GPU
                memory constraints. Defaults to 16. For Chronos-Bolt models,
                higher batch sizes (e.g., 256) are possible.
            alias (str, optional): Name to use for the model in output
                DataFrames and logs. Defaults to "Chronos".

        Notes:
            **Available models:**

            | Model ID                                                               | Parameters |
            | ---------------------------------------------------------------------- | ---------- |
            | [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2)   | 120M         |
            | [`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny)   | 9M         |
            | [`amazon/chronos-bolt-mini`](https://huggingface.co/amazon/chronos-bolt-mini)   | 21M        |
            | [`amazon/chronos-bolt-small`](https://huggingface.co/amazon/chronos-bolt-small) | 48M        |
            | [`amazon/chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base)   | 205M       |
            | [`amazon/chronos-t5-tiny`](https://huggingface.co/amazon/chronos-t5-tiny)   | 8M         |
            | [`amazon/chronos-t5-mini`](https://huggingface.co/amazon/chronos-t5-mini)   | 20M        |
            | [`amazon/chronos-t5-small`](https://huggingface.co/amazon/chronos-t5-small) | 46M        |
            | [`amazon/chronos-t5-base`](https://huggingface.co/amazon/chronos-t5-base)   | 200M       |
            | [`amazon/chronos-t5-large`](https://huggingface.co/amazon/chronos-t5-large) | 710M       |

            **Academic Reference:**

            - Paper: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)

            **Resources:**

            - GitHub: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
            - HuggingFace: [amazon/chronos-models](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - For best performance with large models (e.g., "chronos-t5-large"),
              a CUDA-compatible GPU is recommended.
            - The model weights are loaded with torch_dtype=torch.bfloat16 for
              efficiency on supported hardware.

        """
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias

    @contextmanager
    def _get_model(self) -> BaseChronosPipeline:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = BaseChronosPipeline.from_pretrained(
            self.repo_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _predict(
        self,
        model: BaseChronosPipeline,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """handles distinction between predict and predict_quantiles"""
        if quantiles is not None:
            fcsts = [
                model.predict_quantiles(
                    batch,
                    prediction_length=h,
                    quantile_levels=quantiles,
                )
                for batch in tqdm(dataset)
            ]  # list of tuples
            fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
            if isinstance(model, Chronos2Pipeline):
                fcsts_mean = [f_mean for fcst in fcsts_mean for f_mean in fcst]  # type: ignore
                fcsts_quantiles = [
                    f_quantile
                    for fcst in fcsts_quantiles
                    for f_quantile in fcst  # type: ignore
                ]
            fcsts_mean_np = torch.cat(fcsts_mean).numpy()
            fcsts_quantiles_np = torch.cat(fcsts_quantiles).numpy()
        else:
            fcsts = [
                model.predict(
                    batch,
                    prediction_length=h,
                )
                for batch in tqdm(dataset)
            ]
            if isinstance(model, Chronos2Pipeline):
                fcsts = [f_fcst for fcst in fcsts for f_fcst in fcst]  # type: ignore
            fcsts = torch.cat(fcsts)
            if isinstance(model, ChronosPipeline):
                # for t5 models, `predict` returns a tensor of shape
                # (batch_size, num_samples, prediction_length).
                fcsts_mean = fcsts.mean(dim=1)  # type: ignore
            elif isinstance(model, ChronosBoltPipeline | Chronos2Pipeline):
                # for bolt models, `predict` returns a tensor of shape
                # (batch_size, num_quantiles, prediction_length)
                # for these models, the median is prefered as mean forecasts
                fcsts_mean = fcsts[:, model.quantiles.index(0.5), :]  # type: ignore
            else:
                raise ValueError(f"Unsupported model: {self.repo_id}")
            fcsts_mean_np = fcsts_mean.numpy()  # type: ignore
            fcsts_quantiles_np = None
        return fcsts_mean_np, fcsts_quantiles_np

    def _forecast_chronos2_df(
        self,
        df: pd.DataFrame,
        h: int,
        qc: QuantileConverter,
        X_df: pd.DataFrame | None,
        static_features: list[str] | None,
    ) -> pd.DataFrame:
        # Chronos-2 exposes an official dataframe API: Chronos2Pipeline.predict_df(df=..., future_df=...).
        # Past-only covariates MUST be additional columns in `df`.
        # Known future covariates MUST be passed via `future_df`.
        # IMPORTANT: future_df cannot contain columns not present in df.

        id_col = "unique_id"
        ts_col = "ds"
        target_col = "y"

        required = {id_col, ts_col, target_col}
        missing = [c for c in sorted(required) if c not in df.columns]
        if missing:
            raise ValueError(f"df is missing required columns: {missing}")

        context_df = df.copy()
        context_df[ts_col] = pd.to_datetime(context_df[ts_col])
        context_df = context_df.sort_values([id_col, ts_col]).reset_index(drop=True)

        # past-only covariates live in df
        past_only_covs = list(static_features) if static_features is not None else []
        if past_only_covs:
            miss_cov = [c for c in past_only_covs if c not in context_df.columns]
            if miss_cov:
                raise ValueError(f"Missing past-only covariate columns in df: {miss_cov}")
            bad = [c for c in past_only_covs if not pd.api.types.is_numeric_dtype(context_df[c])]
            if bad:
                raise ValueError(f"df covariate columns must be numeric. Bad columns: {bad}")

        # known-future covariates passed via future_df (but must also exist in df)
        future_df: pd.DataFrame | None = None
        known_future_covs: list[str] = []
        if X_df is not None:
            req_x = {id_col, ts_col}
            miss_x = [c for c in sorted(req_x) if c not in X_df.columns]
            if miss_x:
                raise ValueError(f"X_df is missing required columns: {miss_x}")

            future_df = X_df.copy()
            future_df[ts_col] = pd.to_datetime(future_df[ts_col])
            future_df = future_df.sort_values([id_col, ts_col]).reset_index(drop=True)

            ids_ctx = set(context_df[id_col].unique())
            ids_fut = set(future_df[id_col].unique())
            if ids_ctx != ids_fut:
                missing_in_fut = sorted(ids_ctx - ids_fut)
                extra_in_fut = sorted(ids_fut - ids_ctx)
                raise ValueError(
                    "X_df unique_id set must match df unique_id set. "
                    f"missing_in_X_df={missing_in_fut}, extra_in_X_df={extra_in_fut}"
                )

            known_future_covs = [c for c in future_df.columns if c not in (id_col, ts_col)]
            if not known_future_covs:
                raise ValueError(
                    "X_df was provided but contains no covariate columns "
                    "(expected at least one column besides unique_id and ds)."
                )

            missing_in_df = [c for c in known_future_covs if c not in context_df.columns]
            if missing_in_df:
                raise ValueError(
                    "Chronos-2 requires known-future covariate columns to be present in df as well. "
                    f"Missing in df: {missing_in_df}. "
                    "Fix: include these columns historically in train_df (same names), not only in X_df."
                )

            bad = [c for c in known_future_covs if not pd.api.types.is_numeric_dtype(future_df[c])]
            if bad:
                raise ValueError(f"X_df covariate columns must be numeric. Bad columns: {bad}")

            # Require at least h future rows per unique_id strictly after the last context timestamp.
            last_ts = context_df.groupby(id_col)[ts_col].max()
            for uid, last in last_ts.items():
                df_u = future_df[future_df[id_col] == uid]
                n_future = (df_u[ts_col] > last).sum()
                if n_future < h:
                    raise ValueError(
                        f"X_df (future_df) must contain at least h={h} future rows per unique_id "
                        f"after the last context timestamp. unique_id={uid!r} has {n_future}."
                    )

        all_covs = past_only_covs + known_future_covs

        with self._get_model() as model:
            if not isinstance(model, Chronos2Pipeline):
                raise ValueError(f"Expected Chronos2Pipeline, got {type(model).__name__}")

            pred_df = model.predict_df(
                context_df[[id_col, ts_col, target_col] + all_covs],
                future_df=None
                if future_df is None
                else future_df[[id_col, ts_col] + known_future_covs],
                prediction_length=h,
                quantile_levels=qc.quantiles,
                id_column=id_col,
                timestamp_column=ts_col,
                target=target_col,
            )

        pred_df = pred_df.rename(columns={id_col: "unique_id", ts_col: "ds"})
        if "predictions" not in pred_df.columns:
            raise ValueError("predict_df output is missing required 'predictions' column")

        pred_df[self.alias] = pred_df["predictions"]

        if qc.quantiles is not None:
            # Chronos2Pipeline.predict_df returns quantile columns named by the quantile level (stringified).
            for q in qc.quantiles:
                raw_col = str(q)
                if raw_col in pred_df.columns:
                    pred_df[f"{self.alias}-q-{int(q * 100)}"] = pred_df[raw_col]

            pred_df = qc.maybe_convert_quantiles_to_level(
                pred_df,
                models=[self.alias],
            )

        final_cols = ["unique_id", "ds", self.alias]
        for col in pred_df.columns:
            if col.startswith(self.alias + "-"):
                final_cols.append(col)
        return pred_df[final_cols].copy()

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        X_df: pd.DataFrame | None = None,
        static_features: list[str] | None = None,
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
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)

        is_chronos2 = "chronos-2" in self.repo_id

        if not is_chronos2 and (X_df is not None or static_features is not None):
            raise NotImplementedError(
                "Exogenous variables are only supported for Chronos-2 via Chronos2Pipeline.predict_df. "
                f"Model repo_id={self.repo_id!r} does not support X_df/static_features."
            )

        if is_chronos2:
            return self._forecast_chronos2_df(
                df=df,
                h=h,
                qc=qc,
                X_df=X_df,
                static_features=static_features,
            )

        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                model,
                dataset,
                h,
                quantiles=qc.quantiles,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None and fcsts_quantiles_np is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
