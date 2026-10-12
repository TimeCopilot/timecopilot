import pandas as pd
from foundationforecast.core.exog.covariates import (
    ExogStrategyConfig,
    normalize_exog_strategy,
)
from foundationforecast.core.forecaster import Forecaster as _Forecaster
from foundationforecast.core.forecaster import (
    QuantileConverter,
    _DataProcessor,
    get_seasonality,
    maybe_convert_col_to_datetime,
    maybe_infer_freq,
)

__all__ = [
    "Forecaster",
    "QuantileConverter",
    "_DataProcessor",
    "get_seasonality",
    "maybe_convert_col_to_datetime",
    "maybe_infer_freq",
]


class Forecaster(_Forecaster):
    """TimeCopilot Forecaster base (extends foundationforecast)."""

    def __init__(
        self,
        *,
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = False,
    ) -> None:
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
        )

    def _ensure_forecaster_state(self) -> None:
        if not hasattr(self, "exog_strategy"):
            self.exog_strategy = normalize_exog_strategy(False)
        if not hasattr(self, "reuse_loaded_model"):
            self.reuse_loaded_model = True

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        *,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        self._ensure_forecaster_state()
        return super().cross_validation(
            df,
            h,
            freq=freq,
            n_windows=n_windows,
            step_size=step_size,
            level=level,
            quantiles=quantiles,
            futr_exog_list=futr_exog_list,
        )

    @staticmethod
    def _maybe_infer_freq(df: pd.DataFrame, freq: str | None) -> str:
        return maybe_infer_freq(df, freq)
