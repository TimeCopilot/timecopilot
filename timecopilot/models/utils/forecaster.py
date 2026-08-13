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

    pass
