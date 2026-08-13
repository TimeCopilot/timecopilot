from foundationforecast.models.timegpt import TimeGPT as _TimeGPT
from foundationforecast.models.timegpt import TimeGPTFinetuningConfig

from ..utils.forecaster import Forecaster


class TimeGPT(_TimeGPT, Forecaster):
    pass


__all__ = ["TimeGPT", "TimeGPTFinetuningConfig"]
