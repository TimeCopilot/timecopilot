from foundationforecast.models.timesfm import TimesFM as _TimesFM

from ..utils.forecaster import Forecaster


class TimesFM(_TimesFM, Forecaster):
    pass


__all__ = ["TimesFM"]
