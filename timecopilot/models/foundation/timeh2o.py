from foundationforecast.models.timeh2o import TimeH2O as _TimeH2O

from ..utils.forecaster import Forecaster


class TimeH2O(_TimeH2O, Forecaster):
    pass


__all__ = ["TimeH2O"]
