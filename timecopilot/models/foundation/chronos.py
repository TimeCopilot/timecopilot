from foundationforecast.models.chronos import Chronos as _Chronos
from foundationforecast.models.chronos import ChronosFinetuningConfig

from ..utils.forecaster import Forecaster


class Chronos(_Chronos, Forecaster):
    pass


__all__ = ["Chronos", "ChronosFinetuningConfig"]
