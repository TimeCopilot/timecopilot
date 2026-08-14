from foundationforecast.models.moirai import Moirai as _Moirai

from ..utils.forecaster import Forecaster


class Moirai(_Moirai, Forecaster):
    pass


__all__ = ["Moirai"]
