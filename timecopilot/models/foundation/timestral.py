from foundationforecast.models.timestral import Timestral as _Timestral

from ..utils.forecaster import Forecaster


class Timestral(_Timestral, Forecaster):
    pass


__all__ = ["Timestral"]
