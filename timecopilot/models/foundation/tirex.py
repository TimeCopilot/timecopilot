from foundationforecast.models.tirex import TiRex as _TiRex

from ..utils.forecaster import Forecaster


class TiRex(_TiRex, Forecaster):
    pass


__all__ = ["TiRex"]
