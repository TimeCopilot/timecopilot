from foundationforecast.models.kumo import Kumo as _Kumo

from ..utils.forecaster import Forecaster


class Kumo(_Kumo, Forecaster):
    pass


__all__ = ["Kumo"]
