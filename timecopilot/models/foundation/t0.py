from foundationforecast.models.t0 import T0 as _T0

from ..utils.forecaster import Forecaster


class T0(_T0, Forecaster):
    pass


__all__ = ["T0"]
