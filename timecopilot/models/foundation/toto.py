from foundationforecast.models.toto import Toto as _Toto

from ..utils.forecaster import Forecaster


class Toto(_Toto, Forecaster):
    pass


__all__ = ["Toto"]
