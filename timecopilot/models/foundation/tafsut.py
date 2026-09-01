from foundationforecast.models.tafsut import Tafsut as _Tafsut

from ..utils.forecaster import Forecaster


class Tafsut(_Tafsut, Forecaster):
    pass


__all__ = ["Tafsut"]
