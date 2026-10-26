from foundationforecast.models.tabby import Tabby as _Tabby

from ..utils.forecaster import Forecaster


class Tabby(_Tabby, Forecaster):
    pass


__all__ = ["Tabby"]
