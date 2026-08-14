from foundationforecast.models.sundial import Sundial as _Sundial

from ..utils.forecaster import Forecaster


class Sundial(_Sundial, Forecaster):
    pass


__all__ = ["Sundial"]
