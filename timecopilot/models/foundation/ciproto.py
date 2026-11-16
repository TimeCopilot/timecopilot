from foundationforecast.models.ciproto import CIProto as _CIProto

from ..utils.forecaster import Forecaster


class CIProto(_CIProto, Forecaster):
    pass


__all__ = ["CIProto"]
