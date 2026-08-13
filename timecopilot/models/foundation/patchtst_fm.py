from foundationforecast.models.patchtst_fm import PatchTSTFM as _PatchTSTFM

from ..utils.forecaster import Forecaster


class PatchTSTFM(_PatchTSTFM, Forecaster):
    pass


__all__ = ["PatchTSTFM"]
