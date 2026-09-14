from foundationforecast.models.tabpfn import TABPFN_V2_MODEL, TABPFN_V3_MODEL
from foundationforecast.models.tabpfn import TabPFN as _TabPFN

from ..utils.forecaster import Forecaster


class TabPFN(_TabPFN, Forecaster):
    pass


__all__ = ["TABPFN_V2_MODEL", "TABPFN_V3_MODEL", "TabPFN"]
