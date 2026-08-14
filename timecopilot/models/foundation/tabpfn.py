from foundationforecast.models.tabpfn import TabPFN as _TabPFN

from ..utils.forecaster import Forecaster


class TabPFN(_TabPFN, Forecaster):
    pass


__all__ = ["TabPFN"]
