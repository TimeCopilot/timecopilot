import sys

from .chronos import Chronos, ChronosFinetuningConfig
from .moirai import Moirai
from .timegpt import TimeGPT, TimeGPTFinetuningConfig
from .timesfm import TimesFM
from .toto import Toto

__all__ = [
    "Chronos",
    "ChronosFinetuningConfig",
    "Moirai",
    "TimeGPT",
    "TimeGPTFinetuningConfig",
    "TimesFM",
    "Toto",
]

if sys.version_info >= (3, 11):
    from .tirex import TiRex as TiRex

    __all__.append("TiRex")

if sys.version_info >= (3, 11) and sys.version_info < (3, 14):
    from .flowstate import FlowState as FlowState
    from .patchtst_fm import PatchTSTFM as PatchTSTFM
    from .t0 import T0 as T0

    __all__.extend(["FlowState", "PatchTSTFM", "T0"])

if sys.version_info < (3, 13):
    from .sundial import Sundial as Sundial
    from .tabpfn import TabPFN as TabPFN

    __all__.extend(["Sundial", "TabPFN"])
