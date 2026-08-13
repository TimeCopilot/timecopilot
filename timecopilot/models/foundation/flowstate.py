from foundationforecast.models.flowstate import FlowState as _FlowState

from ..utils.forecaster import Forecaster


class FlowState(_FlowState, Forecaster):
    pass


__all__ = ["FlowState"]
