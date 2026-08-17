# `timecopilot_gift_eval`

GIFT-Eval integration lives in the standalone [`timecopilot-gift-eval`](https://pypi.org/project/timecopilot-gift-eval/) package on PyPI. Install it with TimeCopilot via:

```bash
pip install "timecopilot[gift-eval]"
```

## Migration from `timecopilot.gift_eval`

The in-repo module `timecopilot.gift_eval` was removed in favor of the standalone package. Update imports as follows:

| Before | After |
|--------|-------|
| `from timecopilot.gift_eval.eval import GIFTEval` | `from timecopilot_gift_eval import GIFTEval` |
| `from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor` | `from timecopilot_gift_eval import GluonTSPredictor` |
| `from timecopilot.gift_eval.utils import DATASETS_WITH_TERMS` | `from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS` |

No compatibility shim is provided — install `timecopilot-gift-eval` (or `timecopilot[gift-eval]`) and switch to the `timecopilot_gift_eval` import path.

::: timecopilot_gift_eval.eval
    options:
        members:
            - GIFTEval

::: timecopilot_gift_eval.gluonts_predictor
    options:
        members:
            - GluonTSPredictor

::: timecopilot_gift_eval.protocol
    options:
        members:
            - ForecasterProtocol
