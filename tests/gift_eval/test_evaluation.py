import random
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from timecopilot_gift_eval import GIFTEval, GluonTSPredictor
from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS

from timecopilot.models.stats import SeasonalNaive

TARGET_COLS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    # can be unstable, due to division by zero
    # "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]


@pytest.mark.gift_eval
def test_number_of_datasets(all_results_df: pd.DataFrame):
    assert len(DATASETS_WITH_TERMS) == len(all_results_df)


@pytest.mark.gift_eval
@pytest.mark.parametrize(
    "dataset_name, term",
    random.sample(DATASETS_WITH_TERMS, 20),
)
def test_evaluation(
    dataset_name: str,
    term: str,
    all_results_df: pd.DataFrame,
    storage_path: Path,
):
    predictor = GluonTSPredictor(
        forecaster=SeasonalNaive(alias="Seasonal_Naive"),
        batch_size=512,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        gifteval = GIFTEval(
            dataset_name=dataset_name,
            term=term,
            output_path=temp_dir,
            storage_path=storage_path,
        )
        gifteval.evaluate_predictor(
            predictor,
            batch_size=512,
        )
        eval_df = pd.read_csv(Path(temp_dir) / "all_results.csv")
        expected_eval_df = all_results_df.query("dataset == @gifteval.ds_config")
        assert not eval_df.isna().any().any()
        pd.testing.assert_frame_equal(
            eval_df.reset_index(drop=True)[TARGET_COLS],
            expected_eval_df.reset_index(drop=True)[TARGET_COLS],
            atol=1e-2,
            rtol=1e-2,
            check_dtype=False,
        )
