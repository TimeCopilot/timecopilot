import tempfile
from pathlib import Path

import pandas as pd

from timecopilot.models.stats import SeasonalNaive
from timecopilot_gift_eval import GIFTEval, GluonTSPredictor


def test_concat_results(storage_path: Path):
    predictor = GluonTSPredictor(
        forecaster=SeasonalNaive(),
        batch_size=512,
    )

    def _evaluate_predictor(
        dataset_name: str,
        term: str,
        output_path: Path | str,
        overwrite_results: bool = False,
    ):
        gifteval = GIFTEval(
            dataset_name=dataset_name,
            term=term,
            output_path=output_path,
            storage_path=storage_path,
        )
        gifteval.evaluate_predictor(
            predictor,
            batch_size=512,
            overwrite_results=overwrite_results,
        )

    combinations = [
        ("m4_weekly", "short"),
        ("m4_hourly", "short"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (dataset_name, term) in enumerate(combinations):
            _evaluate_predictor(
                dataset_name=dataset_name,
                term=term,
                output_path=temp_dir,
            )
            eval_df = pd.read_csv(Path(temp_dir) / "all_results.csv")
            assert len(eval_df) == i + 1

        _evaluate_predictor(
            dataset_name="m4_hourly",
            term="short",
            output_path=temp_dir,
            overwrite_results=True,
        )
        eval_df = pd.read_csv(Path(temp_dir) / "all_results.csv")
        assert len(eval_df) == 1
