import logging
from typing import Annotated

import typer

from timecopilot_gift_eval import GIFTEval, GluonTSPredictor
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.tirex import TiRex

logging.basicConfig(level=logging.INFO)


app = typer.Typer()


@app.command()
def run_timecopilot(
    dataset_name: Annotated[
        str,
        typer.Option(help="The name of the dataset to evaluate"),
    ],
    term: Annotated[
        str,
        typer.Option(help="The term to evaluate"),
    ],
    output_path: Annotated[
        str,
        typer.Option(help="The directory to save the results"),
    ],
    storage_path: Annotated[
        str,
        typer.Option(help="The directory were the GIFT data is stored"),
    ],
):
    logging.info(f"Running {dataset_name} {term} {output_path}")
    batch_size = 64
    predictor = GluonTSPredictor(
        forecaster=MedianEnsemble(
            models=[
                Chronos(
                    repo_id="amazon/chronos-2",
                    batch_size=batch_size,
                ),
                TimesFM(
                    repo_id="google/timesfm-2.5-200m-pytorch",
                    batch_size=batch_size,
                ),
                TiRex(
                    batch_size=batch_size,
                ),
            ],
            alias="TimeCopilot",
        ),
        max_length=4_096,
        # data batch size
        batch_size=1_024,
    )
    gifteval = GIFTEval(
        dataset_name=dataset_name,
        term=term,
        output_path=output_path,
        storage_path=storage_path,
    )
    gifteval.evaluate_predictor(predictor, batch_size=512)


if __name__ == "__main__":
    app()
