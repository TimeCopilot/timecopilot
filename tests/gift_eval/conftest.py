from pathlib import Path

import pandas as pd
import pytest

from timecopilot_gift_eval import GIFTEval


@pytest.fixture(scope="session")
def cache_path() -> Path:
    cache_path = Path(".pytest_cache") / "gift_eval"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


@pytest.fixture(scope="session")
def all_results_df(cache_path: Path) -> pd.DataFrame:
    all_results_file = cache_path / "seasonal_naive_all_results.csv"
    if not all_results_file.exists():
        all_results_df = pd.read_csv(
            "https://huggingface.co/spaces/Salesforce/GIFT-Eval/raw/main/results/seasonal_naive/all_results.csv"
        )
        all_results_df.to_csv(all_results_file, index=False)
    return pd.read_csv(all_results_file)


@pytest.fixture(scope="session")
def storage_path(cache_path: Path) -> Path:
    GIFTEval.download_data(cache_path)
    return cache_path
