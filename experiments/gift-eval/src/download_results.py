import logging
from pathlib import Path

import pandas as pd
from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS

logging.basicConfig(level=logging.INFO)


def download_results():
    bucket = "timecopilot-gift-eval"

    dfs = []

    for dataset_name, term in DATASETS_WITH_TERMS:
        csv_path = (
            f"s3://{bucket}/results/timecopilot/{dataset_name}/{term}/all_results.csv"
        )
        logging.info(f"Downloading {csv_path}")
        try:
            df = pd.read_csv(csv_path, storage_options={"anon": False})
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error downloading {csv_path}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    output_dir = Path("results/timecopilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_results.csv", index=False)
    logging.info(f"Saved results to {output_dir / 'all_results.csv'}")


if __name__ == "__main__":
    download_results()
