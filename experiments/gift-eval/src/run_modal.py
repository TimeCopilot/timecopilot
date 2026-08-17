import modal

app = modal.App(name="timecopilot-gift-eval")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    # uploaded to s3 by makefile
    .apt_install("git")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file(".python-version", "/root/.python-version", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .workdir("/root")
    .run_commands("uv pip install . --system --compile-bytecode")
)
secret = modal.Secret.from_name(
    "aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
volume = {
    "/s3-bucket": modal.CloudBucketMount(
        bucket_name="timecopilot-gift-eval",
        secret=secret,
    )
}


@app.function(
    image=image,
    volumes=volume,
    # 6 hours timeout
    timeout=60 * 60 * 6,
    gpu="A10G",
    # as my local
    cpu=8,
)
def run_timecopilot_modal(dataset_name: str, term: str):
    import logging
    from pathlib import Path

    from .run_timecopilot import run_timecopilot

    output_path = Path(f"/s3-bucket/results/timecopilot/{dataset_name}/{term}/")
    if output_path.exists():
        logging.info(f"Output dir {output_path} already exists")
        return
    run_timecopilot(
        dataset_name=dataset_name,
        term=term,
        output_path=output_path,
        storage_path="/s3-bucket/data/gift-eval",
    )


@app.local_entrypoint()
def main():
    import logging

    import fsspec
    from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS

    logging.basicConfig(level=logging.INFO)

    fs = fsspec.filesystem("s3")
    missing_datasets_with_terms = [
        (dataset_name, term)
        for dataset_name, term in DATASETS_WITH_TERMS
        if not fs.exists(
            f"s3://timecopilot-gift-eval/results/timecopilot/{dataset_name}/{term}/all_results.csv"
        )
    ]
    logging.info(f"Running {len(missing_datasets_with_terms)} datasets")
    results = list(
        run_timecopilot_modal.starmap(
            missing_datasets_with_terms,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    logging.info(f"errors: {[r for r in results if r is not None]}")
