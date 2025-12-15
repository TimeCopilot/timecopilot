# First-Place Results on the GIFT-Eval Benchmark

This section documents the evaluation of a foundation model ensemble built using the [TimeCopilot](https://timecopilot.dev) library on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmark.

!!! success ""
    With less than $30 in compute cost, TimeCopilot achieved first place in probabilistic accuracy (CRPS) among open-source solution on this large-scale benchmark, which spans 24 datasets, 144k+ time series, and 177M data points.


TimeCopilot is an open‑source AI agent for time series forecasting that provides a unified interface to multiple forecasting approaches, from foundation models to classical statistical, machine learning, and deep learning methods, along with built‑in ensemble capabilities for robust and explainable forecasting.

<img width="1002" height="1029" alt="image" src="https://github.com/user-attachments/assets/69724886-d37e-46e6-8a10-d82396695b49" />





## Description

This ensemble leverages [**TimeCopilot's MedianEnsemble**](https://timecopilot.dev/api/models/ensembles/#timecopilot.models.ensembles.median.MedianEnsemble) feature, which combines three state-of-the-art foundation models:

- [**Chronos-2** (AWS)](https://timecopilot.dev/api/models/foundation/models/#timecopilot.models.foundation.chronos.Chronos).
- [**TimesFM-2.5** (Google Research)](https://timecopilot.dev/api/models/foundation/models/#timecopilot.models.foundation.timesfm.TimesFM).
- [**TiRex** (NXAI)](https://timecopilot.dev/api/models/foundation/models/#timecopilot.models.foundation.tirex.TiRex).

## Setup

### Prerequisites
- Clone [TimeCopilot's repo](https://github.com/TimeCopilot/timecopilot) and go to `experiments/gift-eval`.
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS CLI configured (for distributed evaluation)
- [Modal](https://modal.com/) account (for distributed evaluation)

### Installation

```bash
# Install dependencies
uv sync
```

## Dataset Management

### Download GIFT-Eval Dataset

```bash
# Download the complete GIFT-Eval dataset
make download-gift-eval-data
```

This downloads all 97 dataset configurations to `./data/gift-eval/`.

### Upload to S3 (Optional)

For distributed evaluation, upload the dataset to S3:

```bash
# Upload dataset to S3 for distributed access
make upload-data-to-s3
```

## Evaluation Methods

### 1. Local Evaluation

Run evaluation on a single dataset locally:

```bash
uv run -m src.run_timecopilot \
  --dataset-name "m4_weekly" \
  --term "short" \
  --output-path "./results/timecopilot/" \
  --storage-path "./data/gift-eval"
```

**Parameters:**

- `--dataset-name`: GIFT-Eval dataset name (e.g., "m4_weekly", "bizitobs_l2c/H")
- `--term`: Forecasting horizon ("short", "medium", "long")
- `--output-path`: Directory to save evaluation results
- `--storage-path`: Path to GIFT-Eval dataset

### 2. Distributed Evaluation (Recommended)

Evaluate all 97 dataset configurations in parallel using [modal](https://modal.com/):

```bash
# Run distributed evaluation on Modal cloud
uv run modal run --detach -m src.run_modal::main
```

This creates one GPU job per dataset configuration, significantly reducing evaluation time.

**Infrastructure:**

- **GPU**: A10G per job
- **CPU**: 8 cores per job  
- **Timeout**: 3 hours per job
- **Storage**: S3 bucket for data and results

### 3. Collect Results

Download and consolidate results from distributed evaluation:

```bash
# Download all results from S3 and create consolidated CSV
uv run python -m src.download_results
```

Results are saved to `results/timecopilot/all_results.csv` in GIFT-Eval format.


## Changelog

### **2025-11-06**

We introduced newer models based on the most recent progress in the field: Chronos-2, TimesFM-2.5 and TiRex.

### **2025-08-05**

GIFT‑Eval recently [enhanced its evaluation dashboard](https://github.com/SalesforceAIResearch/gift-eval?tab=readme-ov-file#2025-08-05) with a new flag that identifies models likely affected by data leakage (i.e., having seen parts of the test set during training). While the test set itself hasn’t changed, this new insight helps us better interpret model performance. To keep our results focused on truly unseen data, we’ve excluded any flagged models from this experiment and added the Sundial model to the ensemble. The previous experiment details remain available [here](https://github.com/TimeCopilot/timecopilot/tree/v0.0.14/experiments/gift-eval).
