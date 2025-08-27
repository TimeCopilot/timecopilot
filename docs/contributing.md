Your contributions are highly appreciated!

## Prerequisites 
Before proceeding, ensure the following tools and credentials are set up:

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Install [pre-commit](https://pre-commit.com/#install).

!!! tip "Tip"
    Once `uv` is installed, you can easily install `pre-commit` by running:
        ```
        uv tool install pre-commit
        ```

- Set up `pre-commit` hook:
    ```
    pre-commit install --install-hooks
    ```
- Generate an OpenAI API Key:
    1. Create an [openai](https://auth.openai.com/log-in) account.
    2. Visit the [API key](https://platform.openai.com/api-keys) page.
    3. Generate a new secret key.  
    You'll need this key in the setup section below. 

## Installation and Setup
To run timecopilot in your local environment:

1. Fork and clone the repository:
    ```
    git clone git@github.com:<your username>/timecopilot.git
    ```
2. Navigate into the project folder:
    ```
    cd timecopilot
    ```
3. Install the required dependencies for local development:
    ```
    uv sync --frozen --all-extras --all-packages --group docs
    ```
4. Export your OpenAI API key as an environment variable:
    ```
    export OPENAI_API_KEY="<your-new-secret-key>"
    ```
5. Test timecopilot with a sample forecast:
    ```
    uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv
    ```

✅ You're ready to start contributing! 

## Running Tests

To run tests, run:

```bash
uv run pytest
```

## Documentation Changes

To run the documentation page in your local environment, run:

```bash
uv run mkdocs serve
```


### Documentation Notes

- Each pull request is tested to ensure it can successfully build the documentation, preventing potential errors.
- Merging into the main branch triggers a deployment of a documentation preview, accessible at [preview.timecopilot.dev](https://preview.timecopilot.dev).
- When a new version of the library is released, the documentation is deployed to [timecopilot.dev](https://timecopilot.dev).

### File Naming Convention

All documentation files should use **kebab-case** (e.g., `model-hub.md`, `forecasting-parameters.md`). Kebab-case is widely used in static site generators and web contexts because it is URL-friendly, consistent, and avoids ambiguity with underscores or `camelCase`. Using a single convention improves readability and prevents broken links in deployment.

For further reference, see the [Google Developer Documentation Style Guide on file names](https://developers.google.com/style/filenames).

## Adding New Datasets

The datasets utilized in our documentation are hosted on AWS at `https://timecopilot.s3.amazonaws.com/public/data/`. If you wish to contribute additional datasets for your changes, please contact [@AzulGarza](http://github.com/AzulGarza) for guidance.

## Forked Dependencies

TimeCopilot uses some forked Python packages, maintained under custom names on PyPI:


- **chronos-forecasting**
    - Forked from: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
    - TimeCopilot fork: [AzulGarza/chronos-forecasting](https://github.com/AzulGarza/chronos-forecasting/tree/feat/timecopilot-chronos-forecasting)
    - Published on PyPI as: [`timecopilot-chronos-forecasting`](https://pypi.org/project/timecopilot-chronos-forecasting/)

- **timesfm**
    - Forked from: [google-research/timesfm](https://github.com/google-research/timesfm)
    - TimeCopilot fork: [AzulGarza/timesfm](https://github.com/AzulGarza/timesfm)
    - Published on PyPI as: [`timecopilot-timesfm`](https://pypi.org/project/timecopilot-timesfm/)

- **tirex**
    - Forked from: [NX-AI/tirex](https://github.com/NX-AI/tirex)
    - TimeCopilot fork: [AzulGarza/tirex](https://github.com/AzulGarza/tirex)
    - Published on PyPI as: [`timecopilot-tirex`](https://pypi.org/project/timecopilot-tirex/)

- **toto**
    - Forked from: [DataDog/toto](https://github.com/DataDog/toto)
    - TimeCopilot fork: [AzulGarza/toto](https://github.com/AzulGarza/toto)
    - Published on PyPI as: [`timecopilot-toto`](https://pypi.org/project/timecopilot-toto/)

- **uni2ts**:
    - Forked from: [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
    - TimeCopilot fork: [AzulGarza/uni2ts](https://github.com/AzulGarza/uni2ts)
    - Published on PyPI as: [`timecopilot-uni2ts`](https://pypi.org/project/timecopilot-uni2ts/)