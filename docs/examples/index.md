# Examples

Interactive notebooks for common TimeCopilot workflows. Pick a section below based on your goal, or browse the sidebar.

For model API details, see the [Model Hub](../model-hub.md).

## Core Workflows

| Notebook | What you'll learn | Prerequisites |
|----------|-------------------|---------------|
| [Forecaster Quickstart](forecaster-quickstart.ipynb) | Forecast and cross-validate with `TimeCopilotForecaster` and classical models | Python 3.10+ |
| [Anomaly Detection](anomaly-detection-forecaster-quickstart.ipynb) | Detect anomalies with the forecaster API | Python 3.10+ |
| [Cryptocurrency Walkthrough](cryptocurrency-quickstart.ipynb) | Handle missing data and multi-series forecasting on real-world data | Python 3.10+ |

## Agent and LLMs

| Notebook | What you'll learn | Prerequisites |
|----------|-------------------|---------------|
| [Agent Quickstart](agent-quickstart.ipynb) | Conversational forecasting with the TimeCopilot agent | OpenAI API key |
| [LLM Providers](llm-providers.ipynb) | Configure OpenAI, Ollama, and other LLM backends | Provider API key or local runtime |
| [AWS Bedrock](aws-bedrock.ipynb) | Run the agent with AWS-hosted models | AWS account with Bedrock access |
| [Google LLMs](google-llms.ipynb) | Run the agent with Google AI Studio / Gemini | Google AI Studio API key |
| [Explaining Models and Ensembles](explaining-foundation-models-and-ensembles.ipynb) | Ask the agent to explain and compare custom model setups | OpenAI API key |

## Foundation Models

| Notebook | What you'll learn | Prerequisites |
|----------|-------------------|---------------|
| [Compare Foundation Models](ts-foundation-models-comparison-quickstart.ipynb) | Benchmark multiple foundation models side by side | Python 3.10+; GPU optional |
| [Chronos Family](chronos-family.ipynb) | Forecast with Chronos 1.x and 2.x checkpoints | Python 3.10+ |
| [TiRex Family](tirex-family.ipynb) | Forecast with TiRex 1.0 and 2.0 | Python 3.11+ |
| [Toto Family](toto-family.ipynb) | Forecast with Toto 1.0 and 2.0 | Python 3.10+ |
| [Finetuning](finetuning.ipynb) | Adapt Chronos 2 and TimeGPT to your data | Python 3.10+; GPU recommended |

## Benchmarks and Ensembles

| Notebook | What you'll learn | Prerequisites |
|----------|-------------------|---------------|
| [GIFT-Eval](gift-eval.ipynb) | Evaluate a foundation model ensemble on GIFT-Eval | Python 3.10+; GPU recommended |
| [Custom Ensembles](custom-ensembles.ipynb) | Build weighted and custom ensembles beyond `MedianEnsemble` | Python 3.10+ |

## Integrations

| Notebook | What you'll learn | Prerequisites |
|----------|-------------------|---------------|
| [sktime](sktime.ipynb) | Use sktime models through TimeCopilot | Python 3.10+; `sktime` extra |
