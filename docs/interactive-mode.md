# Interactive Mode

TimeCopilot ships with a fully interactive terminal experience that lets you select models, configure API keys, load data, and ask follow-up questions — all from a single session. This page walks through every step of the interactive flow, from first launch to ongoing conversations with your data.

---

## Launching interactive mode

Start an interactive session by running the CLI with no subcommand:

```bash
timecopilot
```

You can also pass a specific LLM at launch to skip the model picker entirely:

```bash
timecopilot --llm anthropic:claude-sonnet-4-5-20250929
```

---

## First-time onboarding

When no saved configuration is found, TimeCopilot walks you through a short setup flow.

### 1. LLM selection prompt

The first thing you see is a yes/no question:

```
Would you like to select an LLM? (yes/no)
```

- **yes** (or press Enter) — opens the two-step model picker described below.
- **no** — keeps the default model (`openai:gpt-4o-mini`) and moves straight to the API key step.

### 2. Provider picker

If you chose yes, a table of available providers is displayed:

```
         Select a provider
┏━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ #  ┃ Provider      ┃ Models    ┃
┡━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 1  │ openai        │ 72 models │
│ 2  │ anthropic     │ 19 models │
│ 3  │ google-gla    │ 9 models  │
│ …  │ …             │ …         │
└────┴───────────────┴───────────┘
Select a provider:
```

Enter a number, a provider name, or type `back` to return to the previous step.

### 3. Model picker

After selecting a provider you see its models listed in a similar table. Pick one by number or name, or type `back` to return to the provider list.

### 4. API key

TimeCopilot checks whether the required API key environment variable is set for the selected provider (e.g. `OPENAI_API_KEY` for OpenAI, `ANTHROPIC_API_KEY` for Anthropic). If it is missing you are prompted to enter it:

```
Enter your OPENAI_API_KEY:
```

The key is saved to `timecopilot/.env` so you only need to enter it once. Type `back` to return to model selection instead.

---

## Returning users

When a valid saved configuration already exists, TimeCopilot greets you with a short welcome-back message showing the current model and a command reference table. No onboarding steps are repeated.

---

## The chat loop

Once setup is complete, you land in the main prompt:

```
TimeCopilot:
```

From here you can load data, ask questions, and issue commands.

### Loading data

Provide a file path or URL to kick off an analysis:

```
TimeCopilot: forecast /path/to/sales.csv
```

TimeCopilot accepts `.csv` and `.parquet` files as well as HTTP/S3 URLs. After loading, it selects the best forecasting model, generates predictions, and reports the results — including any anomalies it detects.

### Follow-up questions

After the initial analysis you can ask natural-language follow-up questions in the same session:

```
TimeCopilot: Why does the forecast spike in December?
TimeCopilot: Show me a plot of the forecast vs actual values
TimeCopilot: Compare Chronos and TimesFM for this dataset
```

---

## Commands

The following commands are available at the `TimeCopilot:` prompt:

| Command | Description |
|---------|-------------|
| `help` or `?` | Show the welcome message with examples and command reference |
| `llm` | Switch to a different model (opens the provider → model picker) |
| `back` | Go back to the previous step during onboarding or model selection |
| `exit`, `quit`, or `bye` | Exit TimeCopilot |

!!! tip "Switching models mid-session"

    Type `llm` at any point to open the model picker and switch providers or models without restarting TimeCopilot. Your new selection is saved for future sessions.

---

## Configuration persistence

All configuration is stored in the `timecopilot/.env` file inside your project directory:

- **Selected model** — saved as `TIMECOPILOT_MODEL`
- **API keys** — saved under their standard environment variable names (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

Delete this file to reset your configuration and trigger the onboarding flow again.
