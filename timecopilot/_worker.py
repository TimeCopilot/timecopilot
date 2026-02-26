"""Subprocess entry point for TUI agent execution.

Run as: python -m timecopilot._worker <request.pkl> <response.pkl>

The parent (TUI) spawns this with stdin/stdout/stderr=DEVNULL so that any
library output (logfire, tqdm, matplotlib, Rich) goes to /dev/null instead
of corrupting Textual's terminal.

Communication is via pickle files:
  - request.pkl  → dict with action + parameters
  - response.pkl ← dict with results or error
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


def _run_analyze(request: dict) -> dict:
    """Run TimeCopilot.analyze() and return serializable state."""
    from timecopilot.agent import TimeCopilot

    tc = TimeCopilot(llm=request["llm"])
    result = tc.analyze(df=request["file_path"], query=request["query"])

    # Extract the ForecastAgentOutput as a plain dict
    output_dict = result.output.model_dump()

    # Gather all dataframes
    dataset = tc.dataset
    return {
        "output_dict": output_dict,
        "df": dataset.df,
        "freq": dataset.freq,
        "h": dataset.h,
        "seasonality": dataset.seasonality,
        "fcst_df": getattr(tc, "fcst_df", None),
        "eval_df": getattr(tc, "eval_df", None),
        "features_df": getattr(tc, "features_df", None),
        "anomalies_df": getattr(tc, "anomalies_df", None),
        "eval_forecasters": getattr(tc, "eval_forecasters", []),
        "conversation_history": tc.conversation_history,
    }


def _run_query(request: dict) -> dict:
    """Run TimeCopilot.query() with restored state."""
    from timecopilot.agent import TimeCopilot
    from timecopilot.utils.experiment_handler import ExperimentDataset

    state = request["agent_state"]

    # Reconstruct TimeCopilot with a fresh agent
    tc = TimeCopilot(llm=request["llm"])

    # Restore dataset
    tc.dataset = ExperimentDataset(
        df=state["df"],
        freq=state["freq"],
        h=state["h"],
        seasonality=state["seasonality"],
    )

    # Restore dataframes
    tc.fcst_df = state["fcst_df"]
    tc.eval_df = state["eval_df"]
    tc.features_df = state["features_df"]
    tc.anomalies_df = state["anomalies_df"]
    tc.eval_forecasters = state["eval_forecasters"]
    tc.conversation_history = state.get("conversation_history", [])

    # Run the query (may trigger _maybe_rerun → re-analyze internally)
    result = tc.query(request["query"])

    # Return updated state (query may have triggered a re-analysis)
    dataset = tc.dataset
    return {
        "output": result.output,
        "df": dataset.df,
        "freq": dataset.freq,
        "h": dataset.h,
        "seasonality": dataset.seasonality,
        "fcst_df": getattr(tc, "fcst_df", None),
        "eval_df": getattr(tc, "eval_df", None),
        "features_df": getattr(tc, "features_df", None),
        "anomalies_df": getattr(tc, "anomalies_df", None),
        "eval_forecasters": getattr(tc, "eval_forecasters", []),
        "conversation_history": tc.conversation_history,
    }


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m timecopilot._worker <request.pkl> <response.pkl>")

    req_path = sys.argv[1]
    resp_path = sys.argv[2]

    try:
        with open(req_path, "rb") as f:
            request = pickle.load(f)

        action = request["action"]
        if action == "analyze":
            response = _run_analyze(request)
        elif action == "query":
            response = _run_query(request)
        else:
            response = {"error": f"Unknown action: {action}"}

    except Exception as e:
        response = {"error": f"{type(e).__name__}: {e}"}

    with open(resp_path, "wb") as f:
        pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
