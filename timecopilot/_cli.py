from pathlib import Path

import fire
import logfire
from dotenv import load_dotenv
from rich.console import Console

from timecopilot.agent import TimeCopilot as TimeCopilotAgent

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


class TimeCopilot:
    def __init__(self):
        self.console = Console()

    def forecast(
        self,
        path: str | Path,
        llm: str = "openai:gpt-4o-mini",
        freq: str | None = None,
        h: int | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        retries: int = 3,
        max_length: int | None = None,
    ):
        with self.console.status(
            "[bold blue]TimeCopilot is navigating through time...[/bold blue]"
        ):
            forecasting_agent = TimeCopilotAgent(
                llm=llm, 
                retries=retries, 
                max_length=max_length
            )
            result = forecasting_agent.forecast(
                df=path,
                freq=freq,
                h=h,
                seasonality=seasonality,
                query=query,
                max_length=max_length,
            )

        result.output.prettify(self.console)


def main():
    fire.Fire(TimeCopilot)


if __name__ == "__main__":
    main()
