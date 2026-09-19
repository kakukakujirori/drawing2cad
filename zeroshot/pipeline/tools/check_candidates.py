"""Test competing readings of a drawing, each by its own read-only worker."""

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from zeroshot.pipeline.tools.errors import ToolFeedbackError


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="The candidate's sem_ name.")
    description: str = Field(
        ..., description="The finished shape this reading claims, with its sizes."
    )


class CandidateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: str = Field(
        ...,
        description=(
            "What each view must show if this reading is true: outlines, hidden "
            "lines and printed figures."
        ),
    )
    supported: str = Field(
        ...,
        description="Predictions the drawing confirms, each with its view and location.",
    )
    contradicted: str = Field(
        ...,
        description=(
            "Predictions the drawing contradicts or does not show, each with its "
            "view and location. Empty if none."
        ),
    )
    unresolved: str = Field(..., description="What the views cannot settle.")


@dataclass
class RoundBudget:
    """Model turns every worker of one round may take between them."""

    limit: int
    round: int = -1
    spent: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, round_number: int, turns: int) -> bool:
        with self._lock:
            if round_number != self.round:
                self.round, self.spent = round_number, 0
            if self.spent + turns > self.limit:
                return False
            self.spent += turns
            return True


def create_check_candidates_tool(
    worker: Any,
    worker_turns: int,
    load_image: BaseTool,
    current_round: Callable[[], int],
    budget: RoundBudget,
) -> BaseTool:
    @tool("check_candidates")
    def check_candidates(
        question: str,
        readings: list[Reading],
        images: list[str],
        config: RunnableConfig,
    ) -> str:
        """
        Test two or three competing readings of one ambiguous part of the
        drawing, such as a circle that may be a hole or a boss. Each reading
        goes to its own reviewer, who sees only that reading and the images,
        and reports what it predicts in every view and whether the drawing
        shows it. Use the reports to set the candidates' evidence, refuting
        regions and confidence yourself; the reviewers do not decide.
        Args:
            question: What the views leave open, in one sentence.
            readings: The competing candidates, two or three.
            images: Paths of the view images the reviewers should inspect.
        """
        if not 2 <= len(readings) <= 3:
            raise ToolFeedbackError("Give two or three readings.")
        if not images:
            raise ToolFeedbackError("Give the view images to inspect.")
        blocks = [block for path in images for block in load_image.invoke(path)]
        round_number = current_round()

        def review(reading: Reading) -> dict[str, Any]:
            report: dict[str, Any] = {"name": reading.name}
            if not budget.reserve(round_number, worker_turns):
                return report | {"status": "budget_exhausted"}
            task = (
                f"Question: {question}\nReading to test ({reading.name}): "
                f"{reading.description}\nImages: {', '.join(images)}"
            )
            try:
                # Callbacks only: the parent's graph keys would tie parallel
                # workers to its one tool task.
                result = worker.invoke(
                    {
                        "messages": [
                            HumanMessage([{"type": "text", "text": task}, *blocks])
                        ]
                    },
                    config={"callbacks": config.get("callbacks")},
                )
            except Exception as error:  # noqa: BLE001 -- a failed reviewer must not end the parent
                return report | {"status": "failed", "error": str(error)[:500]}
            check = result.get("structured_response")
            if check is None:
                return report | {"status": "failed", "error": "no report returned"}
            return report | {"status": "completed", **check.model_dump()}

        with ThreadPoolExecutor(max_workers=len(readings)) as pool:
            reports = list(pool.map(review, readings))
        return json.dumps(reports, ensure_ascii=False)

    return check_candidates
