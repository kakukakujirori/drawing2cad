import json
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.check_candidates import (
    CandidateCheck,
    RoundBudget,
    create_check_candidates_tool,
)
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.load_image import create_load_image_tool

_CHECK = CandidateCheck(
    predictions="a circle in front", supported="front", contradicted="", unresolved=""
)


class Reviewer:
    """Answers per reading; records what each reviewer was shown."""

    def __init__(self, fails: str | None = None) -> None:
        self.tasks: list[str] = []
        self.fails = fails
        self._lock = threading.Lock()

    def invoke(self, state: dict[str, Any], config: Any) -> dict[str, Any]:
        message: HumanMessage = state["messages"][0]
        task = message.content[0]["text"]
        with self._lock:
            self.tasks.append(task)
        if self.fails and self.fails in task:
            raise RuntimeError("provider stalled")
        return {"structured_response": _CHECK}


def _tool(tmp_path: Path, reviewer: Reviewer, budget: RoundBudget, round_=0):
    Image.new("RGB", (3, 2)).save(tmp_path / "front.png")
    return create_check_candidates_tool(
        worker=reviewer,
        worker_turns=3,
        load_image=create_load_image_tool(SandboxWorkdir(host_bind_dir=tmp_path)),
        current_round=lambda: round_,
        budget=budget,
    )


def _call(tool, *names: str) -> list[dict[str, Any]]:
    args = {
        "question": "hole or boss?",
        "readings": [{"name": n, "description": f"{n} reading"} for n in names],
        "images": ["front.png"],
    }
    return json.loads(
        tool.invoke(
            {"args": args, "type": "tool_call", "id": "1", "name": "check_candidates"}
        ).content
    )


def test_each_reviewer_sees_its_own_reading_only(tmp_path: Path) -> None:
    reviewer = Reviewer()
    reports = _call(_tool(tmp_path, reviewer, RoundBudget(12)), "sem_hole", "sem_boss")

    assert [r["name"] for r in reports] == ["sem_hole", "sem_boss"]
    assert all(r["status"] == "completed" for r in reports)
    assert sorted("sem_boss" in task for task in reviewer.tasks) == [False, True]
    assert not any("sem_hole" in t and "sem_boss" in t for t in reviewer.tasks)


def test_a_failed_reviewer_leaves_the_others_their_report(tmp_path: Path) -> None:
    tool = _tool(tmp_path, Reviewer(fails="sem_boss"), RoundBudget(12))
    reports = {r["name"]: r for r in _call(tool, "sem_hole", "sem_boss")}

    assert reports["sem_hole"]["status"] == "completed"
    assert reports["sem_boss"]["status"] == "failed"
    assert "provider stalled" in reports["sem_boss"]["error"]


def test_the_budget_spans_the_round_and_renews_with_the_next(tmp_path: Path) -> None:
    budget = RoundBudget(9)
    first = _call(_tool(tmp_path, Reviewer(), budget), "sem_a", "sem_b")
    second = _call(_tool(tmp_path, Reviewer(), budget), "sem_c", "sem_d")
    renewed = _call(_tool(tmp_path, Reviewer(), budget, round_=1), "sem_e", "sem_f")

    assert [r["status"] for r in first] == ["completed", "completed"]
    assert sorted(r["status"] for r in second) == ["budget_exhausted", "completed"]
    assert [r["status"] for r in renewed] == ["completed", "completed"]


@pytest.mark.parametrize("names", [("sem_a",), ("sem_a", "sem_b", "sem_c", "sem_d")])
def test_it_compares_two_or_three_readings(tmp_path: Path, names) -> None:
    tool = _tool(tmp_path, Reviewer(), RoundBudget(12))
    with pytest.raises(ToolFeedbackError, match="two or three"):
        tool.invoke(
            {
                "question": "?",
                "readings": [{"name": n, "description": n} for n in names],
                "images": ["front.png"],
            }
        )
