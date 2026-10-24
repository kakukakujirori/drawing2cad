import base64
import json
import sys
from functools import partial
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.contracts import interpretation
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingView,
    ProjectionMethod,
    Region,
    View,
)
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline_rag.graph import create_annotation_graph, wording_errors


def _write(reading, call_id: str = "write") -> AIMessage:
    payload = base64.b64encode(reading.model_dump_json().encode()).decode()
    command = (
        "python -c \"import base64;open('/work/interpretation.json','wb')"
        f".write(base64.b64decode('{payload}'))\""
    )
    return tool_call("run_shell", {"command": command}, call_id)


def _reading(description: str):
    seed = interpretation(description)
    page = DrawingView(
        name="view_input",
        role="full_page",
        file="/work/drawing.png",
        region=Region(view="view_input", box_px=(0, 0, 20, 20)),
        dimensions=[],
    )
    front = seed.views[0].model_copy(
        update={
            "file": "/work/front.png",
            "region": Region(view="view_input", box_px=(0, 0, 20, 20)),
        }
    )
    return seed.model_copy(update={"views": [page, front]})


def _annotate(workdir, *responses: AIMessage):
    page = workdir.host_bind_dir / "drawing.png"
    Image.new("RGB", (20, 20), "white").save(page)
    Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
    graph = create_annotation_graph(
        agent_builder=partial(
            create_agent,
            role="annotator",
            model=ScriptedChatModel(responses=responses),
            announce_turns=False,
            model_call_retries=0,
            max_turns=5,
        ),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        artifact_presenter=ArtifactPresenter(input="path"),
        input_manifest=InputManifest(
            sample_id="test",
            drawing=[register_view("view_input", View.FULL_PAGE, page)],
            projection_method=ProjectionMethod.THIRD_ANGLE,
        ),
    )
    return graph.invoke({})


def _answer(concerns: dict[str, str]) -> AIMessage:
    return AIMessage(content=json.dumps({"concerns": concerns}))


def test_the_answer_waits_for_a_valid_annotation_and_nothing_is_built():
    with SandboxWorkdir() as workdir:
        result = _annotate(
            workdir,
            _answer({}),
            _write(_reading("a plate")),
            _answer({"concern_web": "x"}),
        )

        assert result["concerns"] == {"concern_web": "x"}
        refusal = result["transcript"][2]
        assert isinstance(refusal, HumanMessage)
        assert refusal.content[0]["text"].startswith("Answer refused")
        (report,) = result["transcript"][5].content  # no reminder to measure
        assert report["text"].startswith("[Interpretation verification]")
        saved = json.loads((workdir.host_bind_dir / "interpretation.json").read_text())
        assert saved["features"][0]["description"] == "a plate"
        assert saved["views"][1]["image_size"] == [20, 20]
        assert not (workdir.host_bind_dir / "model.py").exists()


def test_a_description_with_view_names_is_refused_until_reworded():
    with SandboxWorkdir() as workdir:
        result = _annotate(
            workdir,
            _write(_reading("a plate, solid in the front view"), "bad"),
            _answer({}),
            _write(_reading("a plate, solid seen across it"), "good"),
            _answer({}),
        )

        refusal = result["transcript"][5]
        assert "Answer refused" in refusal.content[0]["text"]
        assert "front view" in refusal.content[0]["text"]
        assert result["concerns"] == {}


def test_a_description_must_carry_over_to_another_part():
    def errors(description):
        return wording_errors(interpretation(description))

    assert errors("A blind 3D hole; seen along its axis, a solid circle.") == []
    assert errors("An X-shaped rib, hidden across its length.") == []
    (error,) = errors(
        "A R10.11 boss along +X, hidden in the top view and view_right; see sem_web."
    )
    for word in ("10.11", "+X", "top view", "view_right", "sem_web"):
        assert word in error
