import json
from functools import partial
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from zeroshot.pipeline.event_logging import JsonlEventWriter
from zeroshot.pipeline.event_logging.projections import RunEventTransformer
from zeroshot.pipeline.tools import create_calculate_drawing_scale_tool


def _measurements(*pairs: tuple[float, float]) -> list[dict]:
    return [
        {"name": f"dim_{i}", "nominal": nominal, "measured": measured}
        for i, (nominal, measured) in enumerate(pairs)
    ]


def _calculate(*pairs: tuple[float, float]) -> dict:
    return create_calculate_drawing_scale_tool().invoke(
        {"measurements": _measurements(*pairs)}
    )


def test_schema_defines_named_measurements_without_a_submitted_dimension():
    tool = create_calculate_drawing_scale_tool()
    schema = tool.get_input_jsonschema()
    assert tool.name == "calculate_drawing_scale"
    assert set(schema["properties"]) == {"measurements"}
    measurement = schema["$defs"]["_Measurement"]
    assert set(measurement["required"]) == {"name", "nominal", "measured"}
    assert measurement["additionalProperties"] is False
    assert (
        "no Dimension has been submitted"
        in measurement["properties"]["name"]["description"]
    )


def test_scale_multiplies_pixels_into_mm_and_rejects_a_bad_reading():
    result = _calculate((10, 100), (25, 250), (40, 400), (30, 600))
    assert result["status"] == "ok"
    assert result["unit"] == "mm/px"
    assert 123 * result["scale"] == pytest.approx(12.3)
    assert result["inliers"] == ["dim_0", "dim_1", "dim_2"]
    assert result["outliers"] == ["dim_3"]
    assert result["measurements"][-1]["residual_px"] == pytest.approx(300)
    assert result["alternatives"] == []


def test_refit_tolerates_pixel_noise_and_reports_consistent_residuals():
    result = _calculate((10, 101), (20, 198), (30, 303), (40, 795))
    assert result["status"] == "ok"
    assert result["scale"] == pytest.approx(0.1, rel=0.01)
    assert result["outliers"] == ["dim_3"]
    for row in result["measurements"]:
        assert row["residual_px"] == pytest.approx(
            row["measured"] - row["nominal"] / result["scale"]
        )
        assert row["threshold_px"] == pytest.approx(2 + 0.02 * row["measured"])
        assert row["inlier"] == (abs(row["residual_px"]) <= row["threshold_px"])


def test_pixel_tolerance_accepts_small_lengths_with_large_relative_error():
    result = _calculate((1, 11), (20, 200), (30, 300))
    assert result["status"] == "ok"
    assert len(result["inliers"]) == 3
    assert result["scale"] == pytest.approx(0.1, rel=0.01)


def test_relative_tolerance_accepts_noise_on_long_lengths():
    result = _calculate((100, 1000), (200, 2030), (300, 2960))
    assert result["status"] == "ok"
    assert len(result["inliers"]) == 3
    assert result["scale"] == pytest.approx(0.1, rel=0.01)


def test_zero_intercept_does_not_explain_away_an_additive_measurement_error():
    result = _calculate((10, 200), (20, 300), (30, 400))
    assert result["status"] == "ambiguous"
    assert result["scale"] is None


@pytest.mark.parametrize(
    "pairs", [((10, 100), (20, 400)), ((10, 100), (20, 200), (30, 600), (40, 800))]
)
def test_conflicting_equal_support_is_not_resolved_arbitrarily(pairs):
    result = _calculate(*pairs)
    assert result["status"] == "ambiguous"
    assert result["scale"] is None
    assert sorted(c["scale"] for c in result["alternatives"]) == pytest.approx(
        [0.05, 0.1]
    )
    assert result["inliers"] == result["outliers"] == []
    assert all(row["residual_px"] is None for row in result["measurements"])


def test_unique_largest_group_still_needs_a_strict_majority():
    result = _calculate((10, 100), (20, 200), (30, 600), (40, 1200))
    assert result["status"] == "no_consensus"
    assert result["scale"] is None
    assert result["alternatives"][0]["inliers"] == ["dim_0", "dim_1"]


def test_one_dimension_returns_a_provisional_scale():
    result = _calculate((25, 100))
    assert result["status"] == "insufficient_evidence"
    assert result["scale"] == 0.25
    assert result["inliers"] == ["dim_0"]


def test_no_dimensions_cannot_establish_a_scale():
    result = _calculate()
    assert result["status"] == "insufficient_evidence"
    assert result["scale"] is None


def test_two_agreeing_dimensions_establish_a_scale():
    result = _calculate((25, 100), (50, 200))
    assert result["status"] == "ok"
    assert result["scale"] == pytest.approx(0.25)


def test_repeated_calls_and_input_order_do_not_change_the_fit():
    tool = create_calculate_drawing_scale_tool()
    measurements = _measurements((10, 101), (20, 198), (30, 303), (40, 795))
    first = tool.invoke({"measurements": measurements})
    assert tool.invoke({"measurements": measurements}) == first
    reversed_result = tool.invoke({"measurements": list(reversed(measurements))})
    assert reversed_result["scale"] == first["scale"]
    assert set(reversed_result["inliers"]) == set(first["inliers"])


def test_calibration_uses_the_resolution_of_the_supplied_measurements():
    original = _calculate((10, 100), (20, 200))
    resized = _calculate((10, 200), (20, 400))
    assert resized["scale"] == pytest.approx(original["scale"] / 2)


def test_duplicate_names_cannot_vote_twice():
    measurement = _measurements((10, 100))[0]
    result = create_calculate_drawing_scale_tool().invoke(
        {"measurements": [measurement, measurement]}
    )
    assert result["status"] == "invalid_input"
    assert result["scale"] is None


@pytest.mark.parametrize("field", ["nominal", "measured"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_lengths_must_be_positive_and_finite(field, value):
    measurement = _measurements((10, 100))[0] | {field: value}
    with pytest.raises(ValidationError):
        create_calculate_drawing_scale_tool().invoke({"measurements": [measurement]})


@pytest.mark.parametrize(
    "change", [{"name": ""}, {"name": "front width"}, {"extra": 1}]
)
def test_invalid_names_and_unknown_measurement_fields_are_rejected(change):
    with pytest.raises(ValidationError):
        create_calculate_drawing_scale_tool().invoke(
            {"measurements": [_measurements((10, 100))[0] | change]}
        )


def test_unrepresentable_scale_returns_json_safe_feedback():
    result = _calculate((1e308, 1e-308))
    assert result["status"] == "invalid_input"
    assert result["scale"] is None
    json.dumps(result, allow_nan=False)


def test_tool_feedback_is_delivered_and_saved_by_existing_event_logging(tmp_path: Path):
    tool = create_calculate_drawing_scale_tool()
    graph = StateGraph(MessagesState)  # pyrefly: ignore[bad-specialization]
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    args = {"measurements": _measurements((10, 100), (20, 200), (30, 600))}
    call = {"name": tool.name, "args": args, "id": "call-scale", "type": "tool_call"}
    path = tmp_path / "events.jsonl"
    with JsonlEventWriter(path, run_id="test", sample_id="sample") as writer:
        stream = graph.compile().stream_events(
            {"messages": [AIMessage(content="", tool_calls=[call])]},
            version="v3",
            transformers=[partial(RunEventTransformer, sink=writer.write)],
        )
        output = stream.output

    assert output is not None
    message = output["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-scale"
    assert isinstance(message.content, str)
    feedback = json.loads(message.content)
    assert feedback["status"] == "ok"
    assert feedback["scale"] == pytest.approx(0.1)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    finished = [record for record in records if record["event"] == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["data"]["tool_call_id"] == "call-scale"
    assert json.loads(finished[0]["data"]["output"]["content"]) == feedback
    started = [record for record in records if record["event"] == "tool_started"]
    assert started[0]["data"]["input"] == args
