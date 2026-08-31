import json
from io import StringIO

from rich.console import Console

from zeroshot.pipeline.event_logging import ConsoleReporter


def test_a_failed_tool_reports_why_and_which_call_it_was() -> None:
    """Rendering one named key printed `None` for a whole run: the payload comes
    from LangGraph verbatim and carries the reason under `message`, not `error`.

    The id matters too — a turn can issue four `load_image` calls at once, and
    without it the console says one of them failed but not which.
    """
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False)
    )

    reporter.render_event(
        {
            "event": "tool_error",
            "timestamp_ms": 1,
            "namespace": [],
            "data": {
                "tool_call_id": "call-3",
                "message": "Not a readable image: /work/attempts/000/out.dxf",
                "tool_name": "load_image",
                "caller": "model",
                "future_field": "UNKNOWN_TOOL_ERROR_FIELD",
            },
        }
    )

    rendered = output.getvalue()
    assert "load_image failed" in rendered
    assert "/work/attempts/000/out.dxf" in rendered
    assert "call-3" in rendered
    # A key this module never heard of still reaches the operator.
    assert "UNKNOWN_TOOL_ERROR_FIELD" in rendered


def test_an_abandoned_attempt_costs_only_its_own_output() -> None:
    """A retried call leaves the attempt it gave up on unfinished forever.

    Nothing may wait on it: the console used to hold a cursor open on such a
    stream and print nothing further for the rest of the run, then fail asking
    it for a message it never produced.
    """
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False)
    )

    # The attempt that was dropped: it announced itself and said nothing more.
    reporter.render_model_item(
        {
            "role": "semantic_hypothesizer",
            "node": "model",
            "namespace": ["semantics:11111111", "propose_0:22222222"],
            "run_id": "abandoned",
            "streamed": True,
            "payload": {"event": "message-start", "role": "assistant"},
        }
    )
    for payload in (
        {"event": "message-start", "role": "assistant"},
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "text-delta", "text": "SURVIVED"},
        },
        {"event": "message-finish", "usage": None},
    ):
        reporter.render_model_item(
            {
                "role": "semantic_hypothesizer",
                "node": "model",
                "namespace": ["semantics:11111111", "propose_0:22222222"],
                "run_id": "retried",
                "streamed": True,
                "payload": payload,
            }
        )

    rendered = output.getvalue()
    assert "SURVIVED" in rendered
    # The attempt that produced nothing announced nothing either.
    assert rendered.count("[model]") == 1


def test_two_callers_a_role_cannot_tell_apart_are_named_by_where_they_run() -> None:
    """The fan-out's proposers share a role and stream at the same time, and a
    run whose stages share one agent gives every stage that role as well.  With
    the role alone the whole run reads as one speaker interrupting itself."""
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False)
    )

    for branch, run_id, text in (
        ("propose_0", "branch-0", "FROM_BRANCH_0"),
        ("propose_1", "branch-1", "FROM_BRANCH_1"),
    ):
        reporter.render_model_item(
            {
                "role": "cad_reconstructor",
                "node": "model",
                "namespace": ["semantics:1111", f"{branch}:2222"],
                "run_id": run_id,
                "streamed": False,
                "payload": {"type": "ai", "content": [{"type": "text", "text": text}]},
            }
        )

    rendered = output.getvalue()
    assert "cad_reconstructor · semantics/propose_0" in rendered
    assert "cad_reconstructor · semantics/propose_1" in rendered
    # The per-run ids in a namespace say nothing to someone watching one run.
    assert "1111" not in rendered
    assert "2222" not in rendered


def test_muted_graph_nodes_suppress_non_head_fanout_model_output() -> None:
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False),
        muted_graph_nodes=["propose_1"],
    )

    for branch, run_id, text in (
        ("propose_0", "branch-0", "HEAD_OUTPUT"),
        ("propose_1", "branch-1", "MUTED_OUTPUT"),
    ):
        reporter.render_model_item(
            {
                "role": "semantic_hypothesizer",
                "node": "model",
                "namespace": ["semantics:1111", f"{branch}:2222"],
                "run_id": run_id,
                "streamed": False,
                "payload": {"type": "ai", "content": [{"type": "text", "text": text}]},
            }
        )

    rendered = output.getvalue()
    assert "HEAD_OUTPUT" in rendered
    assert "MUTED_OUTPUT" not in rendered
    assert "propose_0" in rendered
    assert "propose_1" not in rendered


def test_muted_graph_nodes_suppress_nested_run_events() -> None:
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False),
        muted_graph_nodes=["propose_1"],
    )

    for branch in ("propose_0", "propose_1"):
        reporter.render_event(
            {
                "event": "node_started",
                "timestamp_ms": 1,
                "namespace": ["semantics:1111", f"{branch}:2222"],
                "data": {"node": "model", "triggers": []},
            }
        )

    assert output.getvalue().count("[node] model started") == 1


def _reported(*events: dict) -> str:
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False)
    )
    for data in events:
        reporter.render_event(
            {
                "event": data.pop("event"),
                "timestamp_ms": 1,
                "namespace": [],
                "data": data,
            }
        )
    return output.getvalue()


def test_the_lifecycle_events_a_stage_reports_are_rendered_not_dumped() -> None:
    """Every event name the pipeline emits needs a branch here. One without
    falls through to `[unhandled event]`, which buries the run's own failures
    in the raw payloads of the events around them."""
    rendered = _reported(
        {
            "event": "stage_validation",
            "node": "integrate_stage_submission",
            "error": "the reasoning stage did not return a StageSubmission",
            "failure_count": 2,
        },
        {"event": "stage_submission", "node": "semantics", "submission": {"edits": []}},
        {"event": "audit", "node": "audit", "report": {"accepted": False}},
    )

    assert "[unhandled" not in rendered
    assert "[invalid] integrate_stage_submission — failure 2" in rendered
    assert "did not return a StageSubmission" in rendered
    assert "[submission] semantics" in rendered
    assert "[audit] rejected" in rendered


def test_a_cleared_stage_channel_is_not_reported_as_an_answer() -> None:
    """Every node that touches the channel reports it, so most reports are the
    clearing rather than an answer."""
    rendered = _reported(
        {"event": "stage_submission", "node": "initialize", "submission": None},
        {
            "event": "stage_validation",
            "node": "initialize",
            "error": None,
            "failure_count": 0,
        },
        {"event": "audit", "node": "initialize", "report": None},
    )

    assert rendered == ""


def test_a_call_outside_any_subgraph_is_named_by_its_role_alone() -> None:
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False)
    )

    reporter.render_model_item(
        {
            "role": "output_auditor",
            "node": "model",
            "namespace": [],
            "run_id": "top",
            "streamed": False,
            "payload": {"type": "ai", "content": [{"type": "text", "text": "VERDICT"}]},
        }
    )

    rendered = output.getvalue()
    assert "output_auditor" in rendered
    assert "·" not in rendered


def test_a_model_that_does_not_stream_is_rendered_whole() -> None:
    """Only a streaming model reports itself in parts; the other kind arrives
    as one finished message, and both are this projection's to show."""
    output = StringIO()
    reporter = ConsoleReporter(
        Console(file=output, color_system=None, force_terminal=False, highlight=False)
    )

    reporter.render_model_item(
        {
            "role": "coder",
            "node": "model",
            "namespace": ["semantics:11111111", "propose_0:22222222"],
            "run_id": "whole",
            "streamed": False,
            "payload": {
                "type": "ai",
                "content": [{"type": "text", "text": "WHOLE_ANSWER"}],
                "tool_calls": [
                    {"name": "run_shell", "args": {"command": "WHOLE_COMMAND"}}
                ],
            },
        }
    )

    rendered = output.getvalue()
    assert "WHOLE_ANSWER" in rendered
    assert "tool call: run_shell" in rendered
    assert "WHOLE_COMMAND" in rendered


def test_console_reporter_renders_full_prompt_model_stream_and_tool_output() -> None:
    output = StringIO()
    reporter = ConsoleReporter(
        Console(
            file=output,
            color_system=None,
            force_terminal=False,
            highlight=False,
        )
    )
    prompt = "PROMPT_BEGIN\n" + "p" * 500 + "\nPROMPT_END"
    reasoning = "REASONING_BEGIN\n" + "r" * 500 + "\nREASONING_END"
    answer = "ANSWER_BEGIN\n" + "a" * 500 + "\nANSWER_END"
    command = "COMMAND_BEGIN\n" + "c" * 500 + "\nCOMMAND_END"
    tool_args = json.dumps({"command": command})
    stdout = "STDOUT_BEGIN\n" + "o" * 500 + "\nSTDOUT_END"
    secret = "DO_NOT_PRINT"

    model_events = (
        {"event": "message-start", "id": "message-1", "role": "assistant"},
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "reasoning-delta", "reasoning": reasoning},
        },
        {
            "event": "content-block-finish",
            "index": 0,
            "content": {"type": "reasoning", "reasoning": reasoning},
        },
        {
            "event": "content-block-delta",
            "index": 1,
            "delta": {"type": "text-delta", "text": answer},
        },
        {
            "event": "content-block-finish",
            "index": 1,
            "content": {"type": "text", "text": answer},
        },
        {
            "event": "content-block-delta",
            "index": 2,
            "delta": {
                "type": "block-delta",
                "fields": {
                    "type": "tool_call_chunk",
                    "id": "call-1",
                    "name": "run_shell",
                    "args": tool_args[: len(tool_args) // 2],
                    "index": 2,
                },
            },
        },
        {
            "event": "content-block-delta",
            "index": 2,
            "delta": {
                "type": "block-delta",
                "fields": {
                    "type": "tool_call_chunk",
                    "id": "call-1",
                    "name": "run_shell",
                    "args": tool_args,
                    "index": 2,
                },
            },
        },
        {
            "event": "content-block-finish",
            "index": 2,
            "content": {
                "type": "tool_call",
                "id": "call-1",
                "name": "run_shell",
                "args": {"command": command},
            },
        },
        {
            "event": "content-block-delta",
            "index": 3,
            "delta": {
                "type": "future-delta",
                "payload": "UNKNOWN_MODEL_DELTA",
                "api_key": secret,
            },
        },
        {
            "event": "future-message-event",
            "payload": "UNKNOWN_MODEL_EVENT",
        },
        {"event": "message-finish", "usage": None, "metadata": {}},
    )

    with reporter.run_context(run_id="sample-1:run-1", sample_id="sample-1"):
        reporter.render_event(
            {
                "event": "input",
                "timestamp_ms": 1,
                "namespace": [],
                "data": {
                    "messages": [
                        {"type": "system", "content": prompt},
                    ]
                },
            }
        )
        for model_event in model_events:
            reporter.render_model_item(
                {
                    "role": "agent",
                    "node": "model",
                    "namespace": ["semantics:11111111", "propose_0:22222222"],
                    "run_id": "run-1",
                    "streamed": True,
                    "payload": model_event,
                }
            )
        reporter.render_event(
            {
                "event": "message",
                "timestamp_ms": 2,
                "namespace": [],
                "data": {"messages": ["NORMALIZED_MESSAGE_DUPLICATE"]},
            }
        )
        reporter.render_event(
            {
                "event": "tool_finished",
                "timestamp_ms": 3,
                "namespace": [],
                "data": {
                    "tool_name": "run_shell",
                    "output": {
                        "content": json.dumps(
                            {
                                "status": "COMPLETED",
                                "returncode": 0,
                                "stdout": stdout,
                                "stderr": "",
                            }
                        )
                    },
                },
            }
        )
        reporter.render_event(
            {
                "event": "future_run_event",
                "timestamp_ms": 4,
                "namespace": [],
                "data": {
                    "payload": "UNKNOWN_RUN_EVENT",
                    "api_key": secret,
                },
            }
        )

    rendered = output.getvalue()
    for marker in (
        "PROMPT_BEGIN",
        "PROMPT_END",
        "REASONING_BEGIN",
        "REASONING_END",
        "ANSWER_BEGIN",
        "ANSWER_END",
        "COMMAND_BEGIN",
        "COMMAND_END",
        "STDOUT_BEGIN",
        "STDOUT_END",
    ):
        assert marker in rendered
    assert rendered.count("COMMAND_BEGIN") == 1
    assert rendered.count("COMMAND_END") == 1
    assert rendered.count("tool call: run_shell") == 1
    assert rendered.count("REASONING_BEGIN") == 1
    assert rendered.count("ANSWER_BEGIN") == 1
    assert "[unhandled model delta] future-delta" in rendered
    assert "UNKNOWN_MODEL_DELTA" in rendered
    assert "[unhandled model event] future-message-event" in rendered
    assert "UNKNOWN_MODEL_EVENT" in rendered
    assert "[unhandled event] future_run_event" in rendered
    assert "UNKNOWN_RUN_EVENT" in rendered
    assert "<redacted>" in rendered
    assert secret not in rendered
    assert "NORMALIZED_MESSAGE_DUPLICATE" not in rendered
    assert "run completed" in rendered
