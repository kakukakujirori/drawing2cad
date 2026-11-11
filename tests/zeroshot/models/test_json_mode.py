import asyncio
import copy
import json

import httpx
import pytest
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.tools import tool

from zeroshot.pipeline.models.openrouter import ChatOpenRouter
from zeroshot.pipeline.stages.audit.contracts import AuditSubmission
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.workflow import create_agent

ANSWERS = {
    "responses": [
        {"ticket_id": "ticket_initial", "summary": 'view_front: "A"\n日本語 \\ <item>'},
        {"ticket_id": "ticket_hole", "summary": "Corrected sem_hole."},
    ],
    "stage_report": {
        "concerns": [
            {"id": "concern_depth", "explanation": "dim_depth is provisional."},
            {"id": "concern_radius", "explanation": "sem_fillet needs review."},
        ]
    },
}


@tool
def echo(value: str) -> str:
    """Echo a value."""
    return value


def _call(name, arguments, id="call_1"):
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _model(responses, *, streaming=True, enabled=True):
    """Exercise SDK serialization and response assembly without network access."""
    sent = []

    def respond(request):
        body = json.loads(request.content)
        reply = responses[len(sent)]
        sent.append(body)
        assert "use_json_object_for_structured_output" not in body
        envelope = {
            "id": f"response_{len(sent)}",
            "created": 0,
            "model": "test",
            "provider": "GMICloud",
            "system_fingerprint": None,
        }
        usage = {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cost": 0.001,
        }
        finish = "tool_calls" if reply.get("tool_calls") else "stop"
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    **envelope,
                    "object": "chat.completion",
                    "usage": usage,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish,
                            "message": {"role": "assistant", **reply},
                        }
                    ],
                },
            )
        chunks = []

        def emit(delta, reason=None):
            chunks.append(
                {
                    **envelope,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                }
            )

        emit({"role": "assistant"})
        content = reply.get("content", "")
        for offset in range(0, len(content), 13):
            emit({"content": content[offset : offset + 13]})
        for index, call in enumerate(reply.get("tool_calls", [])):
            emit({"tool_calls": [{"index": index, **call}]})
        emit({}, finish)
        chunks.append(
            {
                **envelope,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            }
        )
        return httpx.Response(
            200,
            text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
            + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    model = ChatOpenRouter(
        model="test",
        api_key="EMPTY",
        streaming=streaming,
        output_version="v1",
        use_json_object_for_structured_output=enabled,
        max_retries=0,
    )
    model.client.sdk_configuration.client = httpx.Client(
        transport=httpx.MockTransport(respond)
    )
    model.client.sdk_configuration.async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    return model, sent


@pytest.mark.parametrize("api", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.parametrize("streaming", [False, True])
def test_sdk_paths_keep_tools_native_and_return_json_content(api, streaming):
    model, sent = _model([{"content": json.dumps(ANSWERS)}], streaming=streaming)
    format_kwargs = ProviderStrategy(TicketAnswers).to_model_kwargs()
    before = copy.deepcopy(format_kwargs)
    bound = model.bind_tools([echo], **format_kwargs)
    messages = [
        SystemMessage(content="Follow the task."),
        HumanMessage(content="Submit."),
    ]
    if api == "invoke":
        answer = bound.invoke(messages)
    elif api == "ainvoke":
        answer = asyncio.run(bound.ainvoke(messages))
    elif api == "stream":
        chunks = list(bound.stream(messages))
        answer = chunks[0]
        for chunk in chunks[1:]:
            answer += chunk
    else:

        async def collect():
            return [chunk async for chunk in bound.astream(messages)]

        chunks = asyncio.run(collect())
        answer = chunks[0]
        for chunk in chunks[1:]:
            answer += chunk

    assert json.loads(answer.text) == ANSWERS
    assert not answer.tool_calls
    assert answer.usage_metadata["output_tokens"] == 7
    assert answer.response_metadata["cost"] == 0.001
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["tools"][0]["function"]["parameters"]["properties"] == {
        "value": {"type": "string"}
    }
    assert '"stage_report"' in sent[0]["messages"][0]["content"]
    assert format_kwargs == before
    assert messages[0].content == "Follow the task."


@pytest.mark.parametrize(
    "content",
    [None, "Follow the task.", [{"type": "text", "text": "Follow the task."}]],
)
def test_schema_is_bound_per_request_without_mutating_or_accumulating_history(content):
    model, sent = _model([{"content": "{}"}] * 3)
    messages = ([] if content is None else [SystemMessage(content=content)]) + [
        HumanMessage(content="Submit.")
    ]
    before = [m.model_dump() for m in messages]
    coding = model.bind_tools([], **ProviderStrategy(TicketAnswers).to_model_kwargs())
    audit = model.bind_tools([], **ProviderStrategy(AuditSubmission).to_model_kwargs())
    coding.invoke(messages)
    audit.invoke(ChatPromptValue(messages=messages))
    coding.invoke(messages)
    assert [m.model_dump() for m in messages] == before
    assert sent[0]["messages"] == sent[2]["messages"]
    for request in sent:
        assert request["messages"][0]["role"] == "system"
        assert json.dumps(request["messages"]).count("Schema: ") == 1
    assert "TicketAnswers" not in json.dumps(sent[1]["messages"])
    assert "AuditSubmission" in json.dumps(sent[1]["messages"])


@pytest.mark.parametrize("prompt", ["Submit.", StringPromptValue(text="Submit.")])
def test_text_inputs_receive_the_output_schema(prompt):
    model, sent = _model([{"content": '{"accepted": true}'}])
    model.bind_tools([], **ProviderStrategy(AuditSubmission).to_model_kwargs()).invoke(
        prompt
    )
    assert sent[0]["messages"][-1] == {"role": "user", "content": "Submit."}
    assert "AuditSubmission" in sent[0]["messages"][0]["content"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
def test_calls_without_a_json_schema_are_unchanged(enabled, response_format):
    model, sent = _model(
        [{"content": "", "tool_calls": [_call("echo", {"value": "hello"})]}],
        enabled=enabled,
    )
    kwargs = {} if response_format is None else {"response_format": response_format}
    answer = model.bind_tools([echo], **kwargs).invoke("Work.")
    assert answer.tool_calls[0]["args"] == {"value": "hello"}
    assert sent[0].get("response_format") == response_format
    assert sent[0]["messages"] == [{"role": "user", "content": "Work."}]


def test_disabled_mode_preserves_provider_schema_enforcement():
    model, sent = _model([{"content": json.dumps(ANSWERS)}], enabled=False)
    kwargs = ProviderStrategy(TicketAnswers, strict=True).to_model_kwargs()
    model.bind_tools([echo], **kwargs).invoke("Submit.")
    assert sent[0]["response_format"] == kwargs["response_format"]
    assert sent[0]["messages"] == [{"role": "user", "content": "Submit."}]


@pytest.mark.parametrize("api", ["invoke", "ainvoke", "stream_events"])
@pytest.mark.parametrize(
    "schema,answer", [(TicketAnswers, ANSWERS), (AuditSubmission, {"accepted": False})]
)
def test_agent_executes_tools_then_retries_forbidden_calls_json_and_schema_errors(
    api, schema, answer
):
    executed = []

    @tool
    def record(value: str) -> str:
        """Record a value."""
        executed.append(value)
        return value

    model, sent = _model(
        [
            {"tool_calls": [_call("record", {"value": "hello"})]},
            {
                "tool_calls": [
                    _call("record", {"value": "must not execute"}, "forbidden")
                ]
            },
            {"content": "```json\n{}\n```"},
            {"content": "{}"},
            {"content": json.dumps(answer)},
        ]
    )
    agent = create_agent(
        "test",
        model,
        [record],
        output_schema=schema,
        response_format_strategy="provider",
        max_turns=2,
        model_call_retries=3,
    )
    inputs = {"messages": [HumanMessage(content="Work then submit.")]}
    if api == "invoke":
        result = agent.invoke(inputs)
    elif api == "ainvoke":
        result = asyncio.run(agent.ainvoke(inputs))
    else:
        result = agent.stream_events(inputs, version="v3").output
    assert result["structured_response"].model_dump() == answer
    assert executed == ["hello"]
    assert result["current_turn"] == 2
    assert len(sent) == 5
    assert len(sent[0]["tools"]) == 1
    for request in sent[1:]:
        assert not request.get("tools")
        assert request["response_format"] == {"type": "json_object"}
        assert any(
            m.get("role") == "tool" and m.get("content") == "hello"
            for m in request["messages"]
        )
    assert "not available" in json.dumps(sent[2]["messages"])
    assert "valid JSON" in json.dumps(sent[3]["messages"])
    assert "required" in json.dumps(sent[4]["messages"])


def test_invalid_submissions_exhaust_the_existing_retry_budget():
    model, sent = _model([{"content": "{}"}] * 3)
    agent = create_agent(
        "test",
        model,
        [],
        output_schema=TicketAnswers,
        response_format_strategy="provider",
        max_turns=1,
        model_call_retries=2,
    )
    result = agent.invoke({"messages": [HumanMessage(content="Submit.")]})
    assert len(sent) == 3
    assert result.get("structured_response") is None
    assert "After 3 attempts" in result["messages"][-1].text
