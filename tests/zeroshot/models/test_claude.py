"""ChatClaude under the staged pipeline's structured-output strategies, offline."""

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import tool

from zeroshot.pipeline.models import claude
from zeroshot.pipeline.models.claude import ChatClaude
from zeroshot.pipeline.models.image_history import INPUT_IMAGE_ID
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.workflow import create_agent

_ANSWER = {"responses": {"ticket_0": "built a box"}, "stage_report": {"concerns": {}}}
_ANSWER_CALL = {
    "type": "tool_use",
    "id": "tu_1",
    "name": "TicketAnswers",
    "input": _ANSWER,
}


@tool
def run_shell(command: str) -> str:
    """Run a shell command."""
    return command


def _model() -> ChatClaude:
    return ChatClaude(
        model="claude-opus-5-5",
        token="test-token",
        thinking={"type": "adaptive", "display": "summarized", "effort": "xhigh"},
    )


def _serve(monkeypatch, replies):
    """Answer each Messages API request with the next reply, keeping the request."""
    requests = []

    class Client:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            del headers
            requests.append(json)
            content = replies[len(requests) - 1]
            return httpx.Response(
                200,
                json={
                    "content": content,
                    "stop_reason": "tool_use"
                    if any(block["type"] == "tool_use" for block in content)
                    else "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(claude.httpx, "Client", Client)
    return requests


@pytest.mark.parametrize(
    ("tool_choice", "sent"),
    [
        ("any", {"type": "auto"}),
        ("run_shell", {"type": "auto"}),
        ({"type": "tool", "name": "run_shell"}, {"type": "auto"}),
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        (
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "auto", "disable_parallel_tool_use": True},
        ),
    ],
)
def test_forced_tool_choice_is_sent_as_auto(tool_choice, sent):
    payload = _model()._build_payload(
        [HumanMessage("go")], tools=[run_shell], tool_choice=tool_choice
    )
    assert payload["tool_choice"] == sent


def test_tool_strategy_answers_through_the_answer_tool(monkeypatch):
    requests = _serve(
        monkeypatch,
        [[_ANSWER_CALL]],
    )
    agent = create_agent(
        role="coder",
        model=_model(),
        tools=[run_shell],
        output_schema=TicketAnswers,
        response_format_strategy="tool",
        max_turns=3,
        model_retries=1,
    )
    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert result["structured_response"] == TicketAnswers.model_validate(_ANSWER)
    assert [t["name"] for t in requests[0]["tools"]] == ["run_shell", "TicketAnswers"]
    assert requests[0]["tool_choice"] == {"type": "auto"}


def test_a_prose_answer_is_asked_again_for_the_answer_tool(monkeypatch):
    requests = _serve(
        monkeypatch,
        [
            [{"type": "text", "text": "I built the box."}],
            [_ANSWER_CALL],
        ],
    )
    agent = create_agent(
        role="coder",
        model=_model(),
        tools=[run_shell],
        output_schema=TicketAnswers,
        response_format_strategy="tool",
        max_turns=3,
        model_retries=1,
    )
    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert result["structured_response"] == TicketAnswers.model_validate(_ANSWER)
    assert "call TicketAnswers" in str(requests[1]["messages"][-1]["content"])


def test_provider_strategy_is_refused_before_the_api(monkeypatch):
    """Why the Claude configs use `tool`: ChatClaude sends no response format, and
    TicketAnswers' free-form maps are outside what Anthropic structured outputs
    accept (every object must set additionalProperties to false)."""
    requests = _serve(monkeypatch, [])
    agent = create_agent(
        role="coder",
        model=_model(),
        tools=[run_shell],
        output_schema=TicketAnswers,
        response_format_strategy="provider",
        max_turns=3,
        model_retries=0,
    )
    with pytest.raises(ValueError, match="response_format_strategy: tool"):
        agent.invoke({"messages": [HumanMessage("go")]})
    assert requests == []


@pytest.mark.parametrize(
    "block",
    [
        # How the staged prompt attaches an input drawing.
        create_image_block(base64="ZHJhd2luZw==", mime_type="image/png", id="x"),
        # How the native prompt does.
        create_image_block(url="data:image/png;base64,ZHJhd2luZw==", id="x"),
    ],
)
def test_langchain_image_blocks_are_sent_in_anthropic_form(block):
    sent = _model()._build_payload([HumanMessage(content=[block])])["messages"]
    assert sent[0]["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "ZHJhd2luZw==",
            },
        }
    ]


def test_image_history_keeps_the_drawing_and_the_newest_tool_images():
    """load_image answers in a tool message, as every native inspection does."""
    drawing = "data:image/png;base64,ZHJhd2luZw=="
    messages = [
        HumanMessage(content=[create_image_block(url=drawing, id=INPUT_IMAGE_ID)])
    ]
    for i in range(9):
        messages += [
            AIMessage(
                content="",
                tool_calls=[{"name": "load_image", "args": {}, "id": f"c{i}"}],
            ),
            ToolMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{i}AAA"},
                    }
                ],
                tool_call_id=f"c{i}",
            ),
        ]
    model = ChatClaude(model="claude-opus-5-5", token="t", image_history_limit=8)
    sent = model._build_payload(messages)["messages"]

    images = [
        part
        for message in sent
        for block in message["content"]
        for part in (
            block["content"] if block.get("type") == "tool_result" else [block]
        )
        if part.get("type") == "image"
    ]
    assert images[0]["source"]["data"] == "ZHJhd2luZw=="
    assert [image["source"]["data"] for image in images[1:]] == [
        f"{i}AAA" for i in range(2, 9)
    ]
    # The transcript the agent keeps is not pruned.
    assert all(isinstance(m.content, list) for m in messages if m.type == "tool")
