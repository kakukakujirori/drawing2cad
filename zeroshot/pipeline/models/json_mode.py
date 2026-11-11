"""Present the output schema when the provider supports JSON mode only.

JSON mode does not enforce the schema. ProviderStrategy still validates the
response locally and the agent's existing retry middleware handles failures.
"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage


def add_output_schema(
    messages: list[BaseMessage], schema: dict[str, Any]
) -> list[BaseMessage]:
    """Explain final submission without modifying the saved conversation."""
    instruction = (
        "Use the available tools to complete the task. When submitting the final "
        "answer, return a JSON object matching this schema, in the response "
        "content (not a tool call). Return only raw JSON: no explanation, "
        "Markdown or code fences outside it. Do not submit before completing "
        "the required tool work. Schema: " + json.dumps(schema, ensure_ascii=False)
    )
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            content = message.content
            content = (
                content + "\n\n" + instruction
                if isinstance(content, str)
                else [*content, {"type": "text", "text": instruction}]
            )
            return [
                *messages[:index],
                message.model_copy(update={"content": content}),
                *messages[index + 1 :],
            ]
    return [SystemMessage(content=instruction), *messages]
