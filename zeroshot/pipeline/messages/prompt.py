from __future__ import annotations

import json
from collections.abc import Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block
from pydantic import BaseModel

from zeroshot.pipeline.messages.prompts import PromptTemplate


def instruction_text(name: str, **context: str) -> str:
    """Render one instruction prompt from `prompts/instructions/`."""
    return PromptTemplate(f"instructions/{name}").render(**context)


def build_instruction(name: str, **context: str) -> HumanMessage:
    return HumanMessage(
        content_blocks=[create_text_block(instruction_text(name, **context))]
    )


def system_prompt_text(
    role: str,
    context: Mapping[str, str],
    output_schema: type[BaseModel] | None = None,
) -> str:
    context = dict(context)

    if output_schema is not None:
        context["output_schema"] = json.dumps(
            output_schema.model_json_schema(),
            indent=2,
        )

    return PromptTemplate(f"roles/{role}").render(**context)


def build_system_prompt(
    role: str,
    context: Mapping[str, str],
    output_schema: type[BaseModel] | None = None,
) -> SystemMessage:
    return SystemMessage(
        content_blocks=[
            create_text_block(system_prompt_text(role, context, output_schema))
        ]
    )
