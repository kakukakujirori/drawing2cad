from __future__ import annotations

import json
from collections.abc import Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block
from pydantic import BaseModel

from zeroshot.pipeline.messages.contracts import view_frame_sentence
from zeroshot.pipeline.messages.prompts import PromptTemplate


def instruction_text(name: str, **context: str) -> str:
    """Render one instruction prompt from `prompts/instructions/`.

    A stage whose directory holds a `guidelines.md` has it rendered first and
    offered to that stage's instructions as `$guidelines`: the contracts and
    standing advice that hold however the stage was entered.

    They ride in the instruction rather than the role so that an agent working
    one stage is told only what that stage needs -- when several stages share
    one system prompt, a role holding all of them would put the coding contract
    in front of the model while it is still reading the drawing.  And they ride
    in every instruction rather than the opening one so that a redo, which
    enters on its own, still carries them.

    `$view_frame` is offered to every prompt because the frame belongs to the
    contract, not to a stage: semantics reports sheet coordinates tagged with a
    view, and each stage after it has to read them back.
    """
    instruction = PromptTemplate(f"instructions/{name}")
    context = {**context, "view_frame": view_frame_sentence()}
    guidelines = instruction.path.parent / "guidelines.md"
    if guidelines.is_file():
        context = {
            **context,
            "guidelines": PromptTemplate(str(guidelines)).render(**context),
        }
    return instruction.render(**context)


def instruction_section(name: str, shown: bool, **context: str) -> str:
    """Render one part of an instruction, for a `$slot` in the rest of it.

    A part the run can leave out still says what it says in `prompts/`, beside
    the instruction it drops into, rather than as a string built at the call
    site: the caller decides whether the model is told, never what it is told.
    """
    return PromptTemplate(f"instructions/{name}").render(**context) if shown else ""


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

    sections = [PromptTemplate(f"roles/{role}").render(**context)]
    if "reconstruction_path" in context:
        sections.append(
            PromptTemplate("roles/reconstruction_history").render(**context)
        )
    return "\n\n".join(sections)


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
