"""The base of the models a stage answers with, and what it forgives."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic_core import PydanticCustomError


def _decoded(value: object) -> object:
    """A JSON object or array the caller delivered as a string, or the value."""
    if not isinstance(value, str) or value.lstrip()[:1] not in "{[":
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded if isinstance(decoded, dict | list) else value


class Submission(BaseModel):
    """A stage's structured answer, parsed from what the provider delivered.

    A tool call arrives with a nested member stringified often enough to
    matter: the answer is whole and only its encoding is wrong. Rejecting it
    costs a round trip, and an author shown its own report back rewrites it
    from memory and breaks a different part.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="wrap")
    @classmethod
    def _accept_a_stringified_member(cls, data: Any, handler: Any) -> Any:
        try:
            return handler(data)
        except (ValidationError, PydanticCustomError, ValueError):
            if not isinstance(data, dict):
                raise
            repaired = {key: _decoded(value) for key, value in data.items()}
            if repaired == data:
                raise
            return handler(repaired)
