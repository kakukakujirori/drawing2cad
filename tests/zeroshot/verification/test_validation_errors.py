"""Each validation error points at the JSON the model wrote, in one short line."""

from typing import Self

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from zeroshot.pipeline.verification.validation_errors import answer_errors, file_errors

_NOT_A_NUMBER = "Input should be a valid number, unable to parse string as a number"


class Part(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    size: float | list[float | None]
    labels: dict[str, float] = {}


class Sheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[Part]

    @model_validator(mode="after")
    def require_unique_names(self) -> Self:
        if len({part.name for part in self.parts}) != len(self.parts):
            raise ValueError("part names must be unique")
        return self


def _file_errors(text: str) -> list[str]:
    with pytest.raises(ValidationError) as caught:
        Sheet.model_validate_json(text)
    return file_errors(caught.value, "sheet.json", text)


def test_an_error_points_past_same_named_keys_and_escaped_brackets() -> None:
    text = """{
  "parts": [
    {"name": "part_a", "size": 1, "labels": {"a } ] \\" { [": 2}},
    {"name": "part_b", "size": "wide"}
  ]
}"""
    # Union branch labels are not keys: both branch errors share one entry.
    assert _file_errors(text) == [
        f"sheet.json:4:24 $.parts[1].size (part_b): {_NOT_A_NUMBER}; "
        + "Input should be a valid array"
    ]


def test_a_missing_field_points_at_its_object_and_an_extra_one_at_its_key() -> None:
    text = '{"parts": [\n  {"name": "part_a",\n   "extra": 1}\n]}'
    assert _file_errors(text) == [
        "sheet.json:2:3 $.parts[0] (part_a): field 'size' is required",
        "sheet.json:3:4 $.parts[0].extra (part_a): Extra inputs are not permitted",
    ]


def test_an_object_constraint_points_at_the_object() -> None:
    text = '{"parts": [{"name": "part_a", "size": 1}, {"name": "part_a", "size": 2}]}'
    assert _file_errors(text) == ["sheet.json:1:1 $: part names must be unique"]


def test_invalid_json_points_where_the_decoder_stopped() -> None:
    text = '{"parts": [\n  {"name": "part_a",}\n]}'
    assert _file_errors(text) == [
        "sheet.json:2:21 Invalid JSON: Expecting property name enclosed in double quotes"
    ]


def test_an_answer_gets_paths_without_lines_or_its_long_input() -> None:
    answer = {
        "parts": [{"name": "part_a", "size": "x" * 500, "labels": {"a b": "tall"}}]
    }
    with pytest.raises(ValidationError) as caught:
        Sheet.model_validate(answer)
    assert answer_errors(caught.value, answer) == [
        f"$.parts[0].size (part_a): {_NOT_A_NUMBER}; Input should be a valid list",
        f'$.parts[0].labels["a b"] (part_a): {_NOT_A_NUMBER}',
    ]


def test_a_key_named_like_a_union_label_is_part_of_the_rejected_value() -> None:
    text = '{"parts": [{"name": "part_a", "size": {"float": 1}}]}'
    assert _file_errors(text) == [
        "sheet.json:1:31 $.parts[0].size (part_a): Input should be a valid number; "
        + "Input should be a valid array"
    ]


def test_json_too_deep_to_locate_still_reports_its_reason() -> None:
    text = '{"parts": ' + "[" * 1200 + "]" * 1200 + "}"
    [report] = _file_errors(text)
    assert report.startswith("sheet.json $: Invalid JSON: recursion limit exceeded")


def test_a_repeated_key_points_at_the_copy_that_was_validated() -> None:
    text = '{"parts": [{"name": "part_a", "size": 1}],\n "parts": [{"name": "part_b"}]}'
    assert _file_errors(text) == [
        "sheet.json:2:12 $.parts[0] (part_b): field 'size' is required"
    ]
