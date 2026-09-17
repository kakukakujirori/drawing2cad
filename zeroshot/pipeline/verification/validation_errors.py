"""Point each Pydantic error at the JSON the model wrote, in one short line."""

import json
import re

from pydantic import ValidationError
from pydantic_core import ErrorDetails

type JsonPath = tuple[str | int, ...]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def file_errors(error: ValidationError, filename: str, raw: str | bytes) -> list[str]:
    """`file:line:col $.path (name): reason` for each error in a written file."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        data = json.loads(text)
        starts = _starts(text)
    except json.JSONDecodeError as invalid:
        return [
            f"{filename}:{invalid.lineno}:{invalid.colno} Invalid JSON: {invalid.msg}"
        ]
    except (RecursionError, ValueError):
        # Too deep or too long a number to locate; the reasons still reach the agent.
        return [f"{filename} {entry}" for _, entry in _entries(error, None)]
    return [
        f"{filename}:{_line_column(text, starts[path])} {entry}"
        for path, entry in _entries(error, data)
    ]


def answer_errors(error: ValidationError, answer: object) -> list[str]:
    """`$.path (name): reason` for each error; a structured answer has no lines."""
    return [entry for _, entry in _entries(error, answer)]


def _entries(error: ValidationError, data: object) -> list[tuple[JsonPath, str]]:
    """Errors at one place, such as one per union branch, share an entry."""
    reasons: dict[tuple[JsonPath, str | None], list[str]] = {}
    for detail in error.errors(include_url=False, include_context=False):
        where = _located(detail["loc"], detail["input"], data)
        reasons.setdefault(where, []).append(_reason(detail))
    entries = []
    for (path, name), found in reasons.items():
        where = f"{_rendered(path)} ({name})" if name else _rendered(path)
        entries.append((path, f"{where}: {'; '.join(dict.fromkeys(found))}"))
    return entries


def _located(
    loc: tuple[int | str, ...], rejected: object, data: object
) -> tuple[JsonPath, str | None]:
    """The rejected value's place, and the innermost name on the way.

    Parts the submission lacks, such as a union branch label or a missing field,
    are skipped. The walk stops at the rejected value, even if a key matches a label.
    """
    path: list[str | int] = []
    node, name = data, _name(data)
    for part in loc:
        if node == rejected:
            break
        match node:
            case dict() if part in node:
                node = node[part]
            case list() if isinstance(part, int) and 0 <= part < len(node):
                node = node[part]
            case _:
                continue
        path.append(part)
        name = _name(node) or name
    return tuple(path), name


def _name(node: object) -> str | None:
    name = node.get("name") if isinstance(node, dict) else None
    return name if isinstance(name, str) else None


def _reason(detail: ErrorDetails) -> str:
    if detail["type"] == "missing":
        return f"field {detail['loc'][-1]!r} is required"
    return detail["msg"].removeprefix("Value error, ")


def _rendered(path: JsonPath) -> str:
    return "$" + "".join(_step(part) for part in path)


def _step(part: str | int) -> str:
    if isinstance(part, int):
        return f"[{part}]"
    return f".{part}" if _IDENTIFIER.fullmatch(part) else f"[{json.dumps(part)}]"


def _starts(text: str) -> dict[JsonPath, int]:
    """Where each value of valid JSON begins; an object member begins at its key.

    A repeated key moves its places to the last copy, the one parsers keep.
    """
    starts: dict[JsonPath, int] = {}
    decoder = json.JSONDecoder()

    def skip(index: int) -> int:
        while text[index : index + 1] in (" ", "\t", "\n", "\r"):
            index += 1
        return index

    def value(index: int, path: JsonPath) -> int:
        index = skip(index)
        starts[path] = index
        if text[index] == "{":
            index = skip(index + 1)
            while text[index] != "}":
                key, after = decoder.raw_decode(text, index)
                end = value(skip(after) + 1, (*path, key))
                starts[(*path, key)] = index
                index = skip(end)
                if text[index] == ",":
                    index = skip(index + 1)
            return index + 1
        if text[index] == "[":
            index, item = skip(index + 1), 0
            while text[index] != "]":
                index = skip(value(index, (*path, item)))
                item += 1
                if text[index] == ",":
                    index = skip(index + 1)
            return index + 1
        return decoder.raw_decode(text, index)[1]

    value(0, ())
    return starts


def _line_column(text: str, index: int) -> str:
    line = text.count("\n", 0, index) + 1
    column = index - text.rfind("\n", 0, index)
    return f"{line}:{column}"
