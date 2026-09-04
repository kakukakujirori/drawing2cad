from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from zeroshot.pipeline.messages.contracts import DrawingSource


def _safe_identifier(name: str, field_name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    if stripped in {".", ".."} or "/" in stripped or "\\" in stripped:
        raise ValueError(f"unsafe {field_name}: {name!r}")
    return stripped


def _present(drawing: DrawingSource) -> None:
    for path in drawing.paths():
        if not path.is_file():
            raise FileNotFoundError(f"Not Found: {path}")


@dataclass(frozen=True)
class InputManifest:
    """The drawings a sample is made of.

    One `DrawingSource` rather than a path because a sample may arrive as one
    sheet, as one file per view, as DXF, as PNG, or as a mixture, and every
    stage after this one should be unable to tell which. A perspective render
    offered alongside the drawing is a sheet like any other, with `label`
    saying which rendering it is.
    """

    sample_id: str
    drawing: DrawingSource

    def __post_init__(self) -> None:
        _present(self.drawing)
        object.__setattr__(
            self, "sample_id", _safe_identifier(self.sample_id, "sample_id")
        )


@dataclass(frozen=True)
class FeedbackManifest:
    """What a verification drew of the solid it built, and what it could not.

    The same shape as the input, so one presenter announces both. `errors` is
    keyed by the label the sheet would have carried, because a sheet that was
    never produced cannot carry its own explanation.
    """

    verification_id: str
    drawing: DrawingSource | None = None
    errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors = MappingProxyType(dict(self.errors))
        if self.drawing is not None:
            _present(self.drawing)
            # An artifact is either present or explained, never both.
            drawn = {sheet.label for sheet in self.drawing.sheets if sheet.label}
            if both := sorted(drawn & set(errors)):
                raise ValueError(f"sheets are both drawn and failed: {both}")

        object.__setattr__(
            self,
            "verification_id",
            _safe_identifier(self.verification_id, "verification_id"),
        )
        object.__setattr__(self, "errors", errors)
