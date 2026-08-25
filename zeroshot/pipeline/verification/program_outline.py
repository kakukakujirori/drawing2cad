"""The coder's program, laid out from the plan as one section per operation.

The coding stage writes a single file that runs as it stands. What this adds is
a marker before each operation's code, written from the plan rather than by the
coder. The markers buy three things at once: a defect belongs to a named step,
a revision can clear one step's code and leave the rest, and the order the
sections appear in is the order the dependencies imply, which the interpreter
then enforces for free -- code that reached forward would raise a NameError.

Sections rather than files or functions. Both of those cut the program up as
well as marking it, and CadQuery is a fluent interface: a chain broken at a
boundary loses the workplane it was standing on, and gains a driver, an
argument convention and an assembled second copy of the file to keep in step.
A comment costs none of that and marks just as well.
"""

import re
import textwrap
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    SemanticHypothesis,
    linearise,
    operation_heading,
    resolve_reference,
)

__all__ = [
    "MARKER",
    "OutlineDeletion",
    "ProgramOutline",
    "SectionReview",
    "render_outline_deletion",
    "render_section_review",
    "review_sections",
    "update_program_outline",
]

# `# ---- ` opens every line the machine owns, marker and continuation alike, so
# that a comment the coder writes can never be mistaken for one of them.
_OWNED = "# ---- "
MARKER = re.compile(rf"^{re.escape(_OWNED)}(op_[a-z][a-z0-9_]*) ")

_WIDTH = 88

# What a file starts as, before any operation has been written. Only the import
# every program needs.
#
# Nothing reads the preamble. It is the run of lines before the first marker,
# carried from one layout to the next as the bytes it is, so however the coder
# arranges its imports and its shared setup is however they stay.
_PREAMBLE = "import cadquery as cq\n"

# Where the finished solid is named. `result` belongs to no operation -- it is
# what the operations between them came to -- so it gets a place of its own
# instead of riding on whichever step happens to land last. Without one it is
# cleared along with that step whenever the step is revised, and the program
# loses its answer over a revision that had nothing to do with it.
_EPILOGUE = f"{_OWNED}result"
_EPILOGUE_LINE = f"{_EPILOGUE} (not a step; kept across every revision)"


@dataclass(frozen=True)
class OutlineDeletion:
    """The area of code that the coder's code was reverted to empty.

    `code_only` -- The comment instruction remains, but the code was deleted.
    `entire_section` -- The entire section, including the comment instruction, was deleted.
    """

    code_only: list[str] = field(default_factory=list)
    entire_section: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SectionReview:
    """The state of the coder's file against the plan it was laid out from."""

    expected: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    out_of_order: bool = False

    @property
    def written(self) -> int:
        return len(self.expected) - len(self.empty) - len(self.missing)

    @property
    def nothing_written(self) -> bool:
        """Whether the file is still the outline it was laid out as.

        A file of markers and no code is not a program that failed; it is a
        program nobody has written, and running it would spend a sandbox to
        find that out. Every marker has to be there for this to be the answer:
        a coder that wrote over the outline has left something, and something
        is judged by building it, not by whether the comments survived.
        """
        return bool(self.expected) and not self.missing and self.written == 0


@dataclass(frozen=True)
class _Section:
    name: str
    marker: str
    body: str

    @property
    def says(self) -> str:
        return _says(self.marker)

    @property
    def empty(self) -> bool:
        """Whether the section holds code, as opposed to blank lines and prose.

        By reading the lines rather than by parsing: a section is asked this
        while the file around it may not compile, and "you have not written
        this one yet" must not turn into a crash the first time the coder
        leaves a bracket open.
        """
        return not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in self.body.splitlines()
        )


def _says(marker: str) -> str:
    """The words of a marker, with the prefix and the line breaks taken out.

    The comparison this feeds answers "does the marker say the same thing", not
    "was it wrapped at the same column". Both have to go: the wrapping is this
    module's own doing and moves whenever the width does, and the prefix is
    written once per line, so counting it would smuggle the line breaks back in
    through the door the normalising was meant to shut.
    """
    return " ".join(
        word for line in marker.splitlines() for word in line[len(_OWNED) :].split()
    )


def _split(source: str) -> tuple[str, list[_Section], str]:
    """The file as its preamble, its sections and its epilogue.

    Anything before the first marker is preamble -- the imports, and whatever
    else the coder put at the top -- and anything after the `result` marker is
    epilogue. Both are kept whole; only the sections between them answer to a
    step and can be cleared.

    A marker line the coder has damaged no longer matches, so the text under it
    stays in the section above rather than disappearing: the review says the
    marker has gone, and the code is still there to be moved back.
    """
    preamble: list[str] = []
    sections: list[tuple[str, list[str], list[str]]] = []
    epilogue: list[str] = []
    ending = False
    for line in source.splitlines(keepends=True):
        if line.startswith(_EPILOGUE):
            ending = True
        elif line.startswith(_OWNED):
            # A marker opens a section; any other line the machine owns is a
            # continuation of one, and is kept with it because together they
            # are the instruction the code below was written against.
            if (found := MARKER.match(line)) and not ending:
                sections.append((found[1], [line], []))
            elif sections and not ending and not sections[-1][2]:
                sections[-1][1].append(line)
        elif ending:
            epilogue.append(line)
        elif sections:
            sections[-1][2].append(line)
        else:
            preamble.append(line)
    return (
        "".join(preamble),
        [_Section(name, "".join(said), "".join(body)) for name, said, body in sections],
        "".join(epilogue),
    )


def _marker(operation: Operation, hypothesis: SemanticHypothesis) -> str:
    """One step's marker: what the plan asks of it, in the file it is asked in.

    The marker carries the resolved text rather than the operation, so that a
    measurement which moved in the hypothesis counts as a change too: a plan
    citing a semantic parameter says the same words after the radius changes, and the
    code written under it is answering the old number.

    This is also the whole record of what a section was written against. There
    is no stamp beside it: a hash of the words would be a second copy of what
    is already legible on the line it would sit on, and `fingerprint.py`
    declined that same trade -- the content, not a mark standing for it.
    """
    detail = " ".join(resolve_reference(operation.detail, hypothesis).split())
    said = [
        operation_heading(operation),
        # Not on hyphens: a plan says "nut-capture pocket" and "lower-eye",
        # and a marker that breaks those across lines is harder to read than
        # one that runs a little long.
        *(textwrap.wrap(detail, _WIDTH - len(_OWNED), break_on_hyphens=False) or [""]),
    ]
    return "\n".join(_OWNED + line for line in said)


################################################################


class ProgramOutline:
    """Write operations as instruction comments, and read back against it."""

    def __init__(self, filepath: Path) -> None:
        # Enforce the abs path to prevent host v.s. sandbox confusion.
        if not filepath.is_absolute():
            raise ValueError(f"{filepath} must be an absolute path.")
        if filepath.suffix != ".py":
            raise ValueError(f"{filepath=} must be a Python file.")
        self._path = filepath
        self._plan: OperationPlan | None = None
        self._own_write: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def own_write(self) -> str | None:
        """The digest of the last content this wrote, or None if it has written none.

        Offered so that whoever watches the file for a turn's work can tell
        that a change was this object's doing rather than the coder's.
        """
        return self._own_write

    def prepare(
        self, plan: OperationPlan, hypothesis: SemanticHypothesis
    ) -> OutlineDeletion:
        """Update the operation plan comments, and notify the difference from the previous."""
        existing = (
            self._path.read_text(encoding="utf-8", errors="replace")
            if self._path.is_file()
            else None
        )
        written, deleted = update_program_outline(plan, hypothesis, existing)
        self._path.write_text(written, encoding="utf-8")

        # update the state
        self._plan = plan
        self._own_write = sha256(written.encode("utf-8")).hexdigest()

        return deleted

    def review(self) -> SectionReview | None:
        """How far the file has got, or None while there is nothing to read it against.

        None rather than an empty reading, because a file nobody laid out is
        not a file with no sections written: the standalone verification tool
        is handed a program somebody wrote by hand, and there is no plan it was
        supposed to follow.
        """
        if self._plan is None or not self._path.is_file():
            return None
        return review_sections(
            self._path.read_text(encoding="utf-8", errors="replace"), self._plan
        )


def update_program_outline(
    plan: OperationPlan,
    hypothesis: SemanticHypothesis,
    existing: str | None = None,
) -> tuple[str, OutlineDeletion]:
    """The file the coder fills in, brought up to date with the plan.

    An update rather than a fresh laying-out: most of the time a file is
    already there and most of its code still answers the plan it was written
    from. What is kept is decided section by section, from the file itself
    rather than from anything this process remembers, which is what makes the
    answer survive a checkpoint restore and lets this stay a plain function.

    A marker that still says what the plan says means the step has not moved
    and its code is left alone. One that says something else means the step was
    revised, and the code is cleared rather than reported as suspect --
    reporting it leaves the coder free to agree with itself and move on, which
    is the failure this exists to prevent.
    """
    preamble, held, epilogue = _split(existing or _PREAMBLE)
    by_name = {section.name: section for section in held}
    wanted = {operation.name for operation in plan.proposal}

    emptied: list[str] = []
    written = [preamble.rstrip("\n"), ""]
    for operation in linearise(plan):
        marker = _marker(operation, hypothesis)
        section = by_name.get(operation.name)
        keeps = section is not None and section.says == _says(marker)
        if section is not None and not keeps and not section.empty:
            emptied.append(operation.name)
        body = section.body.strip("\n") if keeps and section else ""
        written.append(f"{marker}\n{body}\n" if body else f"{marker}\n")

    tail = epilogue.strip("\n")
    written.append(f"{_EPILOGUE_LINE}\n{tail}\n" if tail else f"{_EPILOGUE_LINE}\n")

    return "\n".join(written).lstrip("\n"), OutlineDeletion(
        code_only=emptied,
        entire_section=[section.name for section in held if section.name not in wanted],
    )


def review_sections(source: str, plan: OperationPlan) -> SectionReview:
    """Read the file against the plan it was laid out from."""
    _, held, _ = _split(source)
    expected = [operation.name for operation in linearise(plan)]
    wanted = set(expected)
    seen = [section.name for section in held if section.name in wanted]

    return SectionReview(
        expected=expected,
        empty=[
            section.name for section in held if section.name in wanted and section.empty
        ],
        missing=[name for name in expected if name not in seen],
        out_of_order=seen != [name for name in expected if name in seen],
    )


def render_outline_deletion(deleted: OutlineDeletion) -> str:
    faults: list[str] = []
    if deleted.code_only:
        named = ", ".join(deleted.code_only)
        faults.append(
            f"The code under {named} was written against a different "
            "instruction from the one its marker now carries, so it has been "
            "cleared. Write it again against what the marker says."
        )
    if deleted.entire_section:
        named = ", ".join(deleted.entire_section)
        faults.append(
            f"{named} is no longer in the plan, and the code for it has been "
            "removed along with the marker."
        )
    return " ".join(faults)


def render_section_review(review: SectionReview) -> str:
    """What the coder is told about its own file, or nothing when all is well.

    Progress rather than a fault: "sections 12/15 -- op_bore_through,
    op_fillet_top still to write" is a next step. It is handed over on every
    turn that changed anything, so it doubles as the running account of how far
    along the part is, and it names rather than counts, so it can be acted on
    without going back to compare two lists.
    """
    faults: list[str] = []

    if review.missing or review.empty:
        still = ", ".join(sorted(review.missing + review.empty))
        faults.append(
            f"sections {review.written}/{len(review.expected)} -- {still} "
            "still to write."
        )
    if review.missing:
        named = ", ".join(review.missing)
        faults.append(
            f"The marker for {named} is gone. The markers are written from the "
            "plan; put the code back under the one it belongs to rather than "
            "writing your own."
        )
    if review.out_of_order:
        faults.append(
            "The sections no longer follow the plan's build order, so a step "
            "may run before what it is made from. Leave them in the order they "
            "were written in."
        )
    return " ".join(faults)
