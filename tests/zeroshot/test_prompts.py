from pathlib import Path

import pytest

from zeroshot.pipeline.messages import PromptTemplate


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_a_packaged_prompt_is_addressed_by_name() -> None:
    """Names, not paths: Hydra moves the process into the job's output
    directory before a run, so a relative path in a config would not resolve."""
    prompt = PromptTemplate("coder")

    assert prompt.path.name == "coder.md"
    assert prompt.path.is_file()


def test_placeholders_are_filled_from_the_context() -> None:
    rendered = PromptTemplate("coder").render(
        output_path="/work/model.py",
        verification_dir="/work/attempts",
    )

    assert "/work/model.py" in rendered
    assert "/work/attempts" in rendered
    assert "$output_path" not in rendered
    assert "$verification_dir" not in rendered


def test_a_missing_value_is_refused(tmp_path: Path) -> None:
    """`substitute`, not `safe_substitute`: a value we forgot to pass must not
    reach the model as the literal `$verification_dir`."""
    prompt = PromptTemplate(str(_write(tmp_path / "p.md", "$here and $there")))

    with pytest.raises(KeyError):
        prompt.render(here="only one")


def test_an_unused_value_is_ignored(tmp_path: Path) -> None:
    """One context serves every stage, so a prompt may use none of it."""
    prompt = PromptTemplate(str(_write(tmp_path / "p.md", "no placeholders")))

    assert prompt.render(output_path="/work/model.py") == "no placeholders"


def test_braces_survive_rendering(tmp_path: Path) -> None:
    """Why `$name` and not `{name}`: prompts carry CadQuery snippets."""
    body = 'result = cq.Workplane().box(**{"length": 1})'
    prompt = PromptTemplate(str(_write(tmp_path / "p.md", body)))

    assert prompt.render() == body


def test_surrounding_whitespace_does_not_reach_the_model(tmp_path: Path) -> None:
    """Whether a file ends in a newline is an editor's decision, and must not
    silently change the bytes the model is sent."""
    bare = PromptTemplate(str(_write(tmp_path / "bare.md", "instructions")))
    padded = PromptTemplate(str(_write(tmp_path / "padded.md", "\ninstructions\n\n")))

    assert bare.render() == padded.render() == "instructions"


def test_an_unknown_prompt_is_refused_before_the_run() -> None:
    with pytest.raises(ValueError, match="prompt not found"):
        PromptTemplate("no_such_prompt")


def test_the_digest_follows_the_file(tmp_path: Path) -> None:
    """The prompt is the experiment's main variable, so a run's audit trail
    needs a way to say which text it used."""
    path = _write(tmp_path / "p.md", "first")
    prompt = PromptTemplate(str(path))
    before = prompt.sha256

    _write(path, "second")

    assert prompt.sha256 != before
