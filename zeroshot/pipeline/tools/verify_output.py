from inspect import cleandoc
from pathlib import PurePosixPath

from langchain_core.messages.content import ContentBlock
from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.messages import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification.program_outline import ProgramOutline
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutor
from zeroshot.pipeline.verification.run_render import StepRenderer
from zeroshot.pipeline.verification.verify_output import OutputVerifier


def create_verify_output_tool(
    executor: CadQueryExecutor,
    workdir: SandboxWorkdir,
    renderer: StepRenderer,
    artifact_presenter: ArtifactPresenter | None,
    source_filename: str = "model.py",
    output_dirname: PurePosixPath = PurePosixPath("attempts"),
) -> BaseTool:
    """Offer verification to an agent that must ask for it.

    The staged workflow verifies the coder's writes without being asked, so it
    builds an `OutputVerifier` directly; this is for an agent whose turn is the
    right place to decide.

    The outline is built and never prepared: this agent writes the whole file
    itself, so there is no plan to read it against and `review` answers None.
    """
    verifier = OutputVerifier(
        executor,
        workdir,
        renderer=renderer,
        artifact_presenter=artifact_presenter,
        program=ProgramOutline(workdir.host_bind_dir / source_filename),
        output_dirname=output_dirname,
    )
    description = cleandoc(
        f"""
        Execute and validate the current CadQuery program at
        {workdir.sandbox_bind_dir}/{source_filename}.

        The file must be a self-contained Python program that assigns the final
        CadQuery object to a variable named `result`. Verification runs in a fresh
        isolated directory, so other files in {workdir.sandbox_bind_dir} are not
        available to the program. The tool exports `result` to STEP and accepts it
        only when the exported file contains exactly one valid solid.

        The result reports the execution status, return code, stdout, stderr, and
        any executor error. Generated STEP and its rendered views are saved in
        {workdir.sandbox_bind_dir}/{output_dirname}/<verification_id>/.
        """
    )

    @tool("verify_output", description=description)
    def verify_output() -> list[ContentBlock]:
        return verifier.feedback()

    return verify_output
