from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from zeroshot.pipeline.event_logging import (
    AgentMessageTransformer,
    ConsoleReporter,
    JsonlEventWriter,
    RunEvent,
    RunEventTransformer,
    has_run_completed,
)
from zeroshot.pipeline.messages import ArtifactPresenter, InputManifest
from zeroshot.pipeline.messages.contracts.reconstruction import ReconstructionRun
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import CUSTOM_STATE_TYPES, ReconstructionState
from zeroshot.pipeline.workflow.reconstruction import load_reconstruction

# What the runner hands a graph: the run environment and the artifact contract.
# A graph's own settings are bound into the factory before it gets here, so
# swapping graphs never widens this signature.
GraphFactory = Callable[..., Any]

OnExisting = Literal["fail", "skip", "retry"]
_ON_EXISTING = ("fail", "skip", "retry")


def _latest_program_source(reconstruction: ReconstructionRun) -> str | None:
    """Return the newest committed program available before the resumed stage."""
    return next(
        (
            snapshot.program_source
            for snapshot in reversed(reconstruction.snapshots)
            if snapshot.program_source is not None
        ),
        None,
    )


def _clear_incomplete_run(sample_artifact_root: Path) -> None:
    """Remove what the runner writes, keeping what Hydra wrote for this job.

    Deleting the directory outright would take `.hydra/` and the job log with
    it, and those describe the run about to start.  Everything else is cleared
    by exclusion so a new artifact cannot survive a redo by being forgotten
    here.
    """
    if not sample_artifact_root.is_dir():
        return
    for entry in sample_artifact_root.iterdir():
        if entry.name == ".hydra" or entry.suffix == ".log":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


class PipelineRunner:
    WORKSPACE_DIRNAME = "workspace"

    def __init__(
        self,
        sandbox_runner: SandboxRunner,
        graph_factory: GraphFactory,
        artifact_presenter: ArtifactPresenter,
        artifact_root: str | Path,
        renderer: StepRenderer,
        output_filename: str = "model.py",
        verification_dirname: PurePosixPath = PurePosixPath("attempts"),
        on_existing: OnExisting = "fail",
        console_reporter: ConsoleReporter | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        if on_existing not in _ON_EXISTING:
            raise ValueError(
                f"on_existing must be one of {_ON_EXISTING}: {on_existing!r}"
            )
        self.artifact_presenter = artifact_presenter
        self.sandbox_runner = sandbox_runner
        self.renderer = renderer
        self.artifact_root = Path(artifact_root)
        self.output_filename = output_filename
        self.verification_dirname = verification_dirname
        self.graph_factory = graph_factory
        self.on_existing = on_existing
        self.console_reporter = console_reporter
        self.resume_from = Path(resume_from) if resume_from is not None else None

    def _prepare_workspace(
        self,
        sample_artifact_root: Path,
        events_path: Path,
        reconstruction: ReconstructionRun | None,
    ) -> Path:
        """Reset one sample and restore the files needed by a resumed stage.

        A temporary copy is needed only when `retry` is about to remove the
        attempt that also serves as the resume source. External resume sources
        are copied directly into the new workspace.
        """
        verification = (
            reconstruction.snapshots[-1].verification if reconstruction else None
        )
        attempt = (
            self.resume_from.parent
            / self.verification_dirname
            / verification.verification_id
            if self.resume_from and verification and verification.verification_id
            else None
        )
        if attempt is not None and not attempt.is_dir():
            raise FileNotFoundError(f"resume attempt is missing: {attempt}")

        temporary: tempfile.TemporaryDirectory[str] | None = None
        try:
            if (
                self.on_existing == "retry"
                and attempt is not None
                and attempt.resolve().is_relative_to(sample_artifact_root.resolve())
            ):
                temporary = tempfile.TemporaryDirectory(
                    prefix="drawing2cad-resume-"
                )
                staged = Path(temporary.name) / attempt.name
                shutil.copytree(attempt, staged)
                attempt = staged

            if self.on_existing == "retry":
                _clear_incomplete_run(sample_artifact_root)
            elif events_path.exists():
                raise FileExistsError(
                    "incomplete run left behind, delete it to redo: "
                    f"{sample_artifact_root}"
                )

            sample_artifact_root.mkdir(parents=True, exist_ok=True)
            workspace = sample_artifact_root / self.WORKSPACE_DIRNAME
            workspace.mkdir()

            source = _latest_program_source(reconstruction) if reconstruction else None
            if source is not None:
                (workspace / self.output_filename).write_text(source, encoding="utf-8")
            if attempt is not None:
                destination = workspace / self.verification_dirname / attempt.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(attempt, destination)

            return workspace
        finally:
            if temporary is not None:
                temporary.cleanup()

    def run_sample(self, manifest: InputManifest) -> ReconstructionState | None:
        """Run one sample, or return None when the policy skips it.

        The directory may already exist because Hydra writes its resolved config
        there before the job body runs, so `events.jsonl` is what says whether a
        run happened.  No policy ever *skips* an incomplete one: that would let
        a sweep report every sample as done while some produced nothing.
        """
        sample_artifact_root = self.artifact_root / manifest.sample_id
        events_path = sample_artifact_root / "events.jsonl"
        if has_run_completed(events_path):
            if self.on_existing in {"skip", "retry"}:
                return None
            raise FileExistsError(f"sample already ran: {sample_artifact_root}")

        # Read before `retry` clears the destination. This also permits an
        # interrupted run to resume from its own durable history.
        reconstruction_resume = (
            load_reconstruction(self.resume_from)
            if self.resume_from is not None
            else None
        )
        workspace_path = self._prepare_workspace(
            sample_artifact_root,
            events_path,
            reconstruction_resume,
        )

        run_id = f"{manifest.sample_id}:{uuid4()}"
        checkpoint_path = sample_artifact_root / "checkpoints.sqlite"

        with ExitStack() as stack:
            # instantiate event loggers (available only in this block)
            event_writer = stack.enter_context(
                JsonlEventWriter(
                    sample_artifact_root / "events.jsonl",
                    run_id=run_id,
                    sample_id=manifest.sample_id,
                )
            )
            if self.console_reporter is not None:
                stack.enter_context(
                    self.console_reporter.run_context(
                        run_id=run_id,
                        sample_id=manifest.sample_id,
                    )
                )

            # instantiate sandbox directly in the persistent sample workspace
            workdir = stack.enter_context(SandboxWorkdir(host_bind_dir=workspace_path))

            # copy input files to sandbox
            input_dirname = PurePosixPath("inputs")
            workdir.read_only_subdirs.append(input_dirname)
            staged_manifest = self._stage_inputs(
                manifest=manifest,
                workdir=workdir,
                input_dirname=input_dirname,
            )

            # instantiate the graph
            checkpointer = stack.enter_context(
                SqliteSaver.from_conn_string(str(checkpoint_path))
            )
            checkpointer.serde = JsonPlusSerializer(
                allowed_msgpack_modules=list(CUSTOM_STATE_TYPES)
            )
            graph = self.graph_factory(
                sandbox_runner=self.sandbox_runner,
                sandbox_workdir=workdir,
                renderer=self.renderer,
                artifact_presenter=self.artifact_presenter,
                input_manifest=staged_manifest,
                output_filename=self.output_filename,
                verification_dirname=self.verification_dirname,
                checkpointer=checkpointer,
            )

            # The record is written where events are produced rather than where
            # a consumer gets to them, so no way of watching a run can cost it
            # its log.
            def record_event(event: RunEvent) -> None:
                event_writer.write(event)

            # run the graph
            initial_state = ReconstructionState()
            if reconstruction_resume is not None:
                initial_state["reconstruction"] = reconstruction_resume

            stream = graph.stream_events(
                initial_state,
                config={"configurable": {"thread_id": run_id}},
                version="v3",
                durability="sync",
                transformers=[
                    partial(RunEventTransformer, sink=record_event),
                    AgentMessageTransformer,
                ],
            )
            # One loop, in arrival order: every item either projection publishes
            # is renderable on its own, so neither can hold up the other.
            if self.console_reporter is not None:
                for channel, item in stream.interleave(
                    RunEventTransformer.CHANNEL,
                    AgentMessageTransformer.CHANNEL,
                ):
                    if channel == RunEventTransformer.CHANNEL:
                        self.console_reporter.render_event(item)
                    else:
                        self.console_reporter.render_model_item(item)
            # Drives whatever the console did not, and is the only driver when
            # there is no console at all.
            result = stream.output

        return cast(ReconstructionState, result)

    def _stage_inputs(
        self,
        manifest: InputManifest,
        workdir: SandboxWorkdir,
        input_dirname: str | PurePosixPath = "inputs",
    ) -> InputManifest:
        staged_input_dir = workdir.host_bind_dir / input_dirname
        staged_input_dir.mkdir(parents=True, exist_ok=False)

        staged_dxf_path = staged_input_dir / "techdraw.dxf"
        shutil.copyfile(manifest.dxf_path, staged_dxf_path)

        staged_render_paths: dict[str, Path] = {}

        if self.artifact_presenter.input_render3d_mode != "none":
            for style in self.artifact_presenter.input_render3d_styles:
                source_path = manifest.render3d_paths[style]
                suffix = source_path.suffix or ".png"
                staged_path = staged_input_dir / f"{style}{suffix}"
                shutil.copyfile(source_path, staged_path)
                staged_render_paths[style] = staged_path

        return InputManifest(
            sample_id=manifest.sample_id,
            dxf_path=staged_dxf_path,
            render3d_paths=staged_render_paths,
        )
