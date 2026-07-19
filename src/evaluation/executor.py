"""Crash- and timeout-contained CadQuery source execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
from typing import Any, Mapping


# Spawned workers inherit these before importing NumPy/trimesh. A single thread
# per isolated sample keeps wall-clock timeout behaviour stable under concurrent
# validation.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")


@dataclass(frozen=True)
class CadExecutionResult:
    """Serializable outcome of executing one CadQuery completion."""

    exec_ok: bool = False
    has_result: bool = False
    is_valid: bool = False
    volume: float = 0.0
    valid: bool = False
    step_ok: bool = False
    error: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CadExecutionResult":
        return cls(
            exec_ok=bool(payload.get("exec_ok", False)),
            has_result=bool(payload.get("has_result", False)),
            is_valid=bool(payload.get("is_valid", False)),
            volume=float(payload.get("volume", 0.0)),
            valid=bool(payload.get("valid", False)),
            step_ok=bool(payload.get("step_ok", False)),
            error=None if payload.get("error") is None else str(payload["error"]),
        )


def _short_error(error: BaseException, *, stage: str | None = None) -> str:
    message = " ".join(str(error).split())
    label = type(error).__name__
    value = f"{label}: {message}" if message else label
    if stage:
        value = f"{stage}:{value}"
    return value[:500]


def _recover_shape(namespace: Mapping[str, object], cq: Any) -> object | None:
    def unwrap(value: object) -> object | None:
        if value is None:
            return None
        if isinstance(value, cq.Workplane):
            return value.val()
        if isinstance(value, cq.Shape):
            return value
        return None

    for name in ("result", "r"):
        if name in namespace:
            shape = unwrap(namespace[name])
            if shape is not None:
                return shape
    candidate = None
    for value in namespace.values():
        shape = unwrap(value)
        if shape is not None:
            candidate = shape
    return candidate


def _execute_worker(
    source: str,
    child: Connection,
    mesh_output_path: str | None,
    step_output_path: str | None,
    tessellation_tolerance: float,
    volume_tolerance: float,
) -> None:
    payload = asdict(CadExecutionResult())
    try:
        cq = None
        cadquery_import_error: BaseException | None = None
        try:
            import cadquery as cq
        except BaseException as error:
            cadquery_import_error = error

        namespace: dict[str, object] = {"__name__": "__main__"}
        if cq is not None:
            namespace["cq"] = cq
        try:
            exec(compile(source, "<cadquery-prediction>", "exec"), namespace)
            payload["exec_ok"] = True
        except BaseException as error:
            payload["error"] = _short_error(error, stage="exec")
            child.send(payload)
            return

        if cq is None:
            assert cadquery_import_error is not None
            payload["error"] = _short_error(
                cadquery_import_error, stage="cadquery_import"
            )
            child.send(payload)
            return

        try:
            shape = _recover_shape(namespace, cq)
        except BaseException as error:
            payload["error"] = _short_error(error, stage="result")
            child.send(payload)
            return
        if shape is None:
            payload["error"] = "no_result"
            child.send(payload)
            return
        payload["has_result"] = True

        # Export the recovered shape as STEP whenever a shape exists, so callers
        # obtain a neutral CAD artifact even when the solid is not watertight.
        if step_output_path is not None:
            try:
                shape.exportStep(step_output_path)
                payload["step_ok"] = True
            except BaseException:
                payload["step_ok"] = False

        try:
            payload["is_valid"] = bool(shape.isValid())
        except BaseException:
            payload["is_valid"] = False
        try:
            payload["volume"] = float(shape.Volume())
        except BaseException:
            payload["volume"] = 0.0

        try:
            import numpy as np
            import trimesh

            vertices, faces = shape.tessellate(tessellation_tolerance)
            mesh = trimesh.Trimesh(
                vertices=[(vertex.x, vertex.y, vertex.z) for vertex in vertices],
                faces=faces,
                process=True,
            )
            watertight = bool(len(mesh.faces) >= 3 and mesh.is_watertight)
            payload["valid"] = bool(
                payload["is_valid"]
                and payload["volume"] > volume_tolerance
                and watertight
            )
            if mesh_output_path is not None and payload["valid"]:
                np.savez_compressed(
                    mesh_output_path,
                    vertices=np.asarray(mesh.vertices, dtype=np.float64),
                    faces=np.asarray(mesh.faces, dtype=np.int64),
                )
        except BaseException as error:
            payload["valid"] = False
            payload["error"] = _short_error(error, stage="mesh")
            child.send(payload)
            return
        if not payload["valid"]:
            payload["error"] = "invalid_shape"
        child.send(payload)
    except BaseException as error:
        # Python exceptions are reported. SIGSEGV/SIGABRT and hard exits bypass
        # this block and are recognized from the missing pipe payload by parent.
        payload["error"] = _short_error(error, stage="worker")
        try:
            child.send(payload)
        except BaseException:
            pass
    finally:
        child.close()


def execute_cadquery(
    source: str,
    *,
    timeout_s: float = 120.0,
    mesh_output_path: str | Path | None = None,
    step_output_path: str | Path | None = None,
    tessellation_tolerance: float = 0.1,
    volume_tolerance: float = 1e-6,
) -> CadExecutionResult:
    """Execute source in a fresh ``spawn`` child with a wall-clock timeout.

    When ``mesh_output_path`` is supplied, a valid result is exported as a
    neutral NumPy vertex/face archive. The output must not already exist, which
    prevents a failed process from being mistaken for a successful stale result.

    When ``step_output_path`` is supplied, any recovered shape (even a
    non-watertight one) is exported as STEP; ``step_ok`` reports whether the
    export succeeded.

    This is crash containment, not a security sandbox. Only evaluate completions
    from trusted datasets/models on an appropriately isolated machine.
    """

    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source)!r}")
    if timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    if tessellation_tolerance <= 0.0:
        raise ValueError("tessellation_tolerance must be positive")
    if volume_tolerance < 0.0:
        raise ValueError("volume_tolerance must be non-negative")
    output = None if mesh_output_path is None else Path(mesh_output_path)
    if output is not None:
        if output.exists():
            raise FileExistsError(f"mesh output already exists: {output}")
        if not output.parent.is_dir():
            raise FileNotFoundError(
                f"mesh output parent does not exist: {output.parent}"
            )
    step_output = None if step_output_path is None else Path(step_output_path)
    if step_output is not None:
        if step_output.exists():
            raise FileExistsError(f"step output already exists: {step_output}")
        if not step_output.parent.is_dir():
            raise FileNotFoundError(
                f"step output parent does not exist: {step_output.parent}"
            )

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_execute_worker,
        args=(
            source,
            child,
            None if output is None else str(output),
            None if step_output is None else str(step_output),
            float(tessellation_tolerance),
            float(volume_tolerance),
        ),
        daemon=False,
    )
    process.start()
    child.close()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        parent.close()
        process.close()
        return CadExecutionResult(error="timeout")

    payload: Mapping[str, Any] | None = None
    if parent.poll(0.25):
        try:
            received = parent.recv()
            if isinstance(received, Mapping):
                payload = received
        except (EOFError, OSError):
            payload = None
    parent.close()
    exit_code = process.exitcode
    process.close()
    if payload is None:
        return CadExecutionResult(error=f"process_exit:{exit_code}")
    return CadExecutionResult.from_mapping(payload)


__all__ = ["CadExecutionResult", "execute_cadquery"]
