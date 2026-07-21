"""Batch renderer: STEP files -> ECCV-style techdraw (svg/dxf/pdf) + render_3d PNGs.

Usage:
    python src/data/render/render_dataset.py --input_dir <INPUT> --output_dir <OUTDIR> \
        [--workers N] [--timeout SEC] [--limit N] [--no-render3d] [--no-techdraw]

INPUT is a flat folder containing ``{stem}.step`` files (``{stem}.cadquery.py``
files may sit alongside; they are ignored by the renderer). OUTDIR mirrors the
layout of data/eccv2026-cad-challenge-data:

    OUTDIR/techdraw/{svg,dxf,pdf}/{stem}.*
    OUTDIR/render_3d/{hlg_perspective,transparent_shaded_edges_perspective,
                      hlg_translucent_faces_perspective}/{stem}.png

Each part runs in its own process with a wall-clock timeout because OCC HLR can
hang forever in native code (observed with FreeCAD TechDraw HLR as well); a
try/except cannot recover from that, only SIGKILL can. Re-running with the same
OUTDIR resumes: parts whose manifest entry exists (including current techdraw
metadata when techdraw output is requested) and whose outputs are present are
skipped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.render.config import (  # noqa: E402
    PartResult,
    Render3dPaths,
    TechdrawPaths,
    render3d_paths,
    techdraw_paths,
)

MANIFEST_NAME = "manifest.jsonl"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _outputs_exist(td: TechdrawPaths | None, r3: Render3dPaths | None) -> bool:
    paths = []
    if td is not None:
        paths += [td.svg, td.dxf, td.pdf]
    if r3 is not None:
        paths += [r3.hlg, r3.shaded, r3.hlg_translucent]
    return all(p.exists() for p in paths)


def _worker(
    step_path: Path, td: TechdrawPaths | None, r3: Render3dPaths | None, queue: mp.Queue
) -> None:
    """Runs in a child process; all OCC work happens here."""
    t0 = time.time()
    result = PartResult(name=step_path.stem, ok=True)
    try:
        if td is not None:
            from src.data.render.techdraw import generate_techdraw

            result.extra["techdraw"] = generate_techdraw(step_path, td)
            result.techdraw_ok = True
        if r3 is not None:
            from src.data.render.render3d import generate_render3d

            generate_render3d(step_path, r3)
            result.render3d_ok = True
    except Exception as exc:  # noqa: BLE001 — recorded in the manifest
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = round(time.time() - t0, 2)
    queue.put(result.__dict__)


def _load_done(manifest: Path, *, require_techdraw_info: bool = False) -> set[str]:
    done: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            extra = rec.get("extra")
            has_techdraw_info = isinstance(extra, dict) and isinstance(
                extra.get("techdraw"), dict
            )
            if rec.get("ok") and (not require_techdraw_info or has_techdraw_info):
                done.add(rec["name"])
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-part wall-clock timeout in seconds",
    )
    ap.add_argument("--limit", type=int, default=0, help="render at most N parts")
    ap.add_argument("--no-render3d", action="store_true")
    ap.add_argument("--no-techdraw", action="store_true")
    args = ap.parse_args(argv)

    steps = sorted(args.input_dir.glob("*.step")) + sorted(args.input_dir.glob("*.stp"))
    if args.limit:
        steps = steps[: args.limit]
    if not steps:
        print(f"no .step files found in {args.input_dir}", file=sys.stderr)
        return 1

    out = args.output_dir
    for sub in ("svg", "dxf", "pdf"):
        (out / "techdraw" / sub).mkdir(parents=True, exist_ok=True)
    for sub in (
        "hlg_perspective",
        "transparent_shaded_edges_perspective",
        "hlg_translucent_faces_perspective",
    ):
        (out / "render_3d" / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = out / MANIFEST_NAME
    done = _load_done(
        manifest_path,
        require_techdraw_info=not args.no_techdraw,
    )
    manifest = manifest_path.open("a")

    todo = []
    for step in steps:
        td = None if args.no_techdraw else techdraw_paths(out, step.stem)
        r3 = None if args.no_render3d else render3d_paths(out, step.stem)
        if step.stem in done and _outputs_exist(td, r3):
            continue
        todo.append((step, td, r3))

    print(
        f"{len(steps)} parts, {len(steps) - len(todo)} already done, "
        f"{len(todo)} to render, workers={args.workers}, timeout={args.timeout}s"
    )

    ctx = mp.get_context("spawn")  # OCC state must not be forked
    live: dict[int, tuple] = {}  # pid -> (proc, queue, name, deadline)
    n_ok = n_fail = 0
    queue_iter = iter(todo)

    def _reap(proc, queue, name) -> None:
        nonlocal n_ok, n_fail
        rec = None
        try:
            rec = queue.get_nowait()
        except Exception:  # noqa: BLE001
            pass
        if rec is None:
            rec = PartResult(
                name=name,
                ok=False,
                error="TimeoutError: killed (native hang, likely OCC HLR)",
            ).__dict__
        n_ok += bool(rec["ok"])
        n_fail += not rec["ok"]
        manifest.write(json.dumps(rec) + "\n")
        manifest.flush()
        status = "ok" if rec["ok"] else f"FAIL {rec['error']}"
        print(f"[{n_ok + n_fail}/{len(todo)}] {name}: {status}")

    while True:
        # top up
        while len(live) < args.workers:
            item = next(queue_iter, None)
            if item is None:
                break
            step, td, r3 = item
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(step, td, r3, q), daemon=True)
            p.start()
            live[p.pid] = (p, q, step.stem, time.time() + args.timeout)
        if not live:
            break
        time.sleep(0.2)
        for pid in list(live):
            proc, q, name, deadline = live[pid]
            if not proc.is_alive():
                proc.join()
                _reap(proc, q, name)
                del live[pid]
            elif time.time() > deadline:
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                _reap(proc, q, name)
                del live[pid]

    manifest.close()
    print(f"done: {n_ok} ok, {n_fail} failed -> {out}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
