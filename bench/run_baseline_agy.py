#!/usr/bin/env python3
"""LLM baseline via the **agy CLI agent** (Antigravity CLI), input = the drawing.

For each fixture we hand agy the clean multi-view drawing PNG and ask it to write
and execute a CadQuery script and export the solid to `output.step`, mirroring
what our route produces (same OCC kernel via CadQuery => fair shape comparison).

Per fixture:
  1. make a clean workdir, copy the fixture's drawing.png into it,
  2. run `agy -p "<prompt>" --dangerously-skip-permissions --add-dir <workdir>`
     with cwd=<workdir> (agy reads the PNG, writes a CadQuery script, EXECUTES
     it, exports the solid to output.step in the workdir),
  3. after agy exits, verify output.step exists and copy it into the results
     layout: results/agy_<modelslug>/<uuid>/output.step.
Per-fixture stdout/stderr + exit code go to <fixture>/agy.log; the generated
script and workdir are kept under <fixture>/work/ for debugging.

RESUMABLE: the run dir is date-less (results/agy_<modelslug>/) so re-running the
same command CONTINUES where it left off — fixtures that already have an
output.step are skipped (override with --force). If agy hits a usage/rate limit
mid-run, the loop STOPS cleanly (rather than burning the rest on errors); just
re-run later to pick up the remaining fixtures.

Then score with:  python bench/evaluate.py results/agy_<modelslug>

IMPORTANT — agy must be authenticated and headless auto-approval is required:
  * agy is NOT signed in out of the box. Run `agy` once interactively to sign
    in (and `agy models` to see model ids) BEFORE using this script; a headless
    run against an unauthenticated CLI just errors out (handled gracefully here).
  * `--dangerously-skip-permissions` auto-approves all tool calls; without it
    agy blocks on approval prompts and the headless run hangs. This flag can be
    denied by a harness auto-mode classifier — if so, launch this script from a
    shell where you control agy directly.

Usage:
    python bench/run_baseline_agy.py --dry-run --limit 1   # print cmd+prompt, run nothing
    python bench/run_baseline_agy.py --limit 1             # smoke (needs authed agy)
    python bench/run_baseline_agy.py --all --model <model-id>
    # ^ re-run the same line after a usage-limit stop to resume the remaining fixtures
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

AGY_BIN_DEFAULT = str(Path.home() / ".local" / "bin" / "agy")
# The output STEP the agent must produce, inside its workdir.
OUTPUT_STEP_NAME = "output.step"

# --- Filesystem sandbox (bwrap) --------------------------------------------
# The agent runs with --dangerously-skip-permissions and unrestricted reads. If
# its workdir sits inside the repo (or the repo is otherwise reachable), a
# curious agent will `find` the ground-truth STEP files (they live under
# experiments/ and bench/data/gt) and re-export the *answer* instead of solving
# the drawing — silent test-set contamination (observed with Gemini 3.5 Flash,
# 2026-07-08). `--sandbox-fs` runs agy inside a bubblewrap jail that mounts the
# whole FS read-only, HIDES the repo behind an empty tmpfs, and exposes only an
# opaque per-fixture workdir OUTSIDE the repo. The GT is then physically
# unreachable; the agent still sees the drawing.png staged into the workdir.
DEFAULT_WORK_ROOT = Path.home() / ".cache" / "agy_bench_work"
# agy's auth/config/cache — bound read-WRITE so token refresh still works inside
# the jail (the rest of the FS is read-only).
AGY_CONFIG_DIRS = [
    Path.home() / ".gemini",
    Path.home() / ".antigravity",
    Path.home() / ".config" / "Antigravity",
    Path.home() / ".cache" / "antigravity",
]


def bwrap_wrap(inner_cmd: list[str], workdir: Path, repo: Path,
               rw_paths: list[Path]) -> list[str]:
    """Wrap an agy command in a bubblewrap FS jail that hides `repo`.

    Whole FS read-only, `repo` masked by an empty tmpfs (so the GT under it is
    unreachable), `rw_paths` (+ `workdir`) rebound read-write, cwd=`workdir`.
    """
    args = ["bwrap",
            "--ro-bind", "/", "/",
            "--tmpfs", str(repo),      # mask the repo => GT is gone
            "--tmpfs", "/tmp",
            "--dev-bind", "/dev", "/dev",
            "--proc", "/proc"]
    for p in rw_paths:
        if Path(p).exists():
            args += ["--bind", str(p), str(p)]
    # workdir added last so it wins over the tmpfs/ro-bind above.
    args += ["--bind", str(workdir), str(workdir),
             "--chdir", str(workdir),
             "--die-with-parent"]
    return args + list(inner_cmd)


def _step_geom_lines(path: Path) -> list[str]:
    """STEP body lines minus the FILE_NAME header (which carries an export
    timestamp), so two exports of the *same* solid compare equal."""
    txt = path.read_text(errors="replace").splitlines()
    return [ln for ln in txt if not ln.startswith("FILE_NAME")]


def is_leaked_gt(candidate: Path, gt: Path,
                 tol_lines: int = 20, tol_frac: float = 0.005) -> bool:
    """True iff `candidate` is the ground-truth STEP re-exported.

    Integrity gate against test-set contamination: identical entity count and
    all-but-a-handful of lines identical (only header timestamp + sub-nm float
    rounding may differ). Runs regardless of --sandbox-fs, so a leak via any
    vector is caught post-hoc instead of scoring as a fake success.
    """
    if not (candidate.exists() and gt.exists()):
        return False
    try:
        c = _step_geom_lines(candidate)
        g = _step_geom_lines(gt)
    except Exception:
        return False
    if len(c) != len(g) or not g:
        return False
    diff = sum(1 for a, b in zip(c, g) if a != b)
    return diff <= max(tol_lines, int(tol_frac * len(g)))


def _copy_work_back(workdir: Path, fx: Path) -> None:
    """Copy an external (sandboxed) workdir back under the fixture dir for
    debugging (model.py, output.step, intermediate pngs)."""
    dst = fx / "work"
    try:
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(workdir, dst)
    except Exception:
        pass


def model_slug(model: str | None) -> str:
    if not model:
        return "default"
    return model.replace("/", "-").replace(":", "-").replace(" ", "-")


def build_prompt(cadquery_python: str) -> str:
    """The single headless prompt handed to agy for one fixture.

    Self-contained and explicit: agy runs one-shot headless, so the prompt must
    name the input file, the kernel (CadQuery, for fairness with our route), the
    interpreter to execute with, and the exact output path.
    """
    return (
        f"{C.TASK_DESCRIPTION}\n\n"
        f"The multi-view engineering drawing is the image file `{C.DRAWING_NAME}` "
        f"in the current working directory. It contains standard orthographic "
        f"projections (and possibly section/detail views) with all dimensions in "
        f"millimetres.\n\n"
        f"Do the following, entirely within the current working directory:\n"
        f"1. Open and study `{C.DRAWING_NAME}` to recover the 3D geometry and "
        f"dimensions.\n"
        f"2. Write a Python script `model.py` that builds the part with **CadQuery** "
        f"(import cadquery as cq). Use CadQuery, NOT build123d or OpenSCAD, so the "
        f"result uses the same OpenCASCADE kernel as the reference.\n"
        f"3. The script must export the resulting single solid to a STEP file named "
        f"exactly `{OUTPUT_STEP_NAME}` in the current working directory, e.g. "
        f"`cq.exporters.export(result, '{OUTPUT_STEP_NAME}')`.\n"
        f"4. EXECUTE the script with this exact interpreter (it has CadQuery "
        f"installed): `{cadquery_python} model.py`. If it errors, read the "
        f"traceback, fix `model.py`, and re-run until `{OUTPUT_STEP_NAME}` is "
        f"written successfully.\n"
        f"5. Confirm `{OUTPUT_STEP_NAME}` exists before finishing.\n\n"
        f"Match the drawing's geometry and dimensions as closely as you can. "
        f"Produce exactly one solid."
    )


def agy_command(agy_bin: str, prompt: str, workdir: Path, model: str | None,
                print_timeout_s: float, sandbox: bool) -> list[str]:
    cmd = [
        agy_bin,
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--add-dir", str(workdir),
        # agy expects a Go duration string; whole seconds is unambiguous.
        "--print-timeout", f"{int(print_timeout_s)}s",
    ]
    if model:
        assert model in [
            "Gemini 3.5 Flash (Medium)",
            "Gemini 3.5 Flash (High)",
            "Gemini 3.5 Flash (Low)",
            "Gemini 3.1 Pro (Low)",
            "Gemini 3.1 Pro (High)",
        ], f"Unsupported model: {model}"
        cmd += ["--model", model]
    if sandbox:
        cmd.append("--sandbox")
    return cmd


def looks_unauthenticated(stdout: str, stderr: str) -> bool:
    blob = (stdout + "\n" + stderr).lower()
    needles = ("not signed in", "not authenticated", "sign in", "please log in",
               "please login", "unauthorized", "authentication required",
               "no credentials", "run `agy`", "authentication failed")
    return any(n in blob for n in needles)


def looks_rate_limited(stdout: str, stderr: str) -> bool:
    """Detect a usage/quota/rate limit so the loop can stop cleanly and resume later
    (rather than burning every remaining fixture on the same error)."""
    blob = (stdout + "\n" + stderr).lower()
    needles = ("usage limit", "rate limit", "rate-limit", "quota", "resource exhausted",
               "too many requests", "429", "limit reached", "reached your limit",
               "you've reached", "you have reached", "try again later",
               "insufficient_quota", "over capacity")
    return any(n in blob for n in needles)


def run_one(uuid: str, run_dir: Path, agy_bin: str, model: str | None,
            print_timeout_s: float, wall_timeout_s: float, sandbox: bool,
            cadquery_python: str, dry_run: bool,
            sandbox_fs: bool = False, work_root: Path | None = None) -> str:
    in_dir = C.INPUTS_DIR / uuid
    drawing = in_dir / C.DRAWING_NAME
    fx = run_dir / uuid
    if sandbox_fs:
        # Opaque workdir OUTSIDE the repo (the repo is masked inside the jail).
        # Opaque (not the uuid) so the agent has no key to search other stores by.
        import secrets
        workdir = Path(work_root or DEFAULT_WORK_ROOT) / secrets.token_hex(8)
    else:
        workdir = fx / "work"

    prompt = build_prompt(cadquery_python)
    cmd = agy_command(agy_bin, prompt, workdir, model, print_timeout_s, sandbox)
    if sandbox_fs:
        cmd = bwrap_wrap(cmd, workdir, C.REPO, AGY_CONFIG_DIRS)

    if dry_run:
        print(f"\n===== fixture {uuid} =====")
        print(f"# cwd: {workdir}  (sandbox_fs={sandbox_fs})")
        print(f"# copy: {drawing} -> {workdir / C.DRAWING_NAME}")
        # shell-quote for copy-paste fidelity
        import shlex
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        print("\n----- PROMPT -----")
        print(prompt)
        print("----- END PROMPT -----")
        return "dry-run"

    if not drawing.exists():
        print(f"[agy] {uuid}: missing drawing {drawing}, skipping", file=sys.stderr)
        return "no-input"

    # clean workdir + stage the drawing
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drawing, workdir / C.DRAWING_NAME)

    log_path = fx / "agy.log"
    print(f"[agy] {uuid}: running (model={model_slug(model)}, "
          f"print-timeout={int(print_timeout_s)}s, wall={int(wall_timeout_s)}s)")
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True,
                              text=True, timeout=wall_timeout_s)
        rc = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc = -1
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + "\n[agy] wall-clock timeout"
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    except FileNotFoundError:
        print(f"[agy] agy binary not found at {agy_bin}. Install/point --agy-bin.",
              file=sys.stderr)
        return "no-agy"
    dt = time.time() - t0

    log_path.write_text(
        f"$ {' '.join(cmd)}\n(cwd={workdir}, exit={rc}, {dt:.1f}s, "
        f"timed_out={timed_out})\n\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}\n"
    )

    if looks_unauthenticated(stdout, stderr):
        print(f"[agy] {uuid}: agy appears UNAUTHENTICATED — run `agy` once to "
              f"sign in, then retry. (see {log_path})", file=sys.stderr)
        return "unauth"

    produced = workdir / OUTPUT_STEP_NAME
    if produced.exists() and produced.stat().st_size > 0:
        # Integrity gate: reject a candidate that is just the GT re-exported.
        gt = C.GT_DIR / uuid / C.GT_STEP_NAME
        if is_leaked_gt(produced, gt):
            print(f"[agy] {uuid}: REJECTED — output.step is the ground truth "
                  f"re-exported (test-set contamination). Not accepted as a result. "
                  f"Use --sandbox-fs. (see {log_path})", file=sys.stderr)
            if sandbox_fs:
                _copy_work_back(workdir, fx)   # keep artefacts for forensics
            return "leaked"
        shutil.copy2(produced, fx / "output.step")
        if sandbox_fs:
            _copy_work_back(workdir, fx)        # external workdir -> keep for debugging
        print(f"[agy] {uuid}: output.step ({produced.stat().st_size} B, {dt:.1f}s)")
        return "step"

    # No STEP. Distinguish a usage/rate limit (caller should STOP + resume later)
    # from an ordinary failure (agy tried but produced nothing).
    if looks_rate_limited(stdout, stderr):
        print(f"[agy] {uuid}: hit a USAGE/RATE LIMIT — stopping so you can resume "
              f"later. (see {log_path})", file=sys.stderr)
        return "limit"
    print(f"[agy] {uuid}: no output.step produced (exit={rc}, {dt:.1f}s). "
          f"See {log_path}.", file=sys.stderr)
    return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Gemini 3.1 Pro (High)",
                    help="agy model id (see `agy models`)")
    parser.add_argument("--ids", nargs="*", default=None,
                    help="explicit fixture uuids to run")
    parser.add_argument("--limit", type=int, default=None,
                    help="cap number of fixtures (smoke tests)")
    parser.add_argument("--all", action="store_true",
                    help="run every fixture in the manifest")
    parser.add_argument("--print-timeout", type=float, default=900.0,
                    help="agy --print-timeout, seconds (default 900; sent as Ns)")
    parser.add_argument("--wall-timeout", type=float, default=None,
                    help="hard subprocess wall-clock cap, seconds "
                         "(default: print-timeout + 120)")
    parser.add_argument("--sandbox", action="store_true",
                    help="pass agy --sandbox (restricted terminal)")
    parser.add_argument("--sandbox-fs", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="run agy inside a bubblewrap FS jail that HIDES the repo "
                         "(so the ground-truth STEP files are unreachable and the "
                         "agent cannot read the answer off disk). Uses an opaque "
                         "workdir under --work-root, OUTSIDE the repo. Requires bwrap. "
                         "**ON BY DEFAULT** — pass --no-sandbox-fs to disable (⚠️ then "
                         "the agent can and does find + re-export the GT: contamination).")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT),
                    help=f"parent dir for per-fixture --sandbox-fs workdirs; MUST be "
                         f"outside the repo (default {DEFAULT_WORK_ROOT})")
    parser.add_argument("--agy-bin", default=AGY_BIN_DEFAULT,
                    help=f"path to the agy binary (default {AGY_BIN_DEFAULT})")
    parser.add_argument("--cadquery-python", default=sys.executable,
                    help="interpreter agy is told to execute the script with "
                         "(default the current drawing2cad env, which includes "
                         "CadQuery/OCC)")
    parser.add_argument("--run-name", default=None,
                    help="results/<run_name>/ (default agy_<modelslug>, date-less so "
                         "re-runs resume in place)")
    parser.add_argument("--force", action="store_true",
                    help="re-run fixtures even if they already have an output.step "
                         "(default: skip done ones = resume)")
    parser.add_argument("--dry-run", action="store_true",
                    help="print the exact agy command + prompt per fixture and "
                         "invoke nothing")
    args = parser.parse_args()

    if not C.INPUTS_DIR.exists():
        raise SystemExit(f"No fixtures at {C.INPUTS_DIR}. Run build_fixtures.py.")

    ids = args.ids
    if ids is None:
        ids = C.manifest_ids()
    if not args.all and args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit("No ids selected.")

    if not args.dry_run and not Path(args.agy_bin).exists():
        print(f"[agy] WARNING: agy binary not found at {args.agy_bin}. "
              f"Install it or pass --agy-bin; runs will report 'no-agy'.",
              file=sys.stderr)

    wall = args.wall_timeout if args.wall_timeout is not None else args.print_timeout + 120.0

    work_root = Path(args.work_root)
    if args.sandbox_fs:
        if shutil.which("bwrap") is None:
            raise SystemExit("--sandbox-fs requires bubblewrap (bwrap), not found "
                             "on PATH. Install it (apt install bubblewrap) or drop "
                             "--sandbox-fs (⚠️ then the agent can read the GT).")
        # The work-root MUST be outside the repo, else masking the repo hides it.
        if C.REPO in work_root.resolve().parents or work_root.resolve() == C.REPO:
            raise SystemExit(f"--work-root must be OUTSIDE the repo ({C.REPO}); "
                             f"got {work_root}.")
        if not args.dry_run:
            work_root.mkdir(parents=True, exist_ok=True)
        print(f"[agy] sandbox-fs ON (bwrap): repo hidden, workdirs under {work_root}")
    else:
        print("[agy] ⚠️ sandbox-fs OFF: the agent can read the ground-truth STEP "
              "files off disk. Pass --sandbox-fs to prevent contamination.",
              file=sys.stderr)

    run_name = args.run_name or f"agy_{model_slug(args.model)}"
    run_dir = C.RESULTS_DIR / run_name
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[agy] results -> {run_dir}")

    def is_done(uuid: str) -> bool:
        step = run_dir / uuid / "output.step"
        return step.exists() and step.stat().st_size > 0

    statuses: dict[str, str] = {}
    n_skipped = 0
    stopped_on_limit = False
    for uuid in ids:
        if not args.dry_run and not args.force and is_done(uuid):
            statuses[uuid] = "skip-done"
            n_skipped += 1
            continue
        statuses[uuid] = run_one(
            uuid, run_dir, args.agy_bin, args.model, args.print_timeout, wall,
            args.sandbox, args.cadquery_python, args.dry_run,
            sandbox_fs=args.sandbox_fs, work_root=work_root,
        )
        if statuses[uuid] == "limit":
            stopped_on_limit = True
            print("[agy] usage/rate limit reached — stopping. Re-run the same "
                  "command later to resume the remaining fixtures.", file=sys.stderr)
            break

    if args.dry_run:
        print(f"\n[agy] dry-run: {len(ids)} fixture(s), invoked nothing.")
        return 0

    import json
    # Cumulative truth = count output.step across the WHOLE run dir (survives resumes),
    # not just this invocation's statuses.
    n_step_total = sum(1 for uuid in ids if is_done(uuid))
    n_step_this = sum(1 for s in statuses.values() if s == "step")
    n_leaked = sum(1 for s in statuses.values() if s == "leaked")
    # Merge per-fixture statuses with any prior run_meta so the record is cumulative.
    meta_path = run_dir / "run_meta.json"
    prior = {}
    if meta_path.exists():
        try:
            prior = json.loads(meta_path.read_text()).get("per_fixture", {})
        except Exception:
            prior = {}
    merged = {**prior, **statuses}
    meta_path.write_text(json.dumps({
        "run_name": run_name,
        "system": "baseline_agy",
        "agy_bin": args.agy_bin,
        "model": args.model,
        "cadquery_python": args.cadquery_python,
        "print_timeout_s": args.print_timeout,
        "wall_timeout_s": wall,
        "n_fixtures": len(ids),
        "n_output_step": n_step_total,
        "last_run": {"n_step": n_step_this, "n_skipped_done": n_skipped,
                     "n_leaked_rejected": n_leaked,
                     "stopped_on_limit": stopped_on_limit,
                     "sandbox_fs": args.sandbox_fs},
        "per_fixture": merged,
    }, indent=2))
    print(f"[agy] this run: +{n_step_this} step, {n_skipped} already done | "
          f"cumulative {n_step_total}/{len(ids)} have output.step -> {run_dir}")
    if n_leaked:
        print(f"[agy] ⚠️ {n_leaked} fixture(s) REJECTED as GT-leak (agent re-exported "
              f"the ground truth). Re-run those with --sandbox-fs.", file=sys.stderr)
    if stopped_on_limit:
        print("[agy] stopped early on a usage limit — re-run to resume.", file=sys.stderr)
    if any(s == "unauth" for s in statuses.values()):
        print("[agy] some fixtures reported UNAUTHENTICATED — run `agy` once to "
              "sign in, then re-run.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
