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

INPUT VARIANTS (representation ablation): --input png (default) hands agy the
drawing PNG. --input graph hands it `graph.txt` instead — EXACTLY the serialized
AMVDG text our model receives (scripts/train3d/serialize.py::serialize_3d with
the same quant as run_ours.py) — plus a format legend in the prompt. Same task,
same fixtures, only the input representation varies => measures whether the
AMVDG IR itself adds value for a frontier model. Graph results land in a
separate namespace results/agy-graph_<modelslug>/ (agy-graph-{K}shot_... with
--shots K in 1..3, which prepend K worked in-context examples; K=1 keeps the
literal agy-graph-1shot_...) so report.py pairing can never mix modes.
--max-graph-bytes skips oversized graphs as `too_large` WITHOUT invoking agy
(quota guard).

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
import json
import os
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
# --input graph: the serialized AMVDG text staged into the workdir.
GRAPH_NAME = "graph.txt"

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
               rw_paths: list[Path],
               ro_binds: list[tuple[str, str]] | None = None) -> list[str]:
    """Wrap an agy command in a bubblewrap FS jail that hides `repo`.

    Whole FS read-only, `repo` masked by an empty tmpfs (so the GT under it is
    unreachable), `rw_paths` (+ `workdir`) rebound read-write, cwd=`workdir`.

    `ro_binds` = (src, dest) pairs rebound READ-ONLY *after* the tmpfs mask
    (later mounts win), used to restore a repo-internal interpreter/venv that the
    mask would otherwise hide — see repo_venv_binds(). These carry no GT, so the
    ground truth under the repo stays masked.
    """
    args = ["bwrap",
            "--ro-bind", "/", "/",
            "--tmpfs", str(repo),      # mask the repo => GT is gone
            "--tmpfs", "/tmp",
            "--dev-bind", "/dev", "/dev",
            "--proc", "/proc"]
    # AFTER the tmpfs mask so these win: re-expose only the pinned venv root.
    for src, dest in (ro_binds or []):
        if Path(src).exists():
            args += ["--ro-bind", str(src), str(dest)]
    for p in rw_paths:
        if Path(p).exists():
            args += ["--bind", str(p), str(p)]
    # workdir added last so it wins over the tmpfs/ro-bind above.
    args += ["--bind", str(workdir), str(workdir),
             "--chdir", str(workdir),
             "--die-with-parent"]
    return args + list(inner_cmd)


def repo_venv_binds(cadquery_python: str, repo: Path) -> list[tuple[str, str]]:
    """Read-only bind(s) needed so a repo-internal interpreter survives the jail.

    The bwrap tmpfs masks the whole repo, so if agy is told to execute the
    script with an interpreter whose INVOCATION path is inside the repo (e.g.
    `<repo>/.venv/bin/python`), that path vanishes inside the jail and the run
    can't `import cadquery`. We key on the *unresolved* invocation path, not its
    realpath: here `<repo>/.venv` is a symlink to a conda env OUTSIDE the repo,
    so realpath(interp) is outside — yet the path agy uses is masked all the
    same. Returns [(realsrc, dest)] where dest = the venv root as invoked (the
    masked path to restore) and realsrc = its realpath (the actual store, to
    bind from). Binding only the venv root re-exposes both its bin/ and its
    site-packages while the GT (bench/data/gt, experiments/) stays masked.
    Empty when the interpreter lives outside the repo (already reachable via the
    read-only `/` bind) or the path is relative/unusable.
    """
    interp = Path(cadquery_python)
    if not interp.is_absolute():
        return []
    root = interp.parent.parent                      # venv layout: <root>/bin/py
    under_repo = root == repo or repo in root.parents   # UNRESOLVED path check
    if not under_repo:
        return []
    real = os.path.realpath(str(root))
    if not os.path.isdir(real):
        return []
    return [(real, str(root))]


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


# --- Graph-input mode (representation ablation) ------------------------------
# The graph baseline must feed the agent EXACTLY the text our model receives, so
# graph.txt is produced by the SAME function with the SAME arguments infer.py
# uses: serialize_3d(json.load(open(graph_json)), quant=<quant>) from
# scripts/train3d/serialize.py. serialize.py is stdlib-only, but we still run it
# under a designated interpreter (--serialize-python, default $DRAWING2CAD_PY or
# this python) out-of-process, per the bench convention of shelling out instead
# of importing (common.py).
GRAPH_TASK_DESCRIPTION = (
    "Reconstruct the 3D CAD part described by a multi-view engineering drawing. "
    "The drawing is given NOT as an image but as a plain-text serialization of "
    "its views' geometry, with dimensions in millimetres. Build a single solid "
    "whose geometry and dimensions match the drawing as closely as you can."
)

# Format legend for the serialized AMVDG text, derived from
# scripts/train3d/serialize.py (struct_to_text/_geom_str/_quantize_struct).
# Coverage over the actual bench corpus is asserted mechanically (see the
# legend-coverage check in the graph-baseline docs).
GRAPH_FORMAT_LEGEND = """\
The file is a line-based text serialization of a multi-view engineering
drawing (orthographic views, third-angle). One record per line, in order:
PART header, FEAT features, BLEND edge blends (if any), VIEW blocks of 2D
geometry, DIM dimensions. Tokens are space-separated.

PART unit=mm bbox=<X>,<Y>,<Z> grid=<n> ext=<E>
  bbox = the part's 3D bounding-box extents along model axes X,Y,Z in EXACT
  mm. grid/ext define integer coordinate quantization (see COORDINATES).
FEAT C<k> <kind> axis=<X|Y|Z> r=<mm|-> [dir=<axis>]
  A 3D feature, e.g. kind=cylinder (a hole or boss). axis = its 3D axis
  direction; r = its radius in EXACT mm ('-' if unknown). Geometry rows
  tagged C<k> below are the 2D curves this feature produces in each view.
BLEND <fillet|chamfer> <edges> v<mm|->
  An edge blend applied to the part, radius/setback v in EXACT mm; <edges> =
  all | par:<axis> (edges parallel to axis) | face:<+axis|-axis> | complex.
VIEW <name> axes=<A>,<B>
  Starts a view (front|top|right). Rows until the next VIEW/DIM belong to
  it. axes = which 3D model axes the view shows: the first is the view's x
  (first number of each coordinate pair), the second its y. Here:
  front axes=X,Z, top axes=X,Y, right axes=Y,Z.

Geometry rows inside a VIEW (coordinates are view-local 2D = positions along
the two named model axes):
L <role> <x1>,<y1> <x2>,<y2> [C-tags] [DIA<mm>]     line segment
PL <role> <x1>,<y1> <x2>,<y2> ... <xn>,<yn> [...]   polyline (joined segments)
CI <role> c<x>,<y> r<r> [C-tags] [DIA<mm>]          circle: center + radius
A <role> c<x>,<y> r<r> a<a1>..<a2> [...]            circular arc: runs from
    angle a1 to a2 counterclockwise (degrees from the +x axis toward +y)
EL <role> c<x>,<y> rj<rmaj> rn<rmin> rot<t> [e<t1>..<t2>] [...]  ellipse:
    major/minor radii, major-axis angle t (deg); optional arc span given as
    parametric (eccentric) angles t1..t2, increasing
PT <role> <x>,<y>                                   point

<role>: vis = visible edge; hid = hidden (dashed) edge; vh = a visible and a
  hidden edge coincide (treat as visible); cen = centerline; pha = phantom;
  sec = section cut; brk = break line. Hidden lines fully covered by
  collinear visible lines are omitted; exact duplicate rows are merged.
[C-tags]: comma-separated feature memberships, e.g. C1 or C2,C5 — the curve
  belongs to those FEAT features. Match tags across views to correspond the
  same 3D feature (a hole's circle in top <-> its silhouette lines in front).
DIA<mm>: a diameter annotation attached to this row, EXACT mm (= 2*radius).

DIM d<i> <kind> <role> <value> <view> <axis><lo>..<hi>
  A dimension annotation: <value> is EXACT mm (kind e.g. linear; role e.g.
  width/height/depth). The last field is the measured span: a model-axis
  letter + start..end in quantized view coordinates. Fields may be '-'.

COORDINATES / RECOVERING mm:
  Each view is translated so the part's minimum is 0 on both its axes, and
  the SAME model coordinate prints as the SAME number in every view sharing
  that axis (e.g. a hole-center X in front equals its X in top). Views thus
  assemble directly into one part frame [0,X]x[0,Y]x[0,Z] (bbox, mm).
  With grid=<n> ext=<E> in the header, all row coordinates, circle/arc/
  ellipse radii, and DIM spans are integers on ONE shared grid across all
  views:  mm = value * E / (n - 1).  Angles are already degrees. DIM
  <value>, DIA<mm>, FEAT r=, and PART bbox= are EXACT mm (not quantized) —
  use those for exact dimensions and the grid for positions/proportions.
  Centerlines may extend slightly past the part (even to negative values)."""

_SERIALIZE_SNIPPET = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from serialize import serialize_3d
sys.stdout.write(serialize_3d(json.load(open(sys.argv[2])), quant=int(sys.argv[3])))
"""


def serialize_graph(serialize_python: str, graph_json: Path, quant: int) -> str:
    """{uuid}.graph.json -> the exact serialize_3d(quant=) text our model
    receives (mirrors infer.py's call; see the block comment above)."""
    proc = subprocess.run(
        [serialize_python, "-c", _SERIALIZE_SNIPPET,
         str(C.REPO / "scripts" / "train3d"), str(graph_json), str(quant)],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"serialize_3d failed for {graph_json.name}: "
                           f"{proc.stderr.strip()[-400:]}")
    return proc.stdout


# --shots K worked-example sources — the {uuid}.graph.json pool and the paired
# blend-stripped {uuid}.cadquery.py solutions — plus any extra never-a-fixture
# exclusions are all set on the CLI (--shot-graph-dir / --shot-code-dir /
# --exclude-ids-from; see main()). The ACTIVE bench's own fixtures are always
# excluded regardless (bench_ids_to_exclude), so nothing transient is hardcoded.

# Deterministic k-shot selection, run under serialize-python (has serialize_3d):
#   * candidates = never-a-bench-fixture graphs (from --shot-graph-dir) that
#     also have a blend-stripped GT CadQuery (in --shot-code-dir), by a cheap
#     filename set-intersection (no serialize).
#   * cap: if the candidate pool exceeds --shot-pool-cap, take a deterministic
#     uniform subsample (seed 0) BEFORE serializing. The train split (~9e4) is
#     far too big to serialize whole (~hours); a small pool (< cap, e.g. the val
#     set) is used in full, so its behavior is unchanged.
#   * eligible = the (sub)sampled candidates, serialized and sorted by length.
#   * example 1 = the shortest of the pool.
#   * the remaining k-1 span complexity: picks at evenly spaced percentiles
#     i/k (k=3 -> ~p33/~p67) of the eligible pool, so the model sees a simple,
#     a medium and a complex worked example, not three tiny ones.
#   * byte budget: the total of (serialized graph text + GT code) bytes across
#     chosen examples must stay <= max_shot_bytes; a percentile pick that would
#     blow it walks DOWN to the next shorter still-eligible graph until it fits.
#   * output = chosen examples ordered shortest->longest.
_SHOT_SELECT_SNIPPET = r"""
import glob, json, os, random, sys
sys.path.insert(0, sys.argv[1])
from serialize import serialize_3d
graph_dir, noblend_dir = sys.argv[2], sys.argv[3]
quant, k, budget, cap = (int(sys.argv[4]), int(sys.argv[5]),
                         int(sys.argv[6]), int(sys.argv[7]))
excl = set(json.loads(sys.stdin.read()))
# Cheap pass (no serialize): uuids with BOTH a graph and a blend-stripped GT
# CadQuery, minus the excluded bench fixtures.
graphs = {os.path.basename(p)[: -len(".graph.json")]
          for p in glob.glob(os.path.join(graph_dir, "*.graph.json"))}
codes = {os.path.basename(p)[: -len(".cadquery.py")]
         for p in glob.glob(os.path.join(noblend_dir, "*.cadquery.py"))}
cand = sorted((graphs & codes) - excl)
if not cand:
    sys.exit(3)
# Bound the expensive serialize step for a huge pool (train ~9e4); a uniform
# seed-0 subsample keeps the simple/medium/complex percentile pick representative.
if cap and len(cand) > cap:
    cand = sorted(random.Random(0).sample(cand, cap))
elig = []  # (uuid, text, code)
for uuid in cand:
    try:
        text = serialize_3d(
            json.load(open(os.path.join(graph_dir, uuid + ".graph.json"))),
            quant=quant)
    except Exception:
        continue
    code = open(os.path.join(noblend_dir, uuid + ".cadquery.py")).read()
    elig.append((uuid, text, code))
if not elig:
    sys.exit(3)
elig.sort(key=lambda e: len(e[1]))          # stable: ties keep sampled order
n = len(elig)
cost = [len(t.encode("utf-8")) + len(c.encode("utf-8")) for (_u, t, c) in elig]
chosen = [0]                                # example 1 = shortest, always taken
used = cost[0]
for i in range(1, k):
    idx = int(round((i / k) * (n - 1)))     # evenly spaced percentile
    while idx >= 0 and (idx in chosen or used + cost[idx] > budget):
        idx -= 1                            # walk DOWN to a shorter eligible pick
    if idx >= 0 and idx not in chosen and used + cost[idx] <= budget:
        chosen.append(idx)
        used += cost[idx]
chosen.sort()                               # indices asc == shortest->longest
json.dump([{"uuid": elig[j][0], "text": elig[j][1], "code": elig[j][2],
            "bytes": cost[j]} for j in chosen], sys.stdout)
"""


def _read_id_file(path: Path) -> set[str]:
    """Parse an id source into a set of uuids. Accepts a cadgenbench-style
    manifest (JSON dict with an "ids" list), a bare JSON list, or a plain
    newline-delimited id list — so both a manifest.json and a scratch ids.txt
    work. An unparseable file yields no ids."""
    try:
        txt = path.read_text()
    except OSError:
        return set()
    try:
        obj = json.loads(txt)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        return set(obj.get("ids", []))
    if isinstance(obj, list):
        return set(obj)
    return {ln.strip() for ln in txt.splitlines() if ln.strip()}


def bench_ids_to_exclude(extra_files: list[Path] | None = None) -> set[str]:
    """Uuids that must never be handed to the model as a worked example.

    Always excludes the ACTIVE bench's fixtures (C.manifest_ids(), read from
    data/manifest.json), so this auto-tracks whichever bench data/ points at —
    including after a data->data_new migration — with no path baked in. Plus the
    ids in any `extra_files` (each a manifest.json or a plain id list, e.g.
    another bench corpus passed via --exclude-ids-from). Missing files are
    skipped."""
    ids: set[str] = set(C.manifest_ids())
    for f in (extra_files or []):
        if f.exists():
            ids |= _read_id_file(f)
    return ids


def select_shot_examples(serialize_python: str, quant: int, k: int,
                         max_shot_bytes: int, shot_graph_dir: Path,
                         shot_code_dir: Path,
                         exclude_files: list[Path] | None = None,
                         shot_pool_cap: int = 500) -> list[dict]:
    """--shots K: pick k deterministic worked examples from `shot_graph_dir`
    (a pool of {uuid}.graph.json, e.g. the Z2C train set), each (a) never a
    bench fixture (bench_ids_to_exclude(exclude_files)) and (b) with a blend-
    stripped GT CadQuery under `shot_code_dir` ({uuid}.cadquery.py — blend-free
    matches our model's output convention). A pool larger than `shot_pool_cap`
    is uniformly subsampled (seed 0) before serialization, so a huge source like
    the train split (~9e4) stays ~20 s instead of hours. Example 1 = the
    shortest of the pool; the rest span complexity at evenly spaced percentiles,
    under a total byte budget (see _SHOT_SELECT_SNIPPET)."""
    excl = bench_ids_to_exclude(exclude_files)
    proc = subprocess.run(
        [serialize_python, "-c", _SHOT_SELECT_SNIPPET,
         str(C.REPO / "scripts" / "train3d"), str(shot_graph_dir),
         str(shot_code_dir), str(quant), str(k), str(max_shot_bytes),
         str(shot_pool_cap)],
        input=json.dumps(sorted(excl)), capture_output=True, text=True,
        timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(
            f"[agy] --shots {k}: no eligible worked example ({shot_graph_dir} x "
            f"{shot_code_dir}, {len(excl)} ids excluded): "
            f"{proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout)


def build_graph_prompt(cadquery_python: str,
                       shots: list[dict] | None = None) -> str:
    """The single headless prompt for one fixture in --input graph mode.

    Same task framing as PNG mode (CadQuery model.py, execute with the pinned
    interpreter, export output.step) but the input is the serialized AMVDG
    text `graph.txt`, so the prompt carries the format legend. `shots` (from
    select_shot_examples, ordered shortest->longest) prepends 0..3 worked
    examples as in-context demonstrations. An empty/None list yields the
    zero-shot prompt BYTE-IDENTICAL to the original control."""
    shots = shots or []
    example = ""
    if shots:
        n = len(shots)
        parts = [
            f"Below {'is' if n == 1 else 'are'} {n} worked "
            f"example{'' if n == 1 else 's'} from DIFFERENT parts (same input "
            f"format -> CadQuery), ordered from simplest to most complex. Each "
            f"example script builds the finished solid in the variable "
            f"`result`; your script must ALSO export it to `{OUTPUT_STEP_NAME}` "
            f"as instructed below.\n"
        ]
        for i, shot in enumerate(shots, 1):
            parts.append(
                f"=== Example {i}: input graph ===\n{shot['text']}\n"
                f"=== Example {i}: solution model.py ===\n"
                f"{shot['code'].rstrip()}\n"
            )
        parts.append("=== END EXAMPLES ===\n\n")
        example = "".join(parts)
    return (
        f"{GRAPH_TASK_DESCRIPTION}\n\n"
        f"The drawing is provided as the text file `{GRAPH_NAME}` in the "
        f"current working directory: a plain-text serialization of the "
        f"multi-view drawing's geometry (standard orthographic views, "
        f"third-angle projection, dimensions in millimetres) in the format "
        f"specified below.\n\n"
        f"=== {GRAPH_NAME} FORMAT ===\n{GRAPH_FORMAT_LEGEND}\n"
        f"=== END FORMAT ===\n\n"
        f"{example}"
        f"Do the following, entirely within the current working directory:\n"
        f"1. Read `{GRAPH_NAME}` and use the format specification above to "
        f"recover the 3D geometry and dimensions.\n"
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
            prompt: str, dry_run: bool,
            sandbox_fs: bool = False, work_root: Path | None = None,
            input_mode: str = "png", graph_dir: Path | None = None,
            quant: int = 1024, serialize_python: str = sys.executable,
            max_graph_bytes: int = 0,
            venv_binds: list[tuple[str, str]] | None = None) -> str:
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

    # graph mode: serialize FIRST (also in dry-run — offline, and the size
    # guard must be able to fire before any agy invocation).
    graph_text, graph_json, n_graph_bytes = None, None, 0
    if input_mode == "graph":
        graph_json = (graph_dir or C.SRC_GRAPH_DIR) / f"{uuid}.graph.json"
        if not graph_json.exists():
            print(f"[agy] {uuid}: missing graph {graph_json}, skipping",
                  file=sys.stderr)
            return "no-input"
        try:
            graph_text = serialize_graph(serialize_python, graph_json, quant)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"[agy] {uuid}: {e}", file=sys.stderr)
            return "serialize-error"
        n_graph_bytes = len(graph_text.encode("utf-8"))
        if max_graph_bytes and n_graph_bytes > max_graph_bytes:
            print(f"[agy] {uuid}: {GRAPH_NAME} would be {n_graph_bytes:,} B > "
                  f"--max-graph-bytes {max_graph_bytes:,} — status too_large; "
                  f"agy NOT invoked (no quota spent).", file=sys.stderr)
            return "too_large"

    cmd = agy_command(agy_bin, prompt, workdir, model, print_timeout_s, sandbox)
    if sandbox_fs:
        cmd = bwrap_wrap(cmd, workdir, C.REPO, AGY_CONFIG_DIRS,
                         ro_binds=venv_binds)

    if dry_run:
        print(f"\n===== fixture {uuid} =====")
        print(f"# cwd: {workdir}  (sandbox_fs={sandbox_fs})")
        if input_mode == "graph":
            print(f"# write: {workdir / GRAPH_NAME} <- serialize_3d({graph_json}, "
                  f"quant={quant}) = {n_graph_bytes:,} B")
            print(f"# {GRAPH_NAME} first line: {graph_text.splitlines()[0]}")
        else:
            print(f"# copy: {drawing} -> {workdir / C.DRAWING_NAME}")
        # shell-quote for copy-paste fidelity
        import shlex
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        print("\n----- PROMPT -----")
        print(prompt)
        print("----- END PROMPT -----")
        return "dry-run"

    if input_mode == "png" and not drawing.exists():
        print(f"[agy] {uuid}: missing drawing {drawing}, skipping", file=sys.stderr)
        return "no-input"

    # clean workdir + stage the input
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    # The per-fixture results dir must exist before agy.log is written below.
    # (Bugfix: with --sandbox-fs the workdir is external, so nothing else
    # creates fx and a FRESH sandboxed run crashed on the log write; previous
    # sandboxed runs only survived by resuming into pre-existing fixture dirs.)
    fx.mkdir(parents=True, exist_ok=True)
    if input_mode == "graph":
        (workdir / GRAPH_NAME).write_text(graph_text)
    else:
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
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        stderr += "\n[agy] wall-clock timeout"
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
    parser.add_argument("--input", choices=("png", "graph"), default="png",
                    help="what the agent gets per fixture: png = the rendered "
                         "drawing (control); graph = graph.txt, the serialized "
                         "AMVDG text our model receives (representation "
                         "ablation; prompt carries a format legend; results "
                         "under agy-graph_<modelslug>/)")
    parser.add_argument("--shots", type=int, choices=(0, 1, 2, 3), default=0,
                    help="graph mode: 0 = zero-shot (default; the legend alone "
                         "carries the format, matching the PNG control which "
                         "had no worked example; prompt stays byte-identical to "
                         "the control). K in 1..3 = prepend K worked examples "
                         "(never-a-bench-fixture val graphs + their blend-"
                         "stripped GT CadQuery): example 1 = the shortest, the "
                         "rest span complexity at spaced percentiles under "
                         "--max-shot-bytes. Writes to agy-graph-Kshot_<slug>/ "
                         "(K=1 -> agy-graph-1shot_<slug>/).")
    parser.add_argument("--max-shot-bytes", type=int, default=45000,
                    help="graph mode, --shots>=2: total byte budget over the "
                         "chosen examples (serialized graph text + GT code). A "
                         "percentile pick that would exceed it walks down to the "
                         "next shorter eligible graph (keeps the prompt bounded).")
    parser.add_argument("--shot-graph-dir", type=Path,
                    default=C.REPO / "experiments" / "dataset_z2c_train",
                    help="graph mode, --shots>=1: dir of {uuid}.graph.json to "
                         "draw worked examples from (default the Z2C TRAIN set — "
                         "disjoint by construction from the val-derived bench "
                         "fixtures, so examples never leak a graded part). "
                         "Relative paths resolve against the repo root.")
    parser.add_argument("--shot-code-dir", type=Path,
                    default=C.REPO / "experiments" / "stage_z2c_train_noblend",
                    help="graph mode, --shots>=1: dir of blend-stripped GT "
                         "{uuid}.cadquery.py paired with --shot-graph-dir for the "
                         "worked-example solutions (default stage_z2c_train_noblend). "
                         "Relative paths resolve against the repo root.")
    parser.add_argument("--shot-pool-cap", type=int, default=500,
                    help="graph mode, --shots>=1: if the candidate pool exceeds "
                         "this, uniformly subsample (seed 0) to this many BEFORE "
                         "serializing — bounds selection time on a huge source "
                         "like the train split (~9e4 graphs). 0 = no cap (use the "
                         "whole pool; fine for a small dir like the val set).")
    parser.add_argument("--exclude-ids-from", type=Path, nargs="*", default=[],
                    help="graph mode, --shots>=1: extra id sources (each a "
                         "manifest.json or a newline id list) whose uuids are "
                         "ALSO barred from the worked-example pool, on top of the "
                         "active bench's own fixtures (always excluded). With the "
                         "default train pool this is a no-op safety net (train is "
                         "already disjoint from the val bench); use it when drawing "
                         "examples from a val-derived pool. Relative paths resolve "
                         "against the repo root.")
    parser.add_argument("--quant", type=int, default=1024,
                    help="graph mode: serialize_3d coordinate quantization "
                         "(= serialize.CANON_QUANT). MUST match what our model "
                         "receives at bench time (run_ours.py --quant 1024), "
                         "or the ablation compares different texts.")
    parser.add_argument("--serialize-python",
                    default=os.environ.get("DRAWING2CAD_PY", sys.executable),
                    help="interpreter that imports scripts/train3d/serialize.py "
                         "(stdlib-only; default $DRAWING2CAD_PY or this python)")
    parser.add_argument("--graph-dir", type=Path, default=None,
                    help="graph mode: {uuid}.graph.json source dir (default: "
                         "the manifest's sources.graph_png_dir — the graphs the "
                         "fixtures were built from)")
    parser.add_argument("--max-graph-bytes", type=int, default=2_000_000,
                    help="graph mode: a fixture whose serialized graph.txt "
                         "exceeds this many bytes gets status too_large and is "
                         "skipped WITHOUT invoking agy (quota guard for "
                         "oversized parts, e.g. future data_new sets; 0 = off)")
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
    if args.input != "graph" and args.shots:
        raise SystemExit("--shots is only meaningful with --input graph.")

    # graph mode: resolve the graph source + build the (fixture-independent)
    # prompt once, including the optional worked example.
    graph_dir = None
    shots_list: list[dict] = []
    if args.input == "graph":
        graph_dir = args.graph_dir
        if graph_dir is None:
            src = (C.load_manifest().get("sources") or {}).get("graph_png_dir")
            graph_dir = Path(src) if src else C.SRC_GRAPH_DIR
        if not graph_dir.is_absolute():
            graph_dir = C.REPO / graph_dir
        if not graph_dir.is_dir():
            raise SystemExit(f"--input graph: graph dir not found: {graph_dir}")
        if args.shots >= 1:
            shot_graph_dir = (args.shot_graph_dir if args.shot_graph_dir.is_absolute()
                              else C.REPO / args.shot_graph_dir)
            shot_code_dir = (args.shot_code_dir if args.shot_code_dir.is_absolute()
                             else C.REPO / args.shot_code_dir)
            for label, d in (("--shot-graph-dir", shot_graph_dir),
                             ("--shot-code-dir", shot_code_dir)):
                if not d.is_dir():
                    raise SystemExit(f"--shots {args.shots}: {label} not found: {d}")
            exclude_files = [f if f.is_absolute() else C.REPO / f
                             for f in args.exclude_ids_from]
            print(f"[agy] --shots {args.shots}: selecting worked example(s) "
                  f"from {shot_graph_dir} (pool cap {args.shot_pool_cap}; "
                  f"serialize+sort ~20 s) ...", file=sys.stderr)
            shots_list = select_shot_examples(
                args.serialize_python, args.quant, args.shots,
                args.max_shot_bytes, shot_graph_dir, shot_code_dir,
                exclude_files, args.shot_pool_cap)
            total = sum(s.get("bytes", 0) for s in shots_list)
            print(f"[agy] --shots {args.shots}: {len(shots_list)} example(s), "
                  f"{total:,} B total (budget {args.max_shot_bytes:,}):",
                  file=sys.stderr)
            for i, s in enumerate(shots_list, 1):
                print(f"[agy]   example {i}: {s['uuid']} "
                      f"({len(s['text'])} chars graph, {len(s['code'])} chars "
                      f"code, {s.get('bytes', 0):,} B)", file=sys.stderr)
    prompt = (build_graph_prompt(args.cadquery_python, shots_list)
              if args.input == "graph" else build_prompt(args.cadquery_python))
    if len(prompt) > 100_000:
        raise SystemExit(f"prompt is {len(prompt)} chars — too close to the "
                         f"kernel's ~128 KiB single-argv limit; slim the "
                         f"example/legend.")

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
    venv_binds: list[tuple[str, str]] = []
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
        # If the executing interpreter is invoked via a path inside the repo
        # (e.g. the repo venv), the tmpfs mask would hide it — re-bind its venv
        # root read-only AFTER the mask so `import cadquery` still works while
        # the GT under the repo stays masked.
        venv_binds = repo_venv_binds(args.cadquery_python, C.REPO)
        print(f"[agy] sandbox-fs ON (bwrap): repo hidden, workdirs under {work_root}")
        if venv_binds:
            print(f"[agy] sandbox-fs: re-binding repo-internal interpreter venv "
                  f"read-only: {venv_binds[0][1]} <- {venv_binds[0][0]}")
    else:
        print("[agy] ⚠️ sandbox-fs OFF: the agent can read the ground-truth STEP "
              "files off disk. Pass --sandbox-fs to prevent contamination.",
              file=sys.stderr)

    if args.run_name:
        run_name = args.run_name
    elif args.input == "graph":
        # Distinct namespaces per input mode / shots so report.py pairing can
        # never mix a graph run with the PNG control. K=1 -> agy-graph-1shot
        # (literal preserved for backward compatibility).
        stem = f"agy-graph-{args.shots}shot" if args.shots else "agy-graph"
        run_name = f"{stem}_{model_slug(args.model)}"
    else:
        run_name = f"agy_{model_slug(args.model)}"
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
            args.sandbox, prompt, args.dry_run,
            sandbox_fs=args.sandbox_fs, work_root=work_root,
            input_mode=args.input, graph_dir=graph_dir, quant=args.quant,
            serialize_python=args.serialize_python,
            max_graph_bytes=args.max_graph_bytes,
            venv_binds=venv_binds,
        )
        if statuses[uuid] == "limit":
            stopped_on_limit = True
            print("[agy] usage/rate limit reached — stopping. Re-run the same "
                  "command later to resume the remaining fixtures.", file=sys.stderr)
            break

    n_too_large = sum(1 for s in statuses.values() if s == "too_large")
    if args.dry_run:
        extra = (f" ({n_too_large} would be skipped as too_large)"
                 if n_too_large else "")
        print(f"\n[agy] dry-run: {len(ids)} fixture(s), invoked nothing.{extra}")
        return 0

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
    meta = {
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
    }
    if args.input == "graph":
        # Recorded only in graph mode so PNG-mode run_meta stays byte-identical.
        meta["input"] = "graph"
        meta["graph_dir"] = str(graph_dir)
        meta["quant"] = args.quant
        meta["serialize_python"] = args.serialize_python
        meta["shots"] = args.shots
        if shots_list:
            meta["shot_example_uuids"] = [s["uuid"] for s in shots_list]
            meta["max_shot_bytes"] = args.max_shot_bytes
            if args.shots == 1:
                # Back-compat: keep the old singular key for 1-shot runs.
                meta["shot_example_uuid"] = shots_list[0]["uuid"]
        meta["max_graph_bytes"] = args.max_graph_bytes
        meta["last_run"]["n_too_large"] = n_too_large
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[agy] this run: +{n_step_this} step, {n_skipped} already done | "
          f"cumulative {n_step_total}/{len(ids)} have output.step -> {run_dir}")
    if n_too_large:
        print(f"[agy] {n_too_large} fixture(s) skipped as too_large "
              f"(serialized graph.txt > --max-graph-bytes "
              f"{args.max_graph_bytes:,} B); agy was NOT invoked for them.",
              file=sys.stderr)
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
