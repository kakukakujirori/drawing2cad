#!/usr/bin/env python
"""gt_audit.py -- rule-based trust audit for a staged Zero-To-CAD directory.

Zero-To-CAD-1m ships each part as a pre-built ``{uuid}.step`` (the shipped
reference shape -- what select_zero_to_cad.py stages, what the 2D-drawing
renderer reads, what eval_cq.py loads as the IoU/CD ground truth) plus the
generating ``{uuid}.cadquery.py`` (what build_dataset.py uses as the SFT
``target_code`` text). This script checks whether that trust is warranted, on
BOTH artifacts, for every staged part:

  1. Executes the GT CadQuery program locally (this repo's OCCT/cadquery
     version, not whatever authored the shipped STEP) and runs the same B-rep
     check battery on the resulting shape as on the shipped STEP.
  2. Diffs the two: if executing the "ground truth" program does not
     reproduce the shape used everywhere else in the pipeline, the SFT text
     target and the rendered input drawing describe different objects. This
     is the single highest-value check here -- see compare_signatures in
     solid_checks.py.
  3. Runs geometry sanity checks on each shape (soft OCCT Boolean-argument
     self-interference, evaluator-compatible mesh validity, open/non-manifold
     boundary, disjoint solids, micro edges/faces, kernel tolerance blow-up,
     sampled wall thickness) -- see solid_checks.py for the full battery.
  4. Traces whether each chained solid-modifying call in the GT program
     actually changed the shape (op_trace.py) -- catches e.g. a union whose
     added body was already fully enclosed, a no-op by construction.

Output tiers (see solid_checks.Severity): HARD_INVALID shapes are broken or
unusable downstream (invalid/non-watertight mesh, non-manifold, open,
fragmented, zero-volume) -- candidates for exclusion. SOFT_SUSPECT shapes are
valid but smell (BOP self-interference, thin walls, micro features, tolerance
bloat, no-op ops, GT/STEP divergence) -- candidates for review, not automatic
exclusion; the right threshold for e.g. "how thin is too thin" is a
training-policy call this script deliberately does not make for you.
UNAUDITABLE means the harness itself didn't finish (timeout/crash), not that
the data is bad.

Usage:
    conda run -n drawing2cad python -m src.data.audit.gt_audit \\
        --stage-dir /disk2/drawing2cad_experiments/stage_z2c_train \\
        --out-dir experiments/gt_audit_train --workers 28

Resumable: re-running with the same --out-dir skips uuids already present in
results.jsonl (tolerates a truncated last line from a prior hard kill).
--report-only re-aggregates stats.json and the per-severity tier files from an
existing results.jsonl without re-scanning -- use this to iterate on thresholds.

Process model: a multiprocessing.Pool (spawn context, matching
src/evaluation/executor.py's avoidance of fork -- OCC has enough internal
global state that fork-after-import is a real segfault risk) amortizes
cadquery's ~0.7s import cost across many samples instead of paying it per
sample. Results are drained in COMPLETION order (imap_unordered), not
submission order, so one slow or hung sample never blocks writing the
results the other workers already finished -- the pathology that dominated
wall time before. Two independent timeout layers guard against pathological
GT programs: an in-worker SIGALRM soft-timeout (--per-sample-timeout-s) lets
a worker self-abort and stay alive for the next task, but SIGALRM only fires
when control returns to the Python interpreter, so it cannot preempt a hung
*native* OCC call. The driver-side timeout (--task-timeout-s) is the real
safety net for those: if NO sample completes anywhere in the pool within
that window, the pool is assumed wedged, terminated, and a fresh one is
spawned for the survivors (everything already drained is kept -- never
re-run; see resumability above). A sample that stays a survivor across
repeated wedges is the true culprit and, after a couple of wedges
(max_attempts), is recorded as a hard timeout so the run always terminates.
A crashed worker (SIGSEGV) is handled the same way, via the error the
result iterator raises.

--no-cadquery: some GT splits (e.g. a STEP-only eval set) ship only
{uuid}.step, never the generating {uuid}.cadquery.py. Passing --no-cadquery
switches to a reduced check battery that needs only the STEP -- everything in
solid_checks.py except the sampled wall-thickness-on-divergence special case,
which is moot with no code shape to diverge from. It cannot run op-trace
no-op detection, GT-vs-STEP divergence, or the code-side re-audit, since all
three require executing the GT program (see step_only_audit.py, the
prototype this mode is ported from). Without --no-cadquery, a --stage-dir
that has .step files but zero matching .cadquery.py raises immediately rather
than silently auditing nothing -- see _discover_uuids.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import multiprocessing as mp
import os
import random
import rootutils
import signal
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Spawned workers inherit these before importing NumPy/OCP, mirroring
# src/evaluation/executor.py: one thread per isolated sample keeps wall-clock
# timeout behaviour predictable under concurrent auditing.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")


def _short_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    label = type(exc).__name__
    text = f"{label}: {message}" if message else label
    return text[:500]


class _SoftTimeout(Exception):
    pass


def _raise_soft_timeout(signum, frame) -> None:
    raise _SoftTimeout()


def _audit_one(
    uuid: str,
    stage_dir: str,
    thresholds,  # src.data.audit.solid_checks.Thresholds
    per_sample_timeout_s: float,
) -> dict:
    """Worker entry point. Always returns a record, never raises -- every
    failure mode (bad STEP, bad code, our own bug, a timeout) is captured as
    a field in the record rather than lost.
    """

    record: dict = {"uuid": uuid}
    old_handler = signal.signal(signal.SIGALRM, _raise_soft_timeout)
    signal.alarm(max(1, int(per_sample_timeout_s)))
    try:
        # Imported inside the guarded block (not at module top) so the
        # spawn-context worker process does its own import after inheriting
        # the thread-limiting env vars set above, and so a broken
        # cadquery/OCP install surfaces as a per-sample "unauditable" record
        # (the except BaseException clause below needs no import of its own)
        # instead of an unrecoverable pool-startup crash.
        import cadquery as cq

        from src.data.audit.op_trace import noop_count, trace_source
        from src.data.audit.solid_checks import (
            ShapeSignature,
            audit_shape,
            compare_signatures,
        )

        step_path = os.path.join(stage_dir, f"{uuid}.step")
        code_path = os.path.join(stage_dir, f"{uuid}.cadquery.py")
        # Stashed in the record (not just used locally) so aggregate() can
        # write the per-severity tier files as file paths, and so --report-only
        # re-aggregation doesn't need --stage-dir passed again to know them.
        record["step_path"] = step_path
        record["code_path"] = code_path

        step_shape = None
        try:
            step_shape = cq.importers.importStep(step_path).val()
            record["step_ok"] = True
        except BaseException as exc:
            record["step_ok"] = False
            record["step_error"] = _short_error(exc)

        code_shape, op_trace, code_error = None, [], None
        try:
            source = Path(code_path).read_text(encoding="utf-8")
            code_shape, op_trace, code_error = trace_source(source)
            record["code_ok"] = code_shape is not None
            if code_error:
                record["code_error"] = code_error
        except BaseException as exc:
            record["code_ok"] = False
            record["code_error"] = _short_error(exc)

        step_audit = None
        if step_shape is not None:
            try:
                step_audit = audit_shape(step_shape, thresholds)
            except BaseException as exc:
                record["step_audit_error"] = _short_error(exc)

        # The GT-vs-STEP divergence check needs only the cheap ShapeSignature,
        # not the full battery -- so fingerprint the code shape first and
        # compare. The executed code reproduces the shipped STEP in ~99.6% of
        # samples (only gt_mismatch differs), and when it matches, step_audit
        # already covers that geometry. The signature determines only whether
        # the costly code-side wall-thickness sampling is necessary; topology
        # and downstream mesh validity are always audited below.
        code_signature = None
        if code_shape is not None:
            try:
                code_signature = ShapeSignature.of(code_shape)
            except BaseException as exc:
                record["code_signature_error"] = _short_error(exc)

        divergence = None
        if step_audit is not None and code_signature is not None:
            try:
                diverges, metrics = compare_signatures(
                    code_signature, step_audit.signature, thresholds
                )
                divergence = {"diverges": diverges, **metrics}
            except BaseException as exc:
                record["divergence_error"] = _short_error(exc)

        # Always audit the code shape's topology and downstream mesh. A target
        # program that re-executes to a broken or non-watertight solid must still
        # be caught even when its gross signature matches the shipped STEP:
        # measured on 396 samples, skipping the code shape entirely on a
        # signature match changed 2.5% of verdicts and missed ~0.8% hard-invalid
        # whose defect is on the code shape only. What we skip on a signature
        # match is only wall-thickness sampling -- the costliest check, and a
        # noisy sampled upper-bound already run on the STEP. On a genuine
        # divergence the code shape is different, so include thickness too.
        code_audit = None
        if code_shape is not None:
            diverges = bool(divergence and divergence.get("diverges"))
            try:
                code_audit = audit_shape(
                    code_shape, thresholds, with_thickness=diverges
                )
            except BaseException as exc:
                record["code_audit_error"] = _short_error(exc)

        record["op_trace"] = {
            "n_ops": len(op_trace),
            "n_noop": noop_count(op_trace),
            "noop_methods": [t.method for t in op_trace if t.contributed is False],
            "n_unresolved": sum(1 for t in op_trace if t.contributed is None),
        }

        record["step_audit"] = _audit_to_json(step_audit)
        record["code_audit"] = _audit_to_json(code_audit)
        # Whether wall-thickness sampling ran on the code shape (only on a
        # genuine STEP divergence). A code_audit whose "thickness" is null is a
        # deliberate skip on a signature match, not a failed check.
        record["code_thickness_audited"] = bool(
            code_audit is not None and code_audit.thickness is not None
        )
        record["divergence"] = divergence

        severity, reasons = _combine_severity(
            step_ok=record["step_ok"],
            code_ok=record["code_ok"],
            step_audit=step_audit,
            code_audit=code_audit,
            divergence=divergence,
            n_noop=record["op_trace"]["n_noop"],
        )
        record["severity"] = severity.value
        record["reasons"] = reasons
    except _SoftTimeout:
        # Literal string, not the Severity enum: this branch must stay valid
        # even if the import block above (which pulls in Severity) is what
        # failed or never ran.
        record["severity"] = "unauditable"
        record["reasons"] = ["soft_timeout"]
        record["error"] = f"exceeded per_sample_timeout_s={per_sample_timeout_s}"
    except (
        BaseException
    ) as exc:  # our own bug (or a broken import), not the data's fault
        record["severity"] = "unauditable"
        record["reasons"] = ["harness_exception"]
        record["error"] = _short_error(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return record


def _audit_to_json(audit) -> dict | None:
    if audit is None:
        return None
    payload = asdict(audit)
    payload["severity"] = audit.severity.value
    return payload


def _audit_one_step_only(
    uuid: str,
    stage_dir: str,
    thresholds,  # src.data.audit.solid_checks.Thresholds
    per_sample_timeout_s: float,
) -> dict:
    """Worker entry point for --no-cadquery: the STEP-only subset of _audit_one.

    Ported from data/eccv2026-cad-challenge-data/my_codes/step_only_audit.py
    (prototyped there for a GT split that ships only {uuid}.step). Runs the
    full solid_checks battery on the STEP shape alone -- everything _audit_one
    runs on step_audit -- but skips op-trace, GT-vs-STEP divergence, and the
    code-side re-audit, since all three need the generating .cadquery.py this
    mode assumes does not exist. The record keeps gt_audit's own field names
    (uuid/step_path/step_audit), not step_only_audit.py's (id/...), so
    results.jsonl stays readable by gate.py regardless of which mode produced
    it.
    """

    record: dict = {"uuid": uuid}
    old_handler = signal.signal(signal.SIGALRM, _raise_soft_timeout)
    signal.alarm(max(1, int(per_sample_timeout_s)))
    try:
        import cadquery as cq

        from src.data.audit.solid_checks import audit_shape

        step_path = os.path.join(stage_dir, f"{uuid}.step")
        record["step_path"] = step_path

        step_shape = None
        try:
            step_shape = cq.importers.importStep(step_path).val()
            record["step_ok"] = True
        except BaseException as exc:
            record["step_ok"] = False
            record["step_error"] = _short_error(exc)

        if step_shape is not None:
            try:
                step_audit = audit_shape(step_shape, thresholds, with_thickness=True)
                record["step_audit"] = _audit_to_json(step_audit)
                record["severity"] = step_audit.severity.value
                record["reasons"] = list(step_audit.reasons)
            except BaseException as exc:
                record["step_audit_error"] = _short_error(exc)
                record["severity"] = "unauditable"
                record["reasons"] = ["audit_exception"]
        else:
            # Mirrors _combine_severity: a STEP that won't even import is
            # hard-invalid, not merely unauditable.
            record["severity"] = "hard_invalid"
            record["reasons"] = ["step_import_failed"]
    except _SoftTimeout:
        record["severity"] = "unauditable"
        record["reasons"] = ["soft_timeout"]
        record["error"] = f"exceeded per_sample_timeout_s={per_sample_timeout_s}"
    except BaseException as exc:
        record["severity"] = "unauditable"
        record["reasons"] = ["harness_exception"]
        record["error"] = _short_error(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return record


def _combine_severity(*, step_ok, code_ok, step_audit, code_audit, divergence, n_noop):
    from src.data.audit.solid_checks import Severity

    reasons: list[str] = []
    if not code_ok:
        reasons.append("code_exec_failed")
    if not step_ok:
        reasons.append("step_import_failed")

    hard: set[str] = set()
    soft: set[str] = set()
    for audit in (step_audit, code_audit):
        if audit is None:
            continue
        if audit.severity == Severity.HARD_INVALID:
            hard.update(audit.reasons)
        elif audit.severity == Severity.SOFT_SUSPECT:
            soft.update(audit.reasons)
    reasons.extend(sorted(hard))

    if not code_ok or not step_ok or hard:
        return Severity.HARD_INVALID, reasons

    if n_noop > 0:
        soft.add("noop_operation")
    if divergence is not None and divergence.get("diverges"):
        soft.add("gt_mismatch")
    if soft:
        return Severity.SOFT_SUSPECT, sorted(soft)
    return Severity.OK, []


def _discover_uuids(stage_dir: str, *, require_cadquery: bool = True) -> list[str]:
    """Every {uuid}.step under stage_dir, paired with .cadquery.py by default.

    With require_cadquery=False (--no-cadquery), a {uuid}.step alone is enough
    -- for a GT split that never shipped the generating program.

    With require_cadquery=True (the default), zero uuids surviving the pairing
    filter while .step files exist is treated as a mis-invocation, not "a
    corpus with 0 valid pairs": Zero-To-CAD always stages both artifacts
    together, so the far likelier explanation is a GT split that has no
    .cadquery.py by design (e.g. a STEP-only eval set) and needs
    --no-cadquery, not a silent empty audit.
    """

    step_paths = sorted(glob.glob(os.path.join(stage_dir, "*.step")))
    if not require_cadquery:
        return [os.path.basename(p)[: -len(".step")] for p in step_paths]

    uuids = []
    for step_path in step_paths:
        uuid = os.path.basename(step_path)[: -len(".step")]
        if os.path.exists(os.path.join(stage_dir, f"{uuid}.cadquery.py")):
            uuids.append(uuid)

    if step_paths and not uuids:
        raise ValueError(
            f"{stage_dir}: found {len(step_paths)} .step file(s) but none have "
            f"a matching {{uuid}}.cadquery.py. If this GT dataset does not ship "
            f"cadquery scripts to begin with, re-run with --no-cadquery to run "
            f"the STEP-only checks. Otherwise the .cadquery.py files are "
            f"unexpectedly missing from --stage-dir."
        )
    return uuids


def _load_done_uuids(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["uuid"])
            except Exception:
                continue  # tolerate a truncated last line from a prior kill
    return done


def _run_pool(
    uuids: list[str],
    *,
    stage_dir: str,
    thresholds,
    workers: int,
    maxtasksperchild: int,
    task_timeout_s: float,
    per_sample_timeout_s: float,
    results_path: Path,
    max_attempts: int = 2,
    no_cadquery: bool = False,
) -> None:
    """Audit ``uuids`` and append one JSON record per sample to ``results_path``.

    Results are drained in COMPLETION order via ``imap_unordered``, not in
    submission order, so a single slow or hung sample never blocks writing the
    results the other workers already produced (the old in-order ``future.get``
    stalled every write behind the slowest in-flight future and then discarded
    the whole pool's completed-but-unread work on a timeout -- that serialized
    tail is what turned a ~1-2h job into a ~20h one).

    ``task_timeout_s`` here means "no sample finished anywhere in the pool for
    this long" -- a healthy sample finishes in ~1s, so a full window of silence
    means a worker is wedged in a native OCC call SIGALRM cannot preempt (or a
    worker crashed). We then terminate the pool, keep everything already drained
    (never re-run), and retry the survivors on a fresh pool. Because Pool does
    not bind tasks to workers, we do NOT blame a specific uuid on a single stall
    (the old code recorded ``driver_timeout`` against whichever future it
    happened to be waiting on -- often an innocent, still-queued sample). The
    genuine offender is the one that stays a survivor across rounds; once it has
    survived ``max_attempts`` wedges it is recorded as a hard timeout so the run
    always terminates instead of retrying it forever.
    """

    ctx = mp.get_context("spawn")
    worker = functools.partial(
        _audit_one_step_only if no_cadquery else _audit_one,
        stage_dir=stage_dir,
        thresholds=thresholds,
        per_sample_timeout_s=per_sample_timeout_s,
    )
    pending = list(uuids)
    attempts: Counter = Counter()
    with (
        results_path.open("a", encoding="utf-8") as out_fh,
        tqdm(total=len(pending), desc="auditing") as progress,
    ):

        def _emit(record: dict) -> None:
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()
            progress.update(1)

        while pending:
            batch = pending
            pool = ctx.Pool(processes=workers, maxtasksperchild=maxtasksperchild)
            result_iter = pool.imap_unordered(worker, batch)
            pool.close()
            completed: set[str] = set()
            stalled = False
            for _ in range(len(batch)):
                try:
                    record = result_iter.next(timeout=task_timeout_s)
                except mp.TimeoutError:
                    stalled = True  # nothing finished for a whole window: wedged
                    break
                except StopIteration:
                    break
                except Exception:
                    # A dead worker (SIGSEGV) can poison the iterator; treat it
                    # like a wedge -- terminate, rebuild, retry the survivors.
                    stalled = True
                    break
                uuid = record.get("uuid")
                if uuid is not None:
                    completed.add(uuid)
                _emit(record)
            pool.terminate()
            pool.join()

            remaining = [u for u in batch if u not in completed]
            # A round that made zero progress and still has work left is also a
            # wedge, even if we exited on StopIteration rather than a timeout --
            # fold it into the same give-up path so the loop cannot spin forever.
            if remaining and (stalled or not completed):
                give_up: set[str] = set()
                # Only charge strikes once the survivor set is small enough that
                # "still outstanding" ~= "actually running" (a couple of
                # pool-fulls). While it is large, most survivors were merely
                # queued and never ran, so blaming them would be wrong; we just
                # rebuild and retry, and healthy ones drain next round.
                if len(remaining) <= max(1, 2 * workers):
                    for u in remaining:
                        attempts[u] += 1
                        if attempts[u] >= max_attempts:
                            give_up.add(u)
                for u in sorted(give_up):
                    _emit(
                        {
                            "uuid": u,
                            "severity": "unauditable",
                            "reasons": ["hard_timeout"],
                            "error": (
                                f"no result within task_timeout_s={task_timeout_s} "
                                f"across {attempts[u]} attempts"
                            ),
                        }
                    )
                remaining = [u for u in remaining if u not in give_up]
            pending = remaining


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = int(pos), min(len(sorted_values) - 1, int(pos) + 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _summary(values: list[float]) -> dict:
    finite = sorted(v for v in values if v is not None)
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": finite[0],
        "p05": _percentile(finite, 5),
        "median": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
    }


def aggregate(results_path: Path, out_dir: Path) -> dict:
    records = []
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a truncated/corrupt line from a prior hard kill
                # mid-write (same tolerance as _load_done_uuids): one bad line
                # must not sink the final aggregation of an otherwise-complete
                # multi-hour run. results.jsonl stays the source of truth and
                # can be re-aggregated with --report-only after a manual fix.
                continue

    severity_counts = Counter(r.get("severity", "unknown") for r in records)
    reason_counts = Counter(reason for r in records for reason in r.get("reasons", []))
    brepcheck_hist: Counter = Counter()
    thin_walls, aspect_ratios, tolerances, divergence_vols, noop_counts = (
        [],
        [],
        [],
        [],
        [],
    )
    for r in records:
        # Absent (not zero) for --no-cadquery records and driver-side
        # give-up/timeout records: op-trace never ran for either, so counting
        # them as a measured 0 would understate n_noop_operations_per_sample.
        if "op_trace" in r:
            noop_counts.append(r["op_trace"].get("n_noop", 0))
        for side in ("step_audit", "code_audit"):
            audit = r.get(side)
            if not audit:
                continue
            topo = audit.get("topology", {})
            for status, count in topo.get("brepcheck_status_histogram", {}).items():
                brepcheck_hist[status] += count
            aspect_ratios.append(topo.get("aspect_ratio"))
            tolerances.append(topo.get("max_tolerance_mm"))
            thickness = audit.get("thickness")
            if thickness and thickness.get("min_thickness_relative") is not None:
                thin_walls.append(thickness["min_thickness_relative"])
        divergence = r.get("divergence")
        if divergence and divergence.get("volume_relative_diff") is not None:
            divergence_vols.append(divergence["volume_relative_diff"])

    def _path_lines(severity: str) -> list[str]:
        # One line per flagged sample, tab-separated (uuid, step path, code
        # path) so the file is both directly openable/cat-able and easy to
        # `cut -f2` for a file list, e.g. to feed a follow-up rm/mv/review
        # script. .get(..., "") tolerates results.jsonl rows written before
        # step_path/code_path was added to the per-sample record.
        rows = sorted(
            (r["uuid"], r.get("step_path", ""), r.get("code_path", ""))
            for r in records
            if r.get("severity") == severity
        )
        return [
            f"{uuid}\t{step_path}\t{code_path}" for uuid, step_path, code_path in rows
        ]

    stats = {
        "n_total": len(records),
        "severity_counts": dict(severity_counts),
        "severity_rate": {
            k: v / len(records) if records else 0.0 for k, v in severity_counts.items()
        },
        "reason_counts": dict(reason_counts.most_common()),
        "brepcheck_status_histogram": dict(brepcheck_hist.most_common()),
        "min_thickness_relative": _summary(thin_walls),
        "aspect_ratio": _summary([v for v in aspect_ratios if v is not None]),
        "max_tolerance_mm": _summary([v for v in tolerances if v is not None]),
        "divergence_volume_relative": _summary(divergence_vols),
        "n_noop_operations_per_sample": _summary(noop_counts),
    }

    header = "# uuid\tstep_path\tcode_path\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    # One file per severity tier, named for the exact `severity` field value so
    # the file name is 1:1 with what results.jsonl records and what the training
    # audit gate (configs/data/*.yaml `audit.allow_*`) selects on. The four tier
    # files partition the corpus; they are a human-facing convenience view, not a
    # pipeline input -- results.jsonl stays the source of truth.
    for severity in ("hard_invalid", "soft_suspect", "unauditable", "ok"):
        lines = _path_lines(severity)
        (out_dir / f"{severity}.txt").write_text(
            header + "\n".join(lines) + ("\n" if lines else "")
        )
    return stats


def _build_thresholds(args: argparse.Namespace):
    from src.data.audit.solid_checks import Thresholds

    return Thresholds(
        tolerance_bloat_abs_mm=args.tolerance_bloat_abs_mm,
        small_edge_relative=args.small_edge_relative,
        small_face_relative=args.small_face_relative,
        thin_wall_relative=args.thin_wall_relative,
        aspect_ratio_max=args.aspect_ratio_max,
        divergence_volume_relative=args.divergence_volume_relative,
        divergence_bbox_relative=args.divergence_bbox_relative,
        thickness_samples=args.thickness_samples,
        thickness_seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage-dir", required=True, help="dir of {uuid}.step + {uuid}.cadquery.py"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="results.jsonl / stats.json / per-severity tier .txt files go here",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4)
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="audit only the first N discovered uuids",
    )
    parser.add_argument("--maxtasksperchild", type=int, default=200)
    parser.add_argument("--per-sample-timeout-s", type=float, default=30.0)
    # "No sample finished anywhere in the pool for this long" -> assume a worker
    # is wedged in an unpreemptable native OCC call, terminate and retry the
    # survivors (see _run_pool). A healthy sample finishes in ~1s, so 45s of
    # total pool silence is already a very conservative wedge signal.
    parser.add_argument("--task-timeout-s", type=float, default=45.0)
    parser.add_argument("--thickness-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-bloat-abs-mm", type=float, default=1e-3)
    parser.add_argument("--small-edge-relative", type=float, default=1e-3)
    parser.add_argument("--small-face-relative", type=float, default=1e-3)
    parser.add_argument("--thin-wall-relative", type=float, default=5e-3)
    parser.add_argument("--aspect-ratio-max", type=float, default=500.0)
    parser.add_argument("--divergence-volume-relative", type=float, default=0.01)
    parser.add_argument("--divergence-bbox-relative", type=float, default=0.01)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="skip auditing; just re-aggregate stats.json and tier files from an existing results.jsonl",
    )
    parser.add_argument(
        "--no-cadquery",
        action="store_true",
        help=(
            "audit {uuid}.step files alone, for a GT split that does not ship "
            "the generating {uuid}.cadquery.py by design (e.g. a STEP-only "
            "eval set). Skips op-trace no-op detection, GT-vs-STEP divergence, "
            "and the code-side re-audit -- all three need the .cadquery.py. "
            "Without this flag, a --stage-dir with .step files but zero "
            "matching .cadquery.py raises instead of silently auditing "
            "nothing."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    if not args.report_only:
        uuids = _discover_uuids(args.stage_dir, require_cadquery=not args.no_cadquery)
        if args.limit is not None:
            # _discover_uuids sorts lexicographically by uuid string, so a
            # bare prefix slice risks a hidden bias if uuids were ever
            # assigned in a way that correlates with generation/complexity
            # order. Shuffle (seeded, so --limit runs stay resumable across
            # re-invocations with the same --seed) before slicing.
            uuids = list(uuids)
            random.Random(args.seed).shuffle(uuids)
            uuids = uuids[: args.limit]
        done = _load_done_uuids(results_path)
        pending = [u for u in uuids if u not in done]
        unit = "STEP files" if args.no_cadquery else "pairs"
        print(
            f"discovered {len(uuids)} {unit} in {args.stage_dir}, "
            f"{len(done)} already done, {len(pending)} pending"
        )
        if pending:
            thresholds = _build_thresholds(args)
            t0 = time.time()
            _run_pool(
                pending,
                stage_dir=args.stage_dir,
                thresholds=thresholds,
                workers=args.workers,
                maxtasksperchild=args.maxtasksperchild,
                task_timeout_s=args.task_timeout_s,
                per_sample_timeout_s=args.per_sample_timeout_s,
                results_path=results_path,
                no_cadquery=args.no_cadquery,
            )
            print(f"audited {len(pending)} {unit} in {time.time() - t0:.1f}s")

    stats = aggregate(results_path, out_dir)
    print(json.dumps(stats["severity_counts"], indent=2))
    sc = stats["severity_counts"]
    print(
        f"-> {out_dir}/stats.json, "
        f"hard_invalid.txt ({sc.get('hard_invalid', 0)} uuids), "
        f"soft_suspect.txt ({sc.get('soft_suspect', 0)} uuids), "
        f"ok.txt ({sc.get('ok', 0)} uuids), "
        f"unauditable.txt ({sc.get('unauditable', 0)} uuids)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
