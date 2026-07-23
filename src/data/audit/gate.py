"""Resolve a GT-audit verdict into a train/eval sample allow-list.

The consumption side of ``src/data/audit``: given an audit output directory
(the ``--out-dir`` of ``gt_audit.py``, containing ``results.jsonl``) and an
allow policy, return the set of uuids whose audited verdict is acceptable to
keep. Both the SFT dataloader factory and the generation evaluator call this so
training and evaluation gate on exactly the same policy (no drift).

The directory assumes ``<dataset root>/target_audit``
(``src.data.layout.DatasetRoot.audit_dir``), so a split made of several roots
cannot accidentally gate one root against another root's verdicts.

Policy (see ``configs/data/*.yaml`` ``audit`` block):

- ``ok``            -> always kept (no reasons).
- ``soft_suspect`` -> kept iff every one of its reason codes is in
  ``allow_reasons`` (set-subset: one disallowed reason drops the sample).
- ``hard_invalid`` -> kept only if ``allow_hard`` (a blanket escape hatch;
  these are broken solids -- self-intersecting, non-manifold, open shells).
- ``unauditable``  -> kept only if ``allow_unauditable`` (the harness never
  finished; not a claim the data is good).

The gate is fail-closed on two axes, because both failure modes silently poison
training rather than erroring:

1. A uuid absent from ``results.jsonl`` never passes -- an un-audited or
   wrongly-pointed corpus does not leak through.
2. ``gate_present_ids`` *raises* if any sample actually present (rendered + in
   the manifest) is missing from ``results.jsonl`` at all. The audit is expected
   to cover the whole staged corpus, so a present-but-unaudited sample means the
   audit is incomplete or mis-pointed; enabling the gate against it would
   otherwise quietly shrink the corpus to whatever happened to be audited. A
   count mismatch is treated as the anomaly it is, not silently absorbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

RESULTS_NAME = "results.jsonl"


def load_verdicts(audit_dir: str | Path) -> dict[str, tuple[str, list[str]]]:
    """Map ``uuid -> (severity, reasons)`` from ``<audit_dir>/results.jsonl``.

    Raises ``FileNotFoundError`` if the file is missing, rather than silently
    treating a mis-pointed path as "nothing audited".
    """
    results_path = Path(audit_dir) / RESULTS_NAME
    if not results_path.is_file():
        raise FileNotFoundError(
            f"audit gate enabled but {results_path} is missing -- run "
            f"`python src/data/audit/gt_audit.py --stage-dir <target> "
            f"--out-dir {audit_dir}` first, or set `audit: null` in the data config"
        )
    verdicts: dict[str, tuple[str, list[str]]] = {}
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Same tolerance as gt_audit.aggregate: one corrupt line from a
                # prior hard kill must not sink the whole gate.
                continue
            uuid = record.get("uuid")
            if uuid is None:
                continue
            verdicts[uuid] = (
                record.get("severity", ""),
                list(record.get("reasons", ())),
            )
    return verdicts


def _is_allowed(
    severity: str,
    reasons: Iterable[str],
    *,
    allow_reasons: frozenset[str],
    allow_hard: bool,
    allow_unauditable: bool,
) -> bool:
    if severity == "ok":
        return True
    if severity == "soft_suspect":
        return set(reasons) <= allow_reasons
    if severity == "hard_invalid":
        return allow_hard
    if severity == "unauditable":
        return allow_unauditable
    return False


def audit_allowed_ids(
    audit_dir: str | Path,
    *,
    allow_reasons: Iterable[str] = (),
    allow_hard: bool = False,
    allow_unauditable: bool = False,
) -> set[str]:
    """Return every audited uuid whose verdict passes the allow policy.

    This is the corpus-wide allow-set (no coverage check); consumers that hold
    the set of samples actually present should prefer :func:`gate_present_ids`.
    """
    allow = frozenset(str(r) for r in allow_reasons)
    verdicts = load_verdicts(audit_dir)
    return {
        uuid
        for uuid, (severity, reasons) in verdicts.items()
        if _is_allowed(
            severity,
            reasons,
            allow_reasons=allow,
            allow_hard=allow_hard,
            allow_unauditable=allow_unauditable,
        )
    }


def gate_present_ids(
    audit_dir: str | Path,
    present_ids: Iterable[str],
    *,
    allow_reasons: Iterable[str] = (),
    allow_hard: bool = False,
    allow_unauditable: bool = False,
    context: str = "audit gate",
) -> set[str]:
    """Filter ``present_ids`` (rendered + manifest ids) through the allow policy.

    ``results.jsonl`` is expected to cover the entire staged corpus, so *every*
    present sample must carry an audit verdict. Any present sample missing from
    it means the audit is incomplete or points at the wrong directory -- a hard
    error, never a silent drop, so a partial audit can never quietly shrink the
    training corpus.
    """
    allow = frozenset(str(r) for r in allow_reasons)
    verdicts = load_verdicts(audit_dir)
    present = list(present_ids)
    missing = [sid for sid in present if sid not in verdicts]
    if missing:
        raise ValueError(
            f"{context}: {len(missing)}/{len(present)} present samples are absent "
            f"from {Path(audit_dir) / RESULTS_NAME} (e.g. {missing[:3]}); the audit "
            f"does not cover this split -- audit the full split or set `audit: null`"
        )
    return {
        sid
        for sid in present
        if _is_allowed(
            verdicts[sid][0],
            verdicts[sid][1],
            allow_reasons=allow,
            allow_hard=allow_hard,
            allow_unauditable=allow_unauditable,
        )
    }


def gate_present_ids_from_config(
    audit_config: Mapping,
    audit_dir: str | Path,
    present_ids: Iterable[str],
    *,
    context: str = "audit gate",
) -> set[str]:
    """:func:`gate_present_ids` driven by a data-config ``audit`` mapping.

    The mapping supplies only the allow policy; ``audit_dir`` is derived by the
    caller from the dataset root being gated. The caller skips this entirely
    when ``audit`` is falsy.
    """
    return gate_present_ids(
        audit_dir,
        present_ids,
        allow_reasons=audit_config.get("allow_reasons", ()) or (),
        allow_hard=bool(audit_config.get("allow_hard", False)),
        allow_unauditable=bool(audit_config.get("allow_unauditable", False)),
        context=context,
    )
