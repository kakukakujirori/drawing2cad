#!/usr/bin/env python3
"""Audit quantization extent against the source part scale.

This is an independent inspection script to audit an existing data bundle.

The primary signal is ``extent_ratio = ext / max(abs(bbox_3d))``.  Absolute
``ext`` thresholds confuse legitimately large parts with corrupt projections,
so they are intentionally not used here.

Usage:
    python scripts/train3d/audit_extent.py \
        --bundle experiments/data_z2c_train_noblend/all.jsonl \
        --graph-dir experiments/dataset_z2c_train \
        --warn-ratio 2 \
        --fail-ratio 5 \
        --out /tmp/extent_audit.json
"""
import argparse
import json
import math
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(__file__))
from serialize import (_dropout_feature_hidden, _suppress_covered_hidden,  # noqa: E402
                       serialized_scale, struct_from_graph)


DEFAULT_THRESHOLDS = (2, 3, 5, 10, 20, 100, 1000)


def _percentile(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(q / 100 * len(ys)))]


def distribution(xs):
    finite = [x for x in xs if math.isfinite(x)]
    return {
        "p50": _percentile(finite, 50),
        "p90": _percentile(finite, 90),
        "p99": _percentile(finite, 99),
        "max": max(finite) if finite else None,
        "n_nonfinite": len(xs) - len(finite),
    }


def _geom_contribution(g):
    if "pts" in g:
        vals = [abs(z) for p in g["pts"] for z in p]
        return max(vals, default=0.0)
    cx, cy = g["c"]
    radius = g.get("r") or max(g.get("rmaj", 0.0), g.get("rmin", 0.0))
    return max(abs(cx) + radius, abs(cy) + radius)


def _failure_mode(p):
    typ = p.get("type")
    role = p.get("line_role", "visible")
    if typ == "line" and role == "center":
        return "stray_cylinder_centerline"
    if typ == "polyline":
        return "polyline_discretization"
    if typ in ("arc", "ellipse") and (
            p.get("start_angle") is not None or p.get("p1") is not None):
        return "ill_conditioned_partial_conic"
    if typ == "line":
        return "stray_hlr_line"
    return "out_of_envelope_primitive"


def graph_contributor(graph, expected_ext=None, *, drop_covered=True,
                      hid_dropout=0.0, rng=None):
    """Return the source primitive most likely responsible for serialization ext.

    This mirrors the serializer's per-view shift.  If an existing bundle used
    hidden-line dropout, ``expected_ext`` lets us prefer the candidate whose
    contribution actually matches its recorded header rather than merely the
    largest primitive still present in the source graph.
    """
    st = struct_from_graph(graph, keep_source_ids=True)
    if drop_covered:
        _suppress_covered_hidden(st)
    if hid_dropout:
        _dropout_feature_hidden(st, hid_dropout, rng)
    source = {(v.get("name"), p.get("id")): p
              for v in graph.get("views", []) for p in v.get("primitives", [])}
    candidates = []
    for v in st["views"]:
        for row in v["prims"]:
            contribution = _geom_contribution(row["geom"])
            candidates.append((abs(contribution - expected_ext)
                               if expected_ext is not None else 0.0,
                               -contribution, v["name"], row))
    if not candidates:
        return None
    if expected_ext is not None:
        candidates.sort(key=lambda x: (x[0], x[1]))
    else:
        candidates.sort(key=lambda x: x[1])
    _, neg_contrib, view, row = candidates[0]
    source_ids = row.get("source_ids", [])
    p = source.get((view, source_ids[0]), {}) if source_ids else {}
    return {
        "view": view,
        "primitive_id": p.get("id"),
        "source_ids": source_ids,
        "type": p.get("type") or row.get("typ"),
        "role": p.get("line_role") or row.get("role"),
        "contribution": -neg_contrib,
        "geometry": row["geom"],
        "failure_mode": _failure_mode(p),
    }


def audit_bundle(bundle, graph_dir=None, warn_ratio=2.0, fail_ratio=5.0,
                 absolute_bbox_mm=None):
    ratios = []
    offenders = []
    malformed = []
    n_records = 0
    transforms = {"drop_covered": True, "hid_dropout": 0.0, "seed": 0}
    stats_path = os.path.join(os.path.dirname(os.path.abspath(bundle)), "stats.json")
    try:
        with open(stats_path) as f:
            transforms.update(json.load(f).get("transforms", {}))
    except (OSError, ValueError):
        pass
    with open(bundle) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            n_records += 1
            try:
                record = json.loads(line)
            except Exception as e:
                malformed.append({"line": lineno, "error": str(e)})
                continue
            sid = record.get("id")
            scale = serialized_scale(record.get("input_text", ""))
            if scale is None:
                malformed.append({"line": lineno, "id": sid,
                                  "error": "missing/malformed quantized PART header"})
                continue
            bbox, ext, ratio = scale
            ratios.append(ratio)
            domain_bad = (absolute_bbox_mm is not None
                          and max((abs(x) for x in bbox), default=math.inf)
                          > absolute_bbox_mm)
            if ratio <= warn_ratio and not domain_bad:
                continue
            item = {
                "id": sid,
                "bbox": bbox,
                "extent": ext,
                "extent_ratio": ratio if math.isfinite(ratio) else None,
                "severity": "quarantine" if ratio > fail_ratio else "warning",
            }
            if domain_bad:
                item["absolute_bbox_violation"] = True
                item["severity"] = "quarantine"
            if graph_dir and sid:
                gp = os.path.join(graph_dir, f"{sid}.graph.json")
                try:
                    with open(gp) as gf:
                        seed = zlib.crc32(f"{transforms.get('seed', 0)}:{sid}".encode()) & 0xffffffff
                        item["max_contributor"] = graph_contributor(
                            json.load(gf), ext,
                            drop_covered=transforms.get("drop_covered", True),
                            hid_dropout=transforms.get("hid_dropout", 0.0),
                            rng=random.Random(seed))
                except Exception as e:
                    item["graph_error"] = str(e)
            offenders.append(item)

    thresholds = sorted(set(DEFAULT_THRESHOLDS + (warn_ratio, fail_ratio)))
    return {
        "bundle": bundle,
        "graph_dir": graph_dir,
        "n_records": n_records,
        "n_audited": len(ratios),
        "n_malformed": len(malformed),
        "warn_ratio": warn_ratio,
        "fail_ratio": fail_ratio,
        "absolute_bbox_mm": absolute_bbox_mm,
        "transforms": transforms,
        "extent_ratio": distribution(ratios),
        "counts_above": {str(t): sum(r > t for r in ratios) for t in thresholds},
        "n_warn": sum(x["severity"] == "warning" for x in offenders),
        "n_quarantined": sum(x["severity"] == "quarantine" for x in offenders),
        "n_absolute_bbox_violations": sum(
            bool(x.get("absolute_bbox_violation")) for x in offenders),
        "offenders": offenders,
        "malformed": malformed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="SFT all.jsonl")
    ap.add_argument("--graph-dir", help="source *.graph.json dir for root-cause details")
    ap.add_argument("--warn-ratio", type=float, default=2.0)
    ap.add_argument("--fail-ratio", type=float, default=5.0)
    ap.add_argument("--absolute-bbox-mm", type=float,
                    help="optional independent part-scale domain gate")
    ap.add_argument("--out", required=True, help="output audit JSON")
    args = ap.parse_args()
    if args.warn_ratio < 0 or args.fail_ratio <= args.warn_ratio:
        ap.error("require 0 <= --warn-ratio < --fail-ratio")
    if args.absolute_bbox_mm is not None and args.absolute_bbox_mm <= 0:
        ap.error("--absolute-bbox-mm must be positive")
    report = audit_bundle(args.bundle, args.graph_dir, args.warn_ratio,
                          args.fail_ratio, args.absolute_bbox_mm)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in (
        "n_records", "n_audited", "n_malformed", "extent_ratio",
        "counts_above", "n_warn", "n_quarantined")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
