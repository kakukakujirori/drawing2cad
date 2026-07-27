"""Tests for the ECCV 2026 CAD Challenge metric family.

The port's deterministic half (topology, edge and vertex sampling) is compared
against the challenge's own evaluator entity by entity, and its stochastic half
(the assignment and the F1s) is compared on shared inputs, where both must agree
exactly. That is the only way to check a metric whose surface sampling is
random: comparing end-to-end scores would only compare two random draws.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.metrics.base import MetricSample
from src.metrics.eccv import ECCVChallengeMetric
from src.metrics.eccv.metric import _chamfer, match_entities, match_incidence


OFFICIAL_EVALUATOR = (
    Path(__file__).resolve().parents[2]
    / "data/eccv2026-cad-challenge-data/examples/min_eval/eval.py"
)
OFFICIAL_TARGET_DIR = (
    Path(__file__).resolve().parents[2]
    / "data/eccv2026-cad-challenge-data/train/target_step"
)
OFFICIAL_TARGETS = (
    sorted(OFFICIAL_TARGET_DIR.glob("*.step"))[:10]
    if OFFICIAL_TARGET_DIR.is_dir()
    else []
)

try:
    import cadquery as cq

    HAS_CADQUERY = True
except Exception:
    cq = None
    HAS_CADQUERY = False

HAS_OCC = importlib.util.find_spec("OCC") is not None


def _load_official():
    spec = importlib.util.spec_from_file_location(
        "official_eccv_eval", OFFICIAL_EVALUATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_CADQUERY and HAS_OCC, "CadQuery/pythonocc unavailable")
class ECCVChallengeMetricTest(unittest.TestCase):
    def _score(self, predicted: Path, target: Path, **kwargs) -> dict:
        metric = ECCVChallengeMetric(**kwargs)
        with tempfile.TemporaryDirectory() as workdir:
            sample = MetricSample(
                "box",
                workdir=workdir,
                pred_step_path=predicted,
                gt_step_path=target,
            )
            return dict(metric.score(sample))

    def test_identical_solid_scores_a_perfect_match(self) -> None:
        assert cq is not None
        # In millimetres, i.e. the scale our own datasets use rather than the
        # challenge's normalized targets.
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "box.step"
            cq.exporters.export(cq.Workplane("XY").box(20, 30, 40), str(step))
            row = self._score(step, step, normalize_to_gt_bbox=False)
        for key in ("eccv_surface_f1", "eccv_edge_f1", "eccv_vertex_f1"):
            self.assertAlmostEqual(row[key], 1.0, places=5, msg=key)
        self.assertAlmostEqual(row["eccv_topology_f1"], 1.0, places=5)
        for key in ("eccv_chamfer_surface", "eccv_chamfer_edge", "eccv_chamfer_vertex"):
            self.assertLess(row[key], 0.1, msg=key)
        self.assertEqual(row["eccv_num_pred_faces"], 6)
        self.assertEqual(row["eccv_num_gt_faces"], 6)

    def test_a_part_too_large_for_its_own_units_is_refused(self) -> None:
        assert cq is not None
        # Without a reference frame the per-unit-area density explodes on a
        # millimetre-scale part. Refusing beats silently thinning the samples,
        # which would quietly redefine the metric.
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "box.step"
            cq.exporters.export(cq.Workplane("XY").box(20, 30, 40), str(step))
            with self.assertRaisesRegex(ValueError, "surface samples"):
                self._score(
                    step, step, normalize_to_gt_bbox=False, reference_extent=None
                )

    def test_normalization_recovers_a_rescaled_prediction(self) -> None:
        assert cq is not None
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            target = temporary / "target.step"
            predicted = temporary / "pred.step"
            solid = cq.Workplane("XY").box(2, 3, 4)
            cq.exporters.export(solid, str(target))
            cq.exporters.export(
                solid.val().scale(9.0).translate(cq.Vector(50, 60, 70)), str(predicted)
            )
            normalized = self._score(predicted, target, normalize_to_gt_bbox=True)
            raw = self._score(predicted, target, normalize_to_gt_bbox=False)
        self.assertAlmostEqual(normalized["eccv_surface_f1"], 1.0, places=5)
        # Same shape, wrong size: the absolute-distance threshold rejects it.
        self.assertEqual(raw["eccv_surface_f1"], 0.0)

    def test_reduce_follows_the_official_summary(self) -> None:
        rows = [
            {
                "eccv_valid": True,
                "eccv_surface_f1": 0.8,
                "eccv_edge_f1": 0.6,
                "eccv_vertex_f1": 1.0,
                "eccv_face_edge_f1": 0.4,
                "eccv_edge_vertex_f1": 0.8,
                "eccv_topology_f1": 0.6,
            },
            # Never scored: drags the valid ratio down without diluting the
            # per-axis means, exactly as a missing submission does.
            {"eccv_valid": None},
        ]
        metrics = ECCVChallengeMetric().reduce(rows, prefix="val")
        self.assertEqual(metrics["val/eccv_valid_n"], 1)
        self.assertAlmostEqual(metrics["val/eccv_valid_ratio"], 0.5)
        self.assertAlmostEqual(metrics["val/eccv_surface_f1"], 0.8)
        self.assertAlmostEqual(metrics["val/eccv_topology_f1"], 0.6)
        self.assertAlmostEqual(
            metrics["val/eccv_summary"], 0.5 * (0.8 + 0.6 + 1.0 + 0.6) / 4
        )

    def test_empty_subset_scores_zero_rather_than_raising(self) -> None:
        metrics = ECCVChallengeMetric().reduce([], prefix="val")
        self.assertEqual(metrics["val/eccv_valid_ratio"], 0.0)
        self.assertEqual(metrics["val/eccv_summary"], 0.0)


@unittest.skipUnless(
    HAS_OCC and OFFICIAL_EVALUATOR.is_file() and len(OFFICIAL_TARGETS) > 0,
    "challenge data package not present",
)
class ECCVPortFidelityTest(unittest.TestCase):
    """The port must agree with the challenge's own evaluator."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.metrics.eccv._step_brep import load_step_brep

        cls.official = _load_official()
        cls.samples = []
        for target_path in OFFICIAL_TARGETS:
            official_data = cls.official.load_step_full(target_path)
            ported_data = load_step_brep(target_path, seed=0)
            cls.samples.append((target_path.name, official_data, ported_data))

    def test_deterministic_entities_are_identical(self) -> None:
        for name, official, ported in self.samples:
            with self.subTest(step_file=name):
                self.assertEqual(
                    (official["n_faces"], official["n_edges"], official["n_verts"]),
                    (ported.n_faces, ported.n_edges, ported.n_verts),
                )
                self.assertTrue(
                    np.array_equal(official["fe_matrix"], ported.fe_matrix)
                )
                self.assertTrue(
                    np.array_equal(official["ev_matrix"], ported.ev_matrix)
                )
                self.assertTrue(
                    np.array_equal(official["vertex_pc"], ported.vertex_pc)
                )
                self.assertTrue(np.array_equal(official["edge_pc"], ported.edge_pc))
                self.assertTrue(
                    np.array_equal(official["edge_labels"], ported.edge_labels)
                )

    def test_face_sampling_matches_the_official_density(self) -> None:
        # The sample locations are random, but their per-face counts are not.
        for name, official, ported in self.samples:
            with self.subTest(step_file=name):
                self.assertTrue(
                    np.array_equal(
                        np.bincount(official["face_labels"]),
                        np.bincount(ported.face_labels),
                    )
                )

    def test_scoring_agrees_exactly_on_shared_inputs(self) -> None:
        for name, official, ported in self.samples:
            with self.subTest(step_file=name):
                official_faces, official_face_match = self.official.compute_metric(
                    ported.face_pc,
                    official["face_pc"],
                    ported.face_labels,
                    official["face_labels"],
                )
                ported_faces = match_entities(
                    ported.face_pc,
                    official["face_pc"],
                    ported.face_labels,
                    official["face_labels"],
                    threshold=self.official.F1_threshold,
                )
                self.assertAlmostEqual(
                    official_faces[0], ported_faces.f1, places=12
                )
                self.assertEqual(official_face_match, ported_faces.matches)

                official_edges, official_edge_match = self.official.compute_metric(
                    ported.edge_pc,
                    official["edge_pc"],
                    ported.edge_labels,
                    official["edge_labels"],
                )
                ported_edges = match_entities(
                    ported.edge_pc,
                    official["edge_pc"],
                    ported.edge_labels,
                    official["edge_labels"],
                    threshold=self.official.F1_threshold,
                )
                self.assertAlmostEqual(
                    official_edges[0], ported_edges.f1, places=12
                )
                self.assertEqual(official_edge_match, ported_edges.matches)

                official_topology = self.official.compute_topo_metric(
                    ported.fe_matrix,
                    official["fe_matrix"],
                    official_face_match,
                    official_edge_match,
                )
                ported_topology = match_incidence(
                    ported.fe_matrix,
                    official["fe_matrix"],
                    ported_faces.matches,
                    ported_edges.matches,
                )
                self.assertAlmostEqual(
                    official_topology[0], ported_topology, places=12
                )

    def test_chamfer_distance_agrees_on_shared_inputs(self) -> None:
        from scipy.spatial import cKDTree

        for name, official, ported in self.samples:
            with self.subTest(step_file=name):
                # Surface Chamfer
                pred_tree = cKDTree(ported.face_pc)
                gt_tree = cKDTree(official["face_pc"])
                d1, _ = pred_tree.query(official["face_pc"], k=1)
                d2, _ = gt_tree.query(ported.face_pc, k=1)
                official_cd_surface = (d1.mean() + d2.mean()) / 2
                ported_cd_surface = _chamfer(ported.face_pc, official["face_pc"])
                self.assertAlmostEqual(
                    official_cd_surface, ported_cd_surface, places=12
                )

                # Edge Chamfer
                if len(ported.edge_pc) and len(official["edge_pc"]):
                    pt = cKDTree(ported.edge_pc)
                    gt_t = cKDTree(official["edge_pc"])
                    d1, _ = pt.query(official["edge_pc"], k=1)
                    d2, _ = gt_t.query(ported.edge_pc, k=1)
                    official_cd_edge = (d1.mean() + d2.mean()) / 2
                    ported_cd_edge = _chamfer(ported.edge_pc, official["edge_pc"])
                    self.assertAlmostEqual(
                        official_cd_edge, ported_cd_edge, places=12
                    )

                # Vertex Chamfer
                if len(ported.vertex_pc) and len(official["vertex_pc"]):
                    pt = cKDTree(ported.vertex_pc)
                    gt_t = cKDTree(official["vertex_pc"])
                    d1, _ = pt.query(official["vertex_pc"], k=1)
                    d2, _ = gt_t.query(ported.vertex_pc, k=1)
                    official_cd_vertex = (d1.mean() + d2.mean()) / 2
                    ported_cd_vertex = _chamfer(
                        ported.vertex_pc, official["vertex_pc"]
                    )
                    self.assertAlmostEqual(
                        official_cd_vertex, ported_cd_vertex, places=12
                    )


if __name__ == "__main__":
    unittest.main()
