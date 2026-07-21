import importlib.util
import unittest
from unittest.mock import patch

HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None


@unittest.skipUnless(HAS_CADQUERY, "cadquery is not installed")
class SolidChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cadquery as cq

        from src.data.audit.solid_checks import (
            Severity,
            MeshValidityResult,
            ShapeSignature,
            Thresholds,
            TopologyResult,
            audit_shape,
            classify_severity,
            compare_signatures,
        )

        cls.cq = cq
        cls.Severity = Severity
        cls.MeshValidityResult = MeshValidityResult
        cls.ShapeSignature = ShapeSignature
        cls.Thresholds = Thresholds
        cls.TopologyResult = TopologyResult
        cls.audit_shape = staticmethod(audit_shape)
        cls.classify_severity = staticmethod(classify_severity)
        cls.compare_signatures = staticmethod(compare_signatures)

    def _clean_topology(self, **overrides):
        """A valid single-solid TopologyResult; override fields per test."""
        fields = dict(
            brepcheck_valid=True,
            brepcheck_status_histogram={},
            bop_self_interference=False,
            kernel_flags_small_edge=False,
            has_open_boundary=False,
            free_boundary_edge_count=0,
            non_manifold_edge_count=0,
            max_tolerance_mm=1e-7,
            mean_tolerance_mm=1e-7,
            n_solids=1,
            n_faces=6,
            solid_volumes=[1000.0],
            n_small_edges_relative=0,
            n_small_faces_relative=0,
            volume=1000.0,
            bbox_extents=(10.0, 10.0, 10.0),
            bbox_diagonal=17.32,
            aspect_ratio=1.0,
        )
        fields.update(overrides)
        return self.TopologyResult(**fields)

    def test_clean_box_is_ok(self) -> None:
        box = self.cq.Workplane("XY").box(10, 10, 10).val()
        audit = self.audit_shape(box, self.Thresholds(thickness_samples=100))
        self.assertEqual(audit.severity, self.Severity.OK)
        self.assertEqual(audit.reasons, [])
        self.assertEqual(audit.topology.n_solids, 1)
        self.assertAlmostEqual(audit.topology.volume, 1000.0, places=6)
        self.assertIsNotNone(audit.mesh)
        self.assertTrue(audit.mesh.watertight)

    def test_self_intersecting_bowtie_is_hard_invalid(self) -> None:
        bad = (
            self.cq.Workplane("XY")
            .polyline([(0, 0), (10, 0), (0, 10), (10, 10), (0, 0)])
            .close()
            .extrude(5)
        )
        audit = self.audit_shape(bad.val(), self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.HARD_INVALID)
        self.assertIn("brepcheck_invalid", audit.reasons)
        self.assertTrue(audit.topology.bop_self_interference)

    def test_open_shell_is_hard_invalid_and_unsolidified(self) -> None:
        # A shell missing one face of a box: no TopAbs_SOLID exists at all
        # (n_solids=0) even though real geometry is present (n_faces=5).
        # Mirrors a real defect found in staged Zero-To-CAD-1m STEP files:
        # a TopAbs_COMPOUND of open shells that never got wrapped as a Solid.
        box = self.cq.Workplane("XY").box(10, 10, 10).val()
        open_shell = self.cq.Shell.makeShell(box.Faces()[:5])
        audit = self.audit_shape(open_shell, self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.HARD_INVALID)
        self.assertIn("open_boundary", audit.reasons)
        self.assertIn("unsolidified_shell", audit.reasons)
        self.assertEqual(audit.topology.n_solids, 0)
        self.assertGreater(audit.topology.n_faces, 0)

    def test_touching_solids_split_into_disjoint_components(self) -> None:
        # Two boxes offset so they share only one vertical edge (zero-area
        # contact). OCC keeps these as two separate TopAbs_SOLIDs rather than
        # welding them into one non-manifold solid -- verified empirically --
        # so this is exactly what connected_components/n_solids should catch.
        b1 = self.cq.Workplane("XY").box(10, 10, 10)
        b2 = self.cq.Workplane("XY").box(10, 10, 10).translate((10, 10, 0))
        touching = b1.union(b2, glue=False).val()
        audit = self.audit_shape(touching, self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.HARD_INVALID)
        self.assertIn("disjoint_solids", audit.reasons)
        self.assertEqual(audit.topology.n_solids, 2)
        self.assertAlmostEqual(audit.topology.volume, 2000.0, places=6)

    def test_thin_plate_is_soft_suspect_thin_wall(self) -> None:
        plate = self.cq.Workplane("XY").box(50, 50, 0.2).val()
        audit = self.audit_shape(plate, self.Thresholds(thickness_samples=200))
        self.assertEqual(audit.severity, self.Severity.SOFT_SUSPECT)
        self.assertIn("thin_wall", audit.reasons)
        self.assertIsNotNone(audit.thickness)
        self.assertLess(audit.thickness.min_thickness_mm, 0.3)

    def test_micro_chamfer_flags_micro_edge(self) -> None:
        box = self.cq.Workplane("XY").box(10, 10, 10)
        chamfered = box.edges("|Z").chamfer(0.001).val()
        audit = self.audit_shape(chamfered, self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.SOFT_SUSPECT)
        self.assertIn("micro_edge", audit.reasons)

    def test_signature_comparison_identical_vs_divergent(self) -> None:
        thresholds = self.Thresholds()
        a = self.ShapeSignature.of(self.cq.Workplane("XY").box(10, 10, 10).val())
        b = self.ShapeSignature.of(self.cq.Workplane("XY").box(10, 10, 10).val())
        c = self.ShapeSignature.of(self.cq.Workplane("XY").box(10, 10, 12).val())

        diverges_same, _ = self.compare_signatures(a, b, thresholds)
        diverges_diff, metrics = self.compare_signatures(a, c, thresholds)

        self.assertFalse(diverges_same)
        self.assertTrue(diverges_diff)
        self.assertGreater(
            metrics["volume_relative_diff"], thresholds.divergence_volume_relative
        )

    def test_bop_self_interference_is_soft_when_mesh_is_watertight(self) -> None:
        topology = self._clean_topology(bop_self_interference=True)
        mesh = self.MeshValidityResult(True, 8, 12, True, None)
        severity, reasons = self.classify_severity(
            topology, None, self.Thresholds(), mesh=mesh
        )
        self.assertEqual(severity, self.Severity.SOFT_SUSPECT)
        self.assertIn("bop_self_interference", reasons)
        self.assertNotIn("mesh_not_watertight", reasons)

    def test_non_watertight_mesh_is_hard_and_masks_bop_signal(self) -> None:
        topology = self._clean_topology(bop_self_interference=True)
        mesh = self.MeshValidityResult(True, 8, 12, False, None)
        severity, reasons = self.classify_severity(
            topology, None, self.Thresholds(), mesh=mesh
        )
        self.assertEqual(severity, self.Severity.HARD_INVALID)
        self.assertIn("mesh_not_watertight", reasons)
        self.assertNotIn("bop_self_interference", reasons)

    def test_soft_bop_interference_still_runs_thickness(self) -> None:
        box = self.cq.Workplane("XY").box(10, 10, 10).val()
        topology = self._clean_topology(bop_self_interference=True)
        with patch("src.data.audit.solid_checks.audit_topology", return_value=topology):
            audit = self.audit_shape(box, self.Thresholds(thickness_samples=50))
        self.assertEqual(audit.severity, self.Severity.SOFT_SUSPECT)
        self.assertTrue(audit.mesh.watertight)
        self.assertIsNotNone(audit.thickness)
        self.assertIn("bop_self_interference", audit.reasons)

    def test_fully_enclosing_cut_leaves_empty_shape(self) -> None:
        # A cut whose tool fully encloses the base leaves an empty Compound
        # (verified: 0 solids, 0 faces, 0 volume -- no exception raised).
        box = self.cq.Workplane("XY").box(10, 10, 10)
        tool = self.cq.Workplane("XY").box(20, 20, 20)
        emptied = box.cut(tool).val()
        audit = self.audit_shape(emptied, self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.HARD_INVALID)
        self.assertIn("empty_shape", audit.reasons)
        self.assertNotIn(
            "zero_or_negative_volume", audit.reasons
        )  # n_solids==0 already covers it


if __name__ == "__main__":
    unittest.main()
