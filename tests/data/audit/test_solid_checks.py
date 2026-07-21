import importlib.util
import unittest

HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None


@unittest.skipUnless(HAS_CADQUERY, "cadquery is not installed")
class SolidChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cadquery as cq

        from src.data.audit.solid_checks import (
            Severity,
            ShapeSignature,
            Thresholds,
            audit_shape,
            classify_severity,
            compare_signatures,
        )

        cls.cq = cq
        cls.Severity = Severity
        cls.ShapeSignature = ShapeSignature
        cls.Thresholds = Thresholds
        cls.audit_shape = staticmethod(audit_shape)
        cls.classify_severity = staticmethod(classify_severity)
        cls.compare_signatures = staticmethod(compare_signatures)

    def test_clean_box_is_ok(self) -> None:
        box = self.cq.Workplane("XY").box(10, 10, 10).val()
        audit = self.audit_shape(box, self.Thresholds(thickness_samples=100))
        self.assertEqual(audit.severity, self.Severity.OK)
        self.assertEqual(audit.reasons, [])
        self.assertEqual(audit.topology.n_solids, 1)
        self.assertAlmostEqual(audit.topology.volume, 1000.0, places=6)

    def test_self_intersecting_bowtie_is_hard_invalid(self) -> None:
        bad = (
            self.cq.Workplane("XY")
            .polyline([(0, 0), (10, 0), (0, 10), (10, 10), (0, 0)])
            .close()
            .extrude(5)
        )
        audit = self.audit_shape(bad.val(), self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.HARD_INVALID)
        self.assertIn("self_intersection", audit.reasons)
        self.assertTrue(audit.topology.self_intersects)
        # On a real staged Zero-To-CAD part (not reproduced here to keep this
        # test self-contained) BRepCheck_Analyzer.IsValid() was True while
        # BOPAlgo_ArgumentAnalyzer still correctly flagged BOPAlgo_SelfIntersect
        # on one face -- confirming isValid() alone is not sufficient, which is
        # why check_self_intersection exists as its own check.

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
        self.assertGreater(metrics["volume_relative_diff"], thresholds.divergence_volume_relative)

    def test_fully_enclosing_cut_leaves_empty_shape(self) -> None:
        # A cut whose tool fully encloses the base leaves an empty Compound
        # (verified: 0 solids, 0 faces, 0 volume -- no exception raised).
        box = self.cq.Workplane("XY").box(10, 10, 10)
        tool = self.cq.Workplane("XY").box(20, 20, 20)
        emptied = box.cut(tool).val()
        audit = self.audit_shape(emptied, self.Thresholds(thickness_samples=0))
        self.assertEqual(audit.severity, self.Severity.HARD_INVALID)
        self.assertIn("empty_shape", audit.reasons)
        self.assertNotIn("zero_or_negative_volume", audit.reasons)  # n_solids==0 already covers it


if __name__ == "__main__":
    unittest.main()
