import importlib.util
import unittest

HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None


@unittest.skipUnless(HAS_CADQUERY, "cadquery is not installed")
class OpTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.data.audit.op_trace import noop_count, trace_source

        cls.trace_source = staticmethod(trace_source)
        cls.noop_count = staticmethod(noop_count)

    def test_contributing_ops_are_recorded_as_such(self) -> None:
        source = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10).edges("|Z").chamfer(1)
"""
        shape, trace, error = self.trace_source(source)
        self.assertIsNone(error)
        self.assertIsNotNone(shape)
        methods = [entry.method for entry in trace]
        self.assertIn("chamfer", methods)
        for entry in trace:
            self.assertTrue(entry.contributed, msg=f"{entry.method} should have contributed")
        self.assertEqual(self.noop_count(trace), 0)

    def test_non_intersecting_cut_is_a_noop(self) -> None:
        source = """
import cadquery as cq
base = cq.Workplane("XY").box(10, 10, 10)
tool = cq.Workplane("XY").box(2, 2, 2).translate((100, 100, 100))
result = base.cut(tool)
"""
        shape, trace, error = self.trace_source(source)
        self.assertIsNone(error)
        cut_entries = [entry for entry in trace if entry.method == "cut"]
        self.assertEqual(len(cut_entries), 1)
        self.assertFalse(cut_entries[0].contributed)
        self.assertEqual(self.noop_count(trace), 1)
        # the shape is still recovered and its volume is unaffected by the no-op
        self.assertAlmostEqual(shape.Volume(), 1000.0, places=6)

    def test_broken_source_returns_error_not_exception(self) -> None:
        shape, trace, error = self.trace_source("this is not python (((")
        self.assertIsNone(shape)
        self.assertIsNotNone(error)

    def test_patch_is_fully_restored_after_tracing(self) -> None:
        import cadquery as cq

        original = cq.Workplane.cut
        self.trace_source(
            'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 1, 1)\n'
        )
        self.assertIs(cq.Workplane.cut, original)


if __name__ == "__main__":
    unittest.main()
