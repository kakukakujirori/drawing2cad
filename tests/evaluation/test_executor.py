import tempfile
from pathlib import Path
import unittest

from src.evaluation.executor import CadExecutionResult, execute_cadquery


try:
    import cadquery as cq

    HAS_CADQUERY = True
except Exception:
    cq = None
    HAS_CADQUERY = False


class ExecutorTest(unittest.TestCase):
    def test_result_mapping_defaults_missing_fields(self) -> None:
        result = CadExecutionResult.from_mapping({"exec_ok": True})
        self.assertTrue(result.exec_ok)
        self.assertFalse(result.has_result)
        self.assertEqual(result.volume, 0.0)

    def test_python_exception_is_reported(self) -> None:
        result = execute_cadquery("raise ValueError('bad prediction')", timeout_s=10.0)
        self.assertFalse(result.exec_ok)
        self.assertIn("exec:ValueError", result.error or "")

    def test_hang_is_terminated(self) -> None:
        result = execute_cadquery(
            "import time\ntime.sleep(30)",
            timeout_s=0.5,
        )
        self.assertEqual(result, CadExecutionResult(error="timeout"))

    def test_hard_process_exit_is_reported(self) -> None:
        result = execute_cadquery("import os\nos._exit(9)", timeout_s=10.0)
        self.assertEqual(result.error, "process_exit:9")

    @unittest.skipUnless(HAS_CADQUERY, "CadQuery cannot be imported")
    def test_valid_solid_and_mesh_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mesh_path = Path(temporary) / "box.npz"
            result = execute_cadquery(
                "result = cq.Workplane('XY').box(2, 3, 4)",
                timeout_s=30.0,
                mesh_output_path=mesh_path,
            )
            self.assertTrue(result.exec_ok)
            self.assertTrue(result.has_result)
            self.assertTrue(result.is_valid)
            self.assertAlmostEqual(result.volume, 24.0)
            self.assertTrue(result.valid, result.error)
            self.assertIsNone(result.error)
            self.assertTrue(mesh_path.is_file())


if __name__ == "__main__":
    unittest.main()
