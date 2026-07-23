import tempfile
import unittest
from pathlib import Path

from src.data.audit.gt_audit import _discover_uuids


def _touch(path: Path) -> None:
    path.write_text("")


class DiscoverUuidsTest(unittest.TestCase):
    """_discover_uuids needs no cadquery import (pure filesystem glob), so
    this exercises the --no-cadquery fail-closed guard without depending on
    cadquery being installed.
    """

    def test_default_pairs_step_with_cadquery_and_drops_unpaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp)
            _touch(stage_dir / "aaa.step")
            _touch(stage_dir / "aaa.cadquery.py")
            _touch(stage_dir / "bbb.step")
            _touch(stage_dir / "ccc.step")
            _touch(stage_dir / "ccc.cadquery.py")

            uuids = _discover_uuids(str(stage_dir))

            self.assertEqual(sorted(uuids), ["aaa", "ccc"])

    def test_default_raises_when_steps_exist_but_zero_are_paired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp)
            _touch(stage_dir / "aaa.step")
            _touch(stage_dir / "bbb.step")

            with self.assertRaises(ValueError) as ctx:
                _discover_uuids(str(stage_dir))

            message = str(ctx.exception)
            self.assertIn("--no-cadquery", message)
            self.assertIn(str(stage_dir), message)

    def test_no_cadquery_mode_accepts_step_only_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp)
            _touch(stage_dir / "aaa.step")
            _touch(stage_dir / "bbb.step")

            uuids = _discover_uuids(str(stage_dir), require_cadquery=False)

            self.assertEqual(sorted(uuids), ["aaa", "bbb"])

    def test_no_cadquery_mode_ignores_any_cadquery_files_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp)
            _touch(stage_dir / "aaa.step")
            _touch(stage_dir / "aaa.cadquery.py")

            uuids = _discover_uuids(str(stage_dir), require_cadquery=False)

            self.assertEqual(uuids, ["aaa"])

    def test_empty_directory_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uuids = _discover_uuids(tmp)

            self.assertEqual(uuids, [])


if __name__ == "__main__":
    unittest.main()
