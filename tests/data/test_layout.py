from pathlib import Path
import tempfile
import unittest

from src.data.layout import (
    REQUIRED_SUBDIRS,
    discover_sample_ids,
    resolve_dataset_roots,
)


def _stage(parent: Path, name: str, *, code_targets: bool = True) -> Path:
    """Create a minimally valid dataset root under ``parent``."""
    root = parent / name
    for subdir in REQUIRED_SUBDIRS:
        (root / subdir).mkdir(parents=True)
    (root / "target" / "sample.step").write_text("", encoding="utf-8")
    if code_targets:
        (root / "target" / "sample.cadquery.py").write_text("", encoding="utf-8")
    return root


class ResolveDatasetRootsTest(unittest.TestCase):
    def test_accepts_a_bare_string_as_one_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _stage(Path(temporary), "z2c_val")
            resolved = resolve_dataset_roots(str(root), split="val")
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0].name, "z2c_val")
            self.assertEqual(resolved[0].path, root)
            self.assertEqual(resolved[0].audit_dir, root / "target_audit")

    def test_reports_code_targets_per_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            labeled = _stage(parent, "z2c_val")
            step_only = _stage(parent, "test_vlm", code_targets=False)
            resolved = resolve_dataset_roots(
                [str(labeled), str(step_only)], split="val"
            )
            self.assertEqual(
                [(item.name, item.has_code_targets) for item in resolved],
                [("z2c_val", True), ("test_vlm", False)],
            )

    def test_missing_required_subdirectory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _stage(Path(temporary), "z2c_val")
            (root / "target_audit").rmdir()
            with self.assertRaises(FileNotFoundError) as error:
                resolve_dataset_roots([str(root)], split="val")
            self.assertIn("target_audit", str(error.exception))

    def test_nonexistent_root_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                resolve_dataset_roots(
                    [str(Path(temporary) / "absent")], split="train"
                )

    def test_duplicate_basenames_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = _stage(parent / "a", "z2c_val")
            second = _stage(parent / "b", "z2c_val")
            with self.assertRaises(ValueError) as error:
                resolve_dataset_roots([str(first), str(second)], split="val")
            self.assertIn("unique", str(error.exception))

    def test_empty_root_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_roots([], split="val")


class DiscoverSampleIdsTest(unittest.TestCase):
    def test_ids_come_from_techdraw_dxfs_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dxf_dir = Path(temporary) / "techdraw" / "dxf"
            dxf_dir.mkdir(parents=True)
            for stem in ("000200", "000100", "000300"):
                (dxf_dir / f"{stem}.dxf").write_text("", encoding="utf-8")
            # A non-DXF sibling must not be enumerated.
            (dxf_dir / "notes.txt").write_text("", encoding="utf-8")
            self.assertEqual(
                discover_sample_ids(temporary), ("000100", "000200", "000300")
            )

    def test_missing_dxf_dir_yields_no_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(discover_sample_ids(temporary), ())


if __name__ == "__main__":
    unittest.main()
