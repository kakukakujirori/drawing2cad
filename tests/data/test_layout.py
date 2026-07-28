from pathlib import Path
import tempfile
import unittest

from src.data.layout import (
    REQUIRED_SUBDIRS,
    resolve_dataset_roots,
    take_inventory,
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
                resolve_dataset_roots([str(Path(temporary) / "absent")], split="train")

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


IMAGE_DIR = Path("render_3d") / "hlg_perspective"


def _stage_sample(
    root: Path, sample_id: str, *, image: bool = True, target: bool = True
) -> None:
    """Write the files one sample is made of, optionally leaving some out."""
    for directory in (root / "techdraw" / "dxf", root / IMAGE_DIR, root / "target"):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "techdraw" / "dxf" / f"{sample_id}.dxf").write_text("", encoding="utf-8")
    if image:
        (root / IMAGE_DIR / f"{sample_id}.png").write_text("", encoding="utf-8")
    if target:
        (root / "target" / f"{sample_id}.cadquery.py").write_text("", encoding="utf-8")


class TakeInventoryTest(unittest.TestCase):
    def test_ids_come_from_techdraw_dxfs_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for stem in ("000200", "000100", "000300"):
                _stage_sample(root, stem)
            # A non-DXF sibling must not be enumerated.
            (root / "techdraw" / "dxf" / "notes.txt").write_text("", encoding="utf-8")
            complete, incomplete = take_inventory(root)
            self.assertEqual(complete, ("000100", "000200", "000300"))
            self.assertEqual(incomplete, {})

    def test_missing_dxf_dir_yields_no_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(take_inventory(temporary), ((), {}))

    def test_sample_without_its_raster_is_incomplete_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _stage_sample(root, "000100")
            # The renderer projected the drawing but was killed during render_3d.
            _stage_sample(root, "000200", image=False)
            complete, incomplete = take_inventory(root, image_dirs=[IMAGE_DIR])
            self.assertEqual(complete, ("000100",))
            self.assertEqual(incomplete, {"000200": root / IMAGE_DIR / "000200.png"})

    def test_targets_are_only_required_when_asked_for(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _stage_sample(root, "000100", target=False)
            self.assertEqual(
                take_inventory(root, image_dirs=[IMAGE_DIR])[0], ("000100",)
            )
            complete, incomplete = take_inventory(
                root, image_dirs=[IMAGE_DIR], require_target=True
            )
            self.assertEqual(complete, ())
            self.assertEqual(
                incomplete, {"000100": root / "target" / "000100.cadquery.py"}
            )

    def test_every_configured_raster_must_be_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _stage_sample(root, "000100")
            second = Path("render_3d") / "transparent_shaded_edges_perspective"
            complete, incomplete = take_inventory(root, image_dirs=[IMAGE_DIR, second])
            self.assertEqual(complete, ())
            self.assertEqual(incomplete, {"000100": root / second / "000100.png"})


if __name__ == "__main__":
    unittest.main()
