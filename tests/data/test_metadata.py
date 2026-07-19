import json
from pathlib import Path
import tempfile
import unittest

from src.data import (
    ManifestSampleMetadataProvider,
    MetadataError,
    SampleMetadata,
    VIEW_DIRECTIONS,
)


def _row(sample_id: str = "sample") -> dict:
    return {
        "name": sample_id,
        "ok": True,
        "extra": {
            "techdraw": {
                "scale": 1.25,
                "bbox_format": "xyxy",
                "bbox_coordinate_system": {
                    "unit": "mm",
                    "origin": "sheet_bottom_left",
                    "x_axis": "right",
                    "y_axis": "up",
                },
                "views": {
                    "front": {"bbox": [1, 2, 11, 12]},
                    "top": {"bbox": [1, 20, 11, 30]},
                    "right": {"bbox": [20, 2, 30, 12]},
                },
            }
        },
    }


class ManifestMetadataProviderTest(unittest.TestCase):
    def _write(self, directory: str, rows: list[dict]) -> Path:
        path = Path(directory) / "manifest.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_adapter_returns_typed_canonical_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            failed = {"name": "failed", "ok": False}
            provider = ManifestSampleMetadataProvider(
                self._write(directory, [failed, _row()])
            )
        self.assertEqual(provider.sample_ids, ("sample",))
        metadata = provider.get("sample")
        self.assertIsInstance(metadata, SampleMetadata)
        self.assertEqual(
            tuple(bbox.direction for bbox in metadata.view_bboxes), VIEW_DIRECTIONS
        )
        self.assertEqual(metadata.view_bboxes[0].xyxy_mm, (1.0, 2.0, 11.0, 12.0))
        self.assertEqual(metadata.drawing_scale, 1.25)

    def test_adapter_rejects_coordinate_and_bbox_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            bad_coordinate = _row("coordinate")
            bad_coordinate["extra"]["techdraw"]["bbox_coordinate_system"][
                "origin"
            ] = "top_left"
            with self.assertRaisesRegex(MetadataError, "bbox_coordinate_system"):
                ManifestSampleMetadataProvider(
                    self._write(directory, [bad_coordinate])
                )

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            bad_views = _row("views")
            del bad_views["extra"]["techdraw"]["views"]["right"]
            with self.assertRaisesRegex(MetadataError, "exactly"):
                ManifestSampleMetadataProvider(self._write(directory, [bad_views]))

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            bad_bbox = _row("bbox")
            bad_bbox["extra"]["techdraw"]["views"]["front"]["bbox"] = [1, 2, 1, 3]
            with self.assertRaisesRegex(MetadataError, "positive area"):
                ManifestSampleMetadataProvider(self._write(directory, [bad_bbox]))

    def test_missing_sample_error_names_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            provider = ManifestSampleMetadataProvider(
                self._write(directory, [_row()])
            )
            with self.assertRaisesRegex(KeyError, "absent.*manifest"):
                provider.get("unknown")


if __name__ == "__main__":
    unittest.main()
