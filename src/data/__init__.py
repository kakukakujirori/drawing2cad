"""Public data API for DXF/image drawing-to-CAD inputs."""

from .collator import Drawing2CADBatch, Drawing2CADCollator
from .dataset import (
    DEFAULT_IMAGE_SOURCES,
    DEFAULT_IMAGE_STYLES,
    Drawing2CADDataset,
    Drawing2CADRecord,
    Drawing2CADSample,
    RasterImageSource,
)
from .dxf import (
    DXFParseError,
    DXFPrimitiveConfig,
    DXFPrimitiveData,
    DXFPrimitiveParser,
    DXF_PRIMITIVE_TYPES,
    DXF_PRIMITIVE_TYPE_TO_ID,
    DXF_SAMPLE_FEATURE_NAMES,
    sample_dxf_entity,
)
from .metadata import (
    ManifestSampleMetadataProvider,
    MetadataError,
    SampleMetadata,
    SampleMetadataProvider,
    VIEW_DIRECTIONS,
    VIEW_DIRECTION_TO_ID,
    ViewBBox,
)
from .preprocessing import (
    DEFAULT_INSTRUCTION,
    Drawing2CADPreprocessor,
    PreparedDrawing2CADSample,
)

__all__ = [
    "DEFAULT_IMAGE_SOURCES",
    "DEFAULT_IMAGE_STYLES",
    "DEFAULT_INSTRUCTION",
    "DXFParseError",
    "DXFPrimitiveConfig",
    "DXFPrimitiveData",
    "DXFPrimitiveParser",
    "DXF_PRIMITIVE_TYPES",
    "DXF_PRIMITIVE_TYPE_TO_ID",
    "DXF_SAMPLE_FEATURE_NAMES",
    "Drawing2CADBatch",
    "Drawing2CADCollator",
    "Drawing2CADDataset",
    "Drawing2CADPreprocessor",
    "Drawing2CADRecord",
    "Drawing2CADSample",
    "ManifestSampleMetadataProvider",
    "MetadataError",
    "PreparedDrawing2CADSample",
    "RasterImageSource",
    "SampleMetadata",
    "SampleMetadataProvider",
    "VIEW_DIRECTIONS",
    "VIEW_DIRECTION_TO_ID",
    "ViewBBox",
    "sample_dxf_entity",
]
