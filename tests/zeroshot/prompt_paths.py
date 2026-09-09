"""Expected prompt locations, independent of the process working directory."""

from pathlib import Path

STAGES_DIR = Path(__file__).resolve().parents[2] / "zeroshot/pipeline/stages"
ROLE_PATHS = {
    "drawing_analyzer": STAGES_DIR / "drawings/prompts/role.md",
    "semantic_hypothesizer": STAGES_DIR / "semantics/prompts/role.md",
    "operation_planner": STAGES_DIR / "operations/prompts/role.md",
    "coder": STAGES_DIR / "coding/prompts/role.md",
    "output_auditor": STAGES_DIR / "audit/prompts/role.md",
    "cad_reconstructor": STAGES_DIR / "_base/prompts/cad_reconstructor.md",
}
