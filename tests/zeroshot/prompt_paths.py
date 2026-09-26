"""Expected prompt locations, independent of the process working directory."""

from pathlib import Path

STAGES_DIR = Path(__file__).resolve().parents[2] / "zeroshot/pipeline/stages"
RECONSTRUCTION_CONTEXT_PATH = STAGES_DIR / "_base/prompts/reconstruction_context.md"
ROLE_PATHS = {
    "drawing_interpreter": STAGES_DIR / "interpretation/prompts/role.md",
    "operation_planner": STAGES_DIR / "operations/prompts/role.md",
    "coder": STAGES_DIR / "coding/prompts/role.md",
    "output_auditor": STAGES_DIR / "audit/prompts/role.md",
}
