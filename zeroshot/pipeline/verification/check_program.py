import ast
from dataclasses import dataclass

from zeroshot.pipeline.messages.contracts import OperationPlan


@dataclass(frozen=True)
class ProgramCheck:
    """Whether source preserves the minimum plan-to-code identities."""

    missing_operations: tuple[str, ...] = ()
    unknown_operations: tuple[str, ...] = ()
    result_assigned: bool = False

    @property
    def sound(self) -> bool:
        return (
            not self.missing_operations
            and not self.unknown_operations
            and self.result_assigned
        )


def check_program(
    source: str,
    plan: OperationPlan,
    *,
    filename: str = "model.py",
) -> ProgramCheck:
    """Compare direct program outputs with the operations in the current plan.

    Syntax errors propagate with ``filename`` intact; source-safety and CAD
    execution remain the responsibility of ``CadQueryExecutor``.
    """
    tree = ast.parse(source, filename=filename, mode="exec")

    assigned_names = {
        name for statement in tree.body for name in _assigned_names(statement)
    }
    assigned_returns = {name for name in assigned_names if name.startswith("ret_")}
    result_assigned = "result" in assigned_names

    # Enumerate all op_xxx expected and implemented.
    op_names_expected = {op.name for op in plan.proposal}
    op_names_implemented = {_operation_name(name) for name in assigned_returns}

    # Check the (op_xxx, ret_xxx) pairing.
    missing = op_names_expected - op_names_implemented
    unknown = op_names_implemented - op_names_expected

    return ProgramCheck(
        missing_operations=tuple(sorted(missing)),
        unknown_operations=tuple(sorted(unknown)),
        result_assigned=result_assigned,
    )


def _assigned_names(statement: ast.stmt) -> set[str]:
    """Names assigned directly by one module-level statement."""
    if isinstance(statement, ast.Assign):
        return {
            node.id
            for target in statement.targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return {
            node.id
            for node in ast.walk(statement.target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
    return set()


def _operation_name(return_name: str) -> str:
    """The plan identity named by a reserved program result variable."""
    return f"op_{return_name.removeprefix('ret_')}"
