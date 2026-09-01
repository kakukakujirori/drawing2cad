import ast
from dataclasses import dataclass

from zeroshot.pipeline.messages.contracts import OperationPlan

# Methods that hand back the shape they were called on.
_SHAPE_PRESERVING = frozenset({"clean", "copy"})


@dataclass(frozen=True)
class ProgramCheck:
    """Whether source preserves the minimum plan-to-code identities."""

    missing_operations: tuple[str, ...] = ()
    unknown_operations: tuple[str, ...] = ()
    result_assigned: bool = False
    identity_operations: tuple[str, ...] = ()
    unconsumed_operations: tuple[str, ...] = ()

    @property
    def faults(self) -> tuple[str, ...]:
        """What is wrong with the program, one complaint per kind."""
        faults: list[str] = []
        if (
            self.missing_operations
            or self.unknown_operations
            or not self.result_assigned
        ):
            faults.append(
                "model.py does not match the current OperationPlan: "
                f"missing={self.missing_operations}, "
                f"unknown={self.unknown_operations}, "
                f"result_assigned={self.result_assigned}"
            )
        if self.identity_operations:
            faults.append(
                "These results hand back the shape they were given, so the "
                "operation they name is not implemented: "
                + ", ".join(self.identity_operations)
            )
        if self.unconsumed_operations:
            faults.append(
                "These results are built and then read by nothing, so what "
                "they add never reaches `result`: "
                + ", ".join(self.unconsumed_operations)
            )
        return tuple(faults)

    @property
    def sound(self) -> bool:
        return not self.faults


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

    module_names = {
        name for statement in tree.body for name in assigned_names(statement)
    }
    assigned_returns = {name for name in module_names if name.startswith("ret_")}
    result_assigned = "result" in module_names

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
        identity_operations=_identity_returns(tree),
        unconsumed_operations=_unconsumed_returns(tree, assigned_returns),
    )


def assigned_names(statement: ast.stmt) -> set[str]:
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


def _identity_returns(tree: ast.Module) -> tuple[str, ...]:
    """Find the `ret_*` names that never build anything.

    Every assignment to such a name just passes another `ret_*` through. One
    assignment that builds clears the name: real work followed by a `.clean()`
    still did the work.
    """
    builds: dict[str, bool] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            continue
        is_built = statement.value is not None and not _preserve_input(statement.value)
        for name in assigned_names(statement):
            if name.startswith("ret_"):
                builds[name] = builds.get(name, False) or is_built
    return tuple(sorted(name for name, built in builds.items() if not built))


def _preserve_input(value: ast.expr) -> bool:
    """Whether this is a `ret_*` under nothing but shape-preserving calls.

    An argument, a plain function call or an operator may have built
    something, and stops the walk.
    """
    while isinstance(value, ast.Call):
        if value.args or value.keywords or not isinstance(value.func, ast.Attribute):
            return False
        if value.func.attr not in _SHAPE_PRESERVING:
            return False
        value = value.func.value
    return isinstance(value, ast.Name) and value.id.startswith("ret_")


def _unconsumed_returns(
    tree: ast.Module,
    assigned_returns: set[str],
) -> tuple[str, ...]:
    """Names no other statement reads, so nothing they build reaches `result`."""
    read: set[str] = set()
    for statement in tree.body:
        own = assigned_names(statement)
        read |= {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in own
        }
    return tuple(sorted(assigned_returns - read))
