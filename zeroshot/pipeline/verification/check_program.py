import ast
from dataclasses import dataclass

from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.types import Member

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


def program_output_names(source: str) -> set[str]:
    """Return directly assigned module-level ret_* names without judging the plan."""
    tree = ast.parse(source, filename="model.py", mode="exec")
    return {
        name
        for statement in tree.body
        for name in assigned_names(statement)
        if name.startswith("ret_")
    }


def program_members(source: str) -> dict[str, Member]:
    """Each ret_x as its module-level assignments, citing the op_x it implements.

    Syntax errors propagate, as in ``check_program``.
    """
    # ponytail: only module-level ret_ assignments are compared, so an edited
    # helper they call goes unseen; compare helper definitions if that matters.
    statements: dict[str, list[str]] = {}
    for statement in ast.parse(source, filename="model.py", mode="exec").body:
        for name in assigned_names(statement):
            if name.startswith("ret_"):
                statements.setdefault(name, []).append(ast.unparse(statement))
    return {
        name: Member(text, frozenset({_operation_name(name)}))
        for name, text in statements.items()
    }


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
    """Find the `ret_*` names whose final value only passes another return through.

    Cleaning its own completed shape keeps an operation's work. Replacing it
    with another return discards that work, even if an earlier assignment built.
    A conditional/loop write makes the final value uncertain, not a proven
    identity. Only unconditional assignments can establish that fault again.
    """
    builds: dict[str, bool] = {}
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign | ast.AnnAssign)
            or statement.value is None
        ):
            # A block may update an initialized return, even if it runs zero times.
            # True means "not proven passthrough", not "CAD work definitely ran".
            for name in _possible_stores(statement):
                if name in builds:
                    builds[name] = True
            continue
        preserved = _preserve_input(statement.value)
        for name in assigned_names(statement):
            if name.startswith("ret_"):
                builds[name] = (
                    builds.get(name, False) if preserved == name else preserved is None
                )
    return tuple(sorted(name for name, built in builds.items() if not built))


def _possible_stores(node: ast.AST) -> set[str]:
    """Names a statement may rebind, excluding nested local-scope assignments."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}  # The definition binds its name, but its locals do not.
    if isinstance(node, ast.Lambda):
        return set()
    names = (
        {node.id}
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del)
        else set()
    )
    # Comprehension targets are local; walrus assignments in their expressions are not.
    children = (
        (node.iter, *node.ifs)
        if isinstance(node, ast.comprehension)
        else ast.iter_child_nodes(node)
    )
    for child in children:
        names.update(_possible_stores(child))
    return names


def _preserve_input(value: ast.expr) -> str | None:
    """The `ret_*` beneath only shape-preserving calls, if any.

    An argument, a plain function call or an operator may have built
    something, and stops the walk.
    """
    while isinstance(value, ast.Call):
        if value.args or value.keywords or not isinstance(value.func, ast.Attribute):
            return None
        if value.func.attr not in _SHAPE_PRESERVING:
            return None
        value = value.func.value
    return (
        value.id
        if isinstance(value, ast.Name) and value.id.startswith("ret_")
        else None
    )


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
