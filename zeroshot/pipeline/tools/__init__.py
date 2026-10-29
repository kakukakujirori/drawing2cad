from collections.abc import Collection, Sequence

from langchain_core.tools import BaseTool


def pick_named_tools(
    available: Sequence[BaseTool], names: Collection[str]
) -> list[BaseTool]:
    """Pick the tools named in `names`, in the order of `available`."""
    unknown = set(names) - {tool.name for tool in available}
    if unknown:
        raise ValueError(
            f"unknown tools {sorted(unknown)}; "
            f"available: {[tool.name for tool in available]}"
        )
    return [tool for tool in available if tool.name in names]
