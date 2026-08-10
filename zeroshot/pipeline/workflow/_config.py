"""Runnable configuration helpers for nested workflow graphs."""

from langchain_core.runnables import RunnableConfig


def _child_graph_config(config: RunnableConfig) -> RunnableConfig:
    """Preserve run context without imposing the parent's graph settings."""
    configurable = dict(config.get("configurable", {}))
    configurable.pop("__pregel_durability", None)
    child_config: RunnableConfig = {
        **config,
        "configurable": configurable,
    }
    child_config.pop("recursion_limit", None)
    return child_config
