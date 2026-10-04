"""Shared output-limit counter for the stage agents of one workflow."""

from dataclasses import dataclass


class OutputLimitBudgetExceeded(RuntimeError):
    """Stop this sample before another generation after its output-limit cap."""


@dataclass
class OutputLimitBudget:
    """Cumulative output-limit failures shared by one workflow's agents."""

    maximum: int
    failures: int = 0

    def __post_init__(self) -> None:
        if type(self.maximum) is not int or self.maximum < 1:
            raise ValueError("max_output_limit_failures must be a positive integer")

    def check(self) -> None:
        if self.failures >= self.maximum:
            raise OutputLimitBudgetExceeded(
                f"output_limit_budget_exceeded: {self.failures}/{self.maximum} "
                "generations reached the output limit"
            )

    def record(self) -> None:
        self.failures += 1
        self.check()
