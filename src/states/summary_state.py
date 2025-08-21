import operator
from typing_extensions import Annotated

from dataclasses import dataclass, field

@dataclass(kw_only=True)
class SummaryState:
    topic: str = field(default=None)
    query: str = field(default=None)
    web_resultset: Annotated[list, operator.add] = field(default_factory=list)
    sources: Annotated[list, operator.add] = field(default_factory=list)
    loop_count: int = field(default=0)
    summary: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateInput:
    topic: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateOutput:
    summary: str = field(default=None)
    sources: Annotated[list, operator.add] = field(default_factory=list)