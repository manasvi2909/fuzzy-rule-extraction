from dataclasses import dataclass
from typing import Any


@dataclass
class Rule:
    antecedents: dict[str, str]
    consequent: Any
    weight: float
    support: int
