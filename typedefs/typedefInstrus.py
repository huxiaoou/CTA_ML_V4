from dataclasses import dataclass
from typing import NewType


@dataclass(frozen=True)
class CCfgInstru:
    sectorL0: str
    sectorL1: str


TInstruName = NewType("TInstruName", str)
TUniverse = dict[TInstruName, CCfgInstru]
