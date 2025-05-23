from typing import NewType
from dataclasses import dataclass
from itertools import product
from husfort.qcalendar import CCalendar

TFactorClass = NewType("TFactorClass", str)
TFactorName = NewType("TFactorName", str)
TFactorNames = list[TFactorName]


@dataclass(frozen=True)
class CFactor:
    factor_class: TFactorClass
    factor_name: TFactorName


TFactors = list[CFactor]


@dataclass(frozen=True)
class CCfgFactorGrp:
    factor_class: TFactorClass

    @property
    def factor_names(self) -> TFactorNames:
        raise NotImplementedError

    @property
    def factors(self) -> TFactors:
        res = [CFactor(self.factor_class, factor_name) for factor_name in self.factor_names]
        return TFactors(res)


"""
--- CCfgFactorGrp with Arguments   ---
--- User may not use them directly ---
"""


@dataclass(frozen=True)
class CCfgFactorGrpWin(CCfgFactorGrp):
    wins: list[int]

    # --- name
    def name_vanilla(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}{w:03d}")

    @property
    def names_vanilla(self) -> TFactorNames:
        return [self.name_vanilla(w) for w in self.wins]

    # --- name
    def name_vol(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}{w:03d}VOL")

    @property
    def names_vol(self) -> TFactorNames:
        return [self.name_vol(w) for w in self.wins]

    # --- extra: Delay
    def name_delay(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}{w:03d}D")

    @property
    def names_delay(self) -> TFactorNames:
        return [self.name_delay(w) for w in self.wins]

    # --- extra: Diff
    def name_diff(self) -> TFactorName:
        return TFactorName(f"{self.factor_class}DIF")

    @property
    def names_diff(self) -> TFactorNames:
        return [self.name_diff()]

    # --- extra: Res
    def name_res(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}RES{w:03d}")

    @property
    def names_res(self) -> TFactorNames:
        return [self.name_res(w) for w in self.wins]

    # --- extra: PA
    def name_pa(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}PA{w:03d}")

    @property
    def names_pa(self) -> TFactorNames:
        return [self.name_pa(w) for w in self.wins]

    # --- extra: LA
    def name_la(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}LA{w:03d}")

    @property
    def names_la(self) -> TFactorNames:
        return [self.name_la(w) for w in self.wins]

    # ---------------------------
    # ----- other functions -----
    # ---------------------------
    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla

    def buffer_bgn_date(self, bgn_date: str, calendar: CCalendar, shift: int = -5) -> str:
        return calendar.get_next_date(bgn_date, -max(self.wins) + shift)


@dataclass(frozen=True)
class CCfgFactorGrpWinLambda(CCfgFactorGrp):
    wins: list[int]
    lbds: list[float]

    # --- vanilla
    def name_vanilla(self, win: int, lbd: float) -> TFactorName:
        return TFactorName(f"{self.factor_class}{win:03d}L{int(lbd * 100):02d}")

    @property
    def names_vanilla(self) -> TFactorNames:
        return [self.name_vanilla(win, lbd) for win, lbd in product(self.wins, self.lbds)]

    # --- lbd
    def name_lbd(self, lbd: float) -> TFactorName:
        return TFactorName(f"{self.factor_class}L{int(lbd * 100):02d}")

    @property
    def names_lbd(self) -> TFactorNames:
        return [self.name_lbd(lbd) for lbd in self.lbds]

    # ---------------------------
    # ----- other functions -----
    # ---------------------------
    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla

    def buffer_bgn_date(self, bgn_date: str, calendar: CCalendar, shift: int = -5) -> str:
        return calendar.get_next_date(bgn_date, -max(self.wins) + shift)


@dataclass(frozen=True)
class CCfgFactorGrpLambda(CCfgFactorGrp):
    lbds: list[float]

    def name_vanilla(self, lbd: float) -> TFactorName:
        return TFactorName(f"{self.factor_class}L{int(lbd * 100):02d}")

    @property
    def names_vanilla(self) -> TFactorNames:
        return [self.name_vanilla(lbd) for lbd in self.lbds]

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla
