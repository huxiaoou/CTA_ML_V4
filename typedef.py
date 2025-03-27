import os
from typing import NewType
from enum import StrEnum
from dataclasses import dataclass
from itertools import product
from husfort.qsqlite import CDbStruct

"""
----------------------------
Part I: test returns
----------------------------
"""


class TReturnClass(StrEnum):
    OPN = "Opn"
    CLS = "Cls"


TReturnName = NewType("TReturnName", str)
TReturnNames = list[TReturnName]


@dataclass(frozen=True)
class CRet:
    ret_class: TReturnClass
    win: int
    lag: int

    @property
    def sid(self) -> str:
        return f"{self.win:03d}L{self.lag:d}"

    @property
    def ret_name(self) -> TReturnName:
        return TReturnName(f"{self.ret_class}{self.sid}")

    @property
    def shift(self) -> int:
        return self.win + self.lag

    @staticmethod
    def parse_from_name(return_name: str) -> "CRet":
        """

        :param return_name: like "Cls001L1"
        :return:
        """

        ret_type = TReturnClass(return_name[0:3])
        win = int(return_name[3:6])
        lag = int(return_name[7])
        return CRet(ret_class=ret_type, win=win, lag=lag)


TRets = list[CRet]

"""
----------------------------------
Part II: factors configuration
----------------------------------
"""

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
    @property
    def factor_class(self) -> TFactorClass:
        raise NotImplementedError

    @property
    def factor_names(self) -> TFactorNames:
        raise NotImplementedError

    @property
    def factors(self) -> TFactors:
        res = [CFactor(self.factor_class, factor_name) for factor_name in self.factor_names]
        return TFactors(res)


@dataclass(frozen=True)
class CCfgFactorGrpWin(CCfgFactorGrp):
    wins: list[int]

    @property
    def factor_names(self) -> TFactorNames:
        return TFactorNames([TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins])


class CCfgFactorGrpMTM(CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("MTM")


class CCfgFactorGrpSKEW(CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SKEW")


class CCfgFactorGrpKURT(CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("KURT")


"""
--------------------------------------
Part III: Instruments and Universe
--------------------------------------
"""


@dataclass(frozen=True)
class CCfgInstru:
    sectorL0: str
    sectorL1: str


TInstruName = NewType("TInstruName", str)
TUniverse = NewType("TUniverse", dict[TInstruName, CCfgInstru])

"""
--------------------------------------
Part IV: Simulations
--------------------------------------
"""

"""
--------------------------------
Part V: models
--------------------------------
"""

"""
--------------------------------
Part VI: generic and project
--------------------------------
"""


@dataclass(frozen=True)
class CCfgMktIdx:
    equity: str
    commodity: str

    @property
    def idxes(self) -> list[str]:
        return [self.equity, self.commodity]


@dataclass(frozen=True)
class CCfgAvlbUnvrs:
    win: int
    amount_threshold: float


@dataclass(frozen=True)
class CCfgTrn:
    wins: list[int]


@dataclass(frozen=True)
class CCfgPrd:
    wins: list[int]


@dataclass(frozen=True)
class CCfgSim:
    wins: list[int]


@dataclass(frozen=True)
class CCfgConst:
    COST: float
    SECTORS: list[str]
    LAG: int


"""
--------------------------------
Part VI: generic and project
--------------------------------
"""


@dataclass(frozen=True)
class CCfgDbStruct:
    # --- shared database
    macro: CDbStruct
    forex: CDbStruct
    fmd: CDbStruct
    position: CDbStruct
    basis: CDbStruct
    stock: CDbStruct
    preprocess: CDbStruct
    minute_bar: CDbStruct


@dataclass(frozen=True)
class CCfgProj:
    # --- shared
    calendar_path: str
    root_dir: str
    db_struct_path: str
    alternative_dir: str
    market_index_path: str
    by_instru_pos_dir: str
    by_instru_pre_dir: str
    by_instru_min_dir: str

    # --- project
    project_root_dir: str

    # --- project parameters
    universe: TUniverse
    avlb_unvrs: CCfgAvlbUnvrs
    mkt_idxes: CCfgMktIdx
    const: CCfgConst
    trn: CCfgTrn
    prd: CCfgPrd
    sim: CCfgSim

    @property
    def test_rets_wins(self) -> list[int]:
        return self.sim.wins + self.prd.wins

    @property
    def all_rets(self) -> TRets:
        return [CRet(ret_class=TReturnClass(rc), win=w, lag=self.const.LAG)
                for rc, w in product(TReturnClass, self.test_rets_wins)]

    @property
    def available_dir(self) -> str:
        return os.path.join(self.project_root_dir, "available")

    @property
    def market_dir(self):
        return os.path.join(self.project_root_dir, "market")

    @property
    def test_returns_by_instru_dir(self):
        return os.path.join(self.project_root_dir, "test_returns_by_instru")

    @property
    def test_returns_avlb_raw_dir(self):
        return os.path.join(self.project_root_dir, "test_returns_avlb_raw")

    @property
    def test_returns_avlb_neu_dir(self):
        return os.path.join(self.project_root_dir, "test_returns_avlb_neu")

    @property
    def factors_by_instru_dir(self):
        return os.path.join(self.project_root_dir, "factors_by_instru")

    @property
    def factors_avlb_raw_dir(self):
        return os.path.join(self.project_root_dir, "factors_avlb_raw")

    @property
    def factors_avlb_neu_dir(self):
        return os.path.join(self.project_root_dir, "factors_avlb_neu")


if __name__ == "__main__":
    ret = CRet.parse_from_name("Cls010L1")
    print(f"ret={ret}")
    print(f"ret.ret_class = {ret.ret_class}")
    print(f"ret.ret_name = {ret.ret_name}")
    assert ret.ret_class == "Cls"

    cfg_factor_grp = CCfgFactorGrpMTM(wins=[2, 3, 5])
    print(f"cfg_factor_grp.factor_class = {cfg_factor_grp.factor_class}")
    print(f"cfg_factor_grp.factor_names = {cfg_factor_grp.factor_names}")
    print(f"cfg_factor_grp.factors = {cfg_factor_grp.factors}")
    assert cfg_factor_grp.factor_class == "MTM"

    cfg_factor_grp = CCfgFactorGrpSKEW(wins=[6, 8, 10])
    print(f"cfg_factor_grp.factor_class = {cfg_factor_grp.factor_class}")
    print(f"cfg_factor_grp.factor_names = {cfg_factor_grp.factor_names}")
    print(f"cfg_factor_grp.factors = {cfg_factor_grp.factors}")
    assert cfg_factor_grp.factor_class == "SKEW"
