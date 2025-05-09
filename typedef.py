import os
from typing import NewType
from enum import StrEnum
from dataclasses import dataclass, fields
from itertools import product
from husfort.qsqlite import CDbStruct
from husfort.qcalendar import CCalendar

"""
----------------------------
Part I: test returns
----------------------------
"""


class TReturnClass(StrEnum):
    OPN = "Opn"
    CLS = "Cls"


class TFacRetType(StrEnum):
    RAW = "raw"
    NEU = "neu"


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


class TFactorClass(StrEnum):
    MTM = "MTM"
    SKEW = "SKEW"
    KURT = "KURT"
    RS = "RS"
    BASIS = "BASIS"
    TS = "TS"
    LIQUIDITY = "LIQUIDITY"
    SIZE = "SIZE"
    MF = "MF"
    JUMP = "JUMP"


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


"""
--- CCfgFactorGrp with Arguments   ---
--- User may not use them directly ---
"""


@dataclass(frozen=True)
class _CCfgFactorGrpWin(CCfgFactorGrp):
    wins: list[int]

    # --- name
    def name_vanilla(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}{w:03d}")

    def factor_name(self, w: int) -> TFactorName:
        return self.name_vanilla(w)

    # --- names
    @property
    def names_vanilla(self) -> TFactorNames:
        return [self.name_vanilla(w) for w in self.wins]

    @property
    def factor_names(self) -> TFactorNames:
        return [self.factor_name(w) for w in self.wins]

    # --- other
    def buffer_bgn_date(self, bgn_date: str, calendar: CCalendar, shift: int = -5) -> str:
        return calendar.get_next_date(bgn_date, -max(self.wins) + shift)

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


"""
--- CfgFactorGrp for factors    ---
--- Define this for each factor ---
"""


class CCfgFactorGrpMTM(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.MTM

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_diff


class CCfgFactorGrpSKEW(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.SKEW

    @property
    def factor_names(self) -> TFactorNames:
        return super().factor_names + self.names_delay


class CCfgFactorGrpKURT(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.KURT

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_diff


class CCfgFactorGrpRS(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.RS

    def name_rspa(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}PA{w:03d}")

    def name_rsla(self, w: int) -> TFactorName:
        return TFactorName(f"{self.factor_class}LA{w:03d}")

    @property
    def names_rspa(self) -> TFactorNames:
        return [self.name_rspa(w) for w in self.wins]

    @property
    def names_rsla(self) -> TFactorNames:
        return [self.name_rsla(w) for w in self.wins]

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_rspa + self.names_rsla + self.names_diff


class CCfgFactorGrpBASIS(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.BASIS

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_res


class CCfgFactorGrpTS(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.TS

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_res


class CCfgFactorGrpLIQUIDITY(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.LIQUIDITY

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_diff


class CCfgFactorGrpSIZE(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.SIZE

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_diff


class CCfgFactorGrpMF(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.MF

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_diff


class CCfgFactorGrpJUMP(_CCfgFactorGrpWin):
    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass.JUMP


@dataclass(frozen=True)
class CCfgFactors:
    MTM: CCfgFactorGrpMTM
    SKEW: CCfgFactorGrpSKEW
    KURT: CCfgFactorGrpKURT
    RS: CCfgFactorGrpRS
    BASIS: CCfgFactorGrpBASIS
    TS: CCfgFactorGrpTS
    LIQUIDITY: CCfgFactorGrpLIQUIDITY
    SIZE: CCfgFactorGrpSIZE
    MF: CCfgFactorGrpMF
    JUMP: CCfgFactorGrpJUMP

    @property
    def classes(self) -> list[str]:
        return [f.name for f in fields(self)]


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


class TModelType(StrEnum):
    LINEAR = "LINEAR"
    RIDGE = "RIDGE"
    LOGISTIC = "LOGISTIC"
    LGBM = "LGBM"
    XGB = "XGB"


@dataclass(frozen=True)
class CTestModel:
    model_type: TModelType
    trn_win: int
    using_instru: bool = False
    classification: bool = False
    cv: int = 0
    early_stopping: int = 0
    hyper_param_grids: dict | dict[str, list] = None  # must be provided if cv > 0

    @property
    def save_id(self) -> str:
        ui = "UI" if self.using_instru else "NI"
        return f"{self.model_type}-W{self.trn_win:03d}-CV{self.cv:02d}-ES{self.early_stopping:02d}-{ui}"

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "trn_win": self.trn_win,
            "using_instru": self.using_instru,
            "classification": self.classification,
            "cv": self.cv,
            "early_stopping": self.early_stopping,
            "hyper_param_grids": self.hyper_param_grids,
        }


@dataclass(frozen=True)
class CTestData:
    ret: CRet
    ret_type: TFacRetType
    factors: TFactors
    factor_type: TFacRetType
    universe: TUniverse
    factors_avlb_dir: str
    test_returns_avlb_dir: str

    @property
    def save_id(self) -> str:
        return f"factors-{self.factor_type}-{self.ret.ret_name}-{self.ret_type}"


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
class CCfgPrd:
    wins: list[int]


@dataclass(frozen=True)
class CCfgSim:
    wins: list[int]


@dataclass(frozen=True)
class CCfgConst:
    INIT_CASH: float
    COST_RATE: float
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


TFacUnvrsOpts = dict[tuple[TFacRetType, TReturnClass], TFactors]


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
    instru_info_path: str

    # --- project
    project_root_dir: str

    # --- project parameters
    universe: TUniverse
    avlb_unvrs: CCfgAvlbUnvrs
    mkt_idxes: CCfgMktIdx
    const: CCfgConst
    prd: CCfgPrd
    sim: CCfgSim
    factors: dict
    test_models: list[CTestModel]
    factors_universe_options: TFacUnvrsOpts

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

    @property
    def ic_tests_dir(self):
        return os.path.join(self.project_root_dir, "ic_tests")

    @property
    def mclrn_dir(self):
        return os.path.join(self.project_root_dir, "mclrn")

    @property
    def mclrn_tests_config_file(self):
        return "mclrn_tests_config.yaml"

    @property
    def signals_dir(self):
        return os.path.join(self.project_root_dir, "signals")

    @property
    def simulations_dir(self):
        return os.path.join(self.project_root_dir, "simulations")

    @property
    def evaluations_dir(self):
        return os.path.join(self.project_root_dir, "evaluations")

    @property
    def sims_quick_dir(self):
        return os.path.join(self.project_root_dir, "sims_quick")


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

    model_ridge = CTestModel(
        model_type=TModelType.RIDGE,
        trn_win=60,
        cv=5,
        hyper_param_grids={"alphas": [0.1, 1.0, 10.0]},
    )
    print(f"model_ridge = {model_ridge}")

    test_data = CTestData(
        ret=CRet.parse_from_name("Opn010L1"),
        ret_type=TFacRetType.RAW,
        factors=[
            CFactor(TFactorClass.MTM, TFactorName("MTM001")),
            CFactor(TFactorClass.SKEW, TFactorName("SKEW002")),
        ],
        factor_type=TFacRetType.NEU,
        universe=TUniverse({
            TInstruName("RB.SHF"): CCfgInstru("C", "BLK"),
            TInstruName("M.DCE"): CCfgInstru("C", "OIL")
        }),
        factors_avlb_dir="",
        test_returns_avlb_dir="",
    )
    print(test_data)

    s = f"{model_ridge.save_id}.{test_data.save_id}"
    print(s)
