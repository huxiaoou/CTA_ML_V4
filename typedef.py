import os
from itertools import product
from dataclasses import dataclass
from husfort.qsqlite import CDbStruct
from typedefs.typedefInstrus import TUniverse
from typedefs.typedefReturns import TFacRetType, TReturnClass, CRet, TRets
from typedefs.typedefFactors import TFactors
from typedefs.typedefModels import CTestModel


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
