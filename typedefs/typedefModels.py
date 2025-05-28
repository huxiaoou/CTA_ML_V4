from enum import StrEnum
from dataclasses import dataclass
from typedefs.typedefReturns import CRet, TFacRetType
from typedefs.typedefFactors import TFactors
from typedefs.typedefInstrus import TUniverse


class TModelType(StrEnum):
    BASELINE = "BASELINE"
    LINEAR = "LINEAR"
    RIDGE = "RIDGE"
    LOGISTIC = "LOGISTIC"
    MLP = "MLP"
    SVM = "SVM"
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
