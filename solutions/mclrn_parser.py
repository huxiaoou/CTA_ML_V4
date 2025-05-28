import os
import yaml
from typing import Any
from loguru import logger
from husfort.qutility import SFG, SFY
from husfort.qutility import check_and_mkdir
from solutions.mclrn import (
    CTestMclrn, CTestMclrnBaseLine,
    CTestMclrnLinear, CTestMclrnRidge, CTestMclrnLogistic, CTestMclrnMlp,
    CTestMclrnLGBM, CTestMclrnXGB, CTestMclrnSVM
)
from typedefs.typedefInstrus import TUniverse
from typedefs.typedefReturns import TRets, CRet, TFacRetType
from typedefs.typedefModels import CTestData, CTestModel, TModelType
from typedef import TFacUnvrsOpts

"""
-------------
--- write ---
-------------
"""


def gen_configs_for_mclrn_tests(
        mclrn_dir: str,
        mclrn_tests_config_file: str,
        rets: TRets,
        test_models: list[CTestModel],
):
    iter_args: list[dict] = []
    for ret in rets:
        for test_model in test_models:
            for fac_ret_type in TFacRetType:
                d = {
                    "test_data": {
                        "ret": ret.ret_name,
                        "ret_type": fac_ret_type.value,
                        "factors": ret.ret_class.value,
                        "factor_type": fac_ret_type.value,
                    },
                    "test_model": test_model.to_dict(),
                }
                iter_args.append(d)
    check_and_mkdir(mclrn_dir)
    model_config_path = os.path.join(mclrn_dir, mclrn_tests_config_file)
    with open(model_config_path, "w+") as f:
        yaml.dump_all([iter_args], f)
    logger.info(f"{len(iter_args)} tests are saved to {SFG(model_config_path)}")
    return 0


"""
------------
--- read ---
------------
"""


def load_mclrn_tests(mclrn_dir: str, mclrn_tests_config_file: str) -> list[dict[str, Any]]:
    model_config_path = os.path.join(mclrn_dir, mclrn_tests_config_file)
    with open(model_config_path, "r") as f:
        config_models = yaml.safe_load(f)
    logger.info(f"model configs are loaded from {SFG(model_config_path)}")
    return config_models


def parse_config_to_mclrn_test(
        config: dict[str, Any],
        factors_universe_options: TFacUnvrsOpts,
        universe: TUniverse,
        factors_avlb_raw_dir: str,
        factors_avlb_neu_dir: str,
        test_returns_avlb_raw_dir: str,
        test_returns_avlb_neu_dir: str,
        mclrn_dir: str,
):
    test_data, test_model = config["test_data"], config["test_model"]
    factors = factors_universe_options[(test_data["ret_type"], test_data["factors"])]
    td = CTestData(
        ret=CRet.parse_from_name(test_data["ret"]),
        ret_type=test_data["ret_type"],
        factors=factors,
        factor_type=test_data["factor_type"],
        universe=universe,
        factors_avlb_dir=factors_avlb_raw_dir if test_data["factor_type"] == TFacRetType.RAW
        else factors_avlb_neu_dir,
        test_returns_avlb_dir=test_returns_avlb_raw_dir if test_data["ret_type"] == TFacRetType.RAW
        else test_returns_avlb_neu_dir,
    )
    tm = CTestModel(**test_model)
    x: dict[str, type[CTestMclrn]] = {
        TModelType.BASELINE: CTestMclrnBaseLine,
        TModelType.LINEAR: CTestMclrnLinear,
        TModelType.RIDGE: CTestMclrnRidge,
        TModelType.LOGISTIC: CTestMclrnLogistic,
        TModelType.SVM: CTestMclrnSVM,
        TModelType.MLP: CTestMclrnMlp,
        TModelType.LGBM: CTestMclrnLGBM,
        TModelType.XGB: CTestMclrnXGB,
    }
    if (test_mclrn := x.get(tm.model_type, None)) is None:
        raise ValueError(f"Invalid model type = {SFY(tm.model_type)}")
    test = test_mclrn(test_data=td, test_model=tm, mclrn_dir=mclrn_dir)
    return test


def parse_configs_to_mclrn_tests(
        config_models: list[dict[str, Any]],
        factors_universe_options: TFacUnvrsOpts,
        universe: TUniverse,
        factors_avlb_raw_dir: str,
        factors_avlb_neu_dir: str,
        test_returns_avlb_raw_dir: str,
        test_returns_avlb_neu_dir: str,
        mclrn_dir: str,
) -> list[CTestMclrn]:
    tests: list[CTestMclrn] = []
    for config in config_models:
        test = parse_config_to_mclrn_test(
            config=config,
            factors_universe_options=factors_universe_options,
            universe=universe,
            factors_avlb_raw_dir=factors_avlb_raw_dir,
            factors_avlb_neu_dir=factors_avlb_neu_dir,
            test_returns_avlb_raw_dir=test_returns_avlb_raw_dir,
            test_returns_avlb_neu_dir=test_returns_avlb_neu_dir,
            mclrn_dir=mclrn_dir,
        )
        tests.append(test)
    logger.info(f"{SFG(len(tests))} tests are loaded.")
    return tests


def gen_tests(
        mclrn_dir: str,
        mclrn_tests_config_file: str,
        factors_universe_options: TFacUnvrsOpts,
        universe: TUniverse,
        factors_avlb_raw_dir: str,
        factors_avlb_neu_dir: str,
        test_returns_avlb_raw_dir: str,
        test_returns_avlb_neu_dir: str,

) -> list[CTestMclrn]:
    config_models = load_mclrn_tests(
        mclrn_dir=mclrn_dir,
        mclrn_tests_config_file=mclrn_tests_config_file,
    )
    tests = parse_configs_to_mclrn_tests(
        config_models=config_models,
        factors_universe_options=factors_universe_options,
        universe=universe,
        factors_avlb_raw_dir=factors_avlb_raw_dir,
        factors_avlb_neu_dir=factors_avlb_neu_dir,
        test_returns_avlb_raw_dir=test_returns_avlb_raw_dir,
        test_returns_avlb_neu_dir=test_returns_avlb_neu_dir,
        mclrn_dir=mclrn_dir,
    )
    return tests
