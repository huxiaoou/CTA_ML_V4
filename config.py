import yaml
from husfort.qsqlite import CDbStruct, CSqlTable
from typedefs.typedefInstrus import TUniverse, TInstruName, CCfgInstru
from typedefs.typedefReturns import TReturnClass, TFacRetType
from typedefs.typedefFactors import TFactorClass, TFactorName, CFactor
from typedef import CCfgPrd, CCfgSim, TFacUnvrsOpts, CCfgAvlbUnvrs, CCfgMktIdx, CCfgConst, CTestModel
from typedef import CCfgProj, CCfgDbStruct
from solutions.factor import CCfgFactors

# ---------- project configuration ----------

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

universe = TUniverse({TInstruName(k): CCfgInstru(**v) for k, v in _config["universe"].items()})
factors_universe_options: TFacUnvrsOpts = {}
for fac_ret_type in TFacRetType:
    for ret_class, opts in _config["factors_universe_options"][fac_ret_type].items():
        ret_class: TReturnClass
        opts: list[tuple[str, str]]
        factors_universe_options[(fac_ret_type, ret_class)] = [
            CFactor(TFactorClass(z[0]), TFactorName(z[1])) for z in opts
        ]

proj_cfg = CCfgProj(
    # --- shared data path
    calendar_path=_config["path"]["calendar_path"],
    root_dir=_config["path"]["root_dir"],
    db_struct_path=_config["path"]["db_struct_path"],
    alternative_dir=_config["path"]["alternative_dir"],
    market_index_path=_config["path"]["market_index_path"],
    by_instru_pos_dir=_config["path"]["by_instru_pos_dir"],
    by_instru_pre_dir=_config["path"]["by_instru_pre_dir"],
    by_instru_min_dir=_config["path"]["by_instru_min_dir"],
    instru_info_path=_config["path"]["instru_info_path"],

    # --- project data root dir
    project_root_dir=_config["path"]["project_root_dir"],

    # --- global settings
    universe=universe,
    avlb_unvrs=CCfgAvlbUnvrs(**_config["available"]),
    mkt_idxes=CCfgMktIdx(**_config["mkt_idxes"]),
    const=CCfgConst(**_config["CONST"]),
    prd=CCfgPrd(**_config["prd"]),
    sim=CCfgSim(**_config["sim"]),
    factors=_config["factors"],
    test_models=[CTestModel(**d) for d in _config["test_models"]],
    factors_universe_options=factors_universe_options,
)

# ---------- databases structure ----------
with open(proj_cfg.db_struct_path, "r") as f:
    _db_struct = yaml.safe_load(f)

db_struct_cfg = CCfgDbStruct(
    macro=CDbStruct(
        db_save_dir=proj_cfg.alternative_dir,
        db_name=_db_struct["macro"]["db_name"],
        table=CSqlTable(cfg=_db_struct["macro"]["table"]),
    ),
    forex=CDbStruct(
        db_save_dir=proj_cfg.alternative_dir,
        db_name=_db_struct["forex"]["db_name"],
        table=CSqlTable(cfg=_db_struct["forex"]["table"]),
    ),
    fmd=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["fmd"]["db_name"],
        table=CSqlTable(cfg=_db_struct["fmd"]["table"]),
    ),
    position=CDbStruct(
        db_save_dir=proj_cfg.by_instru_pos_dir,
        db_name=_db_struct["position"]["db_name"],
        table=CSqlTable(cfg=_db_struct["position"]["table"]),
    ),
    basis=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["basis"]["db_name"],
        table=CSqlTable(cfg=_db_struct["basis"]["table"]),
    ),
    stock=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["stock"]["db_name"],
        table=CSqlTable(cfg=_db_struct["stock"]["table"]),
    ),
    preprocess=CDbStruct(
        db_save_dir=proj_cfg.by_instru_pre_dir,
        db_name=_db_struct["preprocess"]["db_name"],
        table=CSqlTable(cfg=_db_struct["preprocess"]["table"]),
    ),
    minute_bar=CDbStruct(
        db_save_dir=proj_cfg.by_instru_min_dir,
        db_name=_db_struct["fMinuteBar"]["db_name"],
        table=CSqlTable(cfg=_db_struct["fMinuteBar"]["table"]),
    ),
)

# --- factors ---
cfg_factors = CCfgFactors(algs_dir="factor_algs", cfg_data=proj_cfg.factors)

if __name__ == "__main__":
    sep = "-" * 80

    print(sep)
    print(f"Size of universe = {len(universe)}")
    for instru, sectors in universe.items():
        print(f"{instru:>6s}: {sectors}")

    print(sep)
    for test_model in proj_cfg.test_models:
        print(test_model)

    print(sep)
    print(factors_universe_options)

    print(sep)
    print(cfg_factors)
