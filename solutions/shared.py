import os
from husfort.qsqlite import CDbStruct, CSqlTable, CSqlVar
from typedef import TReturnClass, CRet, TFactorClass, TFactors


# ----------------------------------------
# ------ sqlite3 database structure ------
# ----------------------------------------

def get_avlb_db(available_dir: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=available_dir,
        db_name="available.db",
        table=CSqlTable(
            name="available",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[
                CSqlVar("return", "REAL"),
                CSqlVar("amount", "REAL"),
                CSqlVar("sectorL0", "TEXT"),
                CSqlVar("sectorL1", "TEXT"),
            ],
        ),
    )


def get_market_db(market_dir: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=market_dir,
        db_name="market.db",
        table=CSqlTable(
            name="market",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[
                CSqlVar("market", "REAL"),
                CSqlVar("C", "REAL"),
                CSqlVar("AUG", "REAL"),
                CSqlVar("MTL", "REAL"),
                CSqlVar("BLK", "REAL"),
                CSqlVar("OIL", "REAL"),
                CSqlVar("CHM", "REAL"),
                CSqlVar("AGR", "REAL"),
                CSqlVar("INH0100_NHF", "REAL"),
                CSqlVar("I881001_WI", "REAL"),
            ],
        )
    )


def gen_test_returns_by_instru_db(
        instru: str,
        test_returns_by_instru_dir: str,
        save_id: TReturnClass,
        ret: CRet,
) -> CDbStruct:
    """

    :param instru: 'RB.SHFE'
    :param test_returns_by_instru_dir: test_returns_by_instru_dir
    :param save_id: 'Opn' or 'Cls'
    :param ret:
    :return:
    """
    return CDbStruct(
        db_save_dir=os.path.join(test_returns_by_instru_dir, save_id),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name=ret.ret_name,
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT"), CSqlVar(ret.ret_name, "REAL")],
        )
    )


def gen_test_returns_avlb_db(
        test_returns_avlb_dir: str,
        save_id: TReturnClass,
        ret: CRet,
) -> CDbStruct:
    """

    :param test_returns_avlb_dir: 'raw' or 'neu'
    :param save_id: 'Opn' or 'Cls'
    :param ret:
    :return:
    """

    return CDbStruct(
        db_save_dir=test_returns_avlb_dir,
        db_name=f"{save_id}.db",
        table=CSqlTable(
            name=ret.ret_name,
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(ret.ret_name, "REAL")],
        )
    )


def gen_factors_by_instru_db(
        instru: str,
        factors_by_instru_dir: str,
        save_id: TFactorClass,
        factors: TFactors,
) -> CDbStruct:
    """

    :param instru: 'RB.SHFE'
    :param factors_by_instru_dir: factors_by_instru_dir
    :param save_id:
    :param factors:
    :return:
    """
    return CDbStruct(
        db_save_dir=os.path.join(factors_by_instru_dir, save_id),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(fac.factor_name, "REAL") for fac in factors],
        )
    )


def gen_factors_avlb_db(
        factors_avlb_dir: str,
        save_id: TFactorClass,
        factors: TFactors,
) -> CDbStruct:
    """

    :param factors_avlb_dir: 'raw' or 'neu'
    :param save_id:
    :param factors:
    :return:
    """

    return CDbStruct(
        db_save_dir=factors_avlb_dir,
        db_name=f"{save_id}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(fac.factor_name, "REAL") for fac in factors],
        )
    )
