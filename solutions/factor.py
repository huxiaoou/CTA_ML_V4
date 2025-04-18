import numpy as np
import pandas as pd
import scipy.stats as sps
import multiprocessing as mp
from typing import Literal
from loguru import logger
from rich.progress import track, Progress
from husfort.qutility import SFG, SFY, error_handler, check_and_makedirs
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from husfort.qcalendar import CCalendar
from typedef import CCfgFactorGrp, TUniverse
from typedef import TFactorClass, TFactors
from solutions.shared import gen_factors_by_instru_db, gen_factors_avlb_db


class _CFactorsByInstruDbOperator:
    def __init__(self, factor_grp: CCfgFactorGrp, factors_by_instru_dir: str):
        self.factor_grp = factor_grp
        self.factors_by_instru_dir: str = factors_by_instru_dir

    def get_instru_db(self, instru: str) -> CDbStruct:
        return gen_factors_by_instru_db(
            instru=instru,
            factors_by_instru_dir=self.factors_by_instru_dir,
            factor_class=self.factor_grp.factor_class,
            factors=self.factor_grp.factors,
        )

    def load_by_instru(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_instru = self.get_instru_db(instru)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="r",
        )
        factor_data = sqldb.read_by_range(bgn_date, stp_date)
        factor_data[self.factor_grp.factor_names] = (
            factor_data[self.factor_grp.factor_names].astype(np.float64).fillna(np.nan)
        )
        return factor_data

    def save_by_instru(self, factor_data: pd.DataFrame, instru: str, calendar: CCalendar):
        """

        :param factor_data: a pd.DataFrame with first 2 columns must be = ["trade_date", "ticker"]
                  then followed by factor names
        :param instru:
        :param calendar:
        :return:
        """
        db_struct_instru = self.get_instru_db(instru)
        check_and_makedirs(db_struct_instru.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="a",
        )
        if sqldb.check_continuity(factor_data["trade_date"].iloc[0], calendar) == 0:
            sqldb.update(factor_data[db_struct_instru.table.vars.names])
        return 0

    def get_factor_data(self, input_data: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        """

        :param input_data:
        :param bgn_date:
        :return: a pd.DataFrame with first 2 columns must be = ["trade_date", "ticker"]
                  then followed by factor names
        """
        input_data = input_data.query(f"trade_date >= '{bgn_date}'")
        factor_data = input_data[["trade_date", "ticker"] + self.factor_grp.factor_names]
        return factor_data

    @staticmethod
    def rename_ticker(data: pd.DataFrame, old_name: str = "ticker_major") -> None:
        data.rename(columns={old_name: "ticker"}, inplace=True)


class _CFactorsByInstruMoreDb(_CFactorsByInstruDbOperator):
    def __init__(
            self,
            factor_grp: CCfgFactorGrp,
            factors_by_instru_dir: str,
            universe: TUniverse,
            db_struct_preprocess: CDbStruct = None,
            db_struct_minute_bar: CDbStruct | None = None,
            db_struct_pos: CDbStruct | None = None,
            db_struct_forex: CDbStruct | None = None,
            db_struct_macro: CDbStruct | None = None,
            db_struct_mkt: CDbStruct | None = None,
    ):
        super().__init__(factor_grp, factors_by_instru_dir)
        self.universe = universe
        self.db_struct_preprocess = db_struct_preprocess
        self.db_struct_minute_bar = db_struct_minute_bar
        self.db_struct_pos = db_struct_pos
        self.db_struct_forex = db_struct_forex
        self.db_struct_macro = db_struct_macro
        self.db_struct_mkt = db_struct_mkt

    def load_preprocess(self, instru: str, bgn_date: str, stp_date: str, values: list[str] = None) -> pd.DataFrame:
        if self.db_struct_preprocess is not None:
            db_struct_instru = self.db_struct_preprocess.copy_to_another(another_db_name=f"{instru}.db")
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date, value_columns=values)
        else:
            raise ValueError("Argument 'db_struct_preprocess' must be provided")

    def load_minute_bar(self, instru: str, bgn_date: str, stp_date: str, values: list[str] = None) -> pd.DataFrame:
        if self.db_struct_minute_bar is not None:
            db_struct_instru = self.db_struct_minute_bar.copy_to_another(another_db_name=f"{instru}.db")
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date, value_columns=values)
        else:
            raise ValueError("Argument 'db_struct_minute_bar' must be provided")

    def load_pos(self, instru: str, bgn_date: str, stp_date: str, values: list[str] = None) -> pd.DataFrame:
        if self.db_struct_pos is not None:
            db_struct_instru = self.db_struct_pos.copy_to_another(another_db_name=f"{instru}.db")
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date, value_columns=values)
        else:
            raise ValueError("Argument 'db_struct_pos' must be provided")

    def load_forex(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        if self.db_struct_forex is not None:
            sqldb = CMgrSqlDb(
                db_save_dir=self.db_struct_forex.db_save_dir,
                db_name=self.db_struct_forex.db_name,
                table=self.db_struct_forex.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date)
        else:
            raise ValueError("Argument 'db_struct_forex' must be provided")

    def load_macro(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        if self.db_struct_macro is not None:
            sqldb = CMgrSqlDb(
                db_save_dir=self.db_struct_macro.db_save_dir,
                db_name=self.db_struct_macro.db_name,
                table=self.db_struct_macro.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date)
        else:
            raise ValueError("Argument 'db_struct_macro' must be provided")

    def load_mkt(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        if self.db_struct_mkt is not None:
            sqldb = CMgrSqlDb(
                db_save_dir=self.db_struct_mkt.db_save_dir,
                db_name=self.db_struct_mkt.db_name,
                table=self.db_struct_mkt.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date)
        else:
            raise ValueError("Argument 'db_struct_mkt' must be provided")


class CFactorsByInstru(_CFactorsByInstruMoreDb):
    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        """
        This function is to be realized by specific factors

        :return : a pd.DataFrame with first 2 columns must be = ["trade_date", "ticker"]
                  then followed by factor names
        """
        raise NotImplementedError

    def process_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar):
        factor_data = self.cal_factor_by_instru(instru, bgn_date, stp_date, calendar)
        self.save_by_instru(factor_data, instru, calendar)
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar, call_multiprocess: bool, processes: int):
        description = f"Calculating factor {SFY(self.factor_grp.factor_class)}"
        if call_multiprocess:
            with Progress() as pb:
                main_task = pb.add_task(description, total=len(self.universe))
                with mp.get_context("spawn").Pool(processes) as pool:
                    for instru in self.universe:
                        pool.apply_async(
                            self.process_by_instru,
                            args=(instru, bgn_date, stp_date, calendar),
                            callback=lambda _: pb.update(main_task, advance=1),
                            error_callback=error_handler,
                        )
                    pool.close()
                    pool.join()
        else:
            for instru in track(self.universe, description=description):
                self.process_by_instru(instru, bgn_date, stp_date, calendar)
        return 0


class CFactorsAvlb(_CFactorsByInstruDbOperator):
    def __init__(
            self,
            factor_grp: CCfgFactorGrp,
            universe: TUniverse,
            factors_by_instru_dir: str,
            factors_avlb_raw_dir: str,
            factors_avlb_neu_dir: str,
            db_struct_avlb: CDbStruct,
    ):
        super().__init__(factor_grp, factors_by_instru_dir)
        self.universe = universe
        self.factors_avlb_raw_dir = factors_avlb_raw_dir
        self.factors_avlb_neu_dir = factors_avlb_neu_dir
        self.db_struct_avlb = db_struct_avlb

    def load_ref_fac(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        ref_dfs: list[pd.DataFrame] = []
        for instru in self.universe:
            df = self.load_by_instru(instru, bgn_date=bgn_date, stp_date=stp_date)
            df["instrument"] = instru
            ref_dfs.append(df)
        res = pd.concat(ref_dfs, axis=0, ignore_index=False)
        res = res.reset_index().sort_values(by=["trade_date"], ascending=True)
        res = res[["trade_date", "instrument"] + self.factor_grp.factor_names]
        return res

    def load_available(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_avlb.db_save_dir,
            db_name=self.db_struct_avlb.db_name,
            table=self.db_struct_avlb.table,
            mode="r",
        )
        avlb_data = sqldb.read_by_range(bgn_date=bgn_date, stp_date=stp_date)
        avlb_data = avlb_data[["trade_date", "instrument", "sectorL1"]]
        return avlb_data

    def fillna_by_sector(self, avlb_i_data: pd.DataFrame) -> pd.DataFrame:
        grp_keys = ["trade_date", "sectorL1"]
        o_data = avlb_i_data.groupby(by=grp_keys)[self.factor_grp.factor_names].apply(
            lambda z: z.fillna(z.mean())
        ).reset_index(level=grp_keys)
        avlb_o_data = pd.merge(
            left=avlb_i_data[["trade_date", "instrument", "sectorL1"]],
            right=o_data[self.factor_grp.factor_names],
            how="inner",
            left_index=True, right_index=True,
        )
        if (l0 := len(avlb_i_data)) != (l1 := len(avlb_o_data)):
            raise ValueError(f"len of raw data = {l0} != len of fil data = {l1}.")
        return avlb_o_data

    def normalize(self, avlb_i_data: pd.DataFrame, q: float = 0.995) -> pd.DataFrame:
        def __normalize(data: pd.DataFrame) -> pd.DataFrame:
            # winsorize
            k = sps.norm.ppf(q)
            mu = data.mean()
            sd = data.std()
            ub, lb = mu + k * sd, mu - k * sd
            t = data.copy()
            for col in data.columns:
                t[col] = t[col].mask(t[col] > ub[col], other=ub[col])
                t[col] = t[col].mask(t[col] < lb[col], other=lb[col])

            # normalize
            z = (t - t.mean()) / t.std()
            return z

        grp_keys = ["trade_date"]
        o_data = avlb_i_data.groupby(by=grp_keys)[self.factor_grp.factor_names].apply(
            __normalize).reset_index(level=grp_keys)
        avlb_o_data = pd.merge(
            left=avlb_i_data[["trade_date", "instrument", "sectorL1"]],
            right=o_data[self.factor_grp.factor_names],
            how="inner",
            left_index=True, right_index=True,
        )
        if (l0 := len(avlb_i_data)) != (l1 := len(avlb_o_data)):
            raise ValueError(f"len of raw data = {l0}  != len of nrm data = {l1}.")
        return avlb_o_data

    def neutralize(self, avlb_i_data: pd.DataFrame) -> pd.DataFrame:
        grp_keys = ["trade_date", "sectorL1"]
        o_data = avlb_i_data.groupby(by=grp_keys)[self.factor_grp.factor_names].apply(
            lambda z: z - z.mean()
        ).reset_index(level=grp_keys)
        avlb_o_data = pd.merge(
            left=avlb_i_data[["trade_date", "instrument", "sectorL1"]],
            right=o_data[self.factor_grp.factor_names],
            how="inner",
            left_index=True, right_index=True,
        )
        if (l0 := len(avlb_i_data)) != (l1 := len(avlb_o_data)):
            raise ValueError(f"len of raw data = {l0}  != len of neu data = {l1}.")
        return avlb_o_data

    def save(self, new_data: pd.DataFrame, calendar: CCalendar, save_type: Literal["raw", "neu"]):
        if save_type == "raw":
            factors_avlb_dir = self.factors_avlb_raw_dir
        elif save_type == "neu":
            factors_avlb_dir = self.factors_avlb_neu_dir
        else:
            raise ValueError(f"Invalid save_type {save_type}")
        db_struct_fac = gen_factors_avlb_db(
            factors_avlb_dir=factors_avlb_dir,
            factor_class=self.factor_grp.factor_class,
            factors=self.factor_grp.factors,
        )
        check_and_makedirs(db_struct_fac.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_fac.db_save_dir,
            db_name=db_struct_fac.db_name,
            table=db_struct_fac.table,
            mode="a",
        )
        if sqldb.check_continuity(new_data["trade_date"].iloc[0], calendar) == 0:
            instru_tst_ret_agg_data = new_data[db_struct_fac.table.vars.names]
            sqldb.update(update_data=instru_tst_ret_agg_data)
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        logger.info(f"Calculate available factor {SFG(self.factor_grp.factor_class)}")
        # avlb raw
        ref_fac_data = self.load_ref_fac(bgn_date, stp_date)
        available_data = self.load_available(bgn_date, stp_date)
        fac_avlb_raw_data = pd.merge(
            left=available_data,
            right=ref_fac_data,
            on=["trade_date", "instrument"],
            how="left",
        ).sort_values(by=["trade_date", "sectorL1"])

        # avlb nrm
        logger.info(f"Fill and Normalize available factor {SFG(self.factor_grp.factor_class)}")
        fac_avlb_fil_data = self.fillna_by_sector(fac_avlb_raw_data)
        fac_avlb_nrm_data = self.normalize(fac_avlb_fil_data)
        self.save(fac_avlb_nrm_data, calendar, save_type="raw")

        # avlb neu
        logger.info(f"Neutralize available factor {SFG(self.factor_grp.factor_class)}")
        fac_avlb_neu_data = self.neutralize(fac_avlb_nrm_data)
        self.save(fac_avlb_neu_data, calendar, save_type="neu")

        logger.info(f"All done for factor {SFG(self.factor_grp.factor_class)}")
        return 0


class CFactorsLoader:
    def __init__(self, factor_class: TFactorClass, factors: TFactors, factors_avlb_dir: str):
        """

        :param factor_class:
        :param factors:
        :param factors_avlb_dir:  factors_avlb_raw_dir or factors_avlb_neu_dir
        """
        self.factor_class = factor_class
        self.factors = factors
        self.factors_avlb_dir = factors_avlb_dir

    @property
    def value_columns(self) -> list[str]:
        return ["trade_date", "instrument"] + [f.factor_name for f in self.factors]

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_fac = gen_factors_avlb_db(
            factors_avlb_dir=self.factors_avlb_dir,
            factor_class=self.factor_class,
            factors=self.factors,
        )
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_fac.db_save_dir,
            db_name=db_struct_fac.db_name,
            table=db_struct_fac.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date, value_columns=self.value_columns)
        return data
