import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qutility import SFY, error_handler, check_and_makedirs
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from husfort.qcalendar import CCalendar
from typedef import CCfgFactorGrp, TUniverse
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
