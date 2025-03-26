import pandas as pd
from rich.progress import track
from husfort.qutility import SFG, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from solutions.shared import gen_test_returns_by_instru_db
from typedef import TUniverse, CRet, TReturnClass


class CTestReturnsByInstru:
    def __init__(
            self,
            ret: CRet,
            universe: TUniverse,
            test_returns_by_instru_dir: str,
            db_struct_preprocess: CDbStruct
    ):
        self.ret = ret
        self.universe = universe
        self.test_returns_by_instru_dir = test_returns_by_instru_dir
        self.db_struct_preprocess = db_struct_preprocess

    def load_preprocess(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_preprocess.db_save_dir,
            db_name=f"{instru}.db",
            table=self.db_struct_preprocess.table,
            mode="r",
        )
        data = sqldb.read_by_range(
            bgn_date=bgn_date, stp_date=stp_date,
            value_columns=["trade_date", "ticker_major", "return_c_major", "return_o_major"]
        )
        return data

    def cal_test_return(
            self,
            instru_ret_data: pd.DataFrame,
            base_bgn_date: str,
            base_end_date: str,
    ) -> pd.DataFrame:
        if self.ret.ret_class == TReturnClass.CLS:
            raw_ret = "return_c_major"
        elif self.ret.ret_class == TReturnClass.OPN:
            raw_ret = "return_o_major"
        else:
            raise ValueError(f"Invalid ret_class: {self.ret.ret_class}")

        instru_ret_data[self.ret.ret_name] = instru_ret_data[raw_ret].rolling(window=self.ret.win).sum().shift(
            -self.ret.shift)
        res = instru_ret_data.query(f"trade_date >= '{base_bgn_date}' & trade_date <= '{base_end_date}'")
        res = res[["trade_date", "ticker_major", self.ret.ret_name]]
        return res

    def process_for_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar):
        iter_dates = calendar.get_iter_list(bgn_date, stp_date)
        base_bgn_date = calendar.get_next_date(iter_dates[0], -self.ret.shift)
        base_end_date = calendar.get_next_date(iter_dates[-1], -self.ret.shift)
        db_struct_instru = gen_test_returns_by_instru_db(
            instru=instru,
            test_returns_by_instru_dir=self.test_returns_by_instru_dir,
            save_id=self.ret.ret_class,
            ret=self.ret,
        )
        check_and_makedirs(db_struct_instru.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="a",
        )
        if sqldb.check_continuity(base_bgn_date, calendar) == 0:
            instru_ret_data = self.load_preprocess(instru, base_bgn_date, stp_date)
            y_instru_data = self.cal_test_return(instru_ret_data, base_bgn_date, base_end_date)
            sqldb.update(update_data=y_instru_data)
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        desc = f"Processing test return {SFG(self.ret.ret_name)}"
        for instru in track(self.universe, description=desc):
            self.process_for_instru(instru, bgn_date=bgn_date, stp_date=stp_date, calendar=calendar)
        return 0

# class CTstRetAgg(_CTstRet):
#     def __init__(
#             self,
#             win: int, lag: int,
#             universe: TUniverse,
#             db_tst_ret_save_dir: str,
#             db_struct_avlb: CDbStruct,
#     ):
#         super().__init__(win, lag, universe, db_tst_ret_save_dir)
#         self.db_struct_avlb = db_struct_avlb
#
#     @property
#     def save_id(self) -> str:
#         return f"{self.win:03d}L{self.lag}"
#
#     @property
#     def ref_id(self) -> str:
#         return self.save_id
#
#     @property
#     def ref_rets(self) -> list[str]:
#         return self.rets
#
#     @property
#     def ref_lbl_cls(self) -> str:
#         return self.ret_lbl_cls
#
#     @property
#     def ref_lbl_opn(self) -> str:
#         return self.ret_lbl_opn
#
#     def load_ref_ret_by_instru(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
#         db_struct_ref = gen_tst_ret_raw_db(
#             instru=instru,
#             db_save_root_dir=self.db_tst_ret_save_dir,
#             save_id=self.ref_id,
#             rets=self.ref_rets,
#         )
#         sqldb = CMgrSqlDb(
#             db_save_dir=db_struct_ref.db_save_dir,
#             db_name=db_struct_ref.db_name,
#             table=db_struct_ref.table,
#             mode="r"
#         )
#         ref_data = sqldb.read_by_range(bgn_date, stp_date)
#         return ref_data
#
#     def load_ref_ret(self, base_bgn_date: str, base_stp_date: str) -> pd.DataFrame:
#         ref_dfs: list[pd.DataFrame] = []
#         for instru in self.universe:
#             df = self.load_ref_ret_by_instru(instru, bgn_date=base_bgn_date, stp_date=base_stp_date)
#             df["instrument"] = instru
#             ref_dfs.append(df)
#         res = pd.concat(ref_dfs, axis=0, ignore_index=False)
#         res = res.reset_index().sort_values(by=["trade_date"], ascending=True)
#         res = res[["trade_date", "instrument"] + self.ref_rets]
#         return res
#
#     def load_available(self, base_bgn_date: str, base_stp_date: str) -> pd.DataFrame:
#         sqldb = CMgrSqlDb(
#             db_save_dir=self.db_struct_avlb.db_save_dir,
#             db_name=self.db_struct_avlb.db_name,
#             table=self.db_struct_avlb.table,
#             mode="r",
#         )
#         avlb_data = sqldb.read_by_range(bgn_date=base_bgn_date, stp_date=base_stp_date)
#         avlb_data = avlb_data[["trade_date", "instrument", "sectorL1"]]
#         return avlb_data
#
#     def save(self, new_data: pd.DataFrame, calendar: CCalendar):
#         db_struct_instru = gen_tst_ret_agg_db(
#             db_save_root_dir=self.db_tst_ret_save_dir,
#             save_id=self.save_id,
#             rets=self.rets,
#         )
#         check_and_makedirs(db_struct_instru.db_save_dir)
#         sqldb = CMgrSqlDb(
#             db_save_dir=db_struct_instru.db_save_dir,
#             db_name=db_struct_instru.db_name,
#             table=db_struct_instru.table,
#             mode="a",
#         )
#         if sqldb.check_continuity(new_data["trade_date"].iloc[0], calendar) == 0:
#             instru_tst_ret_agg_data = new_data[db_struct_instru.table.vars.names]
#             sqldb.update(update_data=instru_tst_ret_agg_data)
#         return 0
#
#     def main_test_return_agg(self, bgn_date: str, stp_date: str, calendar: CCalendar):
#         # logger.info(f"Aggregating test return with lag = {SFG(self.lag)}, win = {SFG(self.win)}")
#         iter_dates = calendar.get_iter_list(bgn_date, stp_date)
#         base_bgn_date = self.get_base_date(iter_dates[0], calendar)
#         base_end_date = self.get_base_date(iter_dates[-1], calendar)
#         base_stp_date = calendar.get_next_date(base_end_date, shift=1)
#
#         ref_tst_ret_data = self.load_ref_ret(base_bgn_date, base_stp_date)
#         available_data = self.load_available(base_bgn_date, base_stp_date)
#         tst_ret_avlb_data = pd.merge(
#             left=available_data,
#             right=ref_tst_ret_data,
#             on=["trade_date", "instrument"],
#             how="left",
#         ).sort_values(by=["trade_date", "sectorL1"])
#         tst_ret_agg_data = tst_ret_avlb_data.query(f"trade_date >= '{base_bgn_date}' & trade_date <= '{base_stp_date}'")
#         self.save(tst_ret_agg_data, calendar)
#         return 0
