import os
import pandas as pd
from loguru import logger
from typedefs.typedefReturns import CRet, TFacRetType, TRets
from typedefs.typedefFactors import CCfgFactorGrp
from husfort.qutility import check_and_makedirs, SFG
from husfort.qsqlite import CMgrSqlDb
from husfort.qcalendar import CCalendar
from husfort.qplot import CPlotLines
from solutions.test_return import CTestReturnLoader
from solutions.factor import CFactorsLoader
from solutions.shared import gen_ic_tests_db


class CICTest:
    def __init__(
            self,
            factor_grp: CCfgFactorGrp,
            factor_type: TFacRetType,
            ret: CRet,
            ret_type: TFacRetType,
            factors_avlb_dir: str,
            test_returns_avlb_dir: str,
            ic_tests_dir: str,
    ):
        self.factor_grp = factor_grp
        self.factor_type = factor_type
        self.ret = ret
        self.ret_type = ret_type
        self.factors_avlb_dir = factors_avlb_dir
        self.test_returns_avlb_dir = test_returns_avlb_dir
        self.ic_tests_dir = ic_tests_dir

    @property
    def save_id(self) -> str:
        return f"{self.factor_grp.factor_class}-{self.factor_type}-{self.ret.ret_name}-{self.ret_type}"

    def load_returns(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        returns_loader = CTestReturnLoader(
            ret=self.ret,
            test_returns_avlb_dir=self.test_returns_avlb_dir,
        )
        return returns_loader.load(bgn_date, stp_date)

    def load_factors(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        factors_loader = CFactorsLoader(
            factor_class=self.factor_grp.factor_class,
            factors=self.factor_grp.factors,
            factors_avlb_dir=self.factors_avlb_dir,
        )
        return factors_loader.load(bgn_date, stp_date)

    def corr(self, data: pd.DataFrame) -> pd.Series:
        s = data[self.factor_grp.factor_names].corrwith(
            data[self.ret.ret_name], axis=0, method="spearman")
        return s

    def save(self, new_data: pd.DataFrame, calendar: CCalendar):
        """

        :param new_data: a pd.DataFrame with columns =
                        ["trade_date"] + self.factor_grp.factor_names
        :param calendar:
        :return:
        """
        ic_test_db_struct = gen_ic_tests_db(
            ic_tests_dir=self.ic_tests_dir,
            factor_class=self.factor_grp.factor_class,
            factors=self.factor_grp.factors,
            factor_type=self.factor_type,
            ret=self.ret,
            ret_type=self.ret_type,
        )
        check_and_makedirs(ic_test_db_struct.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=ic_test_db_struct.db_save_dir,
            db_name=ic_test_db_struct.db_name,
            table=ic_test_db_struct.table,
            mode="a",
        )
        if sqldb.check_continuity(new_data["trade_date"].iloc[0], calendar) == 0:
            update_data = new_data[ic_test_db_struct.table.vars.names]
            sqldb.update(update_data=update_data)
        return 0

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        ic_test_db_struct = gen_ic_tests_db(
            ic_tests_dir=self.ic_tests_dir,
            factor_class=self.factor_grp.factor_class,
            factors=self.factor_grp.factors,
            factor_type=self.factor_type,
            ret=self.ret,
            ret_type=self.ret_type,
        )
        check_and_makedirs(ic_test_db_struct.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=ic_test_db_struct.db_save_dir,
            db_name=ic_test_db_struct.db_name,
            table=ic_test_db_struct.table,
            mode="a",
        )
        data = sqldb.read_by_range(
            bgn_date=bgn_date, stp_date=stp_date,
            value_columns=["trade_date"] + self.factor_grp.factor_names,
        )
        return data

    def plot(self, ic_cumsum: pd.DataFrame):
        check_and_makedirs(save_dir := os.path.join(self.ic_tests_dir, "plots"))
        artist = CPlotLines(
            plot_data=ic_cumsum,
            fig_name=f"{self.save_id}",
            fig_save_dir=save_dir,
            colormap="jet",
            line_style=["-", "-."] * int(ic_cumsum.shape[1] / 2),
            line_width=1.5,
        )
        artist.plot()
        artist.set_axis_x(xtick_count=20, xtick_label_size=8)
        artist.save_and_close()
        return 0

    def report(self, ic_data: pd.DataFrame):
        ic_data["trade_year"] = ic_data.index.map(lambda z: z[0:4])
        dfs: list[pd.DataFrame] = []
        for trade_year, trade_year_data in ic_data.groupby("trade_year"):
            ic_mean = trade_year_data[self.factor_grp.factor_names].mean()
            ic_std = trade_year_data[self.factor_grp.factor_names].std()
            ir = ic_mean / ic_std
            trade_year_sum = pd.DataFrame(
                {"trade_year": trade_year, "IC": ic_mean, "IR": ir}
            ).reset_index().rename(columns={"index": "factor"})
            dfs.append(trade_year_sum)
        sum_df = pd.concat(dfs, axis=0, ignore_index=True)
        report = pd.pivot_table(data=sum_df, values=["IC", "IR"], index="trade_year", columns="factor")
        report.columns = ["-".join(z) for z in report.columns]
        check_and_makedirs(save_dir := os.path.join(self.ic_tests_dir, "reports"))
        report_file = f"{self.save_id}.csv"
        report_path = os.path.join(save_dir, report_file)
        report.to_csv(report_path, float_format="%.6f")
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        buffer_bgn_date = calendar.get_next_date(bgn_date, -self.ret.shift)
        iter_dates = calendar.get_iter_list(buffer_bgn_date, stp_date)
        save_dates = iter_dates[self.ret.shift:]
        base_bgn_date, base_stp_date = iter_dates[0], iter_dates[-self.ret.shift]
        returns_data = self.load_returns(base_bgn_date, base_stp_date)
        factors_data = self.load_factors(base_bgn_date, base_stp_date)
        input_data = pd.merge(
            left=returns_data,
            right=factors_data,
            on=["trade_date", "instrument"],
            how="inner",
        )
        lr, lf, li = len(returns_data), len(factors_data), len(input_data)
        if (li != lr) or (li != lf):
            raise ValueError(f"len of factor data = {lf}, len of return data = {lr}, len of input data = {li}.")
        ic_data = input_data.groupby(by="trade_date").apply(self.corr)
        ic_data["trade_date"] = save_dates
        new_data = ic_data[["trade_date"] + self.factor_grp.factor_names]
        new_data = new_data.reset_index(drop=True)
        self.save(new_data, calendar)
        logger.info(f"IC test for {SFG(self.save_id)} finished.")
        return 0

    def main_summary(self, bgn_date: str, stp_date: str):
        ic_data = self.load(bgn_date, stp_date).set_index("trade_date")
        ic_cumsum = ic_data.cumsum()
        self.plot(ic_cumsum)
        self.report(ic_data)
        return 0


TFactorsAvlbDirType = str
TTestReturnsAvlbDirType = str
TICTestAuxArgs = tuple[TFacRetType, TFactorsAvlbDirType, TTestReturnsAvlbDirType]


def main_ic_tests(
        rets: TRets,
        factor_grp: CCfgFactorGrp,
        aux_args_list: list[TICTestAuxArgs],
        ic_tests_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
):
    for ret in rets:
        for fac_ret_type, factors_avlb_dir, test_returns_avlb_dir in aux_args_list:
            ic_test = CICTest(
                factor_grp=factor_grp,
                factor_type=fac_ret_type,
                ret=ret,
                ret_type=fac_ret_type,
                factors_avlb_dir=factors_avlb_dir,
                test_returns_avlb_dir=test_returns_avlb_dir,
                ic_tests_dir=ic_tests_dir,
            )
            ic_test.main(bgn_date, stp_date, calendar)
            ic_test.main_summary(bgn_date, stp_date)
    return 0
