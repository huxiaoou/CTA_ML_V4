import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qcalendar import CCalendar
from husfort.qutility import qtimer, check_and_makedirs, error_handler
from husfort.qplot import CPlotLines
from husfort.qsqlite import CMgrSqlDb
from typedef import CRet, TReturnClass
from solutions.shared import gen_sims_quick_nav_db
from solutions.test_return import CTestReturnLoader
from solutions.mclrn import CTestMclrn
from solutions.signals import CSignalsLoader

TSimQuickArgs = tuple[CSignalsLoader, CTestReturnLoader]


def covert_tests_to_sims_quick_args(
        tests: list[CTestMclrn],
        signals_dir: str,
        test_returns_avlb_raw_dir: str,
) -> list[TSimQuickArgs]:
    sim_quick_args: list[TSimQuickArgs] = []
    for test in tests:
        signal = CSignalsLoader(signals_dir=signals_dir, signal_id=test.save_id)
        if test.test_data.ret.ret_class == TReturnClass.OPN:
            ret = CRet.parse_from_name("Opn001L1")
        else:
            ret = CRet.parse_from_name("Cls001L1")
        test_return_loader = CTestReturnLoader(ret, test_returns_avlb_raw_dir)
        sim_quick_args.append((signal, test_return_loader))
    return sim_quick_args


class CSimQuick:
    def __init__(
            self,
            signals_loader: CSignalsLoader,
            test_return_loader: CTestReturnLoader,
            cost_rate: float,
            sims_quick_dir: str,
    ):
        self.signals_loader = signals_loader
        self.test_return_loader = test_return_loader
        self.cost_rate = cost_rate
        self.quick_sim_save_dir = sims_quick_dir

    def get_dates(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> tuple[list[str], list[str]]:
        d = self.test_return_loader.ret.shift + 1  # +1 for calculating delta weights
        buffer_bgn_date = calendar.get_next_date(bgn_date, -d)
        iter_dates = calendar.get_iter_list(buffer_bgn_date, stp_date)
        sig_dates = iter_dates[0:-d + 1]
        exe_dates = iter_dates[(d - 1):]
        return sig_dates, exe_dates

    def get_sigs_and_rets(self, base_bgn_date: str, base_stp_date: str) -> pd.DataFrame:
        sigs = self.signals_loader.load(base_bgn_date, base_stp_date)
        rets = self.test_return_loader.load(base_bgn_date, base_stp_date)
        data = pd.merge(left=sigs, right=rets, how="right", on=["trade_date", "instrument"]).fillna(0)
        return data

    def cal_core(self, data: pd.DataFrame, exe_dates: list[str]) -> pd.DataFrame:
        raw_ret = data.groupby(by="trade_date").apply(lambda z: z["weight"] @ z[self.test_return_loader.ret.ret_name])
        daily_weights = pd.pivot_table(data=data, index="trade_date", columns="instrument", values="weight")
        delta_weights = daily_weights.diff().fillna(0)
        delta_weights_sum = delta_weights.abs().sum(axis=1)
        cost = delta_weights_sum * self.cost_rate
        net_ret = raw_ret - cost
        result = pd.DataFrame({
            "raw_ret": raw_ret,
            "delta_weights_sum": delta_weights_sum,
            "cost": cost,
            "ret": net_ret,
            "exe_date": exe_dates,
        })
        return result

    @staticmethod
    def recalibrate_dates(raw_result: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        net_result = raw_result.set_index("exe_date").truncate(before=bgn_date)
        net_result.index.name = "trade_date"
        return net_result

    @staticmethod
    def update_nav(net_result: pd.DataFrame, last_nav: float):
        net_result["nav"] = (net_result["ret"] + 1).cumprod() * last_nav
        return 0

    def load_nav_at_date(self, trade_date: str) -> float:
        db_struct = gen_sims_quick_nav_db(save_dir=self.quick_sim_save_dir, save_id=self.signals_loader.signal_id)
        sqldb = CMgrSqlDb(
            db_save_dir=self.quick_sim_save_dir,
            db_name=db_struct.db_name,
            table=db_struct.table,
            mode="r",
        )
        last_nav = 1.00
        if sqldb.has_table(db_struct.table):
            nav_data = sqldb.read_by_conditions(
                conditions=[("trade_date", "=", trade_date)],
                value_columns=["trade_date", "nav"],
            )
            if not nav_data.empty:
                last_nav = nav_data["nav"].values[0]
        return last_nav

    def load_nav_range(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct = gen_sims_quick_nav_db(save_dir=self.quick_sim_save_dir, save_id=self.signals_loader.signal_id)
        sqldb = CMgrSqlDb(
            db_save_dir=self.quick_sim_save_dir,
            db_name=db_struct.db_name,
            table=db_struct.table,
            mode="r",
        )
        nav_data = sqldb.read_by_range(
            bgn_date=bgn_date, stp_date=stp_date,
            value_columns=["trade_date", "nav"],
        )
        return nav_data.set_index("trade_date")

    def save_nav(self, net_result: pd.DataFrame, calendar: CCalendar):
        db_struct = gen_sims_quick_nav_db(save_dir=self.quick_sim_save_dir, save_id=self.signals_loader.signal_id)
        sqldb = CMgrSqlDb(
            db_save_dir=self.quick_sim_save_dir,
            db_name=db_struct.db_name,
            table=db_struct.table,
            mode="a",
        )
        if sqldb.check_continuity(net_result.index[0], calendar) == 0:
            sqldb.update(update_data=net_result, using_index=True)
        return 0

    def plot(self, nav_data: pd.DataFrame):
        artist = CPlotLines(
            plot_data=nav_data,
            fig_name=f"{self.signals_loader.signal_id}",
            fig_save_dir=self.quick_sim_save_dir,
        )
        artist.plot()
        artist.set_axis_x(xtick_count=12)
        artist.save_and_close()
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        _nav_plot_bgn_date = "20120101"
        sig_dates, exe_dates = self.get_dates(bgn_date, stp_date, calendar)
        base_bgn_date, base_stp_date = sig_dates[0], calendar.get_next_date(sig_dates[-1], shift=1)
        last_date = calendar.get_next_date(bgn_date, shift=-1)
        data = self.get_sigs_and_rets(base_bgn_date, base_stp_date)
        raw_result = self.cal_core(data, exe_dates)
        net_result = self.recalibrate_dates(raw_result, bgn_date)
        last_nav = self.load_nav_at_date(trade_date=last_date)
        self.update_nav(net_result, last_nav)
        self.save_nav(net_result, calendar)
        nav_data = self.load_nav_range(bgn_date=_nav_plot_bgn_date, stp_date=stp_date)
        self.plot(nav_data=nav_data)
        return 0


@qtimer
def main_sims_quick(
        tests: list[CTestMclrn],
        signals_dir: str,
        test_returns_avlb_raw_dir: str,
        cost_rate: float,
        sims_quick_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    check_and_makedirs(sims_quick_dir)
    sim_quick_args = covert_tests_to_sims_quick_args(tests, signals_dir, test_returns_avlb_raw_dir)
    desc = "Do quick simulations"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(sim_quick_args))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for signals_loader, test_return_loader in sim_quick_args:
                    sim_quick = CSimQuick(signals_loader, test_return_loader, cost_rate, sims_quick_dir)
                    pool.apply_async(
                        sim_quick.main,
                        kwds={
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(task_id=main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for signals_loader, test_return_loader in track(sim_quick_args, description=desc):
            sim_quick = CSimQuick(signals_loader, test_return_loader, cost_rate, sims_quick_dir)
            sim_quick.main(bgn_date, stp_date, calendar)
    return 0
