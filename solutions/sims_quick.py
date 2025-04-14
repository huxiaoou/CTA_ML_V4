import os
import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qcalendar import CCalendar
from husfort.qutility import qtimer, check_and_makedirs, error_handler
from husfort.qplot import CPlotLines
from solutions.test_return import CTestReturnLoader
from typedef import CRet, TReturnClass
from solutions.mclrn import CTestMclrn
from solutions.signals import CSignals


def covert_tests_to_sims_quick(
        tests: list[CTestMclrn],
        signals_dir: str,
        test_returns_avlb_raw_dir: str,
) -> list[tuple[CSignals, CTestReturnLoader]]:
    sims: list[tuple[CSignals, CTestReturnLoader]] = []
    for test in tests:
        signal = CSignals(signals_dir=signals_dir, signal_id=test.save_id)
        if test.test_data.ret.ret_class == TReturnClass.OPN:
            ret = CRet.parse_from_name("Opn001L1")
        else:
            ret = CRet.parse_from_name("Cls001L1")
        test_return_loader = CTestReturnLoader(ret, test_returns_avlb_raw_dir)
        sims.append((signal, test_return_loader))
    return sims


class CSimQuick:
    def __init__(
            self,
            signals: CSignals,
            test_return_loader: CTestReturnLoader,
            cost_rate: float,
            sims_quick_dir: str,
    ):
        self.signals = signals
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
        sigs = self.signals.load(base_bgn_date, base_stp_date)
        rets = self.test_return_loader.load(base_bgn_date, base_stp_date)
        data = pd.merge(left=sigs, right=rets, how="right", on=["trade_date", "instrument"]).fillna(0)
        return data

    def cal_raw_res(self, data: pd.DataFrame, exe_dates: list[str]) -> pd.DataFrame:
        raw_rets = data.groupby(by="trade_date").apply(lambda z: z["weight"] @ z[self.test_return_loader.ret.ret_name])
        daily_weights = pd.pivot_table(data=data, index="trade_date", columns="instrument", values="weight")
        delta_weights = daily_weights.diff().fillna(0)
        delta_weights_sum = delta_weights.abs().sum(axis=1)
        cost = delta_weights_sum * self.cost_rate
        net_rets = raw_rets - cost
        raw_res = pd.DataFrame({
            "raw_rets": raw_rets,
            "delta_weights_sum": delta_weights_sum,
            "cost": cost,
            "net_rets": net_rets,
            "exe_date": exe_dates,
        })
        return raw_res

    @staticmethod
    def get_net_res(raw_res: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        net_res = raw_res.set_index("exe_date").truncate(before=bgn_date)
        net_res.index.name = "trade_date"
        return net_res

    def save_data(self, net_res: pd.DataFrame):
        net_file = f"{self.signals.signal_id}.csv.gz"
        net_path = os.path.join(self.quick_sim_save_dir, net_file)
        net_res.to_csv(net_path, float_format="%.6f", index=True)
        return 0

    def plot(self, nav_data: pd.DataFrame):
        artist = CPlotLines(
            plot_data=nav_data,
            fig_name=f"{self.signals.signal_id}",
            fig_save_dir=self.quick_sim_save_dir,
        )
        artist.plot()
        artist.set_axis_x(xtick_count=12)
        artist.save_and_close()
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        sig_dates, exe_dates = self.get_dates(bgn_date, stp_date, calendar)
        base_bgn_date, base_stp_date = sig_dates[0], calendar.get_next_date(sig_dates[-1], shift=1)
        data = self.get_sigs_and_rets(base_bgn_date, base_stp_date)
        raw_res = self.cal_raw_res(data, exe_dates)
        net_res = self.get_net_res(raw_res, bgn_date)
        net_res["nav"] = (net_res["net_rets"] + 1).cumprod()
        self.save_data(net_res)
        self.plot(nav_data=net_res[["nav"]])
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
    sim_args = covert_tests_to_sims_quick(tests, signals_dir, test_returns_avlb_raw_dir)
    desc = "Do quick simulations"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(sim_args))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for signals, test_return_loader in sim_args:
                    sim_quick = CSimQuick(signals, test_return_loader, cost_rate, sims_quick_dir)
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
        for signals, test_return_loader in track(sim_args, description=desc):
            # for signals, test_return_loader in sim_args:
            sim_quick = CSimQuick(signals, test_return_loader, cost_rate, sims_quick_dir)
            sim_quick.main(bgn_date, stp_date, calendar)
    return 0
