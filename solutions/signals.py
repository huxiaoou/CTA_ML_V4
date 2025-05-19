import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track
from husfort.qsqlite import CMgrSqlDb
from husfort.qcalendar import CCalendar
from husfort.qutility import error_handler, check_and_makedirs
from husfort.qsimquick import CSignalsLoaderBase
from rich.progress import Progress
from solutions.shared import gen_prdct_db, gen_sig_db
from solutions.mclrn import CTestMclrn


class CSignalsLoader(CSignalsLoaderBase):
    def __init__(self, signals_dir: str, signal_id: str):
        self.signals_dir = signals_dir
        self._signal_id = signal_id

    @property
    def signal_id(self):
        return self._signal_id

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct = gen_sig_db(save_dir=self.signals_dir, save_id=self.signal_id)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct.db_save_dir,
            db_name=db_struct.db_name,
            table=db_struct.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date)
        return data

    def save(self, data: pd.DataFrame, calendar: CCalendar):
        """

        :param data: a pd.DataFrame with columns = ["trade_date", "instrument", "weight"]
        :param calendar:
        :return:
        """
        check_and_makedirs(self.signals_dir)
        db_struct = gen_sig_db(save_dir=self.signals_dir, save_id=self.signal_id)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct.db_save_dir,
            db_name=db_struct.db_name,
            table=db_struct.table,
            mode="a",
        )
        if sqldb.check_continuity(incoming_date=data["trade_date"].iloc[0], calendar=calendar) == 0:
            sqldb.update(update_data=data)
        return 0


class CSignalsFromPrdct(CSignalsLoader):
    def __init__(self, signals_dir: str, test: CTestMclrn):
        super().__init__(signals_dir, test.save_id)
        self.test = test

    def load_prediction(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        """

        :param bgn_date:
        :param stp_date:
        :return: a pd.DataFrame with columns = ["trade_date", "instrument", ret_name]
        """
        db_struct = gen_prdct_db(
            save_dir=self.test.mclrn_prd_dir,
            save_id=self.test.save_id,
            ret_name=self.test.test_data.ret.ret_name,
        )
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct.db_save_dir,
            db_name=db_struct.db_name,
            table=db_struct.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date)
        return data

    def convert_prediction_to_weights(self, prediction: pd.DataFrame) -> pd.DataFrame:
        """
        :param prediction: a pd.DataFrame with columns = ["trade_date", "instrument", ret_name]
        :param prediction:
        :return: a pd.DataFrame with columns = ["trade_date", "instrument", "weight"]
        """

        def __convert_prd_to_wgt(x: pd.Series) -> pd.Series:
            k = len(x)
            k0, d = k // 2, k % 2
            rou = np.power(1 / 2, 1 / (k0 - 1)) if k0 > 1 else 1
            sgn = np.array([1] * k0 + [0] * d + [-1] * k0)
            val = np.power(rou, list(range(k0)) + [k0] * d + list(range(k0 - 1, -1, -1)))
            s = sgn * val
            abs_sum = np.abs(s).sum()
            w = (s / abs_sum) if abs_sum > 0 else 0
            return pd.Series(data=w, index=x.sort_values(ascending=False).index)

        weights = prediction.groupby(by="trade_date")[self.test.test_data.ret.ret_name].apply(__convert_prd_to_wgt)
        prediction["weight"] = weights.reset_index(level="trade_date", drop=True)
        res = prediction[["trade_date", "instrument", "weight"]]
        return res

    @staticmethod
    def moving_average(weights: pd.DataFrame, win: int) -> pd.DataFrame:
        """

        :param weights: a pd.DataFrame with columns = ["trade_date", "instrument", "weight"]
        :param win: window size for moving average
        :return: a pd.DataFrame with columns = ["trade_date", "instrument", "weight"]
        """
        if win > 1:
            pivot_data = pd.pivot_table(data=weights, index="trade_date", columns="instrument", values="weight")
            pivot_data_ma = pivot_data.fillna(0).rolling(win).mean()
            abs_sum = pivot_data_ma.abs().sum(axis=1)
            pivot_data_nm: pd.DataFrame = pivot_data_ma.div(abs_sum.where(abs_sum > 0, 1), axis=0)
            new_weights = pivot_data_nm.stack().reset_index().rename(columns={0: "weight"})
            res = pd.merge(
                left=weights[["trade_date", "instrument"]],
                right=new_weights,
                on=["trade_date", "instrument"],
                how="left",
            )
            return res
        else:
            return weights

    def main_convert(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        buffer_bgn_date = calendar.get_next_date(bgn_date, shift=-self.test.test_data.ret.win + 1)
        predictions = self.load_prediction(buffer_bgn_date, stp_date)
        weights_buf = self.convert_prediction_to_weights(predictions)
        weights_buf_ma = self.moving_average(weights_buf, win=self.test.test_data.ret.win)
        weights = weights_buf_ma.query(f"trade_date >= '{bgn_date}'")
        self.save(weights, calendar)
        return 0


def main_signals(
        tests: list[CTestMclrn],
        signals_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    signals: list[CSignalsFromPrdct] = [CSignalsFromPrdct(signals_dir, test=test) for test in tests]
    desc = "Convert predictions to signals"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for s in signals:
                    pool.apply_async(
                        s.main_convert,
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
        for s in track(signals, description=desc):
            # for s in signals:
            s.main_convert(bgn_date, stp_date, calendar)
    return 0
