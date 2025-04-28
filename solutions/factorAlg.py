import numpy as np
import pandas as pd
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct
from typedef import (
    TFactorClass, CCfgFactors, TUniverse, CCfgFactorGrp,
    CCfgFactorGrpMTM, CCfgFactorGrpSKEW, CCfgFactorGrpKURT,
    CCfgFactorGrpRS, CCfgFactorGrpBASIS, CCfgFactorGrpTS,
)
from solutions.factor import CFactorsByInstru

"""
-----------------------
Part I: Some math tools
-----------------------
"""


def cal_rolling_corr(df: pd.DataFrame, x: str, y: str, rolling_window: int) -> pd.Series:
    xyb: pd.Series = (df[x] * df[y]).rolling(window=rolling_window).mean()
    xxb: pd.Series = (df[x] * df[x]).rolling(window=rolling_window).mean()
    yyb: pd.Series = (df[y] * df[y]).rolling(window=rolling_window).mean()
    xb: pd.Series = df[x].rolling(window=rolling_window).mean()
    yb: pd.Series = df[y].rolling(window=rolling_window).mean()
    cov_xy: pd.Series = xyb - xb * yb
    cov_xx: pd.Series = xxb - xb * xb
    cov_yy: pd.Series = yyb - yb * yb

    # due to float number precision, cov_xx or cov_yy could be slightly negative
    cov_xx = cov_xx.mask(cov_xx < 1e-10, other=0)
    cov_yy = cov_yy.mask(cov_yy < 1e-10, other=0)

    sqrt_cov_xx_yy: pd.Series = np.sqrt(cov_xx * cov_yy)
    s: pd.Series = cov_xy / sqrt_cov_xx_yy.where(sqrt_cov_xx_yy > 0, np.nan)
    return s


def cal_rolling_beta(df: pd.DataFrame, x: str, y: str, rolling_window: int) -> pd.Series:
    xyb: pd.Series = (df[x] * df[y]).rolling(window=rolling_window).mean()
    xxb: pd.Series = (df[x] * df[x]).rolling(window=rolling_window).mean()
    xb: pd.Series = df[x].rolling(window=rolling_window).mean()
    yb: pd.Series = df[y].rolling(window=rolling_window).mean()
    cov_xy: pd.Series = xyb - xb * yb
    cov_xx: pd.Series = xxb - xb * xb
    s: pd.Series = cov_xy / cov_xx.where(cov_xx > 0, np.nan)
    return s


def cal_top_corr(sub_data: pd.DataFrame, x: str, y: str, sort_var: str, top_size: int, ascending: bool = False):
    sorted_data = sub_data.sort_values(by=sort_var, ascending=ascending)
    top_data = sorted_data.head(top_size)
    r = top_data[[x, y]].corr(method="spearman").at[x, y]
    return r


def auto_weight_sum(x: pd.Series) -> float:
    weight = x.abs() / x.abs().sum()
    return x @ weight


"""
---------------------------------------------------
Part II: factor class from different configuration
---------------------------------------------------
"""


class CFactorMTM(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpMTM, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, name_vanilla in zip(self.cfg.wins, self.cfg.names_vanilla):
            major_data[name_vanilla] = major_data["return_c_major"].rolling(window=win).sum()
        for win, name_diff in zip(self.cfg.wins[0:6], self.cfg.names_diff):
            max_win = max(self.cfg.wins)
            n0, n1 = self.cfg.factor_name(max_win), self.cfg.factor_name(win)
            major_data[name_diff] = major_data[n0] * np.sqrt(win / max_win) - major_data[n1]
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorSKEW(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpSKEW, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, name_vanilla, name_delay in zip(self.cfg.wins, self.cfg.names_vanilla, self.cfg.names_delay):
            major_data[name_vanilla] = -major_data["return_c_major"].rolling(window=win).skew()
            major_data[name_delay] = major_data[name_vanilla].shift(1)
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorKURT(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpKURT, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, name_vanilla in zip(self.cfg.wins, self.cfg.names_vanilla):
            major_data[name_vanilla] = major_data["return_c_major"].rolling(window=win).kurt()
        n0, n1 = self.cfg.name_vanilla(10), self.cfg.name_vanilla(120)
        major_data[self.cfg.name_diff()] = major_data[n0] - major_data[n1]
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorRS(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpRS, **kwargs):
        self.cfg = cfg
        self.__win_min = min(5, min(self.cfg.wins))
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        adj_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "stock"],
        )
        adj_data["stock"] = adj_data["stock"].ffill(limit=self.__win_min).fillna(0)
        for win, rspa, rsla in zip(self.cfg.wins, self.cfg.names_rspa, self.cfg.names_rsla):
            ma = adj_data["stock"].rolling(window=win).mean()
            s = adj_data["stock"] / ma.where(ma > 0, np.nan)
            adj_data[rspa] = 1 - s
            la = adj_data["stock"].shift(win)
            s = adj_data["stock"] / la.where(la > 0, np.nan)
            adj_data[rsla] = 1 - s
        n0, n1 = self.cfg.name_rspa(240), self.cfg.name_rspa(20)
        adj_data[self.cfg.name_diff()] = adj_data[n0] - adj_data[n1]
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorBASIS(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpBASIS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        x, y = "basis_rate", "return_c_major"
        adj_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", x, y],
        )
        for win, name_vanilla, name_res in zip(self.cfg.wins, self.cfg.names_vanilla, self.cfg.names_res):
            adj_data[name_vanilla] = adj_data[x].rolling(window=win, min_periods=int(2 * win / 3)).mean()
            beta = cal_rolling_beta(df=adj_data, x=x, y=y, rolling_window=win)
            adj_data[name_res] = adj_data[y] - adj_data[x] * beta
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorTS(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpTS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    @staticmethod
    def cal_roll_return(x: pd.Series, ticker_n: str, ticker_d: str, prc_n: str, prc_d: str):
        if x[ticker_n] == "" or x[ticker_d] == "":
            return np.nan
        if x[prc_d] > 0:
            cntrct_d, cntrct_n = x[ticker_d].split(".")[0], x[ticker_n].split(".")[0]
            month_d, month_n = int(cntrct_d[-2:]), int(cntrct_n[-2:])
            dlt_month = month_d - month_n
            dlt_month = dlt_month + (12 if dlt_month <= 0 else 0)
            return (x[prc_n] / x[prc_d] - 1) / dlt_month * 12 * 100
        else:
            return np.nan

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "ticker_minor", "close_major", "close_minor", "return_c_major"],
        )
        adj_data[["ticker_major", "ticker_minor"]] = adj_data[["ticker_major", "ticker_minor"]].fillna("")
        adj_data["ts"] = adj_data.apply(
            self.cal_roll_return,
            args=("ticker_major", "ticker_minor", "close_major", "close_minor"),
            axis=1,
        )
        x, y = "ts", "return_c_major"
        for win, name_vanilla, name_res in zip(self.cfg.wins, self.cfg.names_vanilla, self.cfg.names_res):
            adj_data[name_vanilla] = adj_data["ts"].rolling(window=win, min_periods=int(2 * win / 3)).mean()
            beta = cal_rolling_beta(df=adj_data, x=x, y=y, rolling_window=win)
            adj_data[name_res] = adj_data[y] - adj_data[x] * beta
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


"""
---------------------------------------------------
Part III: pick factor
---------------------------------------------------
"""


def pick_factor(
        fclass: TFactorClass,
        cfg_factors: CCfgFactors,
        factors_by_instru_dir: str,
        universe: TUniverse,
        preprocess: CDbStruct,
) -> tuple[CFactorsByInstru, CCfgFactorGrp]:
    if fclass == TFactorClass.MTM:
        cfg = cfg_factors.MTM
        fac = CFactorMTM(
            cfg=cfg,
            factors_by_instru_dir=factors_by_instru_dir,
            universe=universe,
            db_struct_preprocess=preprocess,
        )
    elif fclass == TFactorClass.SKEW:
        cfg = cfg_factors.SKEW
        fac = CFactorSKEW(
            cfg=cfg,
            factors_by_instru_dir=factors_by_instru_dir,
            universe=universe,
            db_struct_preprocess=preprocess,
        )
    elif fclass == TFactorClass.KURT:
        cfg = cfg_factors.KURT
        fac = CFactorKURT(
            cfg=cfg,
            factors_by_instru_dir=factors_by_instru_dir,
            universe=universe,
            db_struct_preprocess=preprocess,
        )
    elif fclass == TFactorClass.RS:
        cfg = cfg_factors.RS
        fac = CFactorRS(
            cfg=cfg,
            factors_by_instru_dir=factors_by_instru_dir,
            universe=universe,
            db_struct_preprocess=preprocess,
        )
    elif fclass == TFactorClass.BASIS:
        cfg = cfg_factors.BASIS
        fac = CFactorBASIS(
            cfg=cfg,
            factors_by_instru_dir=factors_by_instru_dir,
            universe=universe,
            db_struct_preprocess=preprocess,
        )
    elif fclass == TFactorClass.TS:
        cfg = cfg_factors.TS
        fac = CFactorTS(
            cfg=cfg,
            factors_by_instru_dir=factors_by_instru_dir,
            universe=universe,
            db_struct_preprocess=preprocess,
        )
    else:
        raise NotImplementedError(f"Invalid fclass = {fclass}")
    return fac, cfg
