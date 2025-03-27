import numpy as np
import pandas as pd
import talib as ta
import itertools as ittl
from husfort.qcalendar import CCalendar
from typedef import (
    CCfgFactorGrpMTM, CCfgFactorGrpSKEW, CCfgFactorGrpKURT,
    CCfgFactorGrpRS,
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
        win_start_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, factor_name in zip(self.cfg.wins, self.cfg.factor_names):
            major_data[factor_name] = major_data["return_c_major"].rolling(window=win).sum()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorSKEW(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpSKEW, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, factor_name in zip(self.cfg.wins, self.cfg.factor_names):
            major_data[factor_name] = -major_data["return_c_major"].rolling(window=win).skew()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorKURT(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpKURT, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, factor_name in zip(self.cfg.wins, self.cfg.factor_names):
            major_data[factor_name] = major_data["return_c_major"].rolling(window=win).kurt()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorRS(CFactorsByInstru):
    def __init__(self, cfg: CCfgFactorGrpRS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_grp=cfg, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        __min_win = 5
        win_start_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "stock"],
        )
        adj_data["stock"] = adj_data["stock"].ffill(limit=__min_win).fillna(0)
        for win in self.cfg.wins:
            rspa = self.cfg.name_rspa(win)
            rsla = self.cfg.name_rsla(win)

            ma = adj_data["stock"].rolling(window=win).mean()
            s = adj_data["stock"] / ma.where(ma > 0, np.nan)
            adj_data[rspa] = 1 - s

            la = adj_data["stock"].shift(win)
            s = adj_data["stock"] / la.where(la > 0, np.nan)
            adj_data[rsla] = 1 - s
        n0, n1 = self.cfg.name_rspa(240), self.cfg.name_rspa(60)
        adj_data[self.cfg.name_diff()] = adj_data[n0] - adj_data[n1]
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data
