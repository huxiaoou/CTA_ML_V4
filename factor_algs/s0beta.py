import pandas as pd
from husfort.qcalendar import CCalendar
from typedefs.typedefFactors import CCfgFactorGrpWin, TFactorNames, TFactorName
from solutions.factor import CFactorsByInstru
from math_tools.rolling import cal_rolling_beta


class CCfgFactorGrpS0BETA(CCfgFactorGrpWin):
    def __init__(self, **kwargs):
        super().__init__(factor_class="S0BETA", **kwargs)

    def name_diff2(self) -> TFactorName:
        return TFactorName(f"{self.factor_class}DIFF2")

    @property
    def names_diff(self) -> TFactorNames:
        return [self.name_diff(), self.name_diff2()]

    @property
    def factor_names(self) -> TFactorNames:
        return self.names_vanilla + self.names_res + self.names_diff


class CFactorS0BETA(CFactorsByInstru):
    def __init__(self, factor_grp: CCfgFactorGrpS0BETA, **kwargs):
        if not isinstance(factor_grp, CCfgFactorGrpS0BETA):
            raise TypeError("factor_grp must be CCfgFactorGrpS0BETA")
        super().__init__(factor_grp=factor_grp, **kwargs)
        self.cfg = factor_grp
        self.x, self.y = "INH0100_NHF", "return_c_major"

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        buffer_bgn_date = self.cfg.buffer_bgn_date(bgn_date, calendar)
        major_data = self.load_preprocess(
            instru, bgn_date=buffer_bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", self.y],
        )
        market_data = self.load_mkt(bgn_date=buffer_bgn_date, stp_date=stp_date)
        adj_data = pd.merge(
            left=major_data[["trade_date", "ticker_major", self.y]],
            right=market_data[["trade_date", self.x]],
            on="trade_date", how="left",
        )
        for win, name_vanilla, name_res in zip(
                self.cfg.wins, self.cfg.names_vanilla, self.cfg.names_res
        ):
            adj_data[name_vanilla] = cal_rolling_beta(df=adj_data, x=self.x, y=self.y, rolling_window=win)
            adj_data[name_res] = -(adj_data[self.y] - adj_data[name_vanilla] * adj_data[self.x])

        w = 20
        n0, n1 = self.cfg.name_vanilla(w), self.cfg.name_res(w)  # BETA20, RES20
        adj_data[self.cfg.name_diff()] = adj_data[n0] + adj_data[n1]
        adj_data[self.cfg.name_diff2()] = adj_data[n0] * adj_data[self.x].abs() + adj_data[n1]

        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data
