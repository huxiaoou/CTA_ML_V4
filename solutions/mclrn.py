import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import skops.io as sio
from dataclasses import dataclass
from loguru import logger
from rich.progress import track, Progress
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
import lightgbm as lgb
import xgboost as xgb
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CMgrSqlDb
from husfort.qutility import SFG, SFY, check_and_makedirs, error_handler, qtimer
from solutions.shared import gen_factors_avlb_db, gen_test_returns_avlb_db, gen_prdct_db
from typedef import (
    TReturnName, TFactorNames, CFactor,
    CTestData, CTestModel,
)

"""
Part I: Base class for Machine Learning
"""


@dataclass
class CTestMclrn:
    XY_INDEX = ["trade_date", "instrument"]
    RANDOM_STATE = 0

    def __init__(
            self,
            test_data: CTestData,
            test_model: CTestModel,
            mclrn_dir: str,
    ):
        self.test_data = test_data
        self.test_model = test_model
        self.mclrn_dir = mclrn_dir
        self.prototype = NotImplemented
        self.fitted_estimator = NotImplemented
        self.train_score: float | None = None

    @property
    def mclrn_mdl_dir(self) -> str:
        return os.path.join(self.mclrn_dir, "mdl")

    @property
    def mclrn_prd_dir(self) -> str:
        return os.path.join(self.mclrn_dir, "prd")

    @property
    def save_id(self) -> str:
        return f"{self.test_model.save_id}.{self.test_data.save_id}"

    @property
    def x_cols(self) -> TFactorNames:
        return [f.factor_name for f in self.test_data.factors]

    @property
    def y_col(self) -> TReturnName:
        return self.test_data.ret.ret_name

    def reset_estimator(self):
        self.fitted_estimator = None
        return 0

    def load_factor(self, factor: CFactor, bgn_date: str, stp_date: str) -> pd.DataFrame:
        factor_names = [factor.factor_name]
        db_struct_fac = gen_factors_avlb_db(self.test_data.factors_avlb_dir, factor.factor_class, factors=[factor])
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_fac.db_save_dir,
            db_name=db_struct_fac.db_name,
            table=db_struct_fac.table,
            mode="r",
        )
        instru_data = sqldb.read_by_range(bgn_date, stp_date, value_columns=self.XY_INDEX + factor_names)
        instru_data[factor_names] = instru_data[factor_names].astype(np.float64).fillna(np.nan)
        return instru_data

    def load_x(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        factor_dfs: list[pd.DataFrame] = []
        for factor in self.test_data.factors:
            factor_data = self.load_factor(factor, bgn_date, stp_date)
            factor_dfs.append(factor_data.set_index(self.XY_INDEX))
        x_data = pd.concat(factor_dfs, axis=1, ignore_index=False)
        return x_data

    def load_test_return(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        ret_name = self.test_data.ret.ret_name
        db_struct_ref = gen_test_returns_avlb_db(
            self.test_data.test_returns_avlb_dir, self.test_data.ret.ret_class, self.test_data.ret)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_ref.db_save_dir,
            db_name=db_struct_ref.db_name,
            table=db_struct_ref.table,
            mode="r"
        )
        ret_data = sqldb.read_by_range(bgn_date, stp_date, value_columns=self.XY_INDEX + [self.test_data.ret.ret_name])
        ret_data[ret_name] = ret_data[ret_name].astype(np.float64).fillna(np.nan)
        return ret_data

    def load_y(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        ret_data = self.load_test_return(bgn_date=bgn_date, stp_date=stp_date)
        ret_data = ret_data.set_index(self.XY_INDEX).sort_index()
        return ret_data

    @staticmethod
    def aligned_xy(x_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
        aligned_data = pd.merge(left=x_data, right=y_data, left_index=True, right_index=True, how="inner")
        s0, s1, s2 = len(x_data), len(y_data), len(aligned_data)
        if s0 == s1 == s2:
            return aligned_data
        else:
            logger.error(
                f"Length of X = {SFY(s0)}, Length of y = {SFY(s1)}, Length of aligned (X,y) = {SFY(s2)}"
            )
            raise ValueError("(X,y) have different lengths")

    @staticmethod
    def drop_and_fill_nan(aligned_data: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
        idx_null = aligned_data.isnull()
        nan_data = aligned_data[idx_null.any(axis=1)]
        if not nan_data.empty:
            # keep rows where nan prop is <= threshold
            filter_nan = (idx_null.sum(axis=1) / aligned_data.shape[1]) <= threshold
            return aligned_data[filter_nan].fillna(0)
        return aligned_data

    def get_X_y(self, aligned_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return aligned_data[self.x_cols], aligned_data[self.y_col]

    def get_X(self, x_data: pd.DataFrame) -> pd.DataFrame:
        return x_data[self.x_cols]

    def display_fitted_estimator(self) -> None:
        pass

    def fit_estimator(self, x_data: pd.DataFrame, y_data: pd.Series):
        if self.test_model.using_instru:
            x, y = x_data.reset_index(level="instrument"), y_data
            x["instrument"] = x["instrument"].astype("category")
        else:
            x, y = x_data.values, y_data.values
        if self.test_model.cv > 0:
            grid_cv_seeker = GridSearchCV(
                self.prototype,
                self.test_model.hyper_param_grids,
                cv=self.test_model.cv,
            )
            self.fitted_estimator = grid_cv_seeker.fit(x, y)
        else:
            self.fitted_estimator = self.prototype.fit(x, y)
        self.train_score = self.fitted_estimator.score(x, y)
        return 0

    def check_model_existence(self, month_id: str) -> bool:
        month_dir = os.path.join(self.mclrn_mdl_dir, month_id)
        model_file = f"{self.save_id}.skops"
        model_path = os.path.join(month_dir, model_file)
        return os.path.exists(model_path)

    def save_model(self, month_id: str):
        model_file = f"{self.save_id}.skops"
        check_and_makedirs(month_dir := os.path.join(self.mclrn_mdl_dir, month_id))
        model_path = os.path.join(month_dir, model_file)
        sio.dump(self.fitted_estimator, model_path)
        return 0

    def load_model(self, month_id: str, verbose: bool) -> bool:
        model_file = f"{self.save_id}.skops"
        model_path = os.path.join(self.mclrn_mdl_dir, month_id, model_file)
        if os.path.exists(model_path):
            self.fitted_estimator = sio.load(
                model_path,
                trusted=[
                    "collections.defaultdict",
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor",
                    "xgboost.core.Booster", "xgboost.sklearn.XGBRegressor",
                    "sklearn.metrics._scorer._PassthroughScorer",
                    "sklearn.utils._metadata_requests.MetadataRequest",
                    "sklearn.utils._metadata_requests.MethodMetadataRequest",
                ],
            )
            return True
        else:
            if verbose:
                logger.info(f"No model file for {SFY(self.save_id)} at {SFY(month_id)}")
            return False

    def apply_estimator(self, x_data: pd.DataFrame) -> pd.Series:
        if self.test_model.using_instru:
            x = x_data.reset_index(level="instrument")
            x["instrument"] = x["instrument"].astype("category")
        else:
            x = x_data.values
        pred = self.fitted_estimator.predict(X=x)  # type:ignore
        return pd.Series(data=pred, name=self.y_col, index=x_data.index)

    def train(self, model_update_day: str, calendar: CCalendar, verbose: bool):
        model_update_month = model_update_day[0:6]
        if self.check_model_existence(month_id=model_update_month) and verbose:
            logger.info(
                f"Model for {SFY(model_update_month)} @ {SFY(self.save_id)} have been calculated, "
                "program will skip it."
            )
            return 0
        shift, win = self.test_data.ret.shift, self.test_model.trn_win
        trn_b_date = calendar.get_next_date(model_update_day, shift=-shift - win + 1)
        trn_e_date = calendar.get_next_date(model_update_day, shift=-shift)
        trn_s_date = calendar.get_next_date(trn_e_date, shift=1)
        trn_x_data, trn_y_data = self.load_x(trn_b_date, trn_s_date), self.load_y(trn_b_date, trn_s_date)
        trn_aligned_data = self.aligned_xy(trn_x_data, trn_y_data)
        trn_aligned_data = self.drop_and_fill_nan(trn_aligned_data[self.x_cols + [self.y_col]])
        x, y = self.get_X_y(aligned_data=trn_aligned_data)
        self.fit_estimator(x_data=x, y_data=y)
        self.display_fitted_estimator()
        self.save_model(month_id=model_update_month)
        if verbose:
            logger.info(
                f"Train model @ {SFG(model_update_day)}, "
                f"factor selected @ {SFG(trn_e_date)}, "
                f"using train data @ [{SFG(trn_b_date)},{SFG(trn_e_date)}], "
                f"save as {SFG(model_update_month)}"
            )
        return 0

    def process_trn(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool):
        model_update_days = calendar.get_last_days_in_range(bgn_date=bgn_date, stp_date=stp_date)
        for model_update_day in model_update_days:
            self.train(model_update_day, calendar, verbose)
        return 0

    def predict(
            self, prd_month_id: str, prd_month_days: list[str], calendar: CCalendar, verbose: bool,
    ) -> pd.Series:
        trn_month_id = calendar.get_next_month(prd_month_id, -1)
        self.reset_estimator()
        if self.load_model(month_id=trn_month_id, verbose=verbose):
            model_update_day = calendar.get_last_day_of_month(trn_month_id)
            prd_b_date, prd_e_date = prd_month_days[0], prd_month_days[-1]
            prd_s_date = calendar.get_next_date(prd_e_date, shift=1)
            prd_x_data = self.load_x(prd_b_date, prd_s_date)
            x_data = self.get_X(x_data=prd_x_data)
            x_data = self.drop_and_fill_nan(x_data)
            y_h_data = self.apply_estimator(x_data=x_data)
            if verbose:
                trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test_data.ret.shift)
                logger.info(
                    f"Call model @ {SFG(model_update_day)}, "
                    f"factor selected @ {SFG(trn_e_date)}, "
                    f"prediction @ [{SFG(prd_b_date)},{SFG(prd_e_date)}], "
                    f"load model from {SFG(trn_month_id)}"
                )
            return y_h_data.astype(np.float64)
        else:
            return pd.Series(dtype=np.float64)

    def process_prd(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool) -> pd.DataFrame:
        months_groups = calendar.split_by_month(dates=calendar.get_iter_list(bgn_date, stp_date))
        pred_res: list[pd.Series] = []
        for prd_month_id, prd_month_days in months_groups.items():
            month_prediction = self.predict(prd_month_id, prd_month_days, calendar, verbose)
            pred_res.append(month_prediction)
        prediction = pd.concat(pred_res, axis=0, ignore_index=False)
        prediction.index = pd.MultiIndex.from_tuples(prediction.index, names=self.XY_INDEX)
        sorted_prediction = prediction.reset_index().sort_values(["trade_date", "instrument"])
        return sorted_prediction

    def process_save_prediction(self, prediction: pd.DataFrame, calendar: CCalendar):
        db_struct_prdct = gen_prdct_db(self.mclrn_prd_dir, self.save_id, self.test_data.ret.ret_name)
        check_and_makedirs(db_struct_prdct.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_prdct.db_save_dir,
            db_name=db_struct_prdct.db_name,
            table=db_struct_prdct.table,
            mode="a",
        )
        if sqldb.check_continuity(incoming_date=prediction["trade_date"].iloc[0], calendar=calendar) == 0:
            sqldb.update(update_data=prediction)
        return 0

    def main_mclrn_model(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool):
        self.process_trn(bgn_date, stp_date, calendar, verbose)
        prediction = self.process_prd(bgn_date, stp_date, calendar, verbose)
        self.process_save_prediction(prediction, calendar)
        return 0


"""
Part II: Specific class for Machine Learning
"""


class CTestMclrnLinear(CTestMclrn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = LinearRegression(fit_intercept=False)

    def display_fitted_estimator(self) -> None:
        score = self.train_score
        text = f"{self.save_id}, score = {score:>9.6f}"
        print(text)


class CTestMclrnRidge(CTestMclrn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = Ridge(fit_intercept=False)

    def display_fitted_estimator(self) -> None:
        alpha = self.fitted_estimator.best_estimator_.alpha
        score = self.train_score
        text = f"{self.save_id}, best alpha = {alpha:>6.1f}, score = {score:>9.6f}"
        print(text)
        # print(coef)


class CTestMclrnLGBM(CTestMclrn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = lgb.LGBMRegressor(
            # other fixed parameters
            force_row_wise=True,  # cpu device only
            verbose=-1,
            random_state=self.RANDOM_STATE,
            # device_type="gpu", # for small data cpu is much faster
        )

    def display_fitted_estimator(self) -> None:
        best_estimator = self.fitted_estimator.best_estimator_
        score = self.train_score
        text = f"{self.save_id}, " \
               f"n_estimator = {best_estimator.n_estimators:>2d}, " \
               f"num_leaves = {best_estimator.num_leaves:>2d}, " \
               f"learning_rate = {best_estimator.learning_rate:>4.2f}, " \
               f"score = {score:>9.6f}"
        print(text)


class CTestMclrnXGB(CTestMclrn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = xgb.XGBRegressor(
            # other fixed parameters
            verbosity=0,
            random_state=self.RANDOM_STATE,
            enable_categorical=True,
            # device="cuda",  # cpu maybe faster for data not in large scale.
        )

    def display_fitted_estimator(self) -> None:
        best_estimator = self.fitted_estimator.best_estimator_
        score = self.train_score
        text = f"{self.save_id}, " \
               f"n_estimator = {best_estimator.n_estimators:>2d}, " \
               f"max_leaves = {best_estimator.max_leaves:>2d}, " \
               f"learning_rate = {best_estimator.learning_rate:>4.2f}, " \
               f"score = {score:>9.6f}"
        print(text)


@qtimer
def main_train_and_predict(
        tests: list[CTestMclrn],
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
        verbose: bool,
):
    desc = "Training and predicting for machine learning"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for test in tests:
                    pool.apply_async(
                        test.main_mclrn_model,
                        kwds={
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                            "verbose": verbose,
                        },
                        callback=lambda _: pb.update(task_id=main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for test in track(tests, description=desc):
            # for test in tests:
            test.main_mclrn_model(
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0
