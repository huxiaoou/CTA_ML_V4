import os
import numpy as np
import pandas as pd
import skops.io as sio
from dataclasses import dataclass
from loguru import logger
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from solutions.mclrn_custom import BaseLine
import lightgbm as lgb
import xgboost as xgb
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CMgrSqlDb
from husfort.qutility import SFG, SFY, check_and_makedirs, qtimer
from husfort.qmultiprocessing import TTask, mul_process_for_tasks, uni_process_for_tasks, CAgentQueue, EStatusWorker
from solutions.shared import gen_factors_avlb_db, gen_test_returns_avlb_db, gen_prdct_db
from typedefs.typedefFactors import CFactor, TFactorNames
from typedefs.typedefReturns import TReturnName
from typedefs.typedefModels import CTestData, CTestModel

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
        self.trn_score: float | None = None
        self.val_score: float | None = None

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
        x_data, y_data = aligned_data[self.x_cols], aligned_data[self.y_col]
        if self.test_model.using_instru:
            x, y = x_data.reset_index(level="instrument"), y_data
            x["instrument"] = x["instrument"].astype("category")
        else:
            x, y = x_data, y_data
        if self.test_model.classification:
            y = y.mask(y >= 0, other=1)
            y = y.mask(y < 0, other=0)
        return x, y

    def get_X(self, x_only_data: pd.DataFrame) -> pd.DataFrame:
        x_data = x_only_data[self.x_cols]
        if self.test_model.using_instru:
            x = x_data.reset_index(level="instrument")
            x["instrument"] = x["instrument"].astype("category")
        else:
            x = x_data
        return x

    def display_fitted_estimator(self) -> None:
        pass

    def get_fit_params(self, test_x: pd.DataFrame, test_y: pd.Series) -> dict:
        return {"eval_set": [[test_x, test_y]]} if self.test_model.early_stopping > 0 else {}

    def fit_estimator(self, x: pd.DataFrame, y: pd.Series):
        if self.test_model.hyper_param_grids:
            grid_cv_seeker = GridSearchCV(
                self.prototype,
                self.test_model.hyper_param_grids,
                cv=self.test_model.cv,
            )
            if self.test_model.early_stopping > 0:
                x_trn, x_val, y_trn, y_val = train_test_split(
                    x, y, test_size=0.1, random_state=self.RANDOM_STATE, shuffle=False,
                )
            else:
                x_trn, x_val, y_trn, y_val = x, x, y, y
            fit_params = self.get_fit_params(x_val, y_val)
            self.fitted_estimator = grid_cv_seeker.fit(x_trn, y_trn, **fit_params)
            self.trn_score = self.fitted_estimator.score(x_trn, y_trn)  # type: ignore
            self.val_score = self.fitted_estimator.score(x_val, y_val)  # type: ignore
        else:
            fit_params = self.get_fit_params(x, y)
            self.fitted_estimator = self.prototype.fit(x, y, **fit_params)
            self.trn_score = self.fitted_estimator.score(x, y)
            self.val_score = self.fitted_estimator.score(x, y)
        return 0

    def check_model_existence(self, mdl_upd_day: str) -> bool:
        month_dir = os.path.join(self.mclrn_mdl_dir, mdl_upd_day)
        model_file = f"{self.save_id}.skops"
        model_path = os.path.join(month_dir, model_file)
        return os.path.exists(model_path)

    def save_model(self, mdl_upd_day: str):
        model_file = f"{self.save_id}.skops"
        check_and_makedirs(month_dir := os.path.join(self.mclrn_mdl_dir, mdl_upd_day))
        model_path = os.path.join(month_dir, model_file)
        sio.dump(self.fitted_estimator, model_path)
        return 0

    def load_model(self, mdl_upd_day: str, verbose: bool) -> bool:
        model_file = f"{self.save_id}.skops"
        model_path = os.path.join(self.mclrn_mdl_dir, mdl_upd_day, model_file)
        if os.path.exists(model_path):
            untrusted_types = sio.get_untrusted_types(file=model_path)
            self.fitted_estimator = sio.load(model_path, trusted=untrusted_types)
            return True
        else:
            if verbose:
                logger.info(f"No model file for {SFY(self.save_id)} at {SFY(mdl_upd_day)}")
            return False

    def apply_estimator(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_y_h(self, pred: np.ndarray, idx: pd.Index) -> pd.Series:
        return pd.Series(data=pred, name=self.y_col, index=idx).astype(np.float64)

    def train(self, model_update_day: str, calendar: CCalendar, verbose: bool):
        if self.check_model_existence(mdl_upd_day=model_update_day) and verbose:
            logger.info(
                f"Model for {SFY(model_update_day)} @ {SFY(self.save_id)} have been calculated, "
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
        self.fit_estimator(x=x, y=y)
        self.display_fitted_estimator()
        self.save_model(mdl_upd_day=model_update_day)
        if verbose:
            logger.info(
                f"Train and save model @ {SFG(model_update_day)}, "
                f"factor selected @ {SFG(trn_e_date)}, "
                f"using train data @ [{SFG(trn_b_date)},{SFG(trn_e_date)}], "
            )
        return 0

    def process_trn(self, agent_queue: CAgentQueue, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool):
        model_update_days = calendar.get_week_end_days_in_range(bgn_date=bgn_date, stp_date=stp_date)
        agent_queue.set_description(f"{self.save_id:<48s} train")
        agent_queue.set_completed(0)
        agent_queue.set_total(len(model_update_days))
        for model_update_day in model_update_days:
            self.train(model_update_day, calendar, verbose)
            agent_queue.set_advance(advance=1)
        return 0

    def predict(
            self, mdl_upd_day: str, prd_week_days: list[str], calendar: CCalendar, verbose: bool,
    ) -> pd.Series:
        self.reset_estimator()
        if self.load_model(mdl_upd_day=mdl_upd_day, verbose=verbose):
            prd_b_date, prd_e_date = prd_week_days[0], prd_week_days[-1]
            prd_s_date = calendar.get_next_date(prd_e_date, shift=1)
            prd_x_data = self.load_x(prd_b_date, prd_s_date)
            x_only_data = self.drop_and_fill_nan(prd_x_data)
            x = self.get_X(x_only_data=x_only_data)
            pred = self.apply_estimator(x=x)
            y_h_data = self.get_y_h(pred, idx=x_only_data.index)
            if verbose:
                trn_e_date = calendar.get_next_date(mdl_upd_day, shift=-self.test_data.ret.shift)
                logger.info(
                    f"Call model @ {SFG(mdl_upd_day)}, "
                    f"factor selected @ {SFG(trn_e_date)}, "
                    f"prediction @ [{SFG(prd_b_date)},{SFG(prd_e_date)}], "
                )
            return y_h_data
        else:
            return pd.Series(dtype=np.float64)

    def process_prd(
            self, agent_queue: CAgentQueue, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool,
    ) -> pd.DataFrame:
        upd_days_groups: dict[str, list[str]] = calendar.split_by_week_end_days(
            dates=calendar.get_iter_list(bgn_date, stp_date),
            week_end_days=calendar.get_week_end_days_in_range(bgn_date, stp_date),
        )
        agent_queue.set_description(f"{self.save_id:<48s} predict")
        agent_queue.set_completed(0)
        agent_queue.set_total(len(upd_days_groups))
        pred_res: list[pd.Series] = []
        for mdl_upd_day, prd_week_days in upd_days_groups.items():
            if not prd_week_days:
                continue
            month_prediction = self.predict(mdl_upd_day, prd_week_days, calendar, verbose)
            pred_res.append(month_prediction)
            agent_queue.set_advance(advance=1)
        prediction = pd.concat(pred_res, axis=0, ignore_index=False)
        prediction.index = pd.MultiIndex.from_tuples(prediction.index, names=self.XY_INDEX)
        sorted_prediction = prediction.reset_index().sort_values(["trade_date", "instrument"])
        return sorted_prediction

    def process_save_prediction(
            self, agent_queue: CAgentQueue, prediction: pd.DataFrame, calendar: CCalendar,
    ):
        agent_queue.set_description(f"{self.save_id:<48s} save")
        agent_queue.set_completed(0)
        agent_queue.set_total(3)
        db_struct_prdct = gen_prdct_db(self.mclrn_prd_dir, self.save_id, self.test_data.ret.ret_name)
        agent_queue.set_advance(advance=1)
        check_and_makedirs(db_struct_prdct.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_prdct.db_save_dir,
            db_name=db_struct_prdct.db_name,
            table=db_struct_prdct.table,
            mode="a",
        )
        agent_queue.set_advance(advance=1)
        if sqldb.check_continuity(incoming_date=prediction["trade_date"].iloc[0], calendar=calendar) == 0:
            sqldb.update(update_data=prediction)
            agent_queue.set_advance(advance=1)
        return 0

    def main_mclrn_model(
            self,
            agent_queue: CAgentQueue,
            bgn_date: str, stp_date: str, calendar: CCalendar,
            verbose: bool,
    ):
        logger.remove()
        logger.add("mclrn.log")
        self.process_trn(agent_queue, bgn_date, stp_date, calendar, verbose)
        prediction = self.process_prd(agent_queue, bgn_date, stp_date, calendar, verbose)
        self.process_save_prediction(agent_queue, prediction, calendar)
        agent_queue.set_status(EStatusWorker.FINISHED)
        return 0


"""
Part II: Apply estimator
"""


class CTestMclrnReg(CTestMclrn):
    def apply_estimator(self, x: pd.DataFrame) -> np.ndarray:
        # For BASELINE, LINEAR, RIDGE
        pred = self.fitted_estimator.predict(X=x)
        return pred


class CTestMclrnClsProb(CTestMclrn):
    def apply_estimator(self, x: pd.DataFrame) -> np.ndarray:
        # For LOGISTIC, MLP, LGBM, XGB
        prob = self.fitted_estimator.predict_proba(X=x)
        idx = np.where(self.fitted_estimator.classes_ == 1)[0][0]
        pred = prob[:, idx] - 0.50
        return pred


class CTestMclrnClsDec(CTestMclrn):
    def apply_estimator(self, x: pd.DataFrame) -> np.ndarray:
        # For SVM
        pred = self.fitted_estimator.decision_function(X=x)
        return pred


"""
Part III: Specific class for Machine Learning
"""


class CTestMclrnBaseLine(CTestMclrnReg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = BaseLine(method="RNK")

    def display_fitted_estimator(self) -> None:
        text = f"{self.save_id:<52s}| score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnLinear(CTestMclrnReg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = LinearRegression(fit_intercept=False)

    def display_fitted_estimator(self) -> None:
        text = f"{self.save_id:<52s}| score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnRidge(CTestMclrnReg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = Ridge(fit_intercept=False)

    def display_fitted_estimator(self) -> None:
        alpha = self.fitted_estimator.best_estimator_.alpha
        text = f"{self.save_id:<52s}| " \
               f"alpha = {alpha:>5.2f} | " \
               f"score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnLogistic(CTestMclrnClsProb):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = LogisticRegression(
            fit_intercept=False,
            random_state=self.RANDOM_STATE,
        )

    def display_fitted_estimator(self) -> None:
        c = self.fitted_estimator.best_estimator_.C
        text = f"{self.save_id:<52s}| " \
               f"C = {c:>5.2f} | " \
               f"score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnMlp(CTestMclrnClsProb):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = MLPClassifier(
            random_state=self.RANDOM_STATE,
            max_iter=1000,
        )

    def display_fitted_estimator(self) -> None:
        alpha = self.fitted_estimator.best_estimator_.alpha
        layers = self.fitted_estimator.best_estimator_.hidden_layer_sizes
        text = f"{self.save_id:<52s}| " \
               f"layers = {layers} | " \
               f"alpha = {alpha:>6.4f} | " \
               f"score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnSVM(CTestMclrnClsDec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototype = LinearSVC(
            fit_intercept=False,
            tol=0.0001,
            random_state=self.RANDOM_STATE,
        )

    def display_fitted_estimator(self) -> None:
        c = self.fitted_estimator.best_estimator_.C
        text = f"{self.save_id:<52s}| " \
               f"C = {c:>5.2f} | " \
               f"score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnLGBM(CTestMclrnClsProb):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.test_model.early_stopping <= 0:
            raise ValueError(f"test_model.early_stopping must be > 0 for {self.__class__.__name__}")

        # self.prototype = lgb.LGBMRegressor(
        self.prototype = lgb.LGBMClassifier(
            early_stopping_rounds=self.test_model.early_stopping,
            random_state=self.RANDOM_STATE,
            verbose=-1,
            # other fixed parameters
            # force_row_wise=True,  # cpu device only
            # device_type="gpu", # for small data cpu is much faster
        )

    def display_fitted_estimator(self) -> None:
        best_estimator = self.fitted_estimator.best_estimator_
        text = f"{self.save_id:<52s}| " \
               f"n_estimator = {best_estimator.best_iteration_:>3d}/{best_estimator.n_estimators:>3d} | " \
               f"leaves = {best_estimator.max_leaves:>2d} | " \
               f"depth = {best_estimator.max_depth:>2d} | " \
               f"lr = {best_estimator.learning_rate:>4.2f} | " \
               f"score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


class CTestMclrnXGB(CTestMclrnClsProb):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.test_model.early_stopping <= 0:
            raise ValueError(f"test_model.early_stopping must be > 0 for {self.__class__.__name__}")

        # self.prototype = xgb.XGBRegressor(
        self.prototype = xgb.XGBClassifier(
            # other fixed parameters
            early_stopping_rounds=self.test_model.early_stopping,
            verbosity=0,
            random_state=self.RANDOM_STATE,
            enable_categorical=True,
            # device="cuda",  # cpu maybe faster for data not in large scale.
        )

    def get_fit_params(self, test_x: pd.DataFrame, test_y: pd.Series) -> dict:
        d = super().get_fit_params(test_x, test_y)
        d.update({"verbose": False})
        return d

    def display_fitted_estimator(self) -> None:
        best_estimator = self.fitted_estimator.best_estimator_
        text = f"{self.save_id:<52s}| " \
               f"n_estimator = {best_estimator.best_iteration:>3d}/{best_estimator.n_estimators:>3d} | " \
               f"leaves = {best_estimator.max_leaves:>2d} | " \
               f"depth = {best_estimator.max_depth:>2d} | " \
               f"lr = {best_estimator.learning_rate:>4.2f} | " \
               f"score = [{self.trn_score:>7.4f}]/[{self.val_score:>7.4f}]"
        logger.info(text)


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
    tasks: list[TTask] = [
        TTask((test.main_mclrn_model, (bgn_date, stp_date, calendar, verbose)))
        for test in tests
    ]
    if call_multiprocess:
        mul_process_for_tasks(tasks=tasks, processes=processes, callback_log=lambda s: logger.info(s))
    else:
        uni_process_for_tasks(tasks=tasks, callback_log=lambda s: logger.info(s), debug_mode=True)
    return 0
