import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from typing import Literal


class BaseLineEstimator:
    def __init__(self, coe: pd.Series):
        """

        :param coe: to save coe as val and index
                    val: np.ndarray
                    index: list[str]
                    np.ndarray and list are basic types supported by skops.
                    pd.Series is not supported by skops.
        """
        self.val = coe.values
        self.index = coe.index.tolist()

    @property
    def coe(self) -> pd.Series:
        return pd.Series(self.val, index=self.index)

    def __repr__(self) -> str:
        return str(self.coe)

    def predict(self, X: pd.DataFrame) -> np.ndarray | pd.Series:
        return X @ self.coe

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        yh = self.predict(X)
        return r2_score(np.sign(y), np.sign(yh))


class BaseLine:
    def __init__(self, method: Literal["IC", "IR", "SGN", "RNK"]):
        self.method = method

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseLineEstimator:
        data = pd.concat([X, y], axis=1, ignore_index=False)
        ic = data.groupby(by="trade_date").apply(lambda z: z.iloc[:, :-1].corrwith(z.iloc[:, -1]))
        mu = ic.mean()

        if self.method == "IC":
            scale = np.sqrt(mu.abs())
            coe = mu / scale.where(scale > 0, np.nan)
        elif self.method == "IR":
            sd = ic.std()
            coe = mu / sd.where(sd > 0, np.nan)
        elif self.method == "SGN":
            coe = np.sign(mu)
        elif self.method == "RNK":
            rnk = mu.rank()
            wgt = np.power(rnk, 0.25)
            coe = wgt / wgt.sum()
        else:
            raise ValueError(f"Invalid method: {self.method}")
        return BaseLineEstimator(coe=coe.fillna(0))
