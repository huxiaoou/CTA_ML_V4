import numpy as np
import pandas as pd


def auto_weight_sum(x: pd.Series) -> float:
    weight = x.abs() / x.abs().sum()
    return x @ weight


def robust_ret_alg(x: pd.Series, y: pd.Series, scale: float = 1.0) -> pd.Series:
    """

    :param x: must have the same length as y
    :param y:
    :param scale: return scale
    :return:
    """
    return (x / y.where(y != 0, np.nan) - 1) * scale


def robust_ret_log(x: pd.Series, y: pd.Series, scale: float = 1.0) -> pd.Series:
    """

    :param x: must have the same length as y
    :param y:
    :param scale:
    :return: for log return, x, y are supposed to be positive
    """
    return (np.log(x.where(x > 0, np.nan) / y.where(y > 0, np.nan))) * scale


def robust_div(x: pd.Series, y: pd.Series, nan_val: float = np.nan) -> pd.Series:
    """

    :param x: must have the same length as y
    :param y:
    :param nan_val:
    :return:
    """

    return (x / y.where(y != 0, np.nan)).fillna(nan_val)
