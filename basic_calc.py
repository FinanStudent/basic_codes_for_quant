"""
这个文件实现了常见指标的计算
收益率: 累计回报率, 年化收益率, 年化超额
风险: 年化波动, 超额年化波动, 下行风险, 最大回撤率, 最大回撤回补天数, 超额最大回撤
收益风险指标: sharpe, sortino, calmar, treynor
"""
__author__ = 'xhs@我是天空飘着的雾霾'

import numpy as np
import math

"""收益率计算"""


def cumulative_rtn(rtn_series):
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    return (1 + rtn_series).prod() - 1


def annualized_rtn(rtn_series):
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    rtn = cumulative_rtn(rtn_series)
    return (rtn+1)**(250/len(rtn_series))-1


def annualized_alpha(rtn_series, benchmark_series):
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    benchmark_series = benchmark_series.reindex_like(rtn_series)
    alpha = (1+rtn_series).prod()-(1+benchmark_series).prod()
    return (1+alpha)**(250/len(rtn_series))-1


"""风险计算"""


def annualized_std(rtn_series):
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    return rtn_series.std() * math.sqrt(250)


def alpha_annualized_std(rtn_series, benchmark_series):
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    benchmark_series = benchmark_series.reindex_like(rtn_series)
    return annualized_std(rtn_series-benchmark_series)


def downside_risk(rtn_series, threshold_rtn=0):
    # 下行风险
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.mask(rtn_series >= threshold_rtn, np.nan)
    rtn_series = rtn_series.dropna()
    return annualized_std(rtn_series)


def mdd(rtn_series):
    # 最大回撤率
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    cumu_series = (1+rtn_series).cumprod()
    mdd = ((cumu_series.cummax()-cumu_series)/cumu_series.cummax()).max()
    # max_point, mdd = float('-inf'), 0
    # for rtn in cumu_series:
    #     max_point = max(max_point, rtn)
    #     dd = (max_point - rtn) / max_point
    #     mdd = max(mdd, dd)
    return mdd


def mdd_days(rtn_series):
    # 最大回撤回补天数占成立时间比例
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    cumu_series = (1 + rtn_series).cumprod()
    mdds = (cumu_series.cummax()-cumu_series).values
    i1, i2, mdd_day = 0, 0, 0
    for i in range(len(mdds)):
        m = mdds[i]
        if m == 0:
            mdd_day = max(mdd_day, i2-i1)
            i1, i2 = i2, i
    return max(i2-i1, mdd_day)/len(rtn_series)


def alpha_mdd(rtn_series, benchmark_series):
    # 超额最大回撤
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    benchmark_series = benchmark_series.reindex_like(rtn_series)
    cumu_series = (rtn_series-benchmark_series).cumsum()
    max_point, mdd = float('-inf'), 0
    for alpha in cumu_series:
        max_point = max(max_point, alpha)
        dd = (max_point - alpha)
        mdd = max(mdd, dd)
    return mdd


"""收益风险指标"""


def sharpe_ratio(rtn_series, benchmark_series=0):
    excess_rtn = rtn_series - benchmark_series
    excess_rtn = excess_rtn.mask(excess_rtn == np.inf, np.nan)
    excess_rtn = excess_rtn.dropna()
    std = annualized_std(excess_rtn)
    rtn = annualized_rtn(excess_rtn)
    if std == 0:
        return np.nan
    return rtn / std

def alpha_sharpe_ratio(rtn_series, benchmark_series):
    rtn_series = rtn_series.mask(rtn_series == np.inf, np.nan)
    rtn_series = rtn_series.dropna()
    benchmark_series = benchmark_series.reindex_like(rtn_series)
    return annualized_alpha(rtn_series, benchmark_series)/alpha_annualized_std(rtn_series, benchmark_series)


def sortino_ratio(rtn_series, minimun_acceptable_rtn=0):
    excess_rtn = rtn_series - minimun_acceptable_rtn
    excess_rtn = excess_rtn.mask(excess_rtn == np.inf, np.nan)
    excess_rtn = excess_rtn.dropna()
    rtn = annualized_rtn(excess_rtn)
    dr = downside_risk(rtn_series, minimun_acceptable_rtn)
    if dr == 0:
        return np.nan
    return rtn / dr


def calmar_ratio(rtn_series, benchmark_series=0):
    excess_rtn = rtn_series - benchmark_series
    excess_rtn = excess_rtn.mask(excess_rtn == np.inf, np.nan)
    excess_rtn = excess_rtn.dropna()
    max_drawdown = mdd(rtn_series)
    rtn = annualized_rtn(excess_rtn)
    if max_drawdown == 0:
        return np.nan
    return rtn / max_drawdown


def treynor_ratio(rtn_series, benchmark_series, risk_free=0):
    """
    benchmark_series: 市场平均水平
    """
    excess_rtn = rtn_series - risk_free
    excess_rtn = excess_rtn.mask(excess_rtn == np.inf, np.nan)
    excess_rtn = excess_rtn.dropna()
    beta = np.cov(excess_rtn, benchmark_series)[0, 0] / benchmark_series.std()
    if beta == 0:
        return np.nan
    return annualized_rtn(excess_rtn) / beta
