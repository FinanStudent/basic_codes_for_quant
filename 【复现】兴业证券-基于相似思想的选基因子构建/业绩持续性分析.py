"""
海通证券-《量化私募基金的业绩持续性研究与FOF组合构建》
这个文件实现了业绩持续性因子计算
"""
__author__ = 'xhs@我是天空飘着的雾霾'

import pandas as pd
from Accelerator import *
from sklearn.linear_model import LinearRegression as LR
from datetime import date
from DataLoader import *

start = '2012-01-01'  # 因子开始时间
end = date.strftime(date.today(), '%Y-%m-%d')  # 因子结束时间
dl = DataLoader(start, end)
rtn_value = dl.fund_rtn('monthly', 1)  # 月频收益率数据

class CPR_Factor():
    """
    转移概率法
    """
    def __init__(self) -> None:
        rank_WL = rtn_value.sub(rtn_value.quantile(0.5, axis=1), axis=0)
        rank_WL = rank_WL.mask(rank_WL >= 0, 1).mask(rank_WL < 0, -1)
        rank_WL = rank_WL.rolling(2).sum()
        self.rank_WL = rank_WL

    def CPR(self, start_time, end_time):
        df = self.rank_WL.loc[start_time:end_time, :].dropna(thresh=10, axis=1)
        res = df.apply(lambda x: x.value_counts())
        series = (res.loc[2])/res.loc[0]
        series.name = end_time
        return series
    
    def generate_factor(self, window):
        args = [(dl.trade_months[i-window], dl.trade_months[i]) for i in range(window, len(dl.trade_months))]
        calc_set = [(self.CPR, args)]
        accelerator = Accelerator(core=10)
        results = accelerator.execute(calc_set)
        result = pd.DataFrame()
        for k, res in results:
            result = result.append(res)
        result = result.reindex(index=dl.trade_days).ffill()
        return result


class RegressPersist_Factor():
    """
    回归法
    """
    def __init__(self) -> None:
        self.rtn_rank = rtn_value.rank(axis=1)

    def AR1(self, series):
        Y, X = series.dropna(), series.shift(-1).dropna()
        shared_index = Y.index.intersection(X.index)
        Y, X = Y.loc[shared_index], X.loc[shared_index]
        # reg = LR().fit(X, Y)
        # return reg.coef_[0]
        return np.cov(X,Y)[0,1]/np.cov(X,X)[0,1]

    def regress_persist(self, start_time, end_time):
        df = self.rtn_rank.loc[start_time:end_time, :].dropna(thresh=10, axis=1)
        res = df.apply(self.AR1)
        res.name = end_time
        return res
    
    def generate_factor(self, window):
        args = [(dl.trade_months[i-window], dl.trade_months[i]) for i in range(window, len(dl.trade_months))]
        calc_set = [(self.regress_persist, args)]
        accelerator = Accelerator(core=10)
        results = accelerator.execute(calc_set)
        result = pd.DataFrame()
        for k, res in results:
            result = result.append(res)
        result = result.reindex(index=dl.trade_days).ffill()
        return result  

if __name__ == '__main__':
    # 业绩持续性分析1
    fac = CPR_Factor()
    res = fac.generate_factor(24)
    print(res)
    # 业绩持续性分析2
    fac = RegressPersist_Factor()
    res = fac.generate_factor(24)
    print(res)
