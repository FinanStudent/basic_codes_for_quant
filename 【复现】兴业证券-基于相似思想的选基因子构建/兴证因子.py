"""
《兴业证券-基于相似思想的选基因子构建》
平均相似性因子：独特的策略能力
1: 每天对基金过去120交易日的超额进行聚类，得到相似基金
2: 相似基金相互求取余弦相似度的平均值,作为因子值
复制残差因子：独特的策略能力
1: 用过去120交易日的超额求取相关性，得到最相似的10支基金
2: 相似基金对该基金回归，取残差作为因子值
相似动量因子：相似基金出现了领涨
1: 确定lag期限
2: 求取本基金lag后与其他基金收益率的相关性
3: 以相关性为权重对其他基金的当期收益,作为因子值
"""
__author__ = 'xhs@我是天空飘着的雾霾'

import pandas as pd
import numpy as np
import time
import os
from Accelerator import *
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
from datetime import date
import swifter
from DataLoader import *


class Average_Similarity:
    def __init__(self, K=7):
        self.fund_rtn = dl.fund_rtn('daily', 1)
        self.K = K
        self.args = [(dl.tds.get_pre_trade_day(x, 120), x) for x in dl.trade_days]

    def generate_clusters(self, start_time, end_time):
        """
        聚类结果生成
        """
        if start_time is None: return
        data = self.fund_rtn.loc[start_time:end_time, :].dropna(how='all').dropna(axis=1)
        if len(data.columns) < self.K: return
        model = KMeans(n_clusters=self.K, random_state=0)
        model.fit(data.T)
        res = pd.Series(model.labels_, index=data.columns, dtype=np.int8, name=end_time)
        return res
    
    def update_clusters(self):
        # 聚类情况增量更新
        run_start = time.time()
        exist_file = os.path.exists('data/kmeans_clusters.csv')
        if exist_file:
            fund_clusters = pd.read_csv('data/kmeans_clusters.csv', index_col=0)
            last_date = fund_clusters.index[-1]
        else:
            fund_clusters = pd.DataFrame()
            last_date = '0000-00-00'
        with tqdm(total=len(self.args), desc="Progress") as pbar:
            for x, y in self.args:
                try:
                    if (y <= last_date) or (y >= self.fund_rtn.index[-1]): continue
                    fs = self.generate_clusters(x, y)
                    if fs is None: continue
                    fund_clusters = fund_clusters.append(fs)
                finally:
                    pbar.update(1)
        print(fund_clusters)
        self.fund_clusters = fund_clusters
        fund_clusters.to_csv('data/kmeans_clusters.csv')
        run_end = time.time()
        print('聚类更新完成: 用时 %.2f s'%(run_end-run_start))

    def avg_similarity(self, start_time, end_time):
        """
        平均相似性因子
        """
        if start_time is None: return
        data = self.fund_rtn.loc[start_time:end_time, :].dropna(how='all').dropna(axis=1)
        if len(data.columns) < 7: return
        labels = self.fund_clusters.loc[end_time].dropna().astype('int')
        fundnames, fundsimis = [], []
        for i in range(labels.max()+1):
            label_fund = labels[labels==i].index
            label_fund_cs = cosine_similarity(data.loc[:, label_fund].T)
            label_res = (label_fund_cs.sum(axis=1)-1)/(len(label_fund)-1)
            fundnames.append(label_fund)
            fundsimis.append(label_res)
        res = pd.Series(np.concatenate(fundsimis), 
                        index=np.concatenate(fundnames), 
                        dtype=np.float16,
                        name=end_time)
        return res

    def generate_factor(self):
        # # 平均相似度因子生成
        accelerator = Accelerator(core=15)
        calc_set = [(self.avg_similarity, self.args)]
        results = accelerator.execute(calc_set)
        factor_matrix = pd.DataFrame()
        wrong_info, wrong_tickers = [], []
        for k, res in results:  # 计算时间2分钟左右
            if res is None: continue
            elif isinstance(res, str) or isinstance(res, Exception):  # 运行出错的报错信息
                wrong_info.append(res)
                wrong_tickers.append(k)
            else:
                factor_matrix = pd.concat([factor_matrix, res], axis=1)
        factor_matrix = factor_matrix.T.sort_index().astype('float')
        print(factor_matrix)
        print(wrong_tickers)
        print(wrong_info)
        return factor_matrix


class Replicate_Residual:
    def __init__(self):
        self.args = [(dl.tds.get_pre_trade_day(x, 120), x) for x in dl.trade_days]
        self.fund_rtn = dl.fund_rtn('daily', 1)

    def get_resid(self, data, y_col, all_cos):
        """
        回归取残差
        """
        # 求取相似性
        idx = data.columns.get_loc(y_col)
        cos = np.delete(all_cos[idx], idx)
        other_cols = data.columns.drop(y_col)
        cond1 = np.argsort(cos)[-10:]
        # cond2 = np.argwhere(all_cos >= 0.9)
        # X_cols = X.columns[np.intersect1d(cond1, cond2)]
        X_cols = other_cols[cond1]
        model = sm.OLS(data[y_col], data[X_cols]).fit()
        return model.resid.mean()*1e4

    def replicate_residual(self, start_time, end_time):
        """
        复制残差因子
        """
        if (start_time is None) or (end_time >= self.fund_rtn.index[-1]): return
        data = self.fund_rtn.loc[start_time:end_time].dropna(how='all').dropna(axis=1)
        if len(data.columns) < 7: return
        all_cos = cosine_similarity(data.T)
        res = data.swifter.progress_bar(False).apply(lambda x: self.get_resid(data, x.name, all_cos))
        res.name = end_time
        return res
    
    def generate_factor(self):
        # # 复制残差因子生成
        accelerator = Accelerator(core=4)
        calc_set = [(self.replicate_residual, self.args)]
        results = accelerator.execute(calc_set)
        factor_matrix = pd.DataFrame()
        wrong_info, wrong_tickers = [], []
        for k, res in results:
            if res is None: continue
            elif isinstance(res, str) or isinstance(res, Exception):  # 运行出错的报错信息
                wrong_info.append(res)
                wrong_tickers.append(k)
            else:
                factor_matrix = pd.concat([factor_matrix, res], axis=1)
        factor_matrix = factor_matrix.T.sort_index().astype('float')
        print(factor_matrix)
        print(wrong_tickers)
        print(wrong_info)
        return factor_matrix
        

class Similarity_Momentum:
    def __init__(self, lag):
        self.args = [(dl.tds.get_pre_trade_day(x, 120), x) for x in dl.trade_months]
        self.fund_rtn = dl.fund_rtn('daily', 60)
        self.lag = lag

    def col_similarity_momentum(self, data, col):
        """
        相似动量因子单列结果
        """
        rtn = data.iloc[-1].drop(col).values
        col_data = data[col].shift(-self.lag).dropna().values.reshape(1, -1)  # 1行*天数列
        other_data = data.drop(col, axis=1).iloc[:-self.lag,].T.values  # n行*天数列
        cosine = cosine_similarity(col_data, other_data)
        weight = cosine[0]/(cosine[0].sum())
        col_factor = (rtn*weight).sum()
        return col_factor

    def similarity_momentum(self, start_time, end_time):
        """
        相似动量因子生成
        """
        if start_time is None: return
        data = self.fund_rtn.loc[start_time:end_time].dropna(how='all').dropna(axis=1)
        if len(data.columns) < 7: return
        res = data.apply(lambda x: self.col_similarity_momentum(data, x.name))
        res.name = end_time
        return res
    
    def generate_factor(self):
        factor_matrix = pd.DataFrame()
        wrong_info, wrong_tickers = [], []
        with tqdm(total=len(self.args), desc="Progress") as pbar:
            for x, y in self.args:
                try:
                    fs = self.similarity_momentum(x, y)
                    if fs is None: continue
                    factor_matrix = pd.concat([factor_matrix, fs], axis=1)
                except Exception as e:
                    wrong_info.append(e)
                    wrong_tickers.append((x,y))
                finally:
                    pbar.update(1) 
        factor_matrix = factor_matrix.T.sort_index().astype('float')
        factor_matrix = factor_matrix.reindex(index=dl.trade_days)
        print(factor_matrix)
        print(wrong_tickers)
        print(wrong_info)
        return factor_matrix     


if __name__ == '__main__':
    start = '2014-01-01'  # 因子开始时间
    end = date.strftime(date.today(), '%Y-%m-%d')  # 因子结束时间
    dl = DataLoader(start, end)
    # 平均相似度因子
    avgs = Average_Similarity()
    avgs.update_clusters()
    avgFactor = avgs.generate_factor()
    # 复制残差因子
    rr = Replicate_Residual()
    # # rr.replicate_residual('2022-01-01', '2023-06-30')
    rrFactor = rr.generate_factor()
    # 相似动量因子
    sm = Similarity_Momentum(1)
    smFactor = sm.generate_factor()