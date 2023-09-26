"""
本文复现 兴业证券《花开股市，相似几何系列二、三》
"""
__author__ = 'xhs@我是天空飘着的雾霾'
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


class WaveDetector:
    """
    波段划分类
    """
    def __init__(self) -> None:
        self.file_path = 'MarketIndex.h5'

    @staticmethod
    def EMA(N, close_price):
        dp = [0]
        for t in range(1, len(close_price)):
            dp_t = 2*close_price[t]/(N+1) + (N-1)*dp[t-1]/(N+1)
            dp.append(dp_t)
        return dp

    @staticmethod
    def DEA(dif):
        dp = [0]
        for t in range(1, len(dif)):
            dp_t = 2*dif[t]/10 + 8*dp[t-1]/10
            dp.append(dp_t)
        return dp

    @staticmethod
    def integral(val):
        # 同号累加，异号归0
        dp = [val[0]]
        for i in range(1, len(val)):
            if val[i]*dp[i-1] > 0:
                dp.append(dp[i-1]+val[i])
            else:
                dp.append(val[i])
        return dp
    
    def indicator_calc(self, bench_name, start_time, end_time):
        """计算波段划分所需要的指标"""
        bench_val = pd.read_hdf(self.file_path, key=bench_name)
        bench_val.dropna(inplace=True)
        # MACD
        close_price = bench_val.closeIndex.values
        bench_val['EMA12'] = WaveDetector.EMA(12, close_price)
        bench_val['EMA26'] = WaveDetector.EMA(26, close_price)
        bench_val['DIF'] = bench_val['EMA12']-bench_val['EMA26']
        bench_val['DEA'] = WaveDetector.DEA(bench_val['DIF'].values)
        # TR; ATR
        bench_val['TR'] = pd.concat([bench_val['highestIndex']-bench_val['lowestIndex'], bench_val['highestIndex']-bench_val['preCloseIndex'], bench_val['preCloseIndex']-bench_val['lowestIndex']], axis=1).max(axis=1)
        bench_val['ATR'] = bench_val['TR'].rolling(100).mean()
        # Integral
        val = (bench_val['DIF']-bench_val['DEA']).values
        bench_val['integral'] = WaveDetector.integral(val)
        bench_val.set_index('tradeDate', inplace=True)
        bench_val = bench_val.loc[start_time:end_time]
        # plt.figure()
        # ax = bench_val.closeIndex.plot(figsize=(10,5))
        # bench_val[['DIF', 'DEA']].plot(ax=ax, secondary_y=True)
        return bench_val
    
    def direc_calc(self, bench_val, delta):
        """
        初始方向计算
        1 if (integral >= delta) until integral < -delta
        -1 if (integral <= -delta) until integral > delta
        """
        bench = bench_val[['closeIndex', 'integral', 'ATR']].dropna()
        integrals = bench.integral.values
        atrs = bench.ATR.values
        # 初始值
        dir = [1] if integrals[0] >= delta * atrs[0] else [-1]
        for i in range(1, len(bench)): 
            inte = integrals[i]
            delt = delta * atrs[i]
            if (inte >= delt) or (dir[i-1] == 1 and inte >= -delt):
                dir.append(1)
            else: 
                dir.append(-1)
        bench['dir'] = dir
        return bench

    def excep_calc(self, bench_val, beta=0.0):
        """
        当下行中价格破高，或上行中价格破低时，异常状态出现，需要进行修正
        """
        idx_val = bench_val.closeIndex.values
        dir_val = bench_val.dir.values
        anomalys = [-1]
        high_point, low_point = idx_val[0], idx_val[0]  # 跟踪当前高点和低点
        last_low_point, last_high_point = idx_val[0], idx_val[0]  # 前一个高点和低点
        # print('t, direc, anomalys, idxv, last_low_point, last_high_point, low_point, high_point')
        for t in range(1, len(bench_val)):
            direc, idxv = dir_val[t], idx_val[t]
            # 异常判断
            if direc != dir_val[t-1]:
                anomalys.append(1)
            elif anomalys[t-1] == 1:  # 前一时刻非异常状态
                if ((direc == 1 and idxv < (1+beta)*last_low_point)
                    or (direc == -1 and idxv > (1-beta)*last_high_point)):  # 上行期破低，下行期破高
                    anomalys.append(-1)
                else:           
                    anomalys.append(1)
            else:  # 前一时刻异常状态，需要尝试退出异常
                if ((direc == 1 and (idxv > (1+beta)*last_high_point)) 
                    or (direc == -1 and (idxv < (1-beta)*last_low_point))):
                    anomalys.append(1)
                else:
                    anomalys.append(-1)

            if (direc*anomalys[t] != dir_val[t-1]*anomalys[t-1]):  # 上下行切换，更新极值点
                last_low_point, last_high_point = low_point, high_point
                low_point, high_point = idxv, idxv
            else:
                low_point, high_point = min(low_point, idxv), max(high_point, idxv)
                
            # print(bench_val.index[t], direc, anomalys[t], idxv, last_low_point, last_high_point, low_point, high_point)
        return anomalys   

    def extrPoints_detect(self, bench_val):
        """
        找到各区间的极值点
        """
        idx_val = bench_val.closeIndex.values
        dir_val = bench_val.dir.values
        extrme_idxs, extrme_point = [], 0
        i = 1
        while i < len(bench_val):
            extrme_val = idx_val[extrme_point]
            if dir_val[i]*dir_val[i-1] > 0:  # 同向
                if dir_val[i] * (idx_val[i] - extrme_val) >= 0:
                    extrme_point = i
            else:
                extrme_idxs.append(extrme_point)
                dir_val[extrme_point+1:i] = dir_val[i]
                i = extrme_point+1
            i += 1
        extrme_idxs.append(extrme_point)
        return extrme_idxs
    
    def wave_filter(self, bench_val, extrem_idxs, thresh, if_picture=True):
        # 波段筛选
        extrem_points = bench_val.iloc[extrem_idxs].drop_duplicates()
        dirs = extrem_points.dir
        prices = extrem_points.closeIndex
        min_price, max_price = prices[0], prices[0]
        pickups = [extrem_points.index[0]]
        for i in range(1, len(extrem_points)):
            date = extrem_points.index[i]
            price = prices[i]
            dir = dirs.loc[date]
            if dir==1 and (price-min_price)/min_price >= thresh:
                if dirs.loc[pickups[-1]] == -1:
                    max_price = price
                    pickups.append(date)
                else:
                    if max_price <= price:
                        max_price = price
                        pickups.pop()
                        pickups.append(date)
            elif dir==-1 and (max_price-price)/max_price >= thresh/(1+thresh):
                if dirs.loc[pickups[-1]] == 1:
                    min_price = price
                    pickups.append(date)
                else:
                    if min_price >= price:
                        min_price = price
                        pickups.pop()
                        pickups.append(date)
            # print(date, price, dir, pickups, max_price, min_price, (price-min_price)/min_price, (max_price-price)/max_price)
        if if_picture:
            plt.figure()
            ax = bench_val.closeIndex.plot(figsize=(14, 5))
            plt.plot([bench_val.index.get_loc(i) for i in pickups], bench_val.closeIndex.loc[pickups].values)  
            bench_val.dir.plot(ax=ax, secondary_y=True)
        return pickups

    def price_time_location(self, bench_val, extrem_idxs):
        last1 = bench_val.loc[extrem_idxs][['closeIndex']]
        last1['date1'] = last1.index
        last1 = last1.rename(columns={'closeIndex':'index1'}).dropna()

        last2 = bench_val.loc[extrem_idxs][['closeIndex']]
        last2['date2'] = last2.index
        last2 = last2.rename(columns={'closeIndex':'index2'}).shift(1).dropna()

        bench_val = pd.concat([bench_val, last1, last2], axis=1).ffill()
        bench_val.loc[extrem_idxs, ['date1','date2','index1','index2']] = np.nan
        bench_val = bench_val.ffill()
        # 时间效率
        time1 = pd.to_datetime(pd.Series(data=bench_val.index, index=bench_val.index))
        time2 = pd.to_datetime(bench_val.date1)
        time3 = pd.to_datetime(bench_val.date2)
        time_stamp1 = (time1-time2).apply(lambda x: x.days)
        time_stamp2 = (time2-time3).apply(lambda x: x.days)
        bench_val.loc[:, 'time_loc'] = time_stamp1/time_stamp2
        # 价格效率
        price1 = (bench_val.closeIndex-bench_val.index1).abs()
        price2 = (bench_val.index1-bench_val.index2).abs()
        bench_val.loc[:, 'price_loc'] = price1/price2
        # 未来涨跌概率
        bench_val.loc[:, 'raise_prob'] = bench_val.price_loc.rank(ascending=False)/len(bench_val)
        bench_val.loc[:, 'alter_prob'] = bench_val.price_loc.rank(ascending=True)/len(bench_val)
        bench_val.loc[:, 'alter_prob'] = (bench_val.dir*bench_val.alter_prob).apply(lambda x: 1+x if x < 0 else 1-x)
        return bench_val.dropna()
    
    def calc_prob(self, bench_val, date, price):
        if date in bench_val.index:
            raise_prob = bench_val.loc[date, 'raise_prob']
            alter_prob = bench_val.loc[date, 'alter_prob']
        else:
            cases_num = len(bench_val)
            last = bench_val.loc[bench_val.index[bench_val.index <= date]].iloc[-1]
            last_dir = last.loc['dir']
            # 时间效率
            date1, date2 = last.loc['date1'], last.loc['date2']
            date, date1, date2 = pd.to_datetime(date), pd.to_datetime(date1), pd.to_datetime(date2)
            time_loc = (date-date1).days/(date1-date2).days
            alter_prob = len(bench_val[time_loc > bench_val.time_loc])/cases_num
            if last_dir == 1: alter_prob = 1-alter_prob
            # 价格效率
            price1, price2 = last.loc['index1'], last.loc['index2']
            price_loc = abs(price-price1)/abs(price1-price2)
            raise_prob = 1-len(bench_val[price_loc > bench_val.price_loc])/cases_num
        return raise_prob, alter_prob
    
    def run(self, bench_name, delta, thresh, start_time, end_time, if_modify=True):
        """
        bench_name: 划分波段的标的名称
        delta: 边界参数控制
        start_time: 波段起始时间
        end_time: 波段结束时间
        if_modify: True, 是否做异常值调整
        """
        bench_val = self.indicator_calc(bench_name, start_time, end_time)  # 指标计算
        bench_val = self.direc_calc(bench_val, delta)  # 基于MACD和delta修正的初始方向计算
        if if_modify:
            bench_val['exceptions'] = self.excep_calc(bench_val)  # 异常点修正
            bench_val['dir'] *= bench_val['exceptions']  
        extrme_idxs = self.extrPoints_detect(bench_val)  # 获取波段划分的结果
        extrme_idxs = self.wave_filter(bench_val, extrme_idxs, thresh)
        plt.savefig('波段划分结果_%s.png'%bench_name, dpi=500)
        return bench_val, extrme_idxs

    def backtest(self, bench_val, hold_period, start_date, method):
        bench_val['thresh'] = ((bench_val.index1-bench_val.index2)/bench_val.index2).abs().rolling(750).mean()  # 动态单边趋势
        data = bench_val[['closeIndex', 'dir', 'thresh']]
        data.loc[:, 'rtn'] = data['closeIndex'].pct_change()
        start_date = bench_val.index[bench_val.index>=start_date][0]
        start_i = bench_val.index.get_loc(start_date)
        for i in range(start_i, len(data), hold_period):
            direction = data.iloc[i].loc['dir']
            price = data.iloc[i].loc['closeIndex']
            thresh = data.iloc[i].loc['thresh']
            if thresh > 0.5: thresh = 1-thresh
            date = data.index[i]
            date1 = (pd.to_datetime(date)+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            date7 = (pd.to_datetime(date)+datetime.timedelta(days=hold_period)).strftime('%Y-%m-%d')
            prob = self.calc_prob(bench_val, date7, price)
            if method == 'price': raise_prob = prob[0]
            elif method == 'time': raise_prob = prob[1]
            else: raise_prob = (prob[0]+prob[1])/2
            if (raise_prob <= 1-thresh) and (raise_prob >= 0.55):
                signal = 1
            elif (raise_prob <= 0.45) and (raise_prob >= thresh):
                signal = -1
            elif raise_prob<thresh or raise_prob>1-thresh:  # 单边趋势
                signal = direction
            else:  # 信号不明显
                signal = 0
            data.loc[date1:date7, 'signal'] = signal
        # 回测
        data = data.dropna()
        data['long-short'] = data.rtn*data.signal
        data['long-only'] = data.rtn*(data.signal.mask(data.signal<0, 0))
        plt.figure()
        (data[['rtn', 'long-short', 'long-only']]+1).cumprod().plot(figsize=(10, 5))
        plt.savefig('backtest.png', dpi=500)
        return data

if __name__ == '__main__':
    wd = WaveDetector()
    bench_val, extrem_idxs = wd.run('沪深300', 2, 0.05, '2020-01-03', '2023-08-31', True)
    bench_val = wd.price_time_location(bench_val, extrem_idxs[1:])
    _ = wd.backtest(bench_val, 5, '2010-01-03', 'price')