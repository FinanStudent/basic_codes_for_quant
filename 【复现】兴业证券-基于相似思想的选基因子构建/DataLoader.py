"""
这个文件实现了数据加载及部分数据处理工作
"""
__author__ = 'xhs@我是天空飘着的雾霾'

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TradeDateSupportor:
    def __init__(self, start, end):
        trade_calendar = pd.read_csv('data/calendarDate.csv').set_index(['calendarDate'])  # 1990-2023的交易日数据
        self.trade_calendar = trade_calendar.loc[start:end]
        self.trade_days = self.trade_date()
        self.trade_months = self.last_date_of_month()
        self.trade_seasons = self.last_date_of_season()

    def trade_date(self):
        # 获取交易日
        return self.trade_calendar.index.values

    def last_date_of_month(self):
        # 获取月末交易日
        self.trade_calendar.loc[:, 'last_date'] = [x[0:7] for x in self.trade_days]
        trade_dates = self.trade_calendar.drop_duplicates(subset=['last_date'], keep='last').index
        trade_dates = trade_dates.values
        return trade_dates

    def last_date_of_season(self):
        # 获取季末交易日
        lastday_month = self.last_date_of_month()
        vfunc = np.vectorize(lambda x: x[5:7] in ['01', '04', '07', '10'])
        lastday_season = lastday_month[vfunc(lastday_month)]
        return lastday_season

    def get_pre_trade_day(self, date, lag=0):
        if date not in self.trade_days:
            date = self.trade_days[self.trade_days < date][-1]
        idx = np.argwhere(self.trade_days == date)[0][0]
        find_idx = idx-lag
        if find_idx <= 0:
            return None
        else:
            return self.trade_days[find_idx]


class DataLoader:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end
        # 日期获取
        tds = TradeDateSupportor(start, end)
        self.tds = tds
        self.trade_days = tds.trade_days
        self.trade_months = tds.trade_months
        self.trade_seasons = tds.trade_seasons
    
    def net_asset_value(self):
        net_value = pd.read_csv('data/dailyNetValue.csv', index_col=0).join(pd.DataFrame(index=self.trade_days), how='outer')
        net_value = net_value.ffill().loc[self.trade_days]
        return net_value
    
    def fund_rtn(self, freq, period):
        net_value = self.net_asset_value()
        if freq == 'daily':
            rtn = net_value.pct_change(period, fill_method=None)
        elif freq == 'monthly':
            rtn = net_value.loc[self.trade_months].pct_change(period, fill_method=None)
        elif freq == 'seasonally':
            rtn = net_value.loc[self.trade_seasons].pct_change(period, fill_method=None)
        return rtn
