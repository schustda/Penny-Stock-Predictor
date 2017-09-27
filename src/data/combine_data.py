import pandas as pd
import numpy as np
import datetime
from math import isnan
from src.data.define_target import DefineTarget
import pathlib


class CombineData(object):

    def __init__(self,symbol):
        self.symbol = symbol
        self.ihub_posts = pd.read_csv('data/raw_data/ihub/message_boards/'+self.symbol+'.csv')
        self.stock_info = pd.read_csv('data/raw_data/stock/'+self.symbol+'.csv',index_col=0)
        # self.existing = pathlib.Path('data/data/'+self.symbol+'.csv').is_file()
        # if self.existing:
        #     self.data = pd.read_csv('data/data/'+self.symbol+'.csv')

    def _calculate_ohlc(self,df):
        '''
        Parameters
        ----------
        df: pandas dataframe, stock price info

        Output
        ------
        df: pandas dataframe, ohlc and dollar volume of stock

        OHLC Average (open-high-low-close) is a commonly used indicator to
            determine a stock price on a given day
        Dollar Volume is a better predictor than volume alone. i.e. a stock trading
            1M volume at a $0.10 price should be treated differently than a stock
            trading 1M volume at a $10 price
        '''

        #fillna's
        df.fillna(0.00,inplace = True)

        #create ohlc column
        df['ohlc'] = (df['Open']+df['High']+df['Low']+df['Close'])/4

        #create dollar volumne column
        df['dollar_volume'] = df['ohlc'] * df['Volume']
        df.drop(['Open','Close','High','Low','Date','Volume'],axis=1,inplace=True,errors='ignore')
        df.index = pd.to_datetime(df.index)
        self.stock_info = df

    def _remove_weekends_and_holidays(self, df):
        '''
        The stock market is not open on weekends and federal holidays. These
            days will not be included
        '''
        weekend_dates = set(df[df.weekday > 4].index)
        holiday_dates = set(pd.to_datetime(pd.read_csv('data/stock_market_holidays.csv').date))
        return df.drop(holiday_dates.union(weekend_dates),errors='ignore')

    def compile_data(self):

        print ('compiling data for '+self.symbol+'...')

        self._calculate_ohlc(self.stock_info)
        grouped_posts = self.ihub_posts.groupby('date').count().post_number
        grouped_posts.index = pd.to_datetime(grouped_posts.index)

        start_date = max([min(grouped_posts.index.tolist()),min(self.stock_info.index.tolist())])
        end_date = max([max(grouped_posts.index.tolist()),max(self.stock_info.index.tolist())])
        self.df_date = pd.DataFrame(pd.date_range(start_date,end_date)).set_index(0)

        self.combined_data = self.df_date.join(grouped_posts).join(self.stock_info)
        self.combined_data['weekday'] = self.combined_data.index.weekday
        self.combined_data = self._remove_weekends_and_holidays(self.combined_data)
        self.combined_data.index.name = 'date'
        self.combined_data.fillna(0,inplace=True)

        t = DefineTarget(self.combined_data)
        self.combined_data['target'] = t.target
        self.combined_data['symbol'] = self.symbol
        self.combined_data.to_csv('data/data/{0}.csv'.format(self.symbol))

        print ('Complete! \n')

if __name__ == '__main__':
    cbyi = CombineData('cbyi')
    cbyi.compile_data()
    df1 = cbyi.stock_info
    df2 = cbyi.ihub_posts
    df3 = cbyi.df_date
    df4 = cbyi.combined_data
