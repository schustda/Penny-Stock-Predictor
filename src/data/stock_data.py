import pandas as pd
import datetime as dt

class StockData(object):

    def __init__(self,ticker_symbol):
        self.ticker_symbol = ticker_symbol

    def _add_zero_days(self, df):
        '''
        Parameters
        ----------
        df: pandas dataframe, full message board data

        Output
        ------
        df: pandas dataframe, original data with deleted posts added in

        Moderators of a message board forum may remove a post if it violates
        the ihub policy. While the content of these posts is unknown the actual
        post is important when suming the posts per a given day.
        '''

        # df = pd.read_csv('../data/raw_data/stock/cbyi.csv',index_col='Date').iloc[:-18]
        df.index = pd.to_datetime(df.index)

        start = df.index[0]
        # end = min([pd.Timestamp(dt.date.today()),df.index[-1]])
        end = pd.Timestamp(dt.date.today())
        all_days = pd.DatetimeIndex(start = start, end = end, freq = 'B')
        df = df.reindex(all_days)
        df.index.name = 'Date'
        df.fillna({'Volume': 0},inplace=True)
        return df.fillna(method = 'ffill')

    def add_stock_data(self):

        url = 'https://screener.fidelity.com/ftgw/etf/downloadCSV.jhtml?symbol='+str(self.ticker_symbol).upper()
        df = pd.read_csv(url,index_col = 'Date').iloc[:-18]
        self.df = self._add_zero_days(df)
        self.df.to_csv('data/raw_data/stock/'+self.ticker_symbol+'.csv')


if __name__ == '__main__':

    # stock_lst = ['cbyi',
    #     'mine',
    #     'xtrn',
    #     'dolv',
    #     'pgpm',
    #     'cnxs',
    #     'amlh',
    #     'exol',
    #     'coho',
    #     'uoip',
    #     'kget']
    #
    # for stock in stock_lst:
    #     data = StockData(stock)
    #     data.add_stock_data()

    cbyi = StockData('aagc')
    cbyi.add_stock_data()
    df = cbyi.df
