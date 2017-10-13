import pandas as pd
import datetime as dt

class StockData(object):

    def __init__(self,ticker_symbol):
        self.ticker_symbol = ticker_symbol

    def _add_zero_days(self, df):
        '''
        Parameters
        ----------
        df: pandas dataframe, raw stock price history

        Output
        ------
        df: pandas dataframe, original data with 0 volume days added in

        Some tickers will experience no volume in a given day. When importing,
        these days are not included at all. This function adds in the missing
        days with zero volume.
        '''

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)

        # Start is the first day that stock price is recorded
        start = df.index[0]

        # If market is closed, last stock day is current day. Otherwise
        # it is previous day
        end = max([pd.Timestamp(dt.date.today()-dt.timedelta(1)),df.index[-1]])

        # Create datetime index with all business days (no weekends).
        # Holidays are not excluded, but are removed in CombineData
        all_days = pd.DatetimeIndex(start = start, end = end, freq = 'B')

        # Reset the index with the full set, and fill nulls (days that are
        # empty are days with 0 stock volume)
        df = df.reindex(all_days)
        df.index.name = 'Date'
        df.fillna({'Volume': 0},inplace=True)
        return df.fillna(method = 'ffill')

    def add_stock_data(self):

        try:
            url = 'https://screener.fidelity.com/ftgw/etf/downloadCSV.jhtml?symbol='+str(self.ticker_symbol).upper()
            df = pd.read_csv(url,index_col = 'Date').iloc[:-18]
            self.df = self._add_zero_days(df)
            self.df.to_csv('data/raw_data/stock/'+self.ticker_symbol+'.csv')
        except:
            pass

if __name__ == '__main__':

    cbyi = StockData('aagc')
    cbyi.add_stock_data()
    df = cbyi.df
