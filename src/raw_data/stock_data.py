import pandas as pd

class StockData(object):

    def __init__(self,ticker_symbol):
        self.ticker_symbol = ticker_symbol

    def add_stock_data(self):

        url = 'https://screener.fidelity.com/ftgw/etf/downloadCSV.jhtml?symbol='+str(self.ticker_symbol).upper()
        df = pd.read_csv(url)
        df.to_csv('data/raw_data/stock/'+self.ticker_symbol+'.csv')


if __name__ == '__main__':

    stock_lst = ['cbyi',
        'mine',
        'xtrn',
        'dolv',
        'pgpm',
        'cnxs',
        'amlh',
        'exol',
        'coho',
        'uoip',
        'kget']

    for stock in stock_lst:
        data = StockData(stock)
        data.add_stock_data()
