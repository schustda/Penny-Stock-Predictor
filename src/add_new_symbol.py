import pandas as pd
from src.data.ihub_data import IhubData
from src.data.stock_data import StockData
from src.data.combine_data import CombineData

if __name__ == '__main__':

    symbol = input('What is the symbol?  ')
    url = input('What is the ihub message board url?  ')

    df = pd.read_json('data/stock_list.json')
    df[symbol] = pd.Series([symbol,url],index=['symbol','url'])
    df.to_json('data/stock_list.json')

    data = IhubData(df[symbol]['symbol'],df[symbol]['url'],verbose = 1)
    data.pull_posts()
    s = StockData(symbol)
    s.add_stock_data()
    cd = CombineData(symbol)
    cd.compile_data()
