from src.data.ihub_data import IhubData
from src.data.stock_data import StockData
from src.data.combine_data import CombineData
import pandas as pd
import time

if __name__ == '__main__':

    while True:


        df = pd.read_json('data/stock_list.json')
        for symbol in df.columns.tolist():
            print (df[symbol]['url'])
            data = IhubData(df[symbol]['url'],verbose = 1)

            data.pull_posts()
            s = StockData(symbol)
            s.add_stock_data()
            combined_data = CombineData(symbol)
            combined_data.compile_data()

        #update four times a day
        time.sleep(60*60*24 /4)
