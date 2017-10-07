from src.data.ihub_data import IhubData
from src.data.stock_data import StockData
from src.data.combine_data import CombineData
import subprocess
import pandas as pd
import time

if __name__ == '__main__':


    # df = pd.read_json('data/stock_list.json')
    # # updates (in order) message board posts, stock price, compiled data
    # for symbol in df.columns.tolist():
    #     data = IhubData(df[symbol]['symbol'],df[symbol]['url'],verbose = 1)
    #     data.pull_posts()
    #     s = StockData(symbol)
    #     s.add_stock_data()
    #     combined_data = CombineData(symbol)
    #     combined_data.compile_data()


    while True:
        df = pd.read_json('data/stock_list.json')
        first = True
        # updates (in order) message board posts, stock price, compiled data
        for symbol in df.columns.tolist():
            data = IhubData(df[symbol]['symbol'],df[symbol]['url'],verbose = 1)
            data.pull_posts()
            s = StockData(symbol)
            s.add_stock_data()
            cd = CombineData(symbol)
            cd.compile_data()
            if first == True:
                all_data = cd.combined_data
                first = False
            else:
                all_data = pd.concat([all_data,cd.combined_data])
        all_data.dropna(inplace=True)
        all_data.to_csv('data/data/_all.csv')
        rc = subprocess.call('src/scripts/update_data.sh',shell=True)

        # update six times a day
        time.sleep(60*60*24/6)
