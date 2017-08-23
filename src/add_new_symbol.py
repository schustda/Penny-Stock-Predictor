import pandas as pd


if __name__ == '__main__':

    symbol = input('What is the symbol?  ')
    url = input('What is the ihub message board url?  ')

    df = pd.read_json('data/stock_list.json')
    df[symbol] = pd.Series([symbol,url],index=['symbol','url'])
    df.to_json('data/stock_list.json')
