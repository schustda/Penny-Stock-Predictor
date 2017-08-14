import pandas as pd


class DefineTarget(object):

    def __init__(self,ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.data = self._import_data()
        self.stock_size = self._big_or_small()
        self.target = self.add_target()


    def _big_or_small(self):
        sub_penny_count = self.data.where(self.data.ohlc < 0.01).count().post_number
        sub_penny_percentage = sub_penny_count/self.data.shape[0]

        if sub_penny_count > 0.25:
            return 'sub_penny'
        else:
            return 'small_cap'

    def _import_data(self):
        df = pd.read_csv('data/data/'+self.ticker_symbol+'.csv')
        df.dropna(inplace=True)
        df.set_index('date',inplace=True)
        return df

    def add_target(self):
        target = []
        print('defining target...')
        for i in range(self.data.shape[0]-11):
            ohlc = self.data.iloc[i].ohlc
            wk_avg1 = self.data.iloc[i+1:i+6].ohlc.mean()
            wk_avg2 = self.data.iloc[i+1:i+11].ohlc.mean()
            if wk_avg1 > ohlc * 3 and wk_avg2 > ohlc * 2:
                target.append(1)
                print ('got one!')
            else:
                target.append(0)
        return target
        self.data['target'] = target



if __name__ == '__main__':
    target = DefineTarget('cbyi')
    df = target.data



#
# for day in df.index:
#     two_wk_avg =
#
#
# big cap or small cap???
