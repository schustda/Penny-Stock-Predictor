import pandas as pd


class DefineTarget(object):

    def __init__(self,data):
        self.data = data
        # self.ticker_symbol = ticker_symbol
        # self.data = self._import_data()
        self.stock_size = self._big_or_small()
        self.target = self.add_target()


    def _big_or_small(self):
        sub_penny_count = self.data.where(self.data.ohlc < 0.01).count().post_number
        sub_penny_percentage = sub_penny_count/self.data.shape[0]

        if sub_penny_count > 0.25:
            return 'sub_penny'
        else:
            return 'small_cap'

    # def _import_data(self):
    #     df = pd.read_csv('data/data/'+self.ticker_symbol+'.csv')
    #     df.dropna(inplace=True)
    #     df.set_index('date',inplace=True)
    #     return df



    def add_target(self):
        target = []
        print('generating target...')
        for i in range(self.data.shape[0]-11):
            ohlc = self.data.iloc[i].ohlc
            wk_avg1 = self.data.iloc[i+1:i+6].ohlc.mean()
            wk_avg2 = self.data.iloc[i+1:i+11].ohlc.mean()
            wk_avg_vol = self.data.iloc[i-1:i+11].dollar_volume.mean()

            a1 = wk_avg1 > ohlc * 2
            a2 = wk_avg2 > ohlc * 1.5
            a3 = ohlc != 0.0
            a4 = wk_avg2 > 0.00015
            a5 = i > 60
            a6 = wk_avg_vol > 500
            buy = all([a1,a2,a3,a4,a5,a6])

            if buy :
                target.append(1)
            else:
                target.append(0)
        na = [None]*11
        target.extend(na)
        self.data['target'] = target
        return target


if __name__ == '__main__':
    target = DefineTarget('cbyi')
    df = target.data
    t = target.target
#
# for day in df.index:
#     two_wk_avg =
#
#
# big cap or small cap???
