import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from matplotlib.dates import date2num

class PlotData(object):

    def __init__(self,ticker_symbol,start_date,end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv('data/data/'+self.ticker_symbol+'.csv')

    # def _ihub(self, df):
    #     df = pd.read_csv('data/ihub/'+self.stock_identifier+'.csv').groupby('date').count().post_number
    #     df.index = pd.to_datetime(df.index)
    #     return self._standardize(df)
    #
    # def _fidelity(self, df):
    #     df = pd.read_csv('data/fidelity/'+self.stock_identifier+'.csv')
    #     df.index = df.Date
    #     df.index = pd.to_datetime(df.index)
    #     df = df[start_date:end_date]
    #     df_volume = df.Volume
    #     df_price = df.Open
    #     return self._standardize(df_price), self._standardize(df_volume)

    def _standardize(self):
        cols_to_standardize = ['post_number', 'ohlc', 'dollar_volume', 'weekday']
        for col in cols_to_standardize:
            self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())

    def _update(self,var):
        plt.subplot(111)
        plt.ylim([smin.val,smax.val])

    def plot(self):
        self.data.index = pd.to_datetime(self.data.date)
        self.data = self.data[self.start_date:self.end_date]
        self._standardize()
        self.data.interpolate(inplace=True)

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # x = pd.to_datetime(self.data.date).dt.to_pydatetime()
        x = self.data.index

        plt.style.use('ggplot')
        plt.bar(x,self.data.post_number,label = 'ihub posts',alpha = 0.5)
        plt.bar(x,self.data.dollar_volume,label = 'volume',alpha = 0.5)
        plt.plot(x,self.data.ohlc,'r', label = 'stock price')
        # plt.plot(x,self.data.dollar_volume,'k', label = 'volume')

        buy_x = self.data.index[self.data.target == 1]
        buy_y = self.data.ohlc[self.data.target == 1]
        plt.scatter(buy_x,buy_y, s=100,label = 'buy/sell')


        x_min_index = 0
        x_max_index = len(x)-1

        x_min = x[x_min_index]
        x_max = x[x_max_index]

        # timedelta
        x_dt = x_max - x_min

        # plt.axis(x_min, x_max, y_min, y_max)
        y_min = plt.axis()[2]
        y_max = plt.axis()[3]

        plt.axis([x_min, x_max, y_min, y_max])
        plt.legend()

        axcolor = 'lightgoldenrodyellow'
        axpos = plt.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)

        slider_max = len(x) - x_max_index - 1

        # Slider(axes, name, min, max)
        spos = Slider(axpos, 'Pos', matplotlib.dates.date2num(x_min), matplotlib.dates.date2num(x[slider_max]))

        # pretty date names
        plt.gcf().autofmt_xdate()

        def update(val):
            pos = spos.val
            xmin_time = matplotlib.dates.num2date(pos)
            xmax_time = matplotlib.dates.num2date(pos) + x_dt
            # print "x_min: %s, x_max: %s" % (xmin_time.strftime("%H:%M:%S.%f"), xmax_time.strftime("%H:%M:%S.%f"))

            ########################################################
            # RETURNS THE SAME RESULT:

            # xmin_time is datetime.datetime
            # print type(xmin_time)
            # ax.axis([xmin_time, xmax_time, y_min, y_max])

            # xmin_time is numpy.float64
            xmin_time = pos
            print (type(xmin_time))
            ax.axis([xmin_time, xmax_time, y_min, y_max])
            ########################################################
            fig.canvas.draw_idle()

        spos.on_changed(update)

        plt.show()


        # plt.close('all')
        # plt.style.use('seaborn')
        # plt.bar(self.ihub.index,self.ihub.values, label = 'ihub posts')
        # plt.plot(self.stock_price.index,self.stock_price.values,'r', label = 'stock price')
        # plt.plot(self.stock_volume.index,self.stock_volume.values,'k', label = 'volume')
        # plt.legend()
        # plt.show()

if __name__ == '__main__':
    plt.close('all')
    start_year = '2011'
    start_month = '02'
    start_day = '01'

    end_year = '2011'
    end_month = '05'
    end_day = '01'
    start_date = (start_year+start_month+start_day)
    end_date = (end_year+end_month+end_day)
    ticker_symbol = 'cbyi'

    stock = PlotData(ticker_symbol,start_date,end_date)
    df = stock.data
    stock.plot()
