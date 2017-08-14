import pandas as pd
import numpy as np
from math import isnan

class CombineData(object):

    def __init__(self,ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ihub_posts = pd.read_csv('data/raw_data/ihub/message_boards/'+self.ticker_symbol+'.csv')
        self.stock_info = pd.read_csv('data/raw_data/stock/'+self.ticker_symbol+'.csv',index_col=0)

    def _add_deleted_fill_na(self):
        '''
        Parameters
        ----------
        None

        Output
        ------
        ihub_posts: pandas dataframe, original data with deleted posts added in
        '''

        #get missing post numbers
        deleted_post_set = set(range(1,self.ihub_posts.post_number.max())).difference(set(self.ihub_posts.post_number))

        #create df with deleted posts
        df_deleted = pd.DataFrame(np.nan, index=range(len(deleted_post_set)), columns=self.ihub_posts.columns.tolist())
        df_deleted.post_number = deleted_post_set
        df_deleted.subject = '<DELETED>'
        df_deleted.username = '<DELETED>'

        #add to original dataframe
        self.ihub_posts = pd.concat([self.ihub_posts,df_deleted])

        #sort df
        self.ihub_posts.sort_values(['post_number'],axis=0,inplace=True)

        #fillna
        self.ihub_posts.date.fillna(method = 'ffill',inplace = True)
        self.ihub_posts.subject.fillna('None',inplace = True)

    def _calculate_ohlc(self):

        #fillna's
        self.stock_info.fillna(0.00,inplace = True)

        #create ohlc column
        self.stock_info['ohlc'] = (self.stock_info['Open']+self.stock_info['High']+self.stock_info['Low']+self.stock_info['Close'])/4

        #create dollar volumne column
        self.stock_info['dollar_volume'] = self.stock_info['ohlc'] * self.stock_info['Volume']

        #convert to date
        self.stock_info = self.stock_info[0:-19]
        self.stock_info.Date = self.stock_info.Date.map(lambda x: x[0:10])

        #drop columns
        self.stock_info.index = self.stock_info.Date
        self.stock_info.drop(['Open','Close','High','Low','Date','Volume'],axis=1,inplace=True)

        self.stock_info.index = pd.to_datetime(self.stock_info.index)

    def fill_nulls(self, df):

        df.post_number.fillna(0,inplace = True)


        weekends = set([5,6])

        for day in df.index:
            if all([isnan(df.loc[day]['ohlc']),df.loc[day]['weekday'] not in weekends, day not in holiday_set]):
                df = df.set_value(day,'ohlc',0)
                df = df.set_value(day,'dollar_volume',0)

        return df

    def _remove_weekends_and_holidays(self, df):
        weekends = set(df[df.weekday > 4].index)
        holidays = set(pd.to_datetime(pd.read_csv('data/stock_market_holidays.csv').date))
        return df.drop(holidays.union(weekends),errors='ignore')

    def compile_data(self):
        print ('compiling data for '+self.ticker_symbol+'...')

        self._add_deleted_fill_na()
        self._calculate_ohlc()
        self.ihub_posts = self.ihub_posts.groupby('date').count().post_number
        self.ihub_posts.index = pd.to_datetime(self.ihub_posts.index)

        start_date = max([min(self.ihub_posts.index.tolist()),min(self.stock_info.index.tolist())])
        end_date = min([max(self.ihub_posts.index.tolist()),max(self.stock_info.index.tolist())])
        self.df_date = pd.DataFrame(pd.date_range(start_date,end_date)).set_index(0)

        self.combined_data = self.df_date.join(self.ihub_posts).join(self.stock_info)
        self.combined_data['weekday'] = self.combined_data.index.weekday
        self.combined_data = self._remove_weekends_and_holidays(self.combined_data)
        self.combined_data.index.name = 'date'
        self.combined_data.to_csv('data/data/'+self.ticker_symbol+'.csv')

        print ('Complete! \n')

if __name__ == '__main__':
    cbyi = CombineData('cbyi')
    cbyi.compile_data()
    df1 = cbyi.stock_info
    df2 = cbyi.ihub_posts
    df3 = cbyi.df_date
    df4 = cbyi.combined_data


        #fill nulls with date above

    #     new_df = df.copy()
    #     total_num_posts = df.post_number.max()
    #     post_set = set(df.post_number)
    #
    #     for i in range(total_num_posts):
    #         if i not in post_set:
    #             print (i)
    #
    #
    #     post_set = set(df.post_number)
    #     all_posts = set(range(df.post_number.max()))
    #
    # df = pd.read_csv('data/ihub/Cal-Bay-International-Inc-CBYI-5520.csv')
    # num_not_deleted = set(df.post_number)
    # num_posts = set(range(1,df.post_number.max()))
    # deleted_post_numbers = num_posts.difference(num_not_deleted)
    # ddf = pd.DataFrame(np.nan, index=range(len(deleted_post_numbers)), columns=df.columns.tolist())
    # ddf.post_number = deleted_post_numbers
    # ddf.subject = '<DELETED>'
    # ddf.username = '<DELETED>'
    # df2 = pd.concat([df,ddf])
    # df2.sort_values('post_number',inplace = True)
    # # df.drop_duplicates(inplace = True)
    # # df.index = df.post_number
    # # df.drop('post_number',axis = 1,inplace = True)

#114297
