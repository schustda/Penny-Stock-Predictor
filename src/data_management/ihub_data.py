import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import time

class StockData(object):

    def __init__(self,stock_identifier,existing = False, verbose = 0):
        self.stock_identifier = stock_identifier
        self.num_pinned, self.num_posts = self._total_and_num_pinned()
        self.verbose = verbose
        self.ticker_symbol = self.stock_identifier.split('-')[-2].lower()

    def _total_and_num_pinned(self):
        '''
        Parameters
        ----------
        None

        Output
        ------
        num_pinned: int, shows how many 'pinned' posts there are on the specific
            message board
        num_posts: int, shows, to-date, how many messages have been posted on
            the specific board
        '''
        df, _ = self._get_page(self.stock_identifier,most_recent=True,sort = False)
        post_nums = df.index.tolist()
        for i in range(len(post_nums)):
            if post_nums[i] == post_nums[i+1]+1:
                return i, post_nums[i]

    def _clean_dataframe(self, df,sort):
        '''
        Parameters
        ----------
        df: pandas dataframe,
        sort: boolean, as displayed on the webpage, the message board posts are
            displayed in descending order. Sort sorts them

        Output
        ------
        df: pandas dataframe, functions implemented: convert subject and
            username to text, strip line breaks, convert 'date' to datetime
        '''
        df = df.applymap(lambda x: x.text)
        df[[0,1,2]] = df[[0,1,2]].applymap(lambda x: x.strip('-#\n\r').replace('\n', "").replace('\r',''))
        df[3] = pd.to_datetime(df[3]).dt.date
        df[0] = df[0].astype(int)
        if sort:
            df.sort_values([0],inplace = True)
        df.set_index(0,inplace = True)
        return df

    def _get_page(self, post_number = 1, most_recent = False, sort = True, error_list = []):
        '''
        Parameters
        ----------
        post_number: int, goes to a specific page if identified
        most_recent: boolean, returns the currently displayed page if True
        sort: boolean, as displayed on the webpage, the message board posts are
            displayed in descending order. Sort sorts them

        Output
        ------
        df: pandas dataframe, pulled from the internet, parsed, and cleaned
        '''
        pinned = 0
        URL = "https://investorshub.advfn.com/"+self.stock_identifier
        if not most_recent:
            URL += "/?NextStart="+str(post_number)
            posts_per_page = 51-self.num_pinned
            pinned = self.num_pinned

        try:
            content = requests.get(URL).content
            soup = BeautifulSoup(content, "lxml")
            rows = list(soup.find('table', id="ctl00_CP1_gv"))
            table_lst = []
            for row in rows[(2+pinned):-2]:
                cell_lst = [cell for cell in list(row)[1:5]]
                table_lst.append(cell_lst)
            df = pd.DataFrame(table_lst)
            return self._clean_dataframe(df,sort), error_list

        except:
            pass
            print ('ERROR ON PAGE' + post_number)
            error_list.append()
            return pd.DataFrame(), error_list

    def _verbose(self, percent, original_time, time_elapsed):
        a = int(percent/2)
        b = int((100-percent)/2 + 1)
        min_remaining = int(time_elapsed/percent*(100-percent)/60)
        print ('|'+ a*'=' + b*'-' + '|  ' + self.ticker_symbol + ' - ' + str(percent) + '% - ' + str(min_remaining) + ' minutes remaining')

    def update_csv(self):
        '''
        Pulls the existing message board csv and updates it with any messages
        that have been posted that are not currently in the file
        '''

    def create_csv(self):
        '''
        Creates the csv file for the specific ticker symbol within data/ihub/message_board
        '''
        print('pulling posts for ' + self.ticker_symbol)
        first, t, original_time, posts_per_page = True, time(), time(), 51-self.num_pinned
        for post_num in range(posts_per_page-1,self.num_posts,posts_per_page):

            if first:
                df, error_list = self._get_page(post_num)
                first = False
            else:
                page_df, error_list = self._get_page(post_num, error_list = error_list)
                df = pd.concat([df,page_df])

            if self.verbose:
                if time() > t + 60:
                    percent = int(post_num*100/self.num_posts)
                    time_elapsed = time() - original_time
                    self._verbose(percent, original_time, time_elapsed)
                    t = time()

        final_error_list = []
        shallow_error_list = error_list.copy()
        for post_num in shallow_error_list:
            page_df, final_error_list = self._get_page(post_num, error_list = final_error_list)
            df = pd.concat([df,page_df])

        df.sort_index(inplace=True)
        cols = ['post_number','subject','username','date']
        df.columns = cols[1:4]
        df.index.name = cols[0]
        df.drop_duplicates(inplace = True)
        df.to_csv('data/raw_data/ihub/message_boards/'+self.ticker_symbol+'.csv')
        print (self.ticker_symbol + ' complete! \n')
        if len(final_error_list) != 0:
            print('Errors encountered on the following pages:' + final_error_list)

if __name__ == '__main__':
    cbyi = 'Cal-Bay-International-Inc-CBYI-5520'
    mine = 'Minerco-Inc-MINE-17939'
    xtrn = 'Las-Vegas-Railway-Express-XTRN-16650'
    dolv = 'Dolat-Ventures-Inc-DOLV-16401'
    pgpm = 'Pilgrim-Petroleum-Corp-PGPM-5655'
    cnxs = 'Connexus-Corp-CNXS-17863'
    amlh = 'American-Leisure-Holdings-Inc-AMLH-29447'

    stock_lst = [dolv,pgpm,cnxs,amlh]

    for stock in stock_lst:
        data = StockData(stock,verbose = 1)
        data.create_csv()
