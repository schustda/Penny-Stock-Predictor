import requests
import pandas as pd
import numpy as np
import pathlib
from bs4 import BeautifulSoup
from time import time, sleep

class IhubData(object):

    def __init__(self, symbol, url, verbose = 0):
        self.symbol = symbol
        self.url = url
        self.num_pinned, self.num_posts = self._total_and_num_pinned()
        self.verbose = verbose

    def _total_and_num_pinned(self):
        '''
        Output
        ------
        num_pinned: int, shows how many posts are 'pinned' on the top of the
            board.
        num_posts: int, shows, to-date, how many messages have been posted on
            the specific board
        '''
        # Retrieve the first page on the board
        df, _ = self._get_page(self.url,most_recent=True,sort = False)

        # Number of pinned posts determined by the number of posts that are not
        # in 'numerical' order at the top of the page
        post_nums = df.index.tolist()
        for i in range(len(post_nums)):
            if post_nums[i] == post_nums[i+1]+1:
                return i, post_nums[i]

    def _clean_dataframe(self, df, sort):
        '''
        Parameters
        ----------
        df: pandas dataframe,
        sort: boolean, the message board posts are displayed in descending
            order. Sort=True sorts them

        Output
        ------
        df: pandas dataframe, message board table

        functions implemented: convert subject and username to text, strip line
        breaks, convert 'date' to datetime
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
        post_number: int, specific post number of page to be returned
        most_recent: boolean, returns the currently displayed page if True
        sort: boolean, as displayed on the webpage, the message board posts are
            displayed in descending order. Sort sorts them

        Output
        ------
        df: pandas dataframe, pulled from the webpage, parsed, and cleaned
        '''
        pinned = 0
        URL = "https://investorshub.advfn.com/"+self.url
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

        except Exception as e:
            print ('{0} ERROR ON PAGE: {1}'.format(e, str(post_number)))
            error_list.append(post_number)
            return pd.DataFrame(), error_list

    def _verbose(self, percent, original_time, time_elapsed):
        '''
        Method displays progress of retriving message board posts
        '''

        a = int(percent/2)
        b = 50-a
        if percent == 0:
            percent = 0.5
        min_rem = int(time_elapsed/percent*(100-percent)/60)
        print ('|{0}{1}| {2}% - {3} minute(s) remaining'.format(a*'=',b*'-',str(percent),str(min_rem)))

    def _add_deleted_posts(self, df):
        '''
        Parameters
        ----------
        df: pandas dataframe, full message board data

        Output
        ------
        df: pandas dataframe, original data with deleted posts added in

        Moderators of a message board forum may remove a post if it violates
        the ihub policy. While the content of these posts is unknown the actual
        post is important when suming the posts per a given day.
        '''

        #get missing post numbers
        deleted_post_set = set(range(1,df.index.max())).difference(set(df.index))

        #create df with deleted posts
        df_deleted = pd.DataFrame(np.nan, index=range(len(deleted_post_set)), columns=df.columns.tolist())
        df_deleted.index = deleted_post_set
        df_deleted.index.name = 'post_number'
        df_deleted.subject = '<DELETED>'
        df_deleted.username = '<DELETED>'

        #add to original dataframe
        df = pd.concat([df,df_deleted])

        #sort df
        df.sort_index(inplace=True)

        # The dates from the deleted posts will be interpreted as from the
        # day from the previous post
        df.date.fillna(method = 'ffill',inplace = True)

        # Some of the subjects are empty
        df.subject.fillna('None',inplace = True)

        return df

    def pull_posts(self):
        '''
        Output
        ------
        df: pandas dataframe
        '''

        t, original_time, posts_per_page = time(), time(), 50-self.num_pinned

        # Determines whether or not the file has already been created
        # (need to create or update??)
        path = pathlib.Path('data/raw_data/ihub/message_boards/'+self.symbol+'.csv').is_file()

        if not path:

            # if this is the first time this ticker symbol has been run
            print('pulling posts for ' + self.symbol)
            df, error_list = self._get_page(posts_per_page)
            start_page = posts_per_page * 2
            end_page = self.num_posts

        else:
            # if the file has already been created, import it to update and
            # transform it back to native format
            print('updating posts for ' + self.symbol)
            df = pd.read_csv('data/raw_data/ihub/message_boards/'+self.symbol+'.csv')
            df['date'] = pd.to_datetime(df['date']).dt.date

            start_page = df.post_number.max()+posts_per_page-1
            end_page = self.num_posts
            missing_posts = self.num_posts - df.post_number.max()

            df.columns = [0,1,2,3]
            df.index = df[0]
            df.drop(0,axis=1,inplace=True)

            error_list = []

            # Removing erros caused when there are fewer than 50 posts to update
            if start_page > self.num_posts:
                end_page +=1
                start_page = self.num_posts

        # iterate through all of the pages
        for post_num in range(start_page,end_page,posts_per_page):
            # pull from the specific page and add it to the existing dataframe
            page_df, error_list = self._get_page(post_num, error_list = error_list)
            df = pd.concat([df,page_df])

            if self.verbose:
                # Display update ever 60 seconds
                if time() > t + 60:
                    percent = int((post_num-start_page)/(end_page-start_page)*100)
                    time_elapsed = time() - original_time
                    self._verbose(percent, original_time, time_elapsed)
                    t = time()

        # retry pages that previously returned an error
        final_error_list = []
        shallow_error_list = list(error_list)
        for post_num in shallow_error_list:
            page_df, final_error_list = self._get_page(post_num, error_list = final_error_list)
            df = pd.concat([df,page_df])


        # clean and save the new dataframe
        df.sort_index(inplace=True)
        df.columns = ['subject','username','date']
        df.index.name = 'post_number'
        df.drop_duplicates(inplace = True)

        df = self._add_deleted_posts(df)

        # display whether or not the file was created or updated
        if not path:
            print ('complete, ' + str(self.num_posts) + ' posts created')
        else:
            print ('complete, ' + str(missing_posts) + ' post(s) added')

        if len(final_error_list) != 0:
            print('Errors encountered on the following pages: {0}'.format(final_error_list))
        else:
            df.to_csv('data/raw_data/ihub/message_boards/'+self.symbol+'.csv')



if __name__ == '__main__':

    # kget = 'CaliPharms-Inc-KGET-10313'
    # cbyi = 'Cal-Bay-International-Inc-CBYI-5520'
    # data = IhubData('cbyi',cbyi,verbose = 1)
    # data.pull_posts()

    df = pd.read_json('data/stock_list.json')
    data = IhubData(df['sanp']['symbol'],df['sanp']['url'],verbose = 1)
    data.pull_posts()


    # cbyi = 'Cal-Bay-International-Inc-CBYI-5520'
    # mine = 'Minerco-Inc-MINE-17939'
    # xtrn = 'Las-Vegas-Railway-Express-XTRN-16650'
    # dolv = 'Dolat-Ventures-Inc-DOLV-16401'
    # pgpm = 'Pilgrim-Petroleum-Corp-PGPM-5655'
    # cnxs = 'Connexus-Corp-CNXS-17863'
    # amlh = 'American-Leisure-Holdings-Inc-AMLH-29447'
    # exol = 'EXOlifestyle-Inc-EXOL-11015'
    # coho = 'Crednology-Holding-Corp-COHO-4899'
    # uoip = 'UnifiedOnline-Inc-UOIP-5196'
    # kget = 'CaliPharms-Inc-KGET-10313'

    # stock_lst = [cbyi,
    #     mine,
    #     xtrn,
    #     dolv,
    #     pgpm,
    #     cnxs,
    #     amlh,
    #     exol,
    #     coho,
    #     uoip,
    #     kget]
    #
    # for stock in stock_lst:
    #     data = IhubData(stock,verbose = 1)
    #     data.pull_posts()
