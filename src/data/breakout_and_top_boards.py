import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import subprocess

class TopAndBreakoutBoards(object):

    def __init__(self):
        self.bb_url = 'http://investorshub.advfn.com/boards/breakoutboards.aspx'
        self.bb_r = 6
        self.mr_url = 'http://investorshub.advfn.com/boards/most_read.aspx'
        self.mr_r = 5

    def _get_page(self,url,r):
        '''
        Parameters
        ----------
        url: string, webpage to pull from
        r: int, depending on the webpage, pulls certain columns from the table

        Output
        ------
        df: pandas dataframe, pulled from the webpage, parsed, and cleaned

        Used to extract the table from the most_read (mr) or breakout_boards (bb)
            page from iHub's website.
        '''

        content = requests.get(url).content
        soup = BeautifulSoup(content, "lxml")
        rows = list(soup.find('table', id="ctl00_CP1_gv"))
        table_lst = []
        for row in rows[2:-1]:
            cell_lst = [cell for cell in list(row)[r:r+1]]
            table_lst.append(cell_lst)
        df = pd.DataFrame(table_lst).applymap(lambda x: x.text)
        df = df.applymap(lambda x: x.strip('-#\n\r').replace('\n', "").replace('\r','').lower())
        df.columns = [pd.Timestamp("today").strftime("%m-%d-%Y")]
        return df.transpose()

    def update_page_daily(self):
        '''
        Function to develop a history of which stocks have been on the most read
        and breakout boards. First pulls the current bb and mr tables, then updates
        the csv file. Lastly pushes those updated csvs to github.

        Intended to be running on a cloud server.
        '''

        while True:
            bb_old = pd.read_csv('data/raw_data/ihub/breakout_boards.csv',index_col=0)
            bb_current = self._get_page(self.bb_url,self.bb_r)
            bb_old.columns = bb_current.columns
            bb_new = pd.concat([bb_old,bb_current],axis=0)
            bb_new.to_csv('data/raw_data/ihub/breakout_boards.csv')

            mr_old = pd.read_csv('data/raw_data/ihub/most_read.csv',index_col=0)
            mr_current = self._get_page(self.mr_url,self.mr_r)
            mr_old.columns = mr_current.columns
            mr_new = pd.concat([mr_old,mr_current])
            mr_new.to_csv('data/raw_data/ihub/most_read.csv')

            rc = subprocess.call('src/scripts/breakout_and_top_boards.sh',shell=True)

            print ('got pages for '+pd.Timestamp("today").strftime("%m-%d-%Y"), '\n')

            time.sleep((60*60*24) - 60)


if __name__ == '__main__':
    tb = TopAndBreakoutBoards()
    tb.update_page_daily()
