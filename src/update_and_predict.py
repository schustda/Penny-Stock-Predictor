import smtplib
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
from sys import argv
from time import time,sleep
from email.mime.text import MIMEText
from sklearn.externals import joblib
from src.data.ihub_data import IhubData
from src.data.stock_data import StockData
from src.data.combine_data import CombineData
from email.mime.multipart import MIMEMultipart
from src.data.training_data import TrainingData

class Predict(TrainingData):

    def __init__(self,num_days,days_avg,threshold,email_address,password):
        super().__init__(num_days=num_days,days_avg=days_avg,predict=True)
        self.num_days = num_days
        self.days_avg = days_avg
        self.threshold = threshold
        self.predict = True
        self.email_address = email_address
        self.password = password

    def prediction(self):
        model = joblib.load('data/model/model.pkl')
        stocks = pd.read_json('data/stock_list.json')
        buy = []
        for stock in stocks.columns.tolist():
            data = pd.read_csv('data/data/'+stock+'.csv')
            if data.shape[0] <= self.num_days:
                pass
            else:
                x_pred = self._get_data_point(data.shape[0],data).reshape(1,-1)
                train_pred_proba = model.predict(xgb.DMatrix(x_pred))
                train_pred = train_pred_proba.copy()
                train_mask = train_pred > threshold
                train_pred[train_mask] = 1
                train_pred[np.invert(train_mask)] = 0
                if train_pred == 1:
                    buy.append(stock)
                print(stock, bool(train_pred))

        if len(buy) > 0:
            self._email_results(buy)

    def _email_results(self,stock_lst):
        fromaddr = self.email_address
        toaddr = self.email_address
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "BUY SIGNAL ALERT"

        body = "Stocks indicating buy signal: {0}".format(", ".join(stock_lst).upper())
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, self.password)
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()

if __name__ == '__main__':

    num_days = 1200
    days_avg = 12
    threshold = 0.2

    while True:
        t = time()

        rc = subprocess.call('src/scripts/git_pull.sh',shell=True)
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

        #predict
        p  = Predict(num_days = num_days,days_avg=days_avg,threshold=threshold,
        email_address = argv[1], password = argv[2])
        buy = p.prediction()

        # update six times a day
        sleep(60*60*24-(time()-t))
