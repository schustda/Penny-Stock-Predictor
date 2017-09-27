import pandas as pd
import numpy as np
from sklearn.externals import joblib
from src.data.training_data import TrainingData
import xgboost as xgb
import smtplib
from sys import argv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import subprocess


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
            self._email_results

    def email_results(self,stock_lst):
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
        # t = time()
        # rc = subprocess.call('src/scripts/git_pull.sh',shell=True)
        p  = Predict(num_days = num_days,days_avg=days_avg,threshold=threshold,
                email_address = argv[1], password = argv[2])
        buy = p.prediction()
        # time.sleep(60*60*24-(time()-t))
