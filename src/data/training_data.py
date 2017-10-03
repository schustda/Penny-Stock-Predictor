import pandas as pd
import numpy as np
from random import sample,choices
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from time import time

class TrainingData(object):

    def __init__(self, num_days = 50, num_points = 1000, days_avg = 1,
                    errors = 'ignore', method='all', percentage=1,
                    times_to_resample=1, predict=False,verbose=1):
        self.num_days = num_days
        self.num_points = num_points
        self.days_avg = days_avg
        self.errors = errors
        self.method = method
        self.percentage = percentage
        self.times_to_resample = times_to_resample
        self.verbose = verbose
        self.r = int(self.num_days/self.days_avg)
        self.feature_length = int(self.r * 2 + 3)
        self.predict = predict
        if self.predict:
            self.feature_length -= 1

    def _incompatible_points(self, data):
        print('Getting incomplete points...')
        bad_points = set()
        symbol = None
        for point in data.iterrows():
            if point[1].symbol != symbol:
                symbol = point[1].symbol
                bad_points = bad_points.union(set(range(point[0],point[0]+self.num_days)))
        return bad_points

    def _get_points(self,data):
        '''
        Methods available:

        -- 'all'
            Returns all points

        -- 'random_undersampling' -- needs percentage
            Takes all points from 'buy', and percentage from 'non-buy'
            More bias (not representative of population)

        -- 'random_oversampling' -- needs times_to_resample
            Takes all points from 'non-buy', and adds points with re-sampling from 'buy'
            More variance (fitting noise)

        -- 'cluster_random_oversampling' - NOT YET AVAILABLE
            Run two separate clusters on 'buy' and 'non-buy' classes.
            Oversample each cluster so that each class will have the same
                number of points (More prone to overfitting)
        '''
        bad_points = self._incompatible_points(data)
        print('Generating training data points...')
        buy = set(data[data.target==1].index.tolist())-bad_points
        non_buy = set(data[data.target==0].index.tolist())-bad_points

        if self.method == 'all':
            points = buy.union(non_buy)
        elif self.method == 'random_undersampling':
            ru_points_non_buy = set(sample(non_buy,int(len(non_buy)*self.percentage)))
            ru_points_buy = buy
            points = ru_points_non_buy.union(ru_points_buy)
        elif self.method == 'random_oversampling':
            ro_points_buy = choices(list(buy),k=int(len(buy)*self.times_to_resample))
            ro_points_non_buy = list(non_buy)
            points = ro_points_buy + ro_points_non_buy

        return points

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

    def _get_data_point(self,index,data):
        data_point = np.empty(self.feature_length)
        #Posts
        posts = data.post_number.values[index-self.num_days:index]
        post_averages = np.mean(posts.reshape(-1, self.days_avg), axis=1).reshape(1,-1)
        npa = normalize(post_averages.reshape(1,-1))
        data_point[0:self.r] = npa.flatten()

        #posts_significance_factor
        data_point[self.r] = data[data.post_number < posts.mean()].shape[0]/data.shape[0]

        #Volume
        vol = data.dollar_volume.values[index-self.num_days:index]
        vol_averages = np.mean(vol.reshape(-1, self.days_avg), axis=1).reshape(1,-1)
        va = normalize(vol_averages.reshape(1,-1))
        data_point[self.r+1:self.r*2+1] = va.flatten()

        #vol_significance_factor
        data_point[self.r*2+1] = data[data.dollar_volume < vol.mean()].shape[0]/data.shape[0]

        #if trying to make a prediction, there should not be a target
        #if trying to generate training data, target should be included
        if not self.predict:
            data_point[self.r*2+2] = data.loc[index].target

        return data_point

    def generate_training_data(self):
        self.r = int(self.num_days/self.days_avg)
        self.feature_length = int(self.r * 2 + 3)
        print ('num_days: {0}, days_avg: {1}, Method: {2}'.format(self.num_days,self.days_avg,self.method))
        if self.method == 'random_undersampling':
            print ('Percentage : {0}'.format(self.percentage))
        elif self.method == 'random_oversampling':
            print ('Times to resample : {0}'.format(self.times_to_resample))
        t, original_time = time(), time()
        data = pd.read_csv('data/data/_all.csv')
        points = self._get_points(data)
        print ('{0} training points'.format(len(points)))
        print ('Transforming feature space...')
        stock_array = np.empty([len(points),self.feature_length])
        for num,point in enumerate(points):
            stock_array[num] = self._get_data_point(point,data)
            if self.verbose:
                # Display update ever 60 seconds
                if time() > t + 60:
                    percent = int(num/len(points)*100)
                    time_elapsed = time() - original_time
                    self._verbose(percent, original_time, time_elapsed)
                    t = time()

        # if first == True:
        #     full_dataset = stock_array
        #     first=False
        # else:
        #     full_dataset = np.vstack([full_dataset,stock_array])
        # train, test = train_test_split(full_dataset)
        train, test = train_test_split(stock_array,random_state=10)
        np.savetxt('data/model/train.csv',train,delimiter=",")
        np.savetxt('data/model/test.csv',test,delimiter=",")


if __name__ == '__main__':

    # td = CreateTrainingData(num_days = 100, days_avg = 1,method = 'random_oversampling',
    #                         times_to_resample = 20)
    td = CreateTrainingData(num_days = 1200, days_avg = 12,method = 'random_undersampling',
                percentage = 0.2)
    td.generate_training_data()

# FEATURES

    # X number of days before prediction
    # num_days = 100

    #MESSAGE BOARD POSTS
        # Normalize weekly data
        # Significance Factor

    #DOLLAR VOLUME
        # Normalize weekly data
        # Significance Factor

    #BREAKOUT BOARDS
        #Coming as soon as data is available

    #MOST READ BOARDS
        #Coming as soon as data is available
