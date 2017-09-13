import pandas as pd
import numpy as np
from random import randint
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split



def get_points(data,num_points,num_days):
    # first, make sure all of the succssful points are included
    s = data[data.target==1].index.tolist()
    s = list(filter(lambda x: x>num_days,s))
    s.extend([randint(num_days,data.shape[0]-12) for _ in range(num_points)])
    return s

def get_data_point(index,num_days,data):

    data_point = np.empty(int(num_days/5 * 2 + 3))

    #Posts
    posts = data.post_number.values[index-50:index]
    post_averages = np.mean(posts.reshape(-1, 5), axis=1).reshape(1,-1)
    npa = normalize(post_averages.reshape(1,-1))
    data_point[0:10] = npa.flatten()

    #posts_significance_factor
    data_point[10] = data[data.post_number < posts.mean()].shape[0]/data.shape[0]

    #Volume
    vol = data.dollar_volume.values[index-50:index]
    vol_averages = np.mean(vol.reshape(-1, 5), axis=1).reshape(1,-1)
    va = normalize(vol_averages.reshape(1,-1))
    data_point[11:21] = va.flatten()

    #vol_significance_factor
    data_point[21] = data[data.dollar_volume < vol.mean()].shape[0]/data.shape[0]

    data_point[22] = data.loc[index].target

    return data_point



if __name__ == '__main__':



# NUMBER OF RANDOM POINTS FROM EACH DATASET
    num_points = 1000

# FEATURES

    # X number of days before prediction
    num_days = 100

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

    first = True
    df = pd.read_json('data/stock_list.json')
    stocks = df.columns.tolist()

    for stock in stocks:
        print (stock)
        data = pd.read_csv('data/data/'+stock+'.csv')
        points = get_points(data,num_points,num_days)
        for point in points:
            if first:
                full_dataset = get_data_point(point,num_days,data)
                first=False
            else:
                full_dataset = np.vstack([full_dataset,get_data_point(point,num_days,data)])

    train, test = train_test_split(full_dataset)
    np.savetxt('data/model/train.csv',train,delimiter=",")
    np.savetxt('data/model/test.csv',test,delimiter=",")
