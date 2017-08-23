import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.externals import joblib

def get_data_point(index,num_days,data):

    data_point = np.empty(int(num_days/5 * 2 + 2))

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
    return data_point

if __name__ == '__main__':

    num_days = 50
    model = joblib.load('data/model/model.pkl')
    # joblib.dump(model, 'data/model/model.pkl')


    df = pd.read_json('data/stock_list.json')
    stocks = df.columns.tolist()

    for stock in stocks:
        print (stock)
        data = pd.read_csv('data/data/'+stock+'.csv')
        x_pred = get_data_point(data.shape[0],num_days,data).reshape(1,-1)
        print(model.predict(x_pred))
