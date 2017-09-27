import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,recall_score,precision_score
from sklearn.externals import joblib
import xgboost as xgb


if __name__ == '__main__':

    threshold = 0.2

    param_dict = {
    'max_depth':50,
    'silent':1,
    'eta':0.4,
    'gamma':0,
    'objective':'binary:logistic'
    }

    train = pd.read_csv('data/model/train.csv',header=None)
    X_train,y_train = np.array(train.iloc[:,0:-1]),np.array(train.iloc[:,-1])

    test = pd.read_csv('data/model/test.csv',header=None)
    X_test,y_test = np.array(test.iloc[:,0:-1]),np.array(test.iloc[:,-1])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    xgb_model = xgb.train(param_dict,dtrain)

    test_pred_proba = xgb_model.predict(dtest)
    test_pred = test_pred_proba.copy()
    test_mask = test_pred > threshold
    test_pred[test_mask] = 1
    test_pred[np.invert(test_mask)] = 0

    train_pred_proba = xgb_model.predict(dtrain)
    train_pred = train_pred_proba.copy()
    train_mask = train_pred > threshold
    train_pred[train_mask] = 1
    train_pred[np.invert(train_mask)] = 0

    joblib.dump(xgb_model,'data/model/model.pkl')

    print ('Training set scores:')
    print ('AUC - {0}'.format(roc_auc_score(y_train,train_pred_proba)))
    print ('Recall - {0}'.format(recall_score(y_train,train_pred)))
    print ('Precision - {0}'.format(precision_score(y_train,train_pred)))
    print ('\n')
    print ('FINAL HOLDOUT SET SCORES:')
    print ('AUC - {0}'.format(roc_auc_score(y_test,test_pred_proba)))
    print ('Recall - {0}'.format(recall_score(y_test,test_pred)))
    print ('Precision - {0}'.format(precision_score(y_test,test_pred)))
