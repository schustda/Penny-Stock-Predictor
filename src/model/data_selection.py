from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,recall_score,precision_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.data.training_data import CreateTrainingData

class DataSelection(CreateTrainingData):
    def __init__(self,param,units,param2 = 0,units2 = 0):
        super().__init__()
        self.param = param
        self.units = units
        self.param2 = param2
        self.units2 = units2
        self.colors = {'auc':'r','recall':'g','precision':'b'}
        self.lr = self._emptydf('Logistic Regression')
        self.dt = self._emptydf('Decision Tree')
        self.rf = self._emptydf('Random Forest')
        self.gb = self._emptydf('Gradient Boost')
        self.xgb = self._emptydf('XGBoost')
        self.svc = self._emptydf('SVC')

        #Deduced from previous trials
        self.percentage = 0.2
        self.method = 'random_undersampling'
        self.num_days = 1200
        self.days_avg = 10

    def _score_model(self,model,type,threshold = 0.5):
        if type == 'sklearn':
            test_pred = model.predict(self.X_test)
            test_pred_proba = model.predict_proba(self.X_test)[:,1]

            train_pred = model.predict(self.X_train)
            train_pred_proba = model.predict_proba(self.X_train)[:,1]

        elif type in ['svc','xgb']:
            if type == 'xgb':
                test,train = self.dtest,self.dtrain
            else:
                test,train = self.X_test.copy(),self.X_train.copy()
            test_pred_proba = model.predict(test)
            test_pred = test_pred_proba.copy()
            test_mask = test_pred > threshold
            test_pred[test_mask] = 1
            test_pred[np.invert(test_mask)] = 0

            train_pred_proba = model.predict(train)
            train_pred = train_pred_proba.copy()
            train_mask = train_pred > threshold
            train_pred[train_mask] = 1
            train_pred[np.invert(train_mask)] = 0

        return [roc_auc_score(self.y_train,train_pred_proba),
                roc_auc_score(self.y_test,test_pred_proba),
                recall_score(self.y_train,train_pred),
                recall_score(self.y_test,test_pred),
                precision_score(self.y_train,train_pred),
                precision_score(self.y_test,test_pred)]

    def _emptydf(self,name):
        columns = ['train_auc','test_auc','train_recall','test_recall','train_precision','test_precision']
        df = pd.DataFrame(columns=columns,index=range(1,len(self.units)+1))
        df['index'] = df.index
        df['units'] = self.units
        df.index.name = name
        return df

    def _plot(self,ax,data,score):
        color = self.colors[score]

        # Plot Train
        sns.set_color_codes("pastel")
        sns.barplot(x="index", y="train_{0}".format(score), data=data,
                    label="Train", color=color, ax=ax,
                   edgecolor='black')

        # Plot Test
        sns.set_color_codes("muted")
        sns.barplot(x="index", y="test_{0}".format(score), data=data,
                    label="Test", color=color, ax=ax,
                   edgecolor='black')

        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="upper right", frameon=True)
        ax.set_xticklabels(data.units)
        ax.set(ylabel="",xlabel="",ylim=(0,1))
        ax.set_title(data.index.name)

    def _train_models(self,idx):
        train = pd.read_csv('data/model/train.csv',header=None)
        X = np.array(train.iloc[:,0:-1])
        y = np.array(train.iloc[:,-1])
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,
            y,random_state = 10)

        # LogisticRegression
        self.lr_model = LogisticRegression()
        self.lr_model.fit(self.X_train,self.y_train)
        self.lr.loc[idx,'train_auc':'test_precision'] = self._score_model(self.lr_model,'sklearn')

        # DecisionTreeClassifier
        self.dt_model = DecisionTreeClassifier()
        self.dt_model.fit(self.X_train,self.y_train)
        self.dt.loc[idx,'train_auc':'test_precision'] = self._score_model(self.dt_model,'sklearn')

        # RandomForestClassifier
        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.X_train,self.y_train)
        self.rf.loc[idx,'train_auc':'test_precision'] = self._score_model(self.rf_model,'sklearn')

        # GradientBoostingClassifier
        self.gb_model = GradientBoostingClassifier()
        self.gb_model.fit(self.X_train,self.y_train)
        self.gb.loc[idx,'train_auc':'test_precision'] = self._score_model(self.gb_model,'sklearn')

        # SVC
        self.svc_model = SVC()
        self.svc_model.fit(self.X_train,self.y_train)
        self.svc.loc[idx,'train_auc':'test_precision'] = self._score_model(self.svc_model,'svc')

        # XGBoost
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test)
        p = {'max_depth':10, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
        self.xgb_model = xgb.train(p,self.dtrain)
        self.xgb.loc[idx,'train_auc':'test_precision'] = self._score_model(self.xgb_model,'xgb')

    def run(self):
        if self.param2 != 0:
            setattr(self,self.param2,self.units2)
        for idx,value in enumerate(self.units):
            print ('Param value: {0}'.format(value))
            setattr(self,self.param,value)
            self.generate_training_data()
            self._train_models(idx+1)


        if self.param2 != 0:
            title = self.units2
        else:
            title = self.param
        for score_type in ['auc','recall','precision']:

            print ('plotting {0} charts'.format(score_type))
            plt.close('all')

            # Set up figure
            sns.set(style="whitegrid")
            fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle("{0} scores for {1}".format(score_type,title),fontsize = 20)
            sns.despine(left=True,bottom=True)

            # Plot results
            self._plot(ax[0,0],self.lr,score_type)
            self._plot(ax[1,0],self.dt,score_type)
            self._plot(ax[0,1],self.rf,score_type)
            self._plot(ax[1,1],self.gb,score_type)
            self._plot(ax[0,2],self.svc,score_type)
            self._plot(ax[1,2],self.xgb,score_type)

            plt.savefig('images/model_development/data_selection/{0}/{1}.jpg'
                .format(title,score_type))



if __name__ == '__main__':
    # num_days = [5,10,20,50,100,500,1000]
    num_days = list(range(500,1501,100))
    # ds = DataSelection('num_days',num_days)

    days_avg = [1,2,3,4,5,6,8,10,12,15,20]
    ds = DataSelection('days_avg',days_avg)

    percentage = [1,0.85,0.7,.55,0.4,0.25,0.1]
    # ds = DataSelection('percentage',percentage,'method','random_undersampling')

    times_to_resample = [1,2,5,10,15,20,30,50,100]
    ds = DataSelection('times_to_resample',times_to_resample,'method','random_oversampling')

    ds.run()
