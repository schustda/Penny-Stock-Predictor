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

class ModelSelection(object):
    def __init__(self,model_name,testing_param,param_values,param_dict={},final=False,
                        num_iterations=10, threshold = 0.5):
        self.model_name = model_name
        self.testing_param = testing_param
        self.param_values = param_values
        self.param_dict = param_dict
        self.colors = {'auc':'r','recall':'g','precision':'b'}
        self.summary = self._emptydf(self.model_name)
        self.scoring_methods = {"logistic_regression":'sklearn',
                    'decision_tree': 'sklearn',
                    'random_forest': 'sklearn',
                    'gradient_boost': 'sklearn',
                    'svc': 'svc',
                    'xgboost':'xgb',
                    'neural_network':'nn'}
        self.scoring_type = self.scoring_methods[self.model_name]
        self.final = final
        self.num_iterations = num_iterations
        self.threshold = threshold

    def _score_model(self,model):

        if self.scoring_type == 'sklearn':
            test_pred = model.predict(self.X_test)
            test_pred_proba = model.predict_proba(self.X_test)[:,1]

            train_pred = model.predict(self.X_train)
            train_pred_proba = model.predict_proba(self.X_train)[:,1]

        elif self.scoring_type in ['svc','xgb']:
            if self.scoring_type == 'xgb':
                test,train = self.dtest,self.dtrain
            else:
                test,train = self.X_test.copy(),self.X_train.copy()
            test_pred_proba = model.predict(test)
            test_pred = test_pred_proba.copy()
            test_mask = test_pred > self.threshold
            test_pred[test_mask] = 1
            test_pred[np.invert(test_mask)] = 0

            train_pred_proba = model.predict(train)
            train_pred = train_pred_proba.copy()
            train_mask = train_pred > self.threshold
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
        df = pd.DataFrame(columns=columns,index=range(1,len(self.param_values)+1))
        df['index'] = df.index
        df['param_values'] = self.param_values
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

        if self.final == True:
            xticklabels = ''
        else:
            xticklabels = data.param_values

        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="upper right", frameon=True)
        ax.set_xticklabels(xticklabels)
        ax.set(ylabel="",xlabel="",ylim=(0,1))
        ax.set_title(score)

    def _train_models(self,idx):

        # LogisticRegression
        if self.model_name == 'logistic_regression':
            self.lr_model = LogisticRegression(**self.param_dict)
            self.lr_model.fit(self.X_train,self.y_train)
            self.summary.loc[idx,'train_auc':'test_precision'] = self._score_model(self.lr_model)

            # DecisionTreeClassifier
        elif self.model_name == 'decision_tree':
            self.dt_model = DecisionTreeClassifier(**self.param_dict)
            self.dt_model.fit(self.X_train,self.y_train)
            self.summary.loc[idx,'train_auc':'test_precision'] = self._score_model(self.dt_model)

            # RandomForestClassifier
        elif self.model_name == 'random_forest':
            self.rf_model = RandomForestClassifier(**self.param_dict)
            self.rf_model.fit(self.X_train,self.y_train)
            self.summary.loc[idx,'train_auc':'test_precision'] = self._score_model(self.rf_model)

            # GradientBoostingClassifier
        elif self.model_name == 'gradient_boost':
            self.gb_model = GradientBoostingClassifier(**self.param_dict)
            self.gb_model.fit(self.X_train,self.y_train)
            self.summary.loc[idx,'train_auc':'test_precision'] = self._score_model(self.gb_model)

            # SVC
        elif self.model_name == 'svc':
            self.svc_model = SVC(**self.param_dict)
            self.svc_model.fit(self.X_train,self.y_train)
            self.summary.loc[idx,'train_auc':'test_precision'] = self._score_model(self.svc_model)

            # XGBoost
        elif self.model_name == 'xgboost':
            self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            self.dtest = xgb.DMatrix(self.X_test)
            self.xgb_model = xgb.train(self.param_dict,self.dtrain)
            self.summary.loc[idx,'train_auc':'test_precision'] = self._score_model(self.xgb_model)

    def run(self):
        print ('Testing paramaters for {0}'.format(self.testing_param))
        train = pd.read_csv('data/model/train.csv',header=None)
        X = np.array(train.iloc[:,0:-1])
        y = np.array(train.iloc[:,-1])
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y)

        for num in range(1,self.num_iterations):
            print (num)
            for idx,value in enumerate(self.param_values):
                print ('Param value: {0}'.format(value))
                if self.testing_param == 'threshold':
                    self.threshold = value
                    self._train_models(idx+1)
                else:
                    self.param_dict[self.testing_param] = value
                    self._train_models(idx+1)

            if num == 1:
                self.df = self.summary
            else:
                self.df.iloc[:,0:6] *= (num-1)/num
                self.df.iloc[:,0:6] += self.summary.iloc[:,0:6]*(1/num)
        self.summary = self.df

        # Set up figure
        if self.final == True:
            self.figname = '_final'
        else:
            self.figname = self.testing_param
        plt.close('all')
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True)
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout(rect=[0, 0.03, 1, .88])
        fig.suptitle("{0} scores for {1}".format(self.figname,self.model_name),fontsize = 20)
        sns.despine(left=True,bottom=True)

        # Plot results
        scores = ['auc','recall','precision']
        axes = [ax[0],ax[1],ax[2]]
        for score_type,axis in zip(scores,axes):
            print ('plotting {0} charts'.format(score_type))
            self._plot(axis,self.summary,score_type)
        plt.savefig('images/model_development/model_selection/{0}/{1}.jpg'
            .format(self.model_name,self.figname))



if __name__ == '__main__':

    # DecisionTreeClassifier
    dt = 'decision_tree'
    dt_parameters = {
            # 'criterion': ['gini','entropy'],
            # 'splitter': ['best','random'],
            # 'max_depth': [1,2,3,4,5,10,15,20,25,35,45,50,55,60,65,70,75,80,85,90,95,100,None],
            'min_samples_split':[2,3,4,5,10,15,20],
            'min_samples_leaf': [1,2,3,4,5,10,15,20],
            'min_weight_fraction_leaf': [0,.1,.2,.3,.4,.49],
            'max_features':[None,2,3,4,5,6,10,15,20,'auto','sqrt','log2'],
            'max_leaf_nodes': [None,5,10,15,20,50],
            'min_impurity_decrease':[0,.25,.5,.75],
            # 'class_weight':[{0:1,1:i} for i in range(1,10)],
            'presort': [True,False]
            }
    dt_dict = {
            'criterion': 'gini',
            'max_depth':20,
            'min_samples_leaf':3,
            }

    # logistic_regression
    lr = 'logistic_regression'
    lr_parameters = {
            'penalty': ['l2']
            # 'penalty': ['l1','l2'],
            # 'C': [1,2,5,10,15,25,50,100,200,500],
            # 'fit_intercept':[True,False],
            # 'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
            # 'warm_start':[False,True]
            # 'dual': [True,False],
            # 'intercept_scaling':[1,.85,.7,.55,.4,.25,.1]
            }
    lr_dict = {
            'penalty': 'l2',
            'C': 100,
            'fit_intercept':True,
            'solver':'liblinear'
            }

    # xgboost
    xgb_name = 'xgboost'
    xgb_parameters = {
                    # 'max_depth': [50]
                    # 'booster':['gblinear'],
                    # 'max_depth':[0,1,2,5,10,15,20,25,35,50,100],
                    # 'eta':[1,.85,.7,.55,.4,.25,.1,0],
                    # 'gamma':[0,1,2,5,10,25,50,100],
                    # 'objective':['binary:logistic','binary:logitraw'],
                    # 'booster':['gbtree','gblinear','dart'],
                    # 'tree_method':['auto','exact','approx'],
                    # 'process_type': ['default','update'],
                    # 'grow_policy':['depthwise','lossguide']
                    'threshold':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                }
    xgb_dict = {
    'max_depth':50,
    'silent':1,
    'eta':0.4,
    'gamma':0,
    'objective':'binary:logistic'
    # 'booster':'gblinear'
    }

    gbc_name = 'gradient_boost'
    gb_parameters = {
                    # 'criterion':['friedman_mse'],
                    # 'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    'max_depth':[1,2,5,10,15,20,25,50,100]

    }
    gb_dict = {
                    'learning_rate':0.3
    }


    param_dict = xgb_dict
    parameters = xgb_parameters
    model_name = xgb_name
    final = False
    for a,b in parameters.items():

        testing_param = a
        param_values = b

        ms = ModelSelection(model_name=model_name,testing_param=testing_param,
                param_values=param_values,param_dict=param_dict,final=final,
                num_iterations=10)
        ms.run()
