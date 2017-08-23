import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


train = np.loadtxt('train.csv',delimiter=',')

X_train,X_test,y_train,y_test = train_test_split(train[:,0:22],train[:,22])

# lr = LogisticRegression()
# lr.fit(X_train,y_train)
# print (lr.score(X_test,y_test))
# y_pred = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train,y_train)
print (rf.score(X_test,y_test))
y_pred = rf.predict(X_test)

# gb = GradientBoostingClassifier()
# gb.fit(X_train,y_train)
# print (gb.score(X_test,y_test))
# y_pred = gb.predict(X_test)

print (confusion_matrix(y_test,y_pred))
