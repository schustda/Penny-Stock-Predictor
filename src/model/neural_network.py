from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import numpy as np




if __name__ == '__main__':
    np.random.seed(10)
    train = np.loadtxt('data/model/train.csv',delimiter=',')
    X_train,X_test,y_train,y_test = train_test_split(train[:,0:22],train[:,22])

    model = Sequential()

    model.add(Dense(12, input_dim=22, activation='relu'))
    model.add(Dense(22, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=20, batch_size=10, verbose = 1)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_pred = [round(x[0]) for x in y_train_pred]
y_test_pred = [round(x[0]) for x in y_test_pred]


# evaluate the model
print ("Recall score on training set: {0}".format(recall_score(y_train,y_train_pred)))
print ("Recall score on test set: {0}".format(recall_score(y_test,y_test_pred)))
