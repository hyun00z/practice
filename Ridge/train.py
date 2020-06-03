from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import time
import os

def train(config):

    alpha = config.alpha

    data_path = config.data_path
    save_directory = config.save_directory
    data = pd.read_csv(data_path)
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values.reshape(-1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    print("Model build")

    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, Y_train)

    print("Model fit")

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'ridge_model_' + time_stamp + '.sav'
    pickle.dump(ridge_model, open(os.path.join(save_directory, file_name), 'wb'))

    y_train_predict = ridge_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

