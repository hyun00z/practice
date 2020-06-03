from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np


def test(config):
    data_path = config.data_path

    pretrained_file_path = config.pretrained_file_path
    data = pd.read_csv(data_path)
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values.reshape(-1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    model = pickle.load(open(pretrained_file_path, 'rb'))
    print("load pretrained model")

    y_test_predict = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))

