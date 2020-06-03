from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


def test(config):
    data_path = config.data_path

    pretrained_file_path = config.pretrained_file_path
    data = pd.read_csv(data_path)

    x_columns = config.x_columns
    x_columns = x_columns.split(',')
    X = data[x_columns]
    y_column = config.y_column
    Y = data[y_column]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    model = pickle.load(open(pretrained_file_path, 'rb'))
    print("load pretrained model")

    y_test_predict = model.predict(X_test)
    acc = accuracy_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('accuracy score is {}'.format(acc))

