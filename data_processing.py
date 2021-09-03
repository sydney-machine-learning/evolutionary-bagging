from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data(dataset_name, test_size=0.2, random_state=1):
    class Object(object):
        pass
    if dataset_name == 'mnist':
        data = load_digits()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'abalone':
        df = pd.read_csv('data/abalone.csv')
        df['Sex'] = df['Sex'].apply(lambda x: 0 if x=='M' else 1)
        labels = []
        for r in df['Rings']:
            if 0<=r and r<=7:
                label = 1
            elif 8<=r and r<=10:
                label = 2
            elif 11<=r and r<=15:
                label = 3
            elif r>15:
                label = 4
            labels.append(label)
        data = Object()
        data.data = np.asarray(df.loc[:, df.columns!='Rings'])
        data.target = np.asarray(labels)
    elif dataset_name == 'synthetic':
        df = pd.read_csv('data/synth_data.csv')
        df = pd.get_dummies(df, prefix=['a', 'b', 'c'])
        data = Object()
        data.data = np.asarray(df.loc[:, df.columns!='label'])
        data.target = np.asarray(df['label'])
    elif dataset_name == 'red_wine':
        df = pd.read_csv('data/winequality-red.csv')
        df.quality = np.where(df.quality < 6.5, 0, 1)
        data = Object()
        data.data = np.asarray(df.loc[:, df.columns!='quality'])
        data.target = np.asarray(df['quality'])
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print('Train: ', X_train.shape, ' | Test: ', X_test.shape)
    print('Train labels: ', np.unique(y_train, return_counts=True))
    print('Test labels: ', np.unique(y_test, return_counts=True))
    return X_train, X_test, y_train, y_test