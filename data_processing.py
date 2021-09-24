from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import math

def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i/16 * math.pi
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return (x, y)

def spiral(spiral_num):
    return [spiral_xy(i, spiral_num) for i in range(97)]

def load_data(dataset_name, test_size=0.2, random_state=1):
    sc = StandardScaler()
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
    elif dataset_name == 'pima':
        df = pd.read_csv('data/diabetes.csv')
        data = Object()
        data.data = np.asarray(df.loc[:, df.columns!='Outcome'])
        data.target = np.asarray(df['Outcome'])
    elif dataset_name == 'car':
        df = pd.read_csv('data/car.data', sep=',', header=None)
        mapping = {'acc': 1, 'good': 2, 'unacc': 0, 'vgood': 3}
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
        data = Object()
        enc = OneHotEncoder()
        enc.fit(np.asarray(df.iloc[:, :-1]))
        data.data = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
        data.target = np.asarray(df.iloc[:, -1])
    elif dataset_name == 'tic-tac-toe':
        df = pd.read_csv('data/tic-tac-toe.data', sep=',', header=None)
        mapping = {'negative': 0, 'positive': 1}
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
        data = Object()
        enc = OneHotEncoder()
        enc.fit(np.asarray(df.iloc[:, :-1]))
        data.data = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
        data.target = np.asarray(df.iloc[:, -1])
    elif dataset_name == 'ionosphere':
        df = pd.read_csv('data/ionosphere.data', sep=',', header=None)
        mapping = {'b': 0, 'g': 1}
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
        data = Object()
        data.data = np.asarray(df.iloc[:, :-1])
        data.target = np.asarray(df.iloc[:, -1])
    elif dataset_name == '4bit':
        n_bit = 6
        data_dict = dict()
        while len(data_dict) < 2**n_bit:
            new_seq = np.random.randint(0, 2, n_bit)
            s = ''.join([str(bit) for bit in new_seq])
            if s not in data_dict.keys():
                if np.unique(new_seq, return_counts=True)[1][0]%2 == 0:
                    data_dict[s] = 1
                else:
                    data_dict[s] = 0
        data = Object()
        data.data = []
        data.target = []
        for bits, label in data_dict.items():
            bits = [int(bit) for bit in bits]
            data.data.append(bits)
            data.target.append(label)
        data.data = np.asarray(data.data)
        data.target = np.asarray(data.target)
    elif dataset_name == '6bit':
        n_bit = 6
        data_dict = dict()
        while len(data_dict) < 2**n_bit:
            new_seq = np.random.randint(0, 2, n_bit)
            s = ''.join([str(bit) for bit in new_seq])
            if s not in data_dict.keys():
                if np.unique(new_seq, return_counts=True)[1][0]%2 == 0:
                    data_dict[s] = 1
                else:
                    data_dict[s] = 0
        data = Object()
        data.data = []
        data.target = []
        for bits, label in data_dict.items():
            bits = [int(bit) for bit in bits]
            data.data.append(bits)
            data.target.append(label)
        data.data = np.asarray(data.data)
        data.target = np.asarray(data.target)
    elif dataset_name == '8bit':
        n_bit = 8
        data_dict = dict()
        while len(data_dict) < 2**n_bit:
            new_seq = np.random.randint(0, 2, n_bit)
            s = ''.join([str(bit) for bit in new_seq])
            if s not in data_dict.keys():
                if np.unique(new_seq, return_counts=True)[1][0]%2 == 0:
                    data_dict[s] = 1
                else:
                    data_dict[s] = 0
        data = Object()
        data.data = []
        data.target = []
        for bits, label in data_dict.items():
            bits = [int(bit) for bit in bits]
            data.data.append(bits)
            data.target.append(label)
        data.data = np.asarray(data.data)
        data.target = np.asarray(data.target)
    elif dataset_name == 'two-spiral':
        data = Object()
        a = spiral(1)
        data.data = a[:]
        data.data.extend(spiral(-1))
        data.data = np.asarray(data.data)
        data.target = [0]*len(a)
        data.target.extend([1]*len(a))
        data.target = np.asarray(data.target)

    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print('Train: ', X_train.shape, ' | Test: ', X_test.shape)
    print('Train labels: ', np.unique(y_train, return_counts=True))
    print('Test labels: ', np.unique(y_test, return_counts=True))
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    return X_train, X_test, y_train, y_test
