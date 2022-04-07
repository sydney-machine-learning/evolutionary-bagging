import numpy as np
import random
from data_processing import load_data
from sklearn.metrics import f1_score, precision_score, roc_auc_score, recall_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import argparse
import yaml


def prepare_data(dataset_name, up_times=3):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=0.2)
    minor_value = y_train[0].value_counts(sort=True, ascending=True).index[0]
    y_train[0] = [1 if value==minor_value else 0 for value in y_train[0]]
    y_test[0] = [1 if value==minor_value else 0 for value in y_test[0]]

    minor_percent = y_train[0].value_counts(normalize=True)[1]
    imbalance_rate = (1-minor_percent)/minor_percent
    imbalance_rates = np.linspace(1, imbalance_rate, up_times+1)

    data_dict = dict()
    for i, r in enumerate(imbalance_rates):
        if r != imbalance_rate:
            train_minor_count = int(y_train[0].value_counts()[0]/r)
            pos_idx = y_train.index[y_train.iloc[:, 0]==1]
            neg_idx = y_train.index[y_train.iloc[:, 0]==0]
            pos_idx = random.choices(list(pos_idx), k = train_minor_count)
            new_idx = list(neg_idx)
            new_idx.extend(list(pos_idx))
            new_X_train = X_train.iloc[new_idx, :]
            new_y_train = y_train.iloc[new_idx, :]
            pos_idx = y_test.index[y_test.iloc[:, 0]==1]
            neg_idx = y_test.index[y_test.iloc[:, 0]==0]
            pos_idx = random.choices(list(pos_idx), k = train_minor_count)
            new_idx = list(neg_idx)
            new_idx.extend(list(pos_idx))
            new_X_test = X_test.iloc[new_idx, :]
            new_y_test = y_test.iloc[new_idx, :] 
            data_dict[i] = {'ratio': r,
                            'X_train': new_X_train,
                            'X_test': new_X_test,
                            'y_train': new_y_train,
                            'y_test': new_y_test}    
        else:
            data_dict[i] = {'ratio': r,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test}  
    return data_dict   

def get_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return f1, precision, recall, auc

def test_baseline(dataset_name, up_times, n_bags, runs=20):
    data_dict = prepare_data(dataset_name, up_times)
    metrics = {ratio: {met: {'train_rf_metrics': [],
                             'test_rf_metrics': [],
                             'train_bagging_metrics': [],
                             'test_bagging_metrics': []} for met in ['f1', 'precision', 'recall', 'auc']} for ratio in [round(data_dict[i]['ratio'], 2) for i in data_dict]}
    for j in range(runs):
        data_dict = prepare_data(dataset_name, up_times)
        for i in data_dict:
            ratio = round(data_dict[i]['ratio'], 2)
            X_train = data_dict[i]['X_train']
            X_test = data_dict[i]['X_test']
            y_train = data_dict[i]['y_train']
            y_test = data_dict[i]['y_test']
            # random forest
            clf = RandomForestClassifier(n_estimators=n_bags) 
            clf.fit(X_train, y_train.values.ravel())
            train_preds = clf.predict(X_train)
            test_preds = clf.predict(X_test)
            train_f1, train_precision, train_recall, train_auc = get_metrics(y_train, train_preds)
            test_f1, test_precision, test_recall, test_auc = get_metrics(y_test, test_preds)
            for met in ['f1', 'precision', 'recall', 'auc']:
                metrics[ratio][met]['train_rf_metrics'].append(eval(f"train_{met}"))
                metrics[ratio][met]['test_rf_metrics'].append(eval(f"test_{met}"))
            # bagging performance
            clf = BaggingClassifier(n_estimators=n_bags)
            clf.fit(X_train, y_train.values.ravel())
            train_preds = clf.predict(X_train)
            test_preds = clf.predict(X_test)
            train_f1, train_precision, train_recall, train_auc = get_metrics(y_train, train_preds)
            test_f1, test_precision, test_recall, test_auc = get_metrics(y_test, test_preds)
            for met in ['f1', 'precision', 'recall', 'auc']:
                metrics[ratio][met]['train_bagging_metrics'].append(eval(f"train_{met}"))
                metrics[ratio][met]['test_bagging_metrics'].append(eval(f"test_{met}"))
    for ratio in metrics:
        print(f"Ratio: {ratio}")
        for met in ['f1', 'precision', 'recall', 'auc']:
            print(met)
            print('Random forest train metric:   ', round(sum(metrics[ratio][met]['train_rf_metrics'])*100/runs, 2))
            print('                              ', np.std(metrics[ratio][met]['train_rf_metrics']))
            print('Random forest test:           ', round(sum(metrics[ratio][met]['test_rf_metrics'])*100/runs, 2))
            print('                              ', np.std(metrics[ratio][met]['test_rf_metrics']))
            print(f'Bagging train metric:         ', round(sum(metrics[ratio][met]['train_bagging_metrics'])*100/runs, 2))
            print(f'                              ', np.std(metrics[ratio][met]['train_bagging_metrics']))
            print(f'Bagging test metric:          ', round(sum(metrics[ratio][met]['test_bagging_metrics'])*100/runs, 2))
            print(f'                              ', np.std(metrics[ratio][met]['test_bagging_metrics']))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment voting rule')
    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    parser.add_argument('--up_times', type=int,
                        help="Sampling times")
    parser.add_argument('--runs', type=int, default=20,
                        help="Number of runs")
    args = parser.parse_args()
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[args.dataset_name]
    n_bags = dataset_config["n_bags"]
    test_baseline(args.dataset_name, args.up_times, args.runs)