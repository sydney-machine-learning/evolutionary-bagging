from data_processing import load_data
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import *
import xgboost as xgb
import argparse
import numpy as np

def eval_baseline_sklearn(clf, X_train, y_train, X_test, y_test, metric):
    clf.fit(X_train, y_train[0])
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    met_train = eval(f"{metric}_score(y_train, train_preds)")*100
    met_test = eval(f"{metric}_score(y_test, test_preds)")*100
    depth = [e.get_depth() for e in clf.estimators_]
    return met_train, met_test, np.mean(depth)

def run(dataset_name,
        test_size,
        n_bags,
        metric):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size)

    baselines = {"BaggingClassifier": {"train_metric": [], "test_metric": [], "avg_depth": []},
                "ExtraTreesClassifier": {"train_metric": [], "test_metric": [], "avg_depth": []},
                "RandomForestClassifier": {"train_metric": [], "test_metric": [], "avg_depth": []},
                "XGBoost": {"train_metric": [], "test_metric": []}}
    for baseline in baselines.keys():
        if baseline != "XGBoost":
            clf = eval(f"{baseline}(n_estimators=n_bags)")
            met_train, met_test, avg_depth = eval_baseline_sklearn(clf, 
                                                                    X_train, y_train, 
                                                                    X_test, y_test, 
                                                                    metric)

        elif baseline == "XGBoost":
            if len(y_train.loc[:, 0].unique()) > 2:
                clf = xgb.XGBClassifier(n_estimators=n_bags, objective='multi:softprob')
            else:
                clf = xgb.XGBClassifier(n_estimators=n_bags)
            clf.fit(X_train, y_train[0], eval_metric='mlogloss')
            train_preds = clf.predict(X_train)
            test_preds = clf.predict(X_test)
            met_train = eval(f"{metric}_score(y_train, train_preds)")*100
            met_test = eval(f"{metric}_score(y_test, test_preds)")*100

        print(baseline)
        print("Train metric: ", np.mean(met_train))
        print("Test metric:  ", np.mean(met_test))
        try:
            print("Depth: ", np.mean(avg_depth))
        except:
            continue

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of test data')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Classification metric')
    parser.add_argument('--n_bags', type=int,
                        help='Number of bags')
    args = parser.parse_args()

    run(dataset_name=args.dataset_name, 
        test_size=args.test_size,
        n_bags=args.n_bags, 
        metric=args.metric)