from audioop import cross
from cgi import test
from data_processing import load_data
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import tqdm

dataset_names = ["mnist", "breast_cancer", "abalone", 
                 "red_wine", "pima", "car", "tic-tac-toe", 
                 "ionosphere", "churn", "flare", "ring", "two-spiral"]
n_bag_range = list(range(5, 100, 5))

for dataset_name in dataset_names:
    print(dataset_name)
    X_train, _, y_train, _ = load_data(dataset_name, test_size=0)
    X_train = np.asarray(X_train)
    y_train = np.squeeze(np.asarray(y_train))
    cv_scores = []
    for n_bags in tqdm(n_bag_range):
        clf = BaggingClassifier(n_estimators=n_bags)
        score = cross_val_score(clf, X_train, y_train, cv=3).mean()
        cv_scores.append(score)
    n_bag = n_bag_range[np.argmax(cv_scores)]
    print(f"Max score: {max(cv_scores)}")
    print(f"n_bag: {n_bag}")
    print(f"90-quantile score: {np.quantile(cv_scores, 0.9)}")

    print("=======================================")