from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
import numpy as np
from evobagging_methods import EvoBagging
from data_processing import load_data
from scipy import stats
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def voting_metric_roc(bags, X, y, metric):
    preds_list = []
    probs = np.zeros(len(y))
    for bag in bags.values():
        bag['clf'] = DecisionTreeClassifier()
        bag['clf'].fit(bag['X'], bag['y'])
        bag_preds = bag['clf'].predict(X)
        probs += bag['clf'].predict_proba(X)[:, 1]
        preds_list.append(bag_preds)
    temp_preds = np.stack(preds_list)
    final_preds = stats.mode(temp_preds).mode[0]
    met = eval(f"{metric}_score(y, final_preds)")
    probs = probs/len(bags)
    fpr, tpr, _ = roc_curve(y, probs)
    return met, fpr, tpr

def binarize(y_train, y_test):
    y_train = np.array(list(y_train.loc[:, 0]))
    y_test = np.array(list(y_test.loc[:, 0]))
    classes = set(y_train)
    binarized_y = dict()
    for c in classes:
        binarized_y[c] = {'y_train': (y_train==c).astype(int),
                          'y_test': (y_test==c).astype(int)}
    return binarized_y

def run(dataset_name, 
        test_size, 
        metric,
        n_bags, 
        n_iter,
        n_select,
        n_new_bags,
        n_mutation,
        mutation_rate,
        size_coef):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size)
    binarized_y = binarize(y_train, y_test)
    bagging_roc_dict = dict()
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    class_roc_dict = dict()
    for c in binarized_y:
        y_train = binarized_y[c]['y_train']
        y_test = binarized_y[c]['y_test']
        y_train = pd.DataFrame({0: y_train})
        y_test = pd.DataFrame({0:y_test})
        optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                                max_initial_size, n_crossover, n_mutation, 
                                mutation_size, size_coef, metric)
        # bagging FPR and TPR
        clf = BaggingClassifier(n_estimators=n_bags)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        bagging_fpr, bagging_tpr, _ = roc_curve(y_test, probs)
        bagging_roc_dict[c] = {'fpr': bagging_fpr, 'tpr': bagging_tpr}
        # init random bags of random sizes
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        # evaluate
        optimizer.evaluate_bags(bags)
        voting_test = []
        for i in tqdm(range(n_iter)):
            optimizer.evobagging_optimization(bags)
            met, fpr, tpr = voting_metric_roc(bags, X_test, y_test, metric)
            voting_test.append(met)
        class_roc_dict[c] = {'tpr': tpr, 'fpr': fpr}

    for c in class_roc_dict:
        current_tpr = list(class_roc_dict[c]['tpr'])[:]
        current_tpr.extend(list(bagging_roc_dict[c]['tpr'][:]))
        current_fpr = list(class_roc_dict[c]['fpr'][:])
        current_fpr.extend(list(bagging_roc_dict[c]['fpr'][:]))
        current_model = ['EvoBagging']*len(class_roc_dict[c]['tpr'])
        current_model.extend(['Bagging']*len(bagging_roc_dict[c]['tpr']))
        roc_df = pd.DataFrame({'True positive rate': current_tpr,
                               'False positive rate': current_fpr,
                               'Model': current_model})
        g = sns.lineplot(data=roc_df, x='False positive rate', y='True positive rate', hue='Model')
        if c != max(class_roc_dict):
            plt.legend([],[], frameon=False)
        g = g.get_figure()
        g.savefig(f'viz/{dataset_name}_roc_class{c}.png')
        plt.clf()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate ROC curves')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of test data')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Classification metric')
    parser.add_argument('--n_bags', type=int,
                        help='Number of bags')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations')
    parser.add_argument('--n_select', type=int, default=0,
                        help='Number of selected bags each iteration')
    parser.add_argument('--n_new_bags', type=int,
                        help='Generation gap')
    parser.add_argument('--n_mutation', type=int,
                        help='Number of bags to perform mutation on')
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='Percentage of mutated instances in each bag')
    parser.add_argument('--size_coef', type=float,
                        help='Constant K for controlling size')
    parser.add_argument('--procs', type=int, default=16,
                        help='Number of parallel processes')
    args = parser.parse_args()

    run(dataset_name=args.dataset_name, 
        test_size=args.test_size, 
        metric=args.metric,
        n_bags=args.n_bags, 
        n_iter=args.n_iter,
        n_select=args.n_select,
        n_new_bags=args.n_new_bags,
        n_mutation=args.n_mutation,
        mutation_rate=args.mutation_rate,
        size_coef=args.size_coef,
        procs=args.procs)