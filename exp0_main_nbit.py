from data_processing import Object, load_data
from evobagging_methods import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
import xgboost as xgb
import argparse

def test_algo_nbit(dataset_name, 
                  n_exp,
                  metric,
                  n_bags, 
                  n_iter,
                  n_select,
                  n_new_bags,
                  n_mutation,
                  mutation_rate,
                  size_coef, 
                  procs=4):
    X_train, _, y_train, _ = load_data(dataset_name, 0)
    # random forest
    train_rf_metrics = []
    for i in range(n_exp):
        clf = RandomForestClassifier(n_estimators=n_bags) 
        clf.fit(X_train, y_train.values.ravel())
        train_preds = clf.predict(X_train)
        train_rf_metrics.append(eval(f"{metric}_score(y_train, train_preds)"))

    # bagging performance
    train_bagging_scores = []
    for i in range(n_exp):
        clf = BaggingClassifier(n_estimators=n_bags)     
        clf.fit(X_train, y_train.values.ravel())
        train_preds = clf.predict(X_train)
        train_bagging_scores.append(eval(f"{metric}_score(y_train, train_preds)"))
    # extratrees performance
    train_extrees_scores = []
    for i in range(n_exp):
        clf = ExtraTreesClassifier(n_estimators=n_bags)     
        clf.fit(X_train, y_train.values.ravel())
        train_preds = clf.predict(X_train)
        train_extrees_scores.append(eval(f"{metric}_score(y_train, train_preds)"))
    # xgboost performance
    train_xgb_scores = []
    for i in range(n_exp):
        if len(y_train.loc[:, 0].unique()) > 2:
            clf = xgb.XGBClassifier(n_estimators=n_bags, objective='multi:softprob')
        else:
            clf = xgb.XGBClassifier(n_estimators=n_bags)  
        clf.fit(X_train, y_train.values.ravel(), eval_metric='mlogloss')
        train_preds = clf.predict(X_train)
        train_xgb_scores.append(eval(f"{metric}_score(y_train, train_preds)"))
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric, procs)
    all_voting_train = []
    for t in tqdm(range(n_exp)):
        # init random bags of random sizes
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        # evaluate
        payoff_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])
        depth_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])  
        bags = optimizer.evaluate_bags(bags)
        payoff_df.loc[0, :] = [round(bags[j]['payoff']*100, 1) for j in range(n_bags)]
        voting_train, voting_test = [], []
        for i in range(n_iter):
            bags = optimizer.evobagging_optimization(bags)
            payoff_df.loc[i + 1, :] = [round(bags[j]['payoff']*100, 1) for j in range(n_bags)]
            depth_df.loc[i+1, :] = [bags[j]['clf'].get_depth() for j in range(n_bags)]
            voting_train.append(round(optimizer.voting_metric(X_train, y_train, bags)*100, 2))
        best_iter = np.argmax(voting_train)
        all_voting_train.append(voting_train[best_iter])
    evo_acc = np.mean(all_voting_train)
    bag_acc = np.mean(train_bagging_scores)*100
    rf_acc = np.mean(train_rf_metrics)*100
    extrees_acc = np.mean(train_extrees_scores)*100
    xgb_acc = np.mean(train_xgb_scores)*100
    return evo_acc, bag_acc, rf_acc, extrees_acc, xgb_acc

def test_plot_nbit(dataset_name, 
                  start=10,
                  stop=100,
                  procs=4):
    evo = []
    bag = []
    rf = []
    extrees = []
    xgboost = []
    for n in range(start, stop, 2):
        n_new_bags = int(n*0.2)
        if n_new_bags%2 != 0:
            n_new_bags += 1
        n_mutation = int(n/10)
        mutation_rate = 0.05
        size_coef = 100 if dataset_name=='6bit' else 1000
        evo_acc, bag_acc, rf_acc, extrees_acc, xgb_acc = test_algo_nbit(dataset_name, 
                                                                        30,
                                                                        'accuracy',
                                                                        n, 
                                                                        5,
                                                                        0,
                                                                        n_new_bags,
                                                                        n_mutation,
                                                                        mutation_rate,
                                                                        size_coef, 
                                                                        procs)
        evo.append(evo_acc)
        bag.append(bag_acc)
        rf.append(rf_acc)
        extrees.append(extrees_acc)
        xgboost.append(xgb_acc)
    print_df = pd.DataFrame({'EvoBagging': evo, 
                             'Bagging': bag, 
                             'Random Forest': rf,
                             'ExtraTrees': extrees,
                             'XGBoost': xgboost})
    print_df.index = range(start, stop, 2)
    p = sns.lineplot(data=print_df)
    p.set_xlabel("Number of bags")
    p.set_ylabel("Accuracy")
    p.xaxis.set_major_locator(ticker.MultipleLocator(10))
    p.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig = p.get_figure()
    fig.savefig(f'viz/{dataset_name}.png')
    plt.clf()
    return print_df

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Main experiment for nbit')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--start_bags', type=int, default=10,
                        help='Number of starting bags')
    parser.add_argument('--end_bags', type=int, default=100,
                        help='Number of ending bags')                        
    parser.add_argument('--n_exp', type=int, default=30,
                        help='Number of experiments')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Classification metric')
    parser.add_argument('--procs', type=int, default=16,
                        help='Number of parallel processes')
    args = parser.parse_args()

    test_plot_nbit(dataset_name=args.dataset_name, 
                   start_bags=args.start_bags,
                   end_bags=args.end_bags, 
                   n_exp=args.n_exp,
                   metric=args.metric,
                   procs=args.procs)
