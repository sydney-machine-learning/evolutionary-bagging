from data_processing import load_data
from evobagging_methods import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
import argparse

def run(dataset_name, 
        test_size, 
        n_exp,
        metric,
        n_bags, 
        n_iter,
        n_select,
        n_new_bags,
        n_mutation,
        mutation_rate,
        size_coef):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric)
    all_voting_train, all_voting_test = [], []
    depth_evobagging, depth_bagging = [], []
    for t in range(n_exp):
        # init random bags of random sizes
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        # evaluate
        payoff_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])
        depth_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])  
        optimizer.evaluate_bags(bags)
        payoff_df.loc[0, :] = [round(bags[j]['payoff']*100, 1) for j in range(n_bags)]
        voting_train, voting_test = [], []
        for i in tqdm(range(n_iter)):
            bags = optimizer.evobagging_optimization(bags, X_train, y_train, n_select, n_new_bags, 
                                           max_initial_size, n_crossover, n_mutation, 
                                           mutation_size, size_coef, metric)
            payoff_df.loc[i + 1, :] = [round(bags[j]['payoff']*100, 1) for j in range(n_bags)]
            depth_df.loc[i+1, :] = [bags[j]['clf'].get_depth() for j in range(n_bags)]
            voting_train.append(round(optimizer.voting_metric(X_train, y_train, bags)*100, 2))
            voting_test.append(round(optimizer.voting_metric(X_test, y_test, bags)*100, 2))
        # plot fitness
        print_df = payoff_df.mean(axis=1)
        p = sns.lineplot(data=print_df)
        p.set_xlabel("iteration")
        p.set_ylabel("fitness")
        p.xaxis.set_major_locator(ticker.MultipleLocator(5))
        p.xaxis.set_major_formatter(ticker.ScalarFormatter())
        fig = p.get_figure()
        fig.savefig(f'viz/{dataset_name}_fitness_{t}.png')
        plt.clf()
        # get accuracy
        all_voting_train.append(voting_train[-1])
        all_voting_test.append(voting_test[-1])
        # get depth
        avg_depth_evobagging = depth_df.loc[n_iter, :].mean()
        depth_evobagging.append(avg_depth_evobagging)
        clf = BaggingClassifier(n_estimators=n_bags)
        clf.fit(X_train, y_train)
        bagging_depths = [e.get_depth() for e in clf.estimators_]
        avg_depth_bagging = np.mean(bagging_depths)
        depth_bagging.append(avg_depth_bagging)

    print("Accuracy")
    print("Train: ", np.mean(all_voting_train), np.std(all_voting_train))
    print("Test: ", np.mean(all_voting_test), np.std(all_voting_test))
    print("Depth")
    print("EvoBagging: ", np.mean(depth_evobagging), np.std(depth_evobagging))
    print("Bagging: ", np.mean(depth_bagging), np.std(depth_bagging))
    # return print_df for plotting multiple fitness line plots in Jupyter notebook
    return print_df

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Main experiment')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of test data')
    parser.add_argument('--n_exp', type=int, default=30,
                        help='Number of experiments')
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

    args = parser.parse_args()

    run(dataset_name=args.dataset_name, 
        test_size=args.test_size, 
        n_exp=args.n_exp,
        metric=args.metric,
        n_bags=args.n_bags, 
        n_iter=args.n_iter,
        n_select=args.n_select,
        n_new_bags=args.n_new_bags,
        n_mutation=args.n_mutation,
        mutation_rate=args.mutation_rate,
        size_coef=args.size_coef)