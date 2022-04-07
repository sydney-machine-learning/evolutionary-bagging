from data_processing import load_data
from evobagging_methods import *
import seaborn as sns
import yaml
import argparse
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

def run_dataset(dataset_name, 
                metric,
                n_bags, 
                n_iter,
                n_select,
                n_new_bags,
                n_mutation,
                mutation_rate,
                size_coef, 
                procs=4):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=0)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric, procs)
    voting_test = []
    w_voting_test = []
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    for i in range(n_iter):
        bags = optimizer.evobagging_optimization(bags)
        voting_test.append(optimizer.voting_metric(X_test, y_test, bags, False))
        w_voting_test.append(optimizer.voting_metric_weighted(X_test, y_test))
    return voting_test, w_voting_test

def run(dataset_name):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    voting_test, w_voting_test = run_dataset(dataset_name,
        metric='accuracy',
        n_bags=dataset_config['n_bags'], 
        n_iter=dataset_config['n_iter'],
        n_select=0,
        n_new_bags=dataset_config['n_new_bags'],
        n_mutation=dataset_config['n_mutation'],
        mutation_rate=0.05,
        size_coef=dataset_config['size_coef'], 
        procs=4
    )
    print_df = pd.DataFrame.from_dict({'majority voting': voting_test, 
                                       'weighted_voting': w_voting_test}, orient='index').T
    p = sns.lineplot(data=print_df)
    p.set_xticks(range(len(voting_test+1)))
    p.xaxis.set_major_locator(ticker.MultipleLocator(2))
    p.xaxis.set_major_formatter(ticker.ScalarFormatter())
    p.set_xlabel("Iteration")
    p.set_ylabel("Accuracy")
    fig = p.get_figure()
    p.set_xlabel("Iteration")
    p.set_ylabel("% of data covered by the bags")
    fig = p.get_figure()
    fig.savefig('images/ReduceCoverage.png')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment voting rule')
    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    args = parser.parse_args()
    run(args.dataset_name)