from data_processing import load_data
from evobagging_methods import *
import seaborn as sns
import yaml
import argparse

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
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=0.2)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric, procs)
    avg_size, avg_depth = 0, 0
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    for i in range(n_iter):
        bags = optimizer.evobagging_optimization(bags)
    avg_size = np.mean([bag["size"] for bag in bags.values()])
    avg_depth = np.mean([bag["clf"].get_depth() for bag in bags.values()]) 
    met = optimizer.voting_metric(X_test, y_test, bags, False)   
    return avg_size, avg_depth, met

def run(dataset_name, hyp, l_hyp):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    for value in l_hyp:
        avg_size, avg_depth, met = run_dataset(dataset_name,
                                metric='accuracy',
                                n_bags=dataset_config['n_bags'], 
                                n_iter=dataset_config['n_iter'],
                                n_select=0,
                                n_new_bags=dataset_config['n_new_bags'],
                                n_mutation=dataset_config['n_mutation'],
                                mutation_rate=0.05,
                                size_coef=value, 
                                procs=4)
        print(value, avg_size, avg_depth, met)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment reducing bias')
    parser.add_argument('--hyp', type=str, default="size_coef",
                        help="Hyperparams to experiment")
    parser.add_argument('--l_hyp', type=int, default=[100, 1000, 2000, 10000, 20000], 
                        help='<Required> List of values for hyperparam')
    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    args = parser.parse_args()
    run(args.dataset_name, args.hyp, args.l_hyp)