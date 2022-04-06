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
    X_train, _, y_train, _ = load_data(dataset_name, test_size=0)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric, procs)
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    cover_list = []
    cover_idx = set()
    for bag in bags.values():
        cover_idx = cover_idx.union(set(bag['X'].index))
    cover = len(cover_idx)/len(X_train)
    for i in range(n_iter):
        cover_list.append(cover)
        bags = optimizer.evobagging_optimization(bags)
        cover_idx = set()
        for bag in bags.values():
            cover_idx = cover_idx.union(set(bag['X'].index))
        cover = len(cover_idx)/len(X_train)
    return cover_list

def run(dataset_name, l_n_select):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    cover_s = []
    for n_select in l_n_select:
        cover_list = run_dataset(dataset_name,
                            metric='accuracy',
                            n_bags=dataset_config['n_bags'], 
                            n_iter=dataset_config['n_iter'],
                            n_select=n_select,
                            n_new_bags=dataset_config['n_new_bags'],
                            n_mutation=dataset_config['n_mutation'],
                            mutation_rate=0.05,
                            size_coef=dataset_config['size_coef'], 
                            procs=4)
        cover_s.append(cover_list)
    cover_s = pd.DataFrame(cover_s).T
    cover_s.columns = n_select
    p = sns.lineplot(data=cover_s, dashes=False)
    p.set_xlabel("Iteration")
    p.set_ylabel("% of data covered by the bags")
    fig = p.get_figure()
    fig.savefig('images/ReduceCoverage.png')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment selection')
    parser.add_argument('--l_n_select', type=int, default=[8, 12, 16], 
                        help='<Required> List of selection number', required=True)
    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    args = parser.parse_args()
    run(args.dataset_name, args.l_n_select)