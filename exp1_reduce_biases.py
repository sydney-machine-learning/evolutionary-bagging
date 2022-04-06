from data_processing import load_data
from evobagging_methods import *
import seaborn as sns
import yaml
import argparse

import warnings
warnings.filterwarnings("ignore")

def test_bias(bags, X, y):
    biases = []
    for biome in bags.values():
        preds = biome['clf'].predict(X)
        biases.append(sum(preds != y.iloc[:, 0])/len(preds))
    return biases

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
    bias_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])
    for i in range(n_iter):
        bags = optimizer.evobagging_optimization(bags)
        bias_df.loc[i + 1, :] = test_bias(bags, X_train, y_train)
    return list(bias_df.mean(axis=1))

def run(dataset_names):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    bias_df = []
    for dataset_name in dataset_names:
        dataset_config = configs[dataset_name]
        output = run_dataset(dataset_name,
                             metric='accuracy',
                             n_bags=dataset_config['n_bags'], 
                             n_iter=dataset_config['n_iter'],
                             n_select=0,
                             n_new_bags=dataset_config['n_new_bags'],
                             n_mutation=dataset_config['n_mutation'],
                             mutation_rate=0.05,
                             size_coef=dataset_config['size_coef'], 
                             procs=4)
        bias_df.append(output)
    bias_df = pd.DataFrame(bias_df).T
    bias_df.columns = dataset_names
    p = sns.lineplot(data=bias_df, dashes=False)
    p.set_xlabel('Iteration')
    p.set_ylabel('Average bias')
    fig = p.get_figure()
    fig.savefig(f"images/bias_{'_'.join(dataset_names)}.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment reducing bias')
    parser.add_argument('--dataset_names', nargs='+', help='<Required> List of dataset names', required=True)
    args = parser.parse_args()
