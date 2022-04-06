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
    X_train, _, y_train, _ = load_data(dataset_name, test_size=0.2)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric, procs)
    avg_fitness = []
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    avg_fitness.append(np.mean([bag["payoff"] for bag in bags.values()]))
    for i in range(n_iter):
        bags = optimizer.evobagging_optimization(bags)
        avg_fitness.append(np.mean([bag["payoff"] for bag in bags.values()]))
    return avg_fitness

def run(dataset_name, hyp, l_hyp):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    plot_df = pd.DataFrame(columns=["Value", "Iter", "Average Fitness"])
    for value in l_hyp:
        if hyp != "mutation_rate":
            dataset_config[hyp] = value
            mutation_rate = 0.05
        else:
            mutation_rate = value
        avg_fitness = run_dataset(dataset_name,
                                metric='accuracy',
                                n_bags=dataset_config['n_bags'], 
                                n_iter=dataset_config['n_iter'],
                                n_select=0,
                                n_new_bags=dataset_config['n_new_bags'],
                                n_mutation=dataset_config['n_mutation'],
                                mutation_rate=mutation_rate,
                                size_coef=dataset_config['size_coef'], 
                                procs=4)
        new_df = pd.DataFrame({"Hyperparam value": [value]*(dataset_config['n_iter'] + 1),
                                "Iter": range(dataset_config['n_iter'] + 1),
                                "Average Fitness": avg_fitness})
        plot_df = pd.concat([plot_df, new_df], axis=0, ignore_index=True)
    p = sns.lineplot(x="Iter", y="Average Fitness", hue="Hyperparam value", data=plot_df, dashes=False)
    p.set_xlabel("Iteration")
    p.set_ylabel("Average Fitness")
    fig = p.get_figure()
    fig.savefig(f"images/hyp_{hyp}_{dataset_name}.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment reducing bias')
    parser.add_argument('--hyp', type=str, default="mutation_rate",
                        help="Hyperparams to experiment")
    parser.add_argument('--l_hyp', type=int, default=[0.01, 0.05, 0.06, 0.1, 0.2], 
                        help='<Required> List of values for hyperparam')
    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    args = parser.parse_args()
    run(args.dataset_name, args.hyp, args.l_hyp)