from data_processing import load_data
from evobagging_methods import *
import yaml
import argparse
from sklearn.ensemble import BaggingClassifier
import warnings
from exp3_diversity import get_diversity_measures
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
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    bagging_preds = np.zeros((y_test.shape[0], 10))
    evobagging_preds = np.zeros((y_test.shape[0], 10))
    diversity_measures = ["Prediction variance",
                          "Q statistics",
                          "Disagreement",
                          "Double fault",
                          "KW variance",
                          "Entropy",
                          "Generalized diversity"]
    # Bagging variance
    for k in range(10):
        new_train_idx = random.choices(range(X_train.shape[0]), k=len(X_train))
        new_test_idx = random.choices(range(X_test.shape[0]), k=len(X_test))
        new_X_train = X_train.loc[new_train_idx, :].reset_index(drop=True)
        new_X_test = X_test.loc[new_test_idx, :].reset_index(drop=True)
        new_y_train = y_train.loc[new_train_idx, :]
        new_y_test = y_test.loc[new_test_idx, :]
        clf = BaggingClassifier(n_estimators=n_bags)
        new_y_train.reset_index(inplace=True, drop=True)
        new_y_test.reset_index(inplace=True, drop=True)
        clf.fit(new_X_train, new_y_train.values.ravel())
        bagging_preds[:, k] = clf.predict(new_X_test)
        optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                                max_initial_size, n_crossover, n_mutation, 
                                mutation_size, size_coef, metric, procs)
        # init random bags of random sizes
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        # evaluate
        bags = optimizer.evaluate_bags(bags)
        for i in range(n_iter):
            bags = optimizer.evobagging_optimization(bags)
        _, evobagging_preds[:, k] = optimizer.voting_metric(new_X_test, new_y_test, bags, True)
    for model in ["bagging", "evobagging"]:
        print(model)
        model_diversity_measures = get_diversity_measures(eval(f"{model}_preds"), new_y_test.to_numpy())
        for i, measure in enumerate(diversity_measures):
            print(measure + ": ", model_diversity_measures[i])

def run(dataset_name):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    run_dataset(dataset_name,
                metric='accuracy',
                n_bags=dataset_config['n_bags'], 
                n_iter=dataset_config['n_iter'],
                n_select=0,
                n_new_bags=dataset_config['n_new_bags'],
                n_mutation=dataset_config['n_mutation'],
                mutation_rate=0.05,
                size_coef=dataset_config['size_coef'], 
                procs=8)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment variance')
    parser.add_argument('--dataset_name', help='Dataset name', required=True)
    args = parser.parse_args()
    run(args.dataset_name)