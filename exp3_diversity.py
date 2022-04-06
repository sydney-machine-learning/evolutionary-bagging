from data_processing import load_data
from evobagging_methods import *
import yaml
import argparse
from sklearn.ensemble import BaggingClassifier
import warnings
from diversity import *
warnings.filterwarnings("ignore")

def get_diversity_measures(preds, y):
    pred_var = prediction_variance(preds)
    q_stats = q_statistics(preds, y)
    disag = disagreement(preds, y)
    dfault = double_fault(preds, y)
    kw_var = kohavi_wolpert_variance(preds, y)
    e = entropy(preds, y)
    gen_div = generalized_diversity(preds, y)
    return pred_var, q_stats, disag, dfault, kw_var, e, gen_div

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
    diversity_measures = ["Prediction variance",
                          "Q statistics",
                          "Disagreement",
                          "Double fault",
                          "KW variance",
                          "Entropy",
                          "Generalized diversity"]
    bagging_preds = np.zeros((y_test.shape[0], n_bags))
    evobagging_preds = np.zeros((y_test.shape[0], n_bags))
    clf = BaggingClassifier(n_estimators=n_bags)
    clf.fit(X_train, y_train.values.ravel())
    for i, est in enumerate(clf.estimators_):
        preds = est.predict(X_test)
        bagging_preds[:, i] = preds
    optimizer = EvoBagging(X_train, y_train, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, metric, procs)
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    for i in range(n_iter):
        bags = optimizer.evobagging_optimization(bags)

    for i, bag in bags.items():
        preds = bag['clf'].predict(X_test)
        evobagging_preds[:, i] = preds
    for model in ["bagging", "evobagging"]:
        print(model)
        model_diversity_measures = get_diversity_measures(eval(f"{model}_preds"), y_test.to_numpy())
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
                procs=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment variance')
    parser.add_argument('--dataset_name', help='Dataset name', required=True)
    args = parser.parse_args()
    run(args.dataset_name)