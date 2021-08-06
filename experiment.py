import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import random
from decision_tree import *
from ga_methods import mutate, crossover
from ab_methods import adaboost_single_tree
from tqdm import tqdm

class Experiment():
    def __init__(self,
                 test_size=0.2,
                 p=50,
                 r=0.2,
                 m=0.2,
                 step=100):
        self.test_size = test_size
        self.p = p
        self.r = r
        self.m = m
        self.step = step
        self.n_crossover = int(self.r*self.p)
        self.n_select = self.p - self.n_crossover
        self.n_mutate = int(self.m*self.p)
        assert self.n_crossover%2==0, "Odd n_crossover"
        print(f"Select {self.n_select} | Crossover {self.n_crossover} | Mutate {self.n_mutate}")

    def load_data(self):
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, self.test_size, random_state=1)
        print(f"Number of features: {self.X.shape[1]}")
        print(f"Train samples: {self.X_train.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        unique_labels, count_train_labels = np.unique(self.y_train, return_counts=True)
        _, count_test_labels = np.unique(self.y_test, return_counts=True)
        print(f"Labels: {list(unique_labels)}")
        print(f"Count train labels: {list(count_train_labels)}")
        print(f"Coun test labels: {list(count_test_labels)}")

    def eval_model(self, clf, X, y):
        y_test_preds = clf.predict(X)
        test_acc = accuracy_score(y, y_test_preds)
        return test_acc

    def init_bagging_trees(self, X, y, X_train, y_train, X_test, y_test):
        # Initial trees from bagging
        bagging_clf = BaggingClassifier(n_estimators=self.p)
        bagging_clf.fit(X_train, y_train)
        print('Bagging accuracy: ', self.eval_model(bagging_clf, X_test, y_test))
        trees = bagging_clf.estimators_
        trees = [construct_tree(t, X, y) for t in trees]
        return trees

    def run_experiment(self, trees):
        ga_acc_list = []
        ab_acc_list = []
        ga_select = []
        ab_select = []
        for i in tqdm(range(self.step)):
            # GA step
            ga_trees = []
            eval_results = np.asarray([self.eval_model(tree, self.X_test, self.y_test) for tree in trees])
            # select
            select_indices = eval_results.argsort()[-self.n_select:][::-1]
            for i in select_indices:
                ga_trees.append(trees[i])
            # crossover
            crossover_trees = []
            select_indices = eval_results.argsort()[-self.n_crossover:][::-1]
            for i in select_indices:
                crossover_trees.append(trees[i])
            random.shuffle(crossover_trees)
            for i in range(len(crossover_trees), 2):
                tree1 = crossover_trees[i]
                tree2 = crossover_trees[i+1]
                crossover_trees[i:(i+2)] = crossover(tree1, tree2)
            ga_trees.extend(crossover_trees)
            # mutate
            select_indices = random.sample(range(len(ga_trees)), self.n_mutate)
            for i in select_indices:
                mutate(ga_trees[i])

                
            # AdaBoost step
            ab_trees = [adaboost_single_tree(self.X_train, self.y_train, tree) for tree in trees]
                                    
                                    
            # combine tree
            forest = []
            forest.extend(ga_trees)
            forest.extend(ab_trees)
            forest_eval_results = np.asarray([self.eval_model(tree, self.X_test, self.y_test) for tree in forest])
            select_indices = forest_eval_results.argsort()[-self.p:][::-1]
            trees = [forest[i] for i in select_indices]
            ga_acc = round(forest_eval_results[:self.p].mean()*100, 2)
            ab_acc = round(forest_eval_results[self.p:].mean()*100, 2)
            ga_acc_list.append(ga_acc)
            ab_acc_list.append(ab_acc)
            ga_select.append(sum(forest_eval_results < self.p))
            ab_select.append(sum(forest_eval_results >= self.p))
        gaen_y_test_preds = bagging_predict(trees, self.X_test)
        gaen_acc = accuracy_score(self.y_test, gaen_y_test_preds)
        return gaen_acc, ga_acc_list, ab_acc_list, ga_select, ab_select

    def report_results(self, gaen_acc, ga_acc_list, ab_acc_list, ga_select, ab_select):
        plt.plot(range(self.step), ga_acc_list, label = "GA")
        plt.plot(range(self.step), ab_acc_list, label = "AB")
        plt.xlabel('step')
        # Set the y axis label of the current axis.
        plt.ylabel('accuracy')
        # Set a title of the current axes.
        plt.title('GA vs AB: accuracy')
        # show a legend on the plot
        plt.legend()
        plt.savefig(f"Accuracy.png")
        plt.plot(range(self.step), ga_select, label = "GA")
        plt.plot(range(self.step), ab_select, label = "AB")
        plt.xlabel('step')
        # Set the y axis label of the current axis.
        plt.ylabel('select')
        # Set a title of the current axes.
        plt.title('GA vs AB: select')
        # show a legend on the plot
        plt.legend()
        plt.savefig(f"Select.png")
        print(f"Gaen accuracy: {round(gaen_acc*100), 2}")

    def main(self):
        self.load_data()
        trees = self.init_bagging_trees()
        gaen_acc, ga_acc_list, ab_acc_list, ga_select, ab_select = self.run_experiment(trees)
        self.report_results(gaen_acc, ga_acc_list, ab_acc_list, ga_select, ab_select)
        
if __name__=="__main__":
    exp = Experiment()
    exp.main()