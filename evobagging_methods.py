from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random
from scipy import stats
import copy
from multiprocessing import Pool
from functools import partial

class EvoBagging:
    def __init__(self, X_train, y_train, 
                n_select, n_new_bags, 
                max_initial_size, n_crossover, 
                n_mutation, mutation_size, 
                size_coef, metric, procs=4):
        self.X_train = X_train
        self.y_train = y_train
        self.n_select = n_select
        self.n_new_bags = n_new_bags
        self.max_initial_size = max_initial_size
        self.n_crossover = n_crossover
        self.n_mutation = n_mutation
        self.mutation_size = mutation_size
        self.size_coef = size_coef
        self.metric = metric
        self.procs = procs

    def get_score(self, X, y):
        clf = DecisionTreeClassifier(max_depth=X.shape[1])
        clf.fit(X, y)
        preds = clf.predict(X)
        perf = eval(f"{self.metric}_score(y, preds)")
        return perf, clf, preds
        
    def get_payoff(self, bags, idx):
        met, clf, preds = self.get_score(bags[idx]['X'], bags[idx]['y'], self.metric)
        size_multiply = (self.size_coef+bags[idx]['X'].shape[0])/self.size_coef
        payoff = met * size_multiply
        bags[idx]['clf'] = clf
        bags[idx]['metric'] = met
        bags[idx]['preds'] = preds
        bags[idx]['payoff'] = payoff
        bags[idx]['size'] = bags[idx]['X'].shape[0]
        return payoff, met

    def naive_selection(self, bags):
        selected_bag_dict = {}
        selected_ids = []
        bag_idx, payoff_list = [], []
        for idx, bag in bags.items():
            bag_idx.append(idx)
            payoff_list.append(bag['payoff'])
        selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx), reverse=True)][:self.n_select]
        selected_bag_dict = {i: bags[i] for i in selected_ids}
        return selected_bag_dict, selected_ids

    def gen_new_bag(self):
        initial_size = random.randrange(int(self.max_initial_size/2), self.max_initial_size)
        bag_idx = random.choices(list(self.y_train.index), k=initial_size)
        temp_X = self.X_train.loc[bag_idx, :]
        temp_y = self.y_train.loc[bag_idx, :]
        return {'X': temp_X, 'y': temp_y}

    def generation_gap(self, new_bags, bags):
        for _ in range(self.n_new_bags):
            new_bag = self.gen_new_bag(self.X_train, self.y_train, self.max_initial_size)
            new_bag_idx = random.choice(list(set(range(len(bags))) - set(new_bags.keys())))
            new_bags[new_bag_idx] = new_bag
        return new_bags

    def crossover_with_instance_prob(self, parent1, parent2):    
        preds_1 = parent1['preds']
        wrong_idx_1 = preds_1 != parent1['y'][0]
        parent1_leave_idx = parent1['X'].index[wrong_idx_1]
        preds_2 = parent1['preds']
        wrong_idx_2 = preds_2 != parent2['y'][0]
        parent2_leave_idx = parent2['X'].index[wrong_idx_2]
        new_parent1_X = parent1['X'].loc[~parent1['X'].index.isin(parent1_leave_idx)]
        leave_parent1_X = parent1['X'].loc[parent1['X'].index.isin(parent1_leave_idx)]
        new_parent1_y = parent1['y'].loc[~parent1['y'].index.isin(parent1_leave_idx)]
        leave_parent1_y = parent1['y'].loc[parent1['y'].index.isin(parent1_leave_idx)]
        new_parent2_X = parent2['X'].loc[~parent2['X'].index.isin(parent2_leave_idx)]
        leave_parent2_X = parent2['X'].loc[parent2['X'].index.isin(parent2_leave_idx)]
        new_parent2_y = parent2['y'].loc[~parent2['y'].index.isin(parent2_leave_idx)]
        leave_parent2_y = parent2['y'].loc[parent2['y'].index.isin(parent2_leave_idx)]  

        child1, child2 = {}, {}
        child1['X'] = pd.concat([new_parent1_X, leave_parent2_X])
        child1['y'] = pd.concat([new_parent1_y, leave_parent2_y])
        child2['X'] = pd.concat([new_parent2_X, leave_parent1_X])
        child2['y'] = pd.concat([new_parent2_y, leave_parent1_y])

        return child1, child2

    def crossover(self, new_bags, bags):
        _, crossover_pool_idx = self.naive_selection(bags, self.n_crossover)
        random.shuffle(crossover_pool_idx)
        remaining_idx = list(set(range(len(bags))) - set(new_bags.keys()))
        random.shuffle(remaining_idx)
        for j in range(0, self.n_crossover, 2):
            parent1 = bags[crossover_pool_idx[j]]
            parent2 = bags[crossover_pool_idx[j + 1]]
            child1, child2 = self.crossover_with_instance_prob(parent1, parent2)
            new_bags[remaining_idx[j]] = child1
            new_bags[remaining_idx[j + 1]] = child2
        
        return new_bags

    def mutation(self, bags):
        bag_mutation_idx = random.sample(list(bags.keys()), k=self.n_mutation)
        for j in bag_mutation_idx:
            bag_idx = bags[j]['y'].index
            leftover_idx = list(set(self.X_train.index) - set(bag_idx))
            leave_idx = random.sample(list(bag_idx), k=self.mutation_size)
            new_idx = random.choices(list(leftover_idx), k=self.mutation_size)
            keep_bag_X = bags[j]['X'].loc[~bag_idx.isin(leave_idx)]
            keep_bag_y = bags[j]['y'].loc[~bag_idx.isin(leave_idx)]
            new_bag_X = self.X_train.loc[new_idx]
            new_bag_y = self.y_train.loc[new_idx]
            bags[j]['X'] = pd.concat([keep_bag_X, new_bag_X])
            bags[j]['y'] = pd.concat([keep_bag_y, new_bag_y])
        return bags, bag_mutation_idx

    def evaluate_bags(self, bags):
        with Pool(self.procs) as p:
            p.map(partial(self.get_payoff, bags), list(bags.keys()))

    def voting_metric(self, X, y, bags):
        preds_list = []
        for bag in bags.values():
            bag['clf'] = DecisionTreeClassifier()
            bag['clf'].fit(bag['X'], bag['y'])
            bag_preds = bag['clf'].predict(X)
            preds_list.append(bag_preds)
        temp_preds = np.stack(preds_list)
        final_preds = stats.mode(temp_preds).mode[0]
        met = eval(f"{self.metric}_score(y, final_preds)")
        return met

    def evobagging_optimization(self, bags):
        # selection
        new_bags, _ = self.naive_selection(bags)
        # generation gap
        new_bags = self.generation_gap(new_bags, bags)
        # crossover
        new_bags = self.crossover(new_bags, bags)
        # mutation
        new_bags, _ = self.mutation(new_bags)
        # update population
        bags = copy.deepcopy(new_bags)
        # evaluate
        self.evaluate_bags(bags, self.size_coef, self.metric)
        return bags