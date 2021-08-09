import random
import numpy as np
from decision_tree import eval_model

def mutate(tree):
    # node_list: dictionary of nodes
    # feature_range: list of n_features tuples (min_feat, max_feat)
    node = random.choice(tree.node_list)
    # case split node
    if node.feature is not None:
        node_feature_range = tree.feature_range[node.feature]
        min_feat, max_feat = node_feature_range[0], node_feature_range[1]
        node.threshold = random.uniform(min_feat, max_feat)
    else:
        node.value = random.choice(tree.target_values)
    
    
def crossover(tree1, tree2):
    # select nodes
    index1, index2 = 0, 0
    while index1 == 0:
        index1 = random.choice(range(len(tree1.node_list)))
        node1 = tree1.node_list[index1]
    while index2 == 0:
        index2 = random.choice(range(len(tree2.node_list)))
        node2 = tree2.node_list[index2]
    # swap subtrees
    # update parent
    parent1, parent2 = node1.parent, node2.parent
    node1.parent, node2.parent = parent2, parent1
    # update child
    if parent1.left==node1:
        parent1.left = node2
    else: 
        parent1.right = node2
    if parent2.left==node2:
        parent2.left = node1
    else: 
        parent2.right = node1
    # update trees and node_dicts
    tree1 = tree1.get_node_list(tree1.root, [])
    tree2 = tree2.get_node_list(tree2.root, [])
    return tree1, tree2


def ga_iteration(tree_list, 
                 eval_input, 
                 eval_target,
                 n_select,
                 n_crossover,
                 n_mutate):
    ga_trees = []
    eval_results = np.asarray([eval_model(tree, eval_input, eval_target) for tree in tree_list])
    # select
    select_indices = eval_results.argsort()[-n_select:][::-1]
    for i in select_indices:
        ga_trees.append(tree_list[i])
    # crossover
    crossover_trees = []
    select_indices = eval_results.argsort()[-n_crossover:][::-1]
    for i in select_indices:
        crossover_trees.append(tree_list[i])
    random.shuffle(crossover_trees)
    for i in range(len(crossover_trees), 2):
        tree1 = crossover_trees[i]
        tree2 = crossover_trees[i+1]
        crossover_trees[i:(i+2)] = crossover(tree1, tree2)
    ga_trees.extend(crossover_trees)
    # mutate
    select_indices = random.sample(range(len(ga_trees)), n_mutate)
    for i in select_indices:
        mutate(ga_trees[i])
    return ga_trees