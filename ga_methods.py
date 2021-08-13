import random
import numpy as np
from decision_tree import eval_tree

def roulette_wheel_selection(tree_list, pay_off_list, n_select):
    sum_pay_off = sum(pay_off_list)
    probs = [pay_off/sum_pay_off for pay_off in pay_off_list]
    selected_tree_list = random.choices(tree_list, probs, k=n_select)
    return selected_tree_list

def tournament(tree_list, pay_off_list, n_select, tournament_size=0.1):
    tournament_size = int(tournament_size*len(tree_list))
    # deterministic tournament
    selected_tree_list = []
    for _ in range(n_select):
        tournament_round_ids = random.choices(range(len(tree_list)), k=tournament_size)
        tournament_payoffs = [pay_off_list[j] for j in tournament_round_ids]
        max_tournament_payoff = max(tournament_payoffs)
        selected_id = [j for j in tournament_round_ids if pay_off_list[j] == max_tournament_payoff][0]
        selected_tree_list.append(tree_list[selected_id])
    return selected_tree_list

def crossover(tree1, tree2):
    # select nodes
    node_inacc1 = [node.incorrect/(node.correct + node.incorrect + 1e-7) for node in tree1.node_list]
    node_inacc2 = [node.incorrect/(node.correct + node.incorrect + 1e-7) for node in tree2.node_list]
    index1 = random.choices(range(len(tree1.node_list)), weights=node_inacc1)[0]
    node1 = tree1.node_list[index1]
    index2 = random.choices(range(len(tree2.node_list)), weights=node_inacc2)[0]
    node2 = tree2.node_list[index2]
    # swap subtrees
    # update parent
    parent1, parent2 = node1.parent, node2.parent
    node1.parent, node2.parent = parent2, parent1
    # update child
    if parent1 is not None:
        if parent1.left==node1:
            parent1.left = node2
        else:
            parent1.right = node2
    else:
        tree1.root = node2
    if parent2 is not None:
        if parent2.left==node2:
            parent2.left = node1
        else:
            parent2.right = node1
    else:
        tree2.root = node1
    # update trees and node_dicts
    tree1.node_list = tree1.get_node_list(tree1.root, [])
    tree2.node_list = tree2.get_node_list(tree2.root, [])
    return tree1, tree2

def mutate(tree):
    # node_list: dictionary of nodes
    # feature_range: list of n_features tuples (min_feat, max_feat)
    node_inacc = [node.incorrect/(node.correct + node.incorrect + 1e-7) for node in tree.node_list]
    node = random.choices(tree.node_list, weights=node_inacc)[0]
    # case split node
    if node.feature is not None:
        node_feature_range = tree.feature_range[node.feature]
        min_feat, max_feat = node_feature_range[0], node_feature_range[1]
        node.threshold = random.uniform(min_feat, max_feat)
    else:
        node.value = random.choice(tree.target_values)

def get_tree_scaled_payoff(index, tree_list, X, y, x):
    tree = tree_list[index]
    acc = eval_tree(tree, X, y)
    size = tree.get_size()
    payoff = (acc**2)*(x/(size**2+x))
    diff = 0
    for i, other_tree in enumerate(tree_list):
        if i != index:
            n_node_diff = abs(tree.get_size() - other_tree.get_size())
            tree_depth = tree.get_depth()
            n_level_diff = abs(tree_depth - other_tree.get_depth())
            diff += n_node_diff/(2*len(tree.node_list)) + n_level_diff/(2*tree_depth)
        else:
            continue
    mean_diff = (diff + 1e-7)/(len(tree_list) - 1)
    scaled_payoff = payoff*mean_diff
    return scaled_payoff