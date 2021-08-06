import random

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