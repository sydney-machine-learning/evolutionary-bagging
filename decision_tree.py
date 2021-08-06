from __future__ import division, print_function
import numpy as np
import math
from scipy.stats import mode

def divide_on_feature(X, feature, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature] >= threshold
    else:
        split_func = lambda sample: sample[feature] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance


class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    left: DecisionNode
        Next decision node for samples where features value met the threshold.
    right: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature=None, threshold=None,
                 value=None, parent=None, left=None, right=None):
        self.feature = feature          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.parent = None
        self.left = left      # 'Left' subtree
        self.right = right    # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.find_parent(self.root, None)
        self.node_list = self.get_node_list(self.root, [])
        self.feature_range = self.get_feature_range(X)
        self.target_values = self.get_target_values()
        self.loss = None

    def get_feature_range(self, X):
        feature_range = []
        for i in range(X.shape[-1]):
            feature_range.append([min(X[:, i]),
                                max(X[:, i])])
        return feature_range

    def get_target_values(self):
        targets = [node.value for node in self.node_list if node.value is not None]
        return list(set(targets))

    def find_parent(self, node, parent):
        node.parent = parent
        # case leaf node
        if node.feature is None:
            pass
        # case split node
        else:
            self.find_parent(node.left, node)
            self.find_parent(node.right, parent)

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature in range(n_features):
                # All values of feature
                feature_values = np.expand_dims(X[:, feature], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature
                    # meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature": feature, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            left = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature=best_criteria["feature"], threshold=best_criteria[
                                "threshold"], left=left, right=right)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    def get_node_list(self, current_node, node_list):
        if current_node is None:
            return node_list
        if current_node not in node_list:
            node_list.append(current_node)
            if current_node.left is not None:
                node_list = self.get_node_list(current_node.left, node_list)
            else:
                node_list = self.get_node_list(current_node.parent, node_list)
        else:
            if current_node.right not in node_list:
                node_list = self.get_node_list(current_node.right, node_list)
            else:
                node_list = self.get_node_list(current_node.parent, node_list)
        return node_list

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature]
        # Determine if we will follow left or right branch
        branch = tree.right
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left
        elif feature_value == tree.threshold:
            branch = tree.left

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y1) - (1 - p) * \
            calculate_entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)

def bagging_predict(tree_list, X):
    all_preds = []
    for tree in tree_list:
        preds = tree.predict(X)
        all_preds.append(preds)
    all_preds = np.asarray(all_preds)
    major_preds = mode(all_preds, axis=0).mode[0]
    return major_preds

def construct_tree(sklearn_tree, X, y):
    children_left = sklearn_tree.tree_.children_left
    children_right = sklearn_tree.tree_.children_right
    feature_list = sklearn_tree.tree_.feature
    threshold_list = sklearn_tree.tree_.threshold
    value_list = sklearn_tree.tree_.value
    def construct_node(index):
        left_index = children_left[index]
        right_index = children_right[index]
        feature = feature_list[index]
        threshold = threshold_list[index]
        if feature >= 0:
            value = None
        else:
            value = value_list[index]
            value = sklearn_tree.classes_[np.argmax(value)]
        # case leaf
        if left_index < 0:
            node = DecisionNode(feature=None,
                                threshold=threshold,
                                value=value,
                                parent=None,
                                left=None,
                                right=None)
        # case split node
        else:
            left = construct_node(left_index)
            right = construct_node(right_index)
            node = DecisionNode(feature=feature,
                                threshold=threshold,
                                value=None,
                                parent=None,
                                left=left,
                                right=right)
        return node
    # Gather tree
    root = construct_node(0)
    tree = DecisionTree()
    tree.root = root
    tree.one_dim = len(np.shape(y)) == 1
    tree.find_parent(tree.root, None)
    tree.node_list = tree.get_node_list(tree.root, [])
    tree.feature_range = tree.get_feature_range(X)
    tree.target_values = tree.get_target_values()
    return tree