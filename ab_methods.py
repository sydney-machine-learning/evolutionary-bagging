import numpy as np
from decision_tree import construct_tree
from sklearn.tree import DecisionTreeClassifier

def compute_error(y, y_pred, w_i):
    '''
    Calculate the error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation
    
    Note that all arrays should be the same length
    '''
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration. Arguments:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

def adaboost_single_tree(X, y, tree):
    # prediction
    y_pred = tree.predict(X)
    
    # equal weight
    w_i = np.ones(len(y)) * 1 / len(y)
    error = compute_error(y, y_pred, w_i)

    # compute alpha
    alpha = compute_alpha(error)
    w_i = update_weights(w_i, alpha, y, y_pred)

    # update model
    updated_tree = DecisionTreeClassifier()
    updated_tree.fit(X, y, sample_weight=w_i)

    # convert sklearn tree to our tree
    tree = construct_tree(updated_tree, X, y)
    return tree
