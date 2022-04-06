import numpy as np

def pairwise2overall(d, m):
    # d: (M-1)x(M-1)
    overall_diversity = 2*d/(m*(m-1))
    return overall_diversity

def prediction_variance(preds):
    # preds: NxM with N data points and M classifiers
    return np.mean(np.var(preds, axis=1))

def q_statistics(preds, y):
    # preds: NxM with N data points and M classifiers
    n = preds.shape[0]
    m = preds.shape[1]
    d = 0
    for i in range(m):
        for j in range(i + 1, m):
            clf1_preds = preds[:, i]
            clf2_preds = preds[:, j]
            n11, n10, n01, n00 = 0, 0, 0, 0
            for k in range(n):
                if clf1_preds[k] == y[k] and clf2_preds[k] == y[k]:
                    n11 += 1
                elif clf1_preds[k] == y[k] and clf2_preds[k] != y[k]:
                    n10 += 1
                elif clf1_preds[k] != y[k] and clf2_preds[k] == y[k]:
                    n01 += 1
                elif clf1_preds[k] != y[k] and clf2_preds[k] != y[k]:
                    n00 += 1
            numerator = n11*n00 - n10*n01
            if numerator > 0:
                d += numerator/(n11*n00 + n10*n01)
            else:
                d += 1
    overall_diversity = pairwise2overall(d, m)
    return overall_diversity

def disagreement(preds, y):
    # preds: NxM with N data points and M classifiers
    n = preds.shape[0]
    m = preds.shape[1]
    d = 0
    for i in range(m):
        for j in range(i + 1, m):
            clf1_preds = preds[:, i]
            clf2_preds = preds[:, j]
            n10, n01 = 0, 0
            for k in range(n):
                if clf1_preds[k] == y[k] and clf2_preds[k] != y[k]:
                    n10 += 1
                elif clf1_preds[k] != y[k] and clf2_preds[k] == y[k]:
                    n01 += 1
            d += (n10 + n01)/n
    overall_diversity = pairwise2overall(d, m)
    return overall_diversity    

def double_fault(preds, y):
    # preds: NxM with N data points and M classifiers
    n = preds.shape[0]
    m = preds.shape[1]
    d = 0
    for i in range(m):
        for j in range(i + 1, m):
            clf1_preds = preds[:, i]
            clf2_preds = preds[:, j]
            n00 = 0
            for k in range(n):
                if clf1_preds[k] != y[k] and clf2_preds[k] != y[k]:
                    n00 += 1
            d += n00/n
    overall_diversity = pairwise2overall(d, m)
    return overall_diversity

def kohavi_wolpert_variance(preds, y):
    # preds: NxM with N data points and M classifiers
    n = preds.shape[0]
    m = preds.shape[1]
    overall_diversity = 0
    for i in range(n):
        data_point_preds = preds[i, :]
        num_correct_classifiers = np.sum(data_point_preds==y[i])
        overall_diversity += num_correct_classifiers*(m - num_correct_classifiers)
    overall_diversity /= (n*(m**2))
    return overall_diversity

def entropy(preds, y):
    # preds: NxM with N data points and M classifiers
    n = preds.shape[0]
    m = preds.shape[1]
    overall_diversity = 0
    for i in range(n):
        data_point_preds = preds[i, :]
        num_correct_classifiers = np.sum(data_point_preds==y[i])
        overall_diversity += min([num_correct_classifiers, m - num_correct_classifiers])
    overall_diversity = overall_diversity / (n*(m - np.ceil(m/2)))
    return overall_diversity

def generalized_diversity(preds, y):
    # preds: NxM with N data points and M classifiers
    n = preds.shape[0]
    m = preds.shape[1]
    pk = np.zeros(m + 1)
    for i in range(n):
        data_point_preds = preds[i, :]
        num_incorrect_classifiers = np.sum(data_point_preds!=y[i])
        pk[num_incorrect_classifiers] += 1
    pk = pk/n
    krange = np.arange(m + 1)
    kminusrange = np.arange(-1, m)
    temp = krange * pk / m
    overall_diversity = 1 - (np.sum(temp*kminusrange/(m*(m - 1)))/(np.sum(temp/m)))
    return overall_diversity    