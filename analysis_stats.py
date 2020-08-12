import numpy as np
from joblib import Parallel, delayed, cpu_count
from scipy.stats import t, fisher_exact
from sklearn.model_selection import LeavePOut
from sklearn.naive_bayes import GaussianNB


def dunnetts_post_hoc(X0, X, alpha):
    Y = [X0, *X]
    p = len(X)
    N_i = [len(y) for y in Y]
    # s^2 = Sum(Sum((X_ij - |X|)^2))/n
    #n = sum(N_i) - (p+1)
    n = np.sum(N_i) - (p + 1)  # degrees of freedom
    s_num = np.sum([np.power([y-np.mean(x) for y in x],2) for x in Y])
    s = np.sqrt(s_num/n)

    N = [len(x) for x in X]
    m0 = np.mean(X0)
    N0 = len(X0)
    t_cv = t.ppf(1-(alpha/2), n) # get 2-tailed critical value from t-disitribution
    CI = []
    P = []
    for x, Ni in zip(X, N):
        mx = np.mean(x)
        A0 = t_cv*s*np.sqrt(1/Ni + 1/N0)
        Ai = np.abs(mx - m0)
        Ti = Ai/(s * np.sqrt(1/Ni + 1/N0))
        Pi = t.sf(Ti, n)
        P.append(Pi)
        CI.append((Ai-A0, Ai+A0))

    return CI, P


def fisher_test_response_changes(labels, time, pvals):
    pass

def permutation_test(labels, data, alpha=0.05, group_col=0, n_boot=10000, n_cores=1):
    '''data should be for a single tastant and 2 groups
    pvals is a MxN matrix with labels being an M-length array labellings the
    group of each row, if labels has columns, group_col specifies which column
    is for grouping

    Returns
    -------

    '''
    if n_cores is None:
        n_cores = cpu_count() - 1

    groups = np.unique(labels[:,group_col])
    n_grps = len(groups)
    if n_grps == 1:
        print('Only 1 group. Skipping comparison')
        return None, None, None

    if n_grps != 2:
        raise ValueError('Cannot compare more than 2 groups')

    n_samps = {g: np.sum(labels[:,group_col] == g) for g in groups}
    lbls = labels[:, group_col]
    ix = np.where(lbls == groups[0])[0]
    iy = np.where(lbls == groups[1])[0]
    n_sig_x = np.sum(data[ix,:], axis=0)
    n_sig_y = np.sum(data[iy,:], axis=0)
    test_stat = 100*(n_sig_x/len(ix) - n_sig_y/len(iy))
    # Because I need to compare % held units with response changes
    def foo(x):
        return 100*np.mean(x, axis=0)

    results = (Parallel(n_jobs=n_cores, verbose=8)
               (delayed(bootstrap_diff)
                (lbls, data, agg_func=foo)
                for i in range(n_boot)))
    #results = []
    #for i in range(n_boot):
    #    tmp = bootstrap_diff(lbls, n_sig.copy(), agg_func=foo)
    #    results.append(tmp)

    mean_diffs = np.vstack(results)
    out_p = np.sum(np.abs(mean_diffs) > np.abs(test_stat), axis=0)/n_boot

    # apply bonferroni correction
    out_p = out_p * data.shape[1]
    out_n_sig = {groups[0]: n_sig_x, groups[1]: n_sig_y}
    return out_p, test_stat, out_n_sig



def bootstrap_diff(lbls, data, agg_func=np.sum):
    n_samp = data.shape[0]
    idx = np.random.permutation(n_samp)
    grps = np.unique(lbls)
    if len(grps) != 2:
        raise ValueError('Must have 2 groups')

    ix = np.where(lbls == grps[0])[0]
    iy = np.where(lbls == grps[1])[0]
    tmp = data.copy()[idx,:]
    out = agg_func(tmp[ix,:]) - agg_func(tmp[iy,:])
    return out


def bootstrap(data, n_boot=10000, alpha=0.05, n_cores=1, func=np.sum):
    if n_cores is None or n_cores > cpu_count():
        n_cores = cpu_count()-1

    def sample(X):
        idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        return func(X[idx, :], axis=0)

    results = (Parallel(n_jobs=n_cores, verbose=8)
               (delayed(sample)(data)
                for _ in range(n_boot)))
    means = np.vstack(results)
    lower = np.percentile(means, 100*alpha/2, axis=0)
    upper = np.percentile(means, 100*(1-alpha/2), axis=0)
    return np.mean(means, axis=0), lower, upper




class NBClassifier(object):
    def __init__(self, y, X, row_id=None):
        self.Y = y
        self.X = X
        self.row_id = row_id
        self.result = None

    def fit(self):
        X = self.X
        Y = self.Y
        lpo = LeavePOut(1)
        n_splits = lpo.get_n_splits(Y)
        correct = 0
        predictions = np.zeros((n_splits,), object)
        for train_idx, test_idx in lpo.split(Y):
            train_x = X[train_idx, :]
            train_y = Y[train_idx]
            test_x = X[test_idx, :]
            test_y = Y[test_idx]
            gnb = GaussianNB()
            y_pred = gnb.fit(train_x, train_y).predict(test_x)
            predictions[test_idx] = y_pred[0]
            correct += (test_y == y_pred).sum()

        accuracy = 100* (correct/n_splits)
        self.result = ClassifierResult(accuracy, X, Y, y_pred, row_id=self.row_id)
        return self.result


class LDAClassifier(object):
    def __init__(self, y, X, row_id=None):
        self.Y = y
        self.X = X
        self.row_id = row_id
        self.result = None

    def fit(self):
        X = self.X
        Y = self.Y
        lda = LDA(n_components=2)
        new_X = lda.fit_transform(X, Y)
        y_pred = lda.predict(X)
        accuracy = 100* ((y_pred == Y).sum() / Y.shape[0])
        self.result = ClassifierResult(accuracy, new_X, Y, y_pred,
                                       row_id=self.row_id, model=lda)
        return self.result


def ClassifierResult(object):
    def __init__(self, accuracy, x, y, y_pred, row_id=None, model=None):
        self.accuracy = accuracy
        self.X = x
        self.Y = y
        self.Y_predicted = y_pred
        self.model = model
        self.row_id = row_id
