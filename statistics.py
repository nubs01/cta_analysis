import numpy as np
from scipy.stats import t


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
