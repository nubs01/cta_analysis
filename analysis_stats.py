import numpy as np
import pandas as pd
import itertools as it
from joblib import Parallel, delayed, cpu_count
from scipy.stats import t, fisher_exact, shapiro, levene, chisquare, kruskal, chi2_contingency
from scipy.optimize import curve_fit
from sklearn.model_selection import LeavePOut
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pingouin as pg


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


def chisquared_test(labels, data, alpha=0.05, group_col=0):
    '''performs chi-squared test to compare the distributions between 2 groups
    data should be binary integers (0 or 1 in each element). Could also work
    with integer counts.
    '''
    groups = np.unique(labels[:,group_col])
    n_grps = len(groups)
    if n_grps == 1:
        print('Only 1 group. Skipping Comparison')
        return None, None, None
    elif n_grps != 2:
        raise ValueError('Cannot compare more than 2 groups.')

    n_samps = {g: np.sum(labels[:, group_col] == g) for g in groups}
    lbls = labels[:, group_col]
    ix = np.where(lbls == groups[0])[0]
    iy = np.where(lbls == groups[1])[0]
    n_x = np.sum(data[ix,:], axis=0)
    n_y = np.sum(data[iy,:], axis=0)
    stat, p = chisquare(n_x, f_exp=n_y)
    comp_data = {groups[0]: n_x, groups[1]: n_y}
    return stat, p, comp_data


def chi2_with_posthoc(df, alpha=0.05, group_col='exp_group', taste='Saccharin',
                      exp_group='Cre', ctrl_group='GFP'):
    df = df[df['taste'] == taste]
    tmp_df = df.groupby([group_col, 'time_bin'])['n_diff'].sum().reset_index()
    table1 = tmp_df.pivot(index=group_col, columns='time_bin', values='n_diff')
    bin_order = sorted(table1.columns, key=lambda x:float(x.split(' - ')[0]))
    table1 = table1[bin_order]
    stat_all, p_all = chisquare(table1.loc[exp_group], f_exp=table1.loc[ctrl_group])
    tmp_df = df.groupby([group_col, 'time_bin'])['n_same'].sum().reset_index()
    table2 = tmp_df.pivot(index=group_col, columns='time_bin', values='n_same')
    table2 = table2[bin_order]
    percent_diff = 100*table1 / (table1+table2)
    percent_diff = percent_diff[bin_order]

    # Now go through each time and look at the 2x2 contingency tables
    out_data = []
    out_tables = {}
    n_groups = len(df['time_bin'].unique())
    for (t, tg), grp in df.groupby(['time', 'time_bin']):
        tbl = grp.groupby(group_col)[['n_diff', 'n_same']].sum()
        s, p = chisquare(tbl.loc[exp_group], f_exp=tbl.loc[ctrl_group])
        padj = p*n_groups
        reject = True if padj <= alpha else False
        tmp = {'time': t, 'time_bin': tg, 'stat': s, 'p': p, 'p-adj': padj,
               'reject': reject}
        out_data.append(tmp)
        out_tables[tg] = tbl

    posthoc_df = pd.DataFrame(out_data)
    return stat_all, p_all, table1, posthoc_df, out_tables, percent_diff

def chi2_contingency_for_taste_responsive_cells(df, alpha=0.05,
                                                value_cols=['responsive',
                                                            'non-responsive'],
                                                group_cols=['exp_group',
                                                            'time_group']):
    table1 = df.groupby(group_cols)[value_cols].sum()
    stat, p, dof, expected = chi2_contingency(table1.to_numpy()[:, -len(value_cols):])

    reject = (p<=alpha)
    statistics = {'omnibus': {'A': 'all', 'B': 'all', 'statistic': stat,
                              'pval':p, 'dof': dof, 'reject': reject}}
    if p <= alpha:
        tmp = [df[x].unique() for x in group_cols]
        pairs = list(it.combinations(it.product(*tmp),2))
        # only compare pairs with some common factor
        pairs = [[(x1,y1), (x2,y2)] for (x1,y1),(x2,y2) in pairs if x1==x2 or y1==y2]
        n_pairs = len(pairs)
        for i, pair in enumerate(pairs):
            s, p, d, ex  = chi2_contingency(table1.loc[pair])
            # bonferoni correction
            p = p * n_pairs
            reject = (p<=alpha)
            tmp = {'A': '%s_%s' % pair[0], 'B': '%s_%s' % pair[1],
                   'statistic': s, 'pval': p, 'dof': d, 'reject': reject}
            statistics[i] = tmp

    return statistics

def chi2_contingency_with_posthoc(df, group_cols, value_cols, alpha=0.05,
                                  label_formatter='%s_%s\n%s'):
    s, p, dof, expected = chi2_contingency(df[value_cols].to_numpy())
    reject = (p <= alpha)
    statistics = {'omnibus': {'A': 'all', 'B': 'all', 'statistic': s,
                              'pval': p, 'dof': dof, 'reject': reject}}
    df2 = df.set_index(group_cols)
    tmp = [df[x].unique() for x in group_cols]
    pairs = list(it.combinations(it.product(*tmp), 2))
    pairs = [[(x1, y1, z1), (x2, y2, z2)] for (x1, y1, z1), (x2, y2, z2)
             in pairs if ((x1 == x2) & (y1 == y2)) or z1==z2]
    pairs = [[x,y] for x,y in pairs if x in df2.index and y in df2.index]
    n_pairs = len(pairs)
    for i, pair in enumerate(pairs):
        dat = df2.loc[pair, value_cols]
        dat = dat.loc[:, (dat!=0).any(axis=0)]
        s, p, d, ex = chi2_contingency(dat.to_numpy())
        # Bonferroni correction
        p = p * n_pairs
        reject = (p<=alpha)
        tmp = {'A': label_formatter % pair[0], 'B': label_formatter % pair[1],
               'statistic': s, 'pval': p, 'dof': d, 'reject': reject}
        statistics[i] = tmp

    return statistics


def permutation_test(labels, data, alpha=0.05, group_col=0, n_boot=1000, n_cores=1):
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


def bootstrap(data, n_boot=1000, alpha=0.05, n_cores=1, func=np.sum):
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
        gnb = GaussianNB()
        gnb.fit(self.X, self.Y)
        self.model=gnb

    def predict(self, X):
        return self.model.predict(X)

    def leave1out_fit(self):
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

        full_model = GaussianNB()
        full_model.fit(X,Y)
        accuracy = 100* (correct/n_splits)
        self.model = full_model
        self.result = ClassifierResult(accuracy, X, Y, predictions, row_id=self.row_id, model=full_model)
        return self.result


class LDAClassifier(object):
    def __init__(self, y, X, row_id=None, n_components=2):
        self.Y = y
        self.X = X
        self.row_id = row_id
        self.result = None
        self.model = None
        self.n_components = np.min([n_components, X.shape[1], len(np.unique(y))-1])

    def fit(self):
        X = self.X
        Y = self.Y
        lda = LDA(n_components=self.n_components)
        new_X = lda.fit_transform(X, Y)
        y_pred = lda.predict(X)
        self.model = lda

    def predict(self, X):
        return self.model.predict(X)

    def leave1out_fit(self):
        X = self.X
        Y = self.Y
        lpo = LeavePOut(1)
        n_splits = lpo.get_n_splits(Y)
        correct = 0
        predictions = np.zeros((n_splits,), object)
        for train_idx, test_idx in lpo.split(Y):
            lda = LDA(n_components=self.n_components)
            new_X = lda.fit_transform(X[train_idx,:], Y[train_idx])
            y_pred = lda.predict(X[test_idx,:])
            correct += (Y[test_idx] == y_pred).sum()
            predictions[test_idx] = y_pred[0]

        full_model = LDA(n_components=self.n_components)
        new_X = full_model.fit_transform(X, Y)
        self.model = full_model
        accuracy = 100* ((predictions == Y).sum() / Y.shape[0])
        self.result = ClassifierResult(accuracy, new_X, Y, predictions,
                                       row_id=self.row_id, model=full_model)
        return self.result


class ClassifierResult(object):
    def __init__(self, accuracy, x, y, y_pred, row_id=None, model=None):
        # fully trained model, leave1out accuracy, leave1out predictions
        self.accuracy = accuracy
        self.X = x
        self.Y = y
        self.Y_predicted = y_pred
        self.model = model
        self.row_id = row_id


def anova_3way(df, value_col, factor_cols, alpha=0.05):
    aov_tables = []
    comparisons = []
    comp_str = value_col + ' ~ '
    factor_strs = ['C(%s, Sum)' % x for x in factor_cols]
    if len(factor_cols) > 3:
        raise ValueError('Too Many Factors')

    # First check to see if the 3-way interaction is significant
    comp_str += ' + '.join(factor_strs) + ' + ' + '*'.join(factor_strs)
    test_key = ':'.join(factor_strs)
    model1 = ols(comp_str, data=df).fit()
    aov1 = sm.stats.anova_lm(model1, typ=3)
    # Include normality test on residuals
    aov1.loc['Shapiro_Wilk', ['F', 'PR(>F)']] = shapiro(model1.resid)
    if aov1.loc[test_key, 'PR(>F)'] <= alpha:
        return [aov1], [comp_str]

    aov_tables.append(aov1)
    comparisons.append(comp_str)

    # If 3-way interaction is not significant then drop 3-way and re-run and
    # look at 2-ways
    comp_str = value_col + ' ~ '
    comp_str += ' + '.join(factor_strs)
    combo_keys = []
    for a,b in it.combinations(factor_strs, 2):
        comp_str += ' + %s:%s' % (a,b)
        combo_keys.append('%s:%s' % (a,b))

    while len(combo_keys) > 0:
        model2 = ols(comp_str, data=df).fit()
        aov2 = sm.stats.anova_lm(model2, typ=3)
        # Include normality test on residuals
        aov2.loc['Shapiro_Wilk', ['F', 'PR(>F)']] = shapiro(model2.resid)
        aov_tables.append(aov2)
        comparisons.append(comp_str)
        # Find any non-significant 2-factor interactions and drop them
        all_sig = True
        drop = []
        for i, x in enumerate(combo_keys):
            if aov2.loc[x, 'PR(>F)'] > alpha:
                all_sig = False
                comp_str = comp_str.replace(' + %s' % x, '')
                drop.append(i)

        if all_sig:
            return aov_tables, comparisons

        drop.reverse()
        for i in drop:
            combo_keys.pop(i)

    # If we get here is mean none of the 2-factor or 3-factor interactions were
    # significant, so run it one more time with just single factors and return
    comp_str = value_col + ' ~ ' + ' + '.join(factor_strs)
    model3 = ols(comp_str, data=df).fit()
    aov3 = sm.stats.anova_lm(model3, typ=3)
    # Include normality test on residuals
    aov3.loc['Shapiro_Wilk', ['F', 'PR(>F)']] = shapiro(model3.resid)
    aov_tables.append(aov3)
    comparisons.append(comp_str)
    return aov_tables, comparisons


def test_anova_assumptions(df, dv, between, within):
    '''Test assumptions for using mixed_anova
    '''
    # Sample size: At least 20 samples per group
    # Normality: The dependent variable is normally distributed (shapiro-wilke test)
    # Homogeneity of variance: equal variances (levene's test)
    # Sphericity: Mauchly's Test (if failed can still use Greenhouse-Geisser adjusted p-vals)
    # Homogeneity of inter-correlations: Box's M test
    out = {'sample_size': True, 'normality': True, 'variance': True, 'sphericity': True, 'intercorr': True}
    all_samples = []
    for name, group in df.groupby([between, within]):
        if len(group) < 20:
            out['sample_size'] = False

        stat, p = shaprio(group[dv])
        if p < 0.05:
            out['normality'] = False

        all_samples.append(group[dv].to_numpy())

    W, p = levene(*all_samples)
    if p < 0.05:
        out['variance'] = False

    return out


def kw_and_gh(df, group_col, value_col):
    df = df.dropna(subset=[group_col, value_col])
    dat = [group[value_col].values for _, group in df.groupby(group_col)]
    kw_stat, kw_p = kruskal(*dat)
    try:
        gameshowell_df = pg.pairwise_gameshowell(data=df, dv=value_col, between=group_col).round(4)
        def apply_rejection(pval):
            if pval <= 0.05:
                return True
            else:
                return False

        gameshowell_df['reject'] = gameshowell_df['pval'].apply(apply_rejection)
    except Exception as ex:
        gameshowell_df = None
        print(f'kruskal wallis p: {kw_p}, games-howell failed')
        print(ex)

    return kw_stat, kw_p, gameshowell_df

def get_diff_df(df, group_cols, diff_col, value_col):
    diff_groups = df[diff_col].unique()
    assert len(diff_groups) == 2, 'Expected 2 comparison groups'
    df = df.copy()
    df = df.dropna(subset=[value_col])
    df2 = df.pivot_table(columns=diff_col, values=value_col, index=group_cols,
                         aggfunc=[np.mean, np.std, len]).reset_index()
    out = []
    for i, row in df2.iterrows():
        dm = -np.diff(row['mean'])[0]
        sem = np.sqrt(np.sum(row['std']**2)/np.sum(row['len']))
        row['mean_diff'] = dm
        row['sem_diff'] = sem
        out.append(row)

    keep_cols = [*group_cols, 'mean_diff', 'sem_diff']
    keep_cols = [(x, '') for x in keep_cols]
    out = pd.DataFrame(out)
    out = out[keep_cols]
    out.columns = out.columns.droplevel(1)
    return out


def gaussian_fit(x, y):
    pars, cov = curve_fit(f=gaussian, xdata=x, ydata=y, p0=[1000,100], bounds=(0, 2000))
    stdevs = np.sqrt(np.diag(cov))
    y_fit = gaussian(x, *pars)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res/ss_tot)
    return pars, stdevs, r2

def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def anova(df, between=None, within=None, dv=None, subject=None):
    if not isinstance(between ,list):
        groups = [between]
    else:
        groups = between

    if within:
        groups = [*groups, within]

    aov = df.anova(dv=dv, between=groups)
    ptt = df.pairwise_ttests(dv=dv, between=between, within=within,
                             subject=subject, padjust='bonf')
    return aov, ptt
