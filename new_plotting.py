import aggregation as agg
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from scipy.stats import sem
import itertools as it
from plotting import ORDERS, add_suplabels, change_hue
import analysis_stats as stats
from scipy.ndimage.filters import gaussian_filter1d


def plot_confusion_correlations(df, save_file=None):
    """look at ID and pal confusion vs n_cells and # of trials or each tastant"""
    data_cols = ['ID_confusion', 'pal_confusion']
    comparison_vars = ['n_cells', 'nacl_trials', 'ca_trials',
                       'quinine_trials', 'sacc_trials']

    # Actually just convert string cols to numbers and look at correlation matrix
    convert_vars = ['exp_name', 'exp_group', 'time_group', 'cta_group', 'state_group']
    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x:i for i,x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]]

    fig, ax = plt.subplots(1,1,figsize=(9,8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])

    cm = df2.rcorr(method='spearman', stars=False, padjust='bonf', decimals=15)
    cm[cm == '-'] = 1
    cm = cm.astype('float')
    m1 = np.triu(cm)
    m2 = np.tril(cm)
    g = sns.heatmap(cm, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax,
                    mask=m1)
    sns.heatmap(cm, annot=True, fmt='.1g', mask=m2, cbar=False, square=True, ax=ax)
    ax.text(0.05, .9, 'corr')
    ax.text(0.5, 0.25, 'p')
    ax.plot([0,1], [0,1], color='k', linewidth=2)
    statistics = df2.pairwise_corr(padjust='bonf', method='spearman')
    # g = sns.heatmap(df2.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0,
    #                 square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12,8)
    g.set_title('Confusion Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn += '.txt'
        agg.write_dict_to_txt({'Confusion Correlation Statistics': statistics}, fn)


def plot_coding_correlations(df, save_file=None):
    """look at ID and pal confusion vs n_cells and # of trials or each tastant"""
    df = fix_coding_df(df)
    data_cols = ['id_acc', 'pal_acc']
    comparison_vars = ['n_cells', 'n_held_cells']

    # Actually just convert string cols to numbers and look at correlation matrix
    convert_vars = ['exp_name', 'exp_group', 'time_group', 'cta_group',
                    'state_group']
    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x:i for i,x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]]

    fig, ax = plt.subplots(1,1,figsize=(9,8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])

    cm = df2.rcorr(method='spearman', stars=False, padjust='bonf', decimals=15)
    cm[cm == '-'] = 1
    cm = cm.astype('float')
    m1 = np.triu(cm)
    m2 = np.tril(cm)
    g = sns.heatmap(cm, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax,
                    mask=m1)
    sns.heatmap(cm, annot=True, fmt='.1g', mask=m2, cbar=False, square=True, ax=ax)
    ax.text(0.05, .9, 'corr')
    ax.text(0.5, 0.25, 'p')
    ax.plot([0,1], [0,1], color='k', linewidth=2)
    statistics = df2.pairwise_corr(padjust='bonf', method='spearman')
    # g = sns.heatmap(df2.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0,
    #                 square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12,8)
    g.set_title('Coding Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn += '.txt'
        agg.write_dict_to_txt({'Coding Correlation Statistics': statistics}, fn)


def plot_timing_correlations(df, save_file=None):
    data_cols = ['t_start', 't_end', 'duration']
    convert_vars = ['exp_name', 'exp_group', 'cta_group', 'time_group', 'taste']
    comparison_vars = ['palatability', 'n_cells']

    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x:i for i,x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]].dropna()

    fig, ax = plt.subplots(1,1,figsize=(9,8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])

    cm = df2.rcorr(method='spearman', stars=False, padjust='bonf', decimals=15)
    cm[cm == '-'] = 1
    cm = cm.astype('float')
    m1 = np.triu(cm)
    m2 = np.tril(cm)
    g = sns.heatmap(cm, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax,
                    mask=m1)
    sns.heatmap(cm, annot=True, fmt='.1g', mask=m2, cbar=False, square=True, ax=ax)
    ax.text(0.05, .9, 'corr')
    ax.text(0.5, 0.25, 'p')
    ax.plot([0,1], [0,1], color='k', linewidth=2)
    statistics = df2.pairwise_corr(padjust='bonf', method='spearman')
    # g = sns.heatmap(df2.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0,
    #                 square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12,8)
    g.set_title('Timing Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn += '.txt'
        agg.write_dict_to_txt({'Timing Correlation Statistics': statistics}, fn)


def plot_confusion_data(df, save_file=None, group_col='exp_group', kind='bar',
                        plot_points=False):
    df = df.copy()
    # Make extra column for composite grouping
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['time_group'].unique()
    group_order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    state_order = ORDERS['state_group']
    cond_order = []
    for g,h in it.product(group_order, hue_order):
        cond_order.append('%s_%s' % (g,h))

    fig = plt.figure(figsize=(14,10))
    outer_ax = add_suplabels(fig, 'Bootstrapped Confusion Analysis', '', '% '
                             'Sacc trials classified as NaCl')
    nrows = len(states)
    ncols = 2
    axes = np.array([[fig.add_subplot(nrows, ncols, j + ncols*i + 1)
                      for j in range(ncols)] for i in range(nrows)]) 
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[1,0])
    axes[0,1].get_shared_y_axes().join(axes[0,1], axes[1,1])
    statistics = {}
    for sg, (id_ax, pal_ax) in zip(state_order, axes):
        grp = df.query('state_group == @sg')
        id_kw_stat, id_kw_p, id_gh_df = stats.kw_and_gh(grp, 'grouping', 'ID_confusion')
        pal_kw_stat, pal_kw_p, pal_gh_df = stats.kw_and_gh(grp, 'grouping', 'pal_confusion')
        id_sum = grp.groupby(['exp_group', 'time_group'])['ID_confusion'].describe()
        pal_sum = grp.groupby(['exp_group', 'time_group'])['pal_confusion'].describe()
        statistics[sg] = {'id': {'kw_stat': id_kw_stat, 'kw_p': id_kw_p,
                                 'posthoc': id_gh_df, 'summary': id_sum},
                          'pal': {'kw_stat': pal_kw_stat, 'kw_p': pal_kw_p,
                                  'posthoc': pal_gh_df, 'summary': pal_sum}}

        g1 = plot_box_and_paired_points(grp, group_col, 'ID_confusion',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=id_ax, kind=kind,
                                        plot_points=plot_points)

        g2 = plot_box_and_paired_points(grp, group_col, 'pal_confusion',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=pal_ax, kind=kind,
                                        plot_points=plot_points)

        #g1.axhline(50, linestyle='--', alpha=0.5, color='k')
        #g2.axhline(33.3, linestyle='--', alpha=0.5, color='k')

        g1.set_ylabel(sg.replace('_', ' '))
        g1.legend_.remove()
        g1.set_xlabel('')
        g2.set_xlabel('')
        g2.set_ylabel('')
        if id_ax.is_first_row():
            g1.set_title('ID Confusion')
            g2.set_title('Pal Confusion')


        if not pal_ax.is_last_row():
            g2.legend_.remove()
        else:
            g2.legend(bbox_to_anchor=[1.2,1.2,0,0])

        #n_cells = grp.groupby('grouping').size().to_dict()
        n_cells = None
        g1_y = plot_sig_stars(g1, id_gh_df, cond_order, n_cells=n_cells)
        g2_y = plot_sig_stars(g2, pal_gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics


def plot_timing_data(df, save_file=None, group_col='exp_group', kind='bar', plot_points=False):
    assert len(df.taste.unique()) == 1, 'Please run one taste at a time'
    df = df.copy().dropna(subset=[group_col, 'state_group', 'time_group',
                                  't_start', 't_end'])

    # Make extra column for composite grouping
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    taste = df['taste'].unique()[0]
    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['time_group'].unique()
    group_order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    state_order = ORDERS['state_group']
    cond_order = []
    for g,h in it.product(group_order, hue_order):
        cond_order.append('%s_%s' % (g,h))

    fig = plt.figure(figsize=(11,7))
    outer_ax = add_suplabels(fig, f'{taste} HMM Timing Analysis', '',
                             'transition time (ms)')
    plot_grps = [('early', 't_end'), ('late', 't_start')]
    titles = {'t_end': 'End Times', 't_start': 'Start Times'}
    axes = np.array([fig.add_subplot(1, len(plot_grps), i+1)
                     for i in range(len(plot_grps))])
    statistics = {}
    for (sg, vg), ax in zip(plot_grps, axes):
        grp = df.query('state_group == @sg')
        kw_stat, kw_p, gh_df = stats.kw_and_gh(grp, 'grouping', vg)
        summary = grp.groupby(['exp_group', 'time_group'])[vg].describe()
        statistics[sg] = {titles[vg]: {'kw_stat': kw_stat, 'kw_p': kw_p,
                                          'posthoc': gh_df, 'summary': summary}}

        g1 = plot_box_and_paired_points(grp, group_col, vg,
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=ax, kind=kind,
                                        plot_points=plot_points)

        g1.set_ylabel('')
        g1.set_xlabel('')
        g1.set_title('%s %s' %(sg, titles[vg]))

        if ax.is_last_col():
            g1.legend(bbox_to_anchor=[1.2,0.8,0,0])
        else:
            g1.legend_.remove()

        n_cells = grp.groupby('grouping').size().to_dict()
        g1_y = plot_sig_stars(g1, gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics

def plot_timing_distributions(df, state='early', value_col='t_end', save_file=None):
    df = df.copy()
    df = df[df['valid']]
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'], x['cta_group']), axis=1)
    groups = df.grouping.unique()
    time_groups = df.time_group.unique()
    time_groups = ORDERS['time_group']

    g = sns.displot(data=df.query('state_group == @state'), x=value_col,
                    hue='time_group', row='grouping', hue_order = time_groups,
                    kde=True)
    g.set_titles('{row_name}')
    g.fig.set_size_inches([7.9, 8.2])
    plt.tight_layout()
    g.fig.subplots_adjust(top=0.85)
    title = '%s state %s times' % (state, value_col.split('_')[-1])
    g.fig.suptitle(title)

    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        return None
    else:
        return g


def fix_coding_df(df):
    # Melt df to get state_group column
    id_cols = ['exp_name', 'exp_group', 'time_group', 'cta_group']
    other_cols = ['n_cells', 'n_held_cells']
    data_cols = {'id_acc': ('early_ID_acc', 'late_ID_acc'),
                 'pal_acc': ('early_pal_acc', 'late_pal_acc')}
    df2 = None
    for k,v in data_cols.items():
        tmp = df.melt(id_vars=[*id_cols, *other_cols],
                      value_vars=v,
                      var_name='state_group', value_name=k)
        tmp['state_group'] = tmp['state_group'].apply(lambda x: x.split('_')[0])
        if df2 is None:
            df2 = tmp
        else:
            df2 = pd.merge(df2, tmp, on=[*id_cols, *other_cols, 'state_group'],
                           validate='1:1')

    # NaN in n_cells, means not enough cells were present to fit hmms (<3)
    df = df2.dropna().copy()
    return df


def plot_coding_data(df, save_file=None, group_col='exp_group',
                     plot_points=False, kind='bar'):
    df = fix_coding_df(df)
    # Make extra column for composite grouping
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['time_group'].unique()
    group_order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    state_order = ORDERS['state_group']
    cond_order = []
    for g,h in it.product(group_order, hue_order):
        cond_order.append('%s_%s' % (g,h))

    fig = plt.figure(figsize=(14,10))
    outer_ax = add_suplabels(fig, 'HMM Coding Analysis', '', 'Classification Accuracy (%)')
    nrows = len(states)
    ncols = 2
    axes = np.array([[fig.add_subplot(nrows, ncols, j + ncols*i + 1)
                      for j in range(ncols)] for i in range(nrows)]) 
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[1,0])
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,1])
    axes[0,1].get_shared_y_axes().join(axes[0,1], axes[1,1])
    statistics = {}
    for sg, (id_ax, pal_ax) in zip(state_order, axes):
        grp = df.query('state_group == @sg')
        id_kw_stat, id_kw_p, id_gh_df = stats.kw_and_gh(grp, 'grouping', 'id_acc')
        pal_kw_stat, pal_kw_p, pal_gh_df = stats.kw_and_gh(grp, 'grouping', 'pal_acc')
        id_sum = grp.groupby(['exp_group', 'time_group'])['id_acc'].describe()
        pal_sum = grp.groupby(['exp_group', 'time_group'])['pal_acc'].describe()
        statistics[sg] = {'id': {'kw_stat': id_kw_stat, 'kw_p': id_kw_p,
                                 'posthoc': id_gh_df, 'summary': id_sum},
                          'pal': {'kw_stat': pal_kw_stat, 'kw_p': pal_kw_p,
                                  'posthoc': pal_gh_df, 'summary': pal_sum}}

        g1 = plot_box_and_paired_points(grp, group_col, 'id_acc',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=id_ax, kind=kind,
                                        plot_points=plot_points)

        g2 = plot_box_and_paired_points(grp, group_col, 'pal_acc',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=pal_ax, kind=kind,
                                        plot_points=plot_points)

        g1.axhline(100/3, linestyle='--', alpha=0.5, color='k')
        g2.axhline(100/3, linestyle='--', alpha=0.5, color='k')

        g1.set_ylabel(sg.replace('_', ' '))
        g1.legend_.remove()
        g1.set_xlabel('')
        g2.set_xlabel('')
        g2.set_ylabel('')
        if id_ax.is_first_row():
            g1.set_title('ID Coding Accuracy')
            g2.set_title('Pal Coding Accuracy')


        if not pal_ax.is_last_row():
            g2.legend_.remove()
        else:
            g2.legend(bbox_to_anchor=[1.2,1.2,0,0])

        #n_cells = grp.groupby('grouping').size().to_dict()
        n_cells = None
        g1_y = plot_sig_stars(g1, id_gh_df, cond_order, n_cells=n_cells)
        g2_y = plot_sig_stars(g2, pal_gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics

def plot_box_and_paired_points(df, x, y, hue, order=None, hue_order=None,
                               subjects=None, estimator=np.mean,
                               error_func=sem, kind='box', plot_points=True, **kwargs):
    groups = df[x].unique()
    hues = df[hue].unique()
    if order is None:
        order = groups

    if hue_order is None:
        hue_order = hues

    # early state end time
    if kind == 'box':
        ax = sns.boxplot(data=df, x=x, hue=hue, y=y, order=order,
                         hue_order=hue_order, **kwargs)
    elif kind == 'bar':
        ax = sns.barplot(data=df, x=x, hue=hue, y=y, order=order,
                         hue_order=hue_order, **kwargs)
    elif kind == 'violin':
        ax = sns.violinplot(data=df, x=x, hue=hue, y=y, order=order,
                            hue_order=hue_order, **kwargs)
    else:
        raise ValueError('kind must be bar or box or violin')

    if not plot_points:
        return ax

    xpts = []
    for p in ax.patches:
        x1 = p.get_x() + p.get_width()/2
        xpts.append(x1)

    xpts.sort()
    max_jitter = np.min(np.diff(xpts))
    plot_pts = []
    xmap = {}
    for (g, h), xp in zip(it.product(order, hue_order), xpts):
        xmap[(g,h)] = xp

    for subj, grp in df.groupby(subjects):
        for g in grp[x].unique():
            xvals = []
            yvals = []
            yerr = []
            for h in hue_order:
                if h not in grp[hue].values:
                    continue

                tmp = grp[(grp[hue] == h) & (grp[x] == g)]
                r = (np.random.rand(1)[0] - 0.5) * max_jitter/4
                yvals.append(estimator(tmp[y]))
                xvals.append(xmap[(g, h)] + r)
                yerr.append(error_func(tmp[y]))

            ax.errorbar(xvals, yvals, yerr=yerr, alpha=0.4, marker='.',
                        markersize=10, color='grey', linewidth=2)

    return ax


def plot_sig_stars(ax, posthoc_df, cond_order, n_cells=None):
    if posthoc_df is None:
        return

    truedf = posthoc_df[posthoc_df['reject']]

    xpts = []
    ypts = []
    for p, cond in zip(ax.patches, cond_order):
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        xpts.append(x)
        ypts.append(y)

    idx = np.argsort(xpts)
    xpts = [xpts[i] for i in idx]
    ypts = [ypts[i] for i in idx]
    pts = {cond: (x,y) for cond,x,y in zip(cond_order, xpts, ypts)}

    slines = [] # x1, x2, y1, y2
    sdists = []
    max_y = 0

    for i, row in truedf.iterrows():
        g1 = row['A']
        g2 = row['B']
        p = row['pval']
        if p <=0.001:
            ss = '***'
        elif p<=0.01:
            ss='**'
        elif p<=0.05:
            ss='*'
        else:
            continue

        x1, y1 = pts[g1]
        x2, y2 = pts[g2]
        y1 = 1.2*y1
        y2 = 1.2*y2
        dist = abs(x2-x1)
        sdists.append(dist)
        slines.append((x1, x2, y1, y2, ss))
        if y1 > max_y:
            max_y = y1

        if y2 > max_y:
            max_y = y2

    if n_cells:
        for k,v in n_cells.items():
            x1, y1 = pts[k]
            ax.text(x1, .1*max_y, f'N={v}', horizontalalignment='center', color='white')

    if truedf.empty:
        return

    sdists = np.array(sdists)
    idx = list(np.argsort(sdists))
    idx.reverse()
    scaling = max_y * 8/100
    ytop = max_y + scaling*len(truedf)
    maxy = ytop
    for i in idx:
        x1, x2, y1, y2, ss = slines[i]
        mid = (x1 + x2)/2
        ax.plot([x1, x1, x2, x2], [y1, ytop, ytop, y2], linewidth=1, color='k')
        ax.text(mid, ytop, ss, horizontalalignment='center', fontsize=14, fontweight='bold')
        ytop -= scaling

    return maxy+5


def plot_BIC(ho, proj, save_file=None):
    ho = fix_hmm_overview(ho, proj)
    ho = ho.query('notes == "sequential - BIC test"')
    ho = ho[ho['n_cells'] >= 3]
    fig, ax = plt.subplots()
    g = sns.barplot(data=ho, x='n_states', y='BIC', ax=ax)
    cond_order = sorted(ho['n_states'].unique())
    kw_s, kw_p, gh = stats.kw_and_gh(ho, 'n_states', 'BIC')
    #plot_sig_stars(g, gh, cond_order)
    statistics = {'BIC Stats': {'KW stat': kw_s, 'KW p': kw_p, 'games_howell': gh}}
    if save_file:
        fig.savefig(save_file)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn+'.txt')
        plt.close(fig)
    else:
        return fig, ax, statistics


def fix_hmm_overview(ho, proj):
    ho = ho.copy()
    df = proj._exp_info
    ho['exp_group'] = ho['exp_name'].map(df.set_index('exp_name')['exp_group'].to_dict())
    df['cta_group'] = df['CTA_learned'].apply(lambda x: 'CTA' if x else 'No CTA')
    ho['cta_group'] = ho['exp_name'].map(df.set_index('exp_name')['cta_group'].to_dict())
    ho['time_group'] = ho['rec_dir'].apply(lambda x: 'preCTA'
                                           if ('pre' in x or 'Train' in x)
                                           else 'postCTA')
    return ho


def plot_hmm_trial_breakdown(df, proj, save_file=None):
    df = fix_hmm_overview(df, proj)
    df = df.query('exp_group != "Cre" or cta_group != "CTA"')
    df = df.query('taste != "Water" and n_cells >= 3').copy()
    # df = df.query('exclude == False')
    df['grouping'] = df.apply(lambda x: '%s_%s\n%s' % (x['exp_group'],
                                                      x['cta_group'],
                                                      x['time_group']), axis=1) 
    id_cols = ['exp_group', 'cta_group', 'time_group', 'grouping', 'taste']
    df2 = df.groupby([*id_cols, 'state_presence']).size().reset_index()
    df2 = df2.rename(columns={0: 'count'})
    df2['percent'] = df2.groupby(['grouping', 'taste'])['count'].apply(lambda x: 100*x/sum(x))

    o1 = ORDERS['exp_group']
    o2 = ORDERS['cta_group']
    o3 = ORDERS['time_group']
    row_order = ORDERS['taste'].copy()
    if 'Water' in row_order:
        row_order.pop(row_order.index('Water'))

    hue_order = ORDERS['state_presence']

    cond_order = ['%s_%s\n%s' % x for x in it.product(o1, o2, o3)
                  if (x[0] != "Cre" or x[1] != "CTA")]
    statistics = {}
    g = sns.catplot(data=df2, x='grouping', y='percent', row='taste',
                    hue='state_presence', kind='bar', order=cond_order,
                    row_order=row_order, hue_order=hue_order)
    g.fig.set_size_inches([14,20])
    g.set_xlabels('')

    df3 = df.groupby([*id_cols, 'state_presence']).size()
    df3 = df3.unstack('state_presence', fill_value=0).reset_index()
    cond_y = [df2[df2.grouping == x].percent.max() + 5 for x in cond_order]

    for taste, group in df3.groupby('taste'):
        row = row_order.index(taste)
        ax = g.axes[row, 0]
        ax.set_ylabel(taste)
        ax.set_title('')
        tmp_stats = stats.chi2_contingency_with_posthoc(group,
                                                        ['exp_group',
                                                         'cta_group',
                                                         'time_group'],
                                                        hue_order,
                                                        label_formatter='%s_%s\n%s')

        sdf = pd.DataFrame.from_dict(tmp_stats, orient='index')
        statistics[taste] = sdf
        tsdf = sdf[sdf['reject']]
        if len(tsdf) == 0:
            continue

        ytop = max(cond_y) + 5
        for i, row in tsdf.iterrows():
            if i == 'omnibus':
                continue

            x1 = cond_order.index(row['A'])
            x2 = cond_order.index(row['B'])
            ax.plot([x1, x1, x2, x2], [cond_y[x1], ytop, ytop, cond_y[x2]], color='k')
            ss = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else '*'
            mid = (x1 + x2)/2
            ax.text(mid, ytop + 1, ss, fontsize=14)
            ytop += 8

    g.fig.suptitle('% trials containing HMM states')
    #plt.tight_layout()
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn+'.txt')
    else:
        return g, statistics


def plot_taste_responsive_units(tasty_df, save_file=None):
    tasty_df = tasty_df[tasty_df['single_unit'] & (tasty_df['area'] == 'GC')].copy()
    order = ORDERS['exp_group']
    hue_order = ORDERS['time_group']
    df = tasty_df.groupby(['exp_group', 'time_group',
                           'taste', 'taste_responsive']).size()
    df = df.unstack('taste_responsive', fill_value=0).reset_index()
    df = df.rename(columns={True:'responsive', False:'non-responsive'})
    def percent(x):
        return 100*x['responsive']/(x['responsive'] + x['non-responsive'])

    df['percent_responsive'] = df.apply(percent, axis=1)

    tastes = list(df.taste.unique())
    groups = list(it.product(df.exp_group.unique(), df.time_group.unique()))
    fig, axes = plt.subplots(nrows=len(tastes), ncols=len(groups), figsize=(12,12))
    for (eg, tg, tst), grp in df.groupby(['exp_group', 'time_group', 'taste']):
        row = tastes.index(tst)
        col = groups.index((eg, tg))
        if len(tastes) == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        labels = ['responsive', 'non-responsive']
        values = [grp[x].sum() for x in labels]
        ax.pie(values, autopct='%1.1f%%')
        if ax.is_first_col():
            ax.set_ylabel(tst)

        if ax.is_first_row():
            ax.set_title(f'{eg}\n{tg}')

        if ax.is_last_row() and ax.is_last_col():
            ax.legend(labels, bbox_to_anchor=[1.6, 2.5, 0, 0])

    plt.subplots_adjust(top=0.85)
    fig.suptitle('% Taste Responsive Units')

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)

    df2 = tasty_df.groupby(['exp_name', 'exp_group',
                            'time_group', 'rec_group',
                            'unit_num'])['taste_responsive'].any()
    df2 = df2.reset_index().groupby(['exp_group', 'time_group', 'taste_responsive']).size()
    df2 = df2.unstack('taste_responsive', fill_value=0)
    df2 = df2.rename(columns={True: 'responsive', False: 'non-responsive'}).reset_index()
    df2['percent_responsive'] =  df2.apply(percent, axis=1)
    df2['n_cells'] = df2.apply(lambda x: x['responsive'] + x['non-responsive'], axis=1)
    df2['labels'] = df2.apply(lambda x: '%s_%s' % (x['exp_group'], x['time_group']), axis=1)
    n_cells = df2.set_index('labels')['n_cells'].to_dict()

    fig2, axes2 = plt.subplots(figsize=(12,9))
    statistics = stats.chi2_contingency_for_taste_responsive_cells(df2)
    g = sns.barplot(data=df2, x='exp_group', hue='time_group',
                    y='percent_responsive', order=order,
                    hue_order=hue_order, ax=axes2)
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    ph_df = pd.DataFrame.from_dict(statistics, orient='index')
    ph_df = ph_df.iloc[1:]
    plot_sig_stars(axes2, ph_df, cond_order, n_cells=n_cells)
    axes2.set_title('% Taste Responsive Units')

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn = fn + '-stats'
        statistics['counts'] = df2
        agg.write_dict_to_txt(statistics, fn + '.txt')
        fig2.savefig(fn+'.svg')
        plt.close(fig2)
        return
    else:
        return fig, axes, fig2, axes2


def plot_pal_responsive_units(pal_df, save_dir=None):
    pal_df = pal_df[pal_df['single_unit'] & (pal_df['area'] == 'GC')].copy()
    order = ORDERS['exp_group']
    hue_order = ORDERS['time_group']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    pal_df['abs_corr'] = pal_df['spearman_r'].apply(np.abs)
    pal_df['group_col'] = pal_df.apply(lambda x: '%s_%s' % (x['exp_group'], x['time_group']), axis=1)

    # Plot 1
    fig, ax = plt.subplots(figsize=(10,9))
    g = sns.barplot(data=pal_df, x='exp_group', y='abs_corr', hue='time_group',
                    order=order, hue_order=hue_order, ax=ax)
    g.set_title('Peak Spearman Correlation to Palatability')
    g.set_ylabel('|Spearman R|')
    g.set_xlabel('')
    kw_stat, kw_p, gh_df = stats.kw_and_gh(pal_df, 'group_col', 'spearman_r')
    n_cells = pal_df.groupby('group_col')['spearman_r'].size().to_dict()
    plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)
    if save_dir:
        fn = os.path.join(save_dir, 'palatability_spearman_corr.svg')
        fig.savefig(fn)
        fn2 = fn.replace('.svg', '.txt')
        out = {'kw_stat': kw_stat, 'kw_p': kw_p, 'Games-Howell posthoc': gh_df}
        agg.write_dict_to_txt(out, fn2)
        plt.close(fig)

    # taste disrcrim plot
    df = pal_df.groupby(['exp_group', 'time_group',
                         'taste_discriminative']).size()
    df = df.unstack('taste_discriminative', fill_value=0).reset_index()
    df = df.rename(columns={True:'discriminative', False:'non-discriminative'})
    def percent(x):
        return 100*x['discriminative']/(x['discriminative'] + x['non-discriminative'])

    df['percent_discriminative'] = df.apply(percent, axis=1)

    groups = list(it.product(df.exp_group.unique(), df.time_group.unique()))
    fig, axes = plt.subplots( ncols=len(groups), figsize=(12,12))
    for (eg, tg), grp in df.groupby(['exp_group', 'time_group']):
        col = groups.index((eg, tg))
        ax = axes[col]

        labels = ['discriminative', 'non-discriminative']
        values = [grp[x].sum() for x in labels]
        ax.pie(values, autopct='%1.1f%%')

        if ax.is_first_row():
            ax.set_title(f'{eg}\n{tg}')

        if ax.is_last_row() and ax.is_last_col():
            ax.legend(labels, bbox_to_anchor=[1.6, 2.5, 0, 0])

    plt.subplots_adjust(top=0.85)
    fig.suptitle('% Taste Discriminative Units')

    if save_dir:
        fn = os.path.join(save_dir, 'taste_discriminative.svg')
        fig.savefig(fn)
        plt.close(fig)

    df2 = pal_df.groupby(['exp_name', 'exp_group',
                          'time_group', 'rec_group',
                          'unit_num'])['taste_discriminative'].any()
    df2 = df2.reset_index().groupby(['exp_group', 'time_group', 'taste_discriminative']).size()
    df2 = df2.unstack('taste_discriminative', fill_value=0)
    df2 = df2.rename(columns={True: 'discriminative', False: 'non-discriminative'}).reset_index()
    df2['percent_discriminative'] =  df2.apply(percent, axis=1)
    df2['n_cells'] = df2.apply(lambda x: x['discriminative'] + x['non-discriminative'], axis=1)
    df2['labels'] = df2.apply(lambda x: '%s_%s' % (x['exp_group'], x['time_group']), axis=1)
    n_cells = df2.set_index('labels')['n_cells'].to_dict()

    fig2, axes2 = plt.subplots(figsize=(12,9))
    statistics = stats.chi2_contingency_for_taste_responsive_cells(df2,
                                                                   value_cols=['discriminative',
                                                                               'non-discriminative'])
    g = sns.barplot(data=df2, x='exp_group', hue='time_group',
                    y='percent_discriminative', order=order,
                    hue_order=hue_order, ax=axes2)
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    ph_df = pd.DataFrame.from_dict(statistics, orient='index')
    ph_df = ph_df.iloc[1:]
    plot_sig_stars(axes2, ph_df, cond_order, n_cells=n_cells)
    axes2.set_title('% Taste Discriminative Units')

    if save_dir:
        fn = os.path.join(save_dir, 'taste_discriminative_comparison')
        statistics['counts'] = df2
        agg.write_dict_to_txt(statistics, fn + '.txt')
        fig2.savefig(fn+'.svg')
        plt.close(fig2)
        return
    else:
        return fig, axes, fig2, axes2

#def plot_mean_spearman_correlation(df, save_file=None):
def plot_mean_spearman_correlation(pal_file, proj, save_file=None):
    data  = np.load(pal_file)
    # labels are : exp_group, time_group, rec_dir, unit_num
    l = list(data['labels'])
    sr = data['spearman_r']
    t = data['time']
    index = pd.MultiIndex.from_tuples(l, names=['exp_group', 'time_group',
                                                'rec_dir', 'unit_num'])
    df = pd.DataFrame(sr, columns=t, index=index)
    df = df.reset_index().melt(id_vars=['exp_group', 'time_group',
                                        'rec_dir', 'unit_num'],
                               value_vars=t,
                               var_name='time_bin',
                               value_name='spearman_r')

    df = agg.apply_grouping_cols(df, proj)
    # Drop Cre - No CTA
    df = df.query('exp_group != "Cre" or cta_group != "CTA"')
    df = df.copy()
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'], x['cta_group']), axis=1)
    df['abs_r'] = df['spearman_r'].abs()
    df['resp_time'] = df['time_bin'].apply(lambda x: 'Early (0-750ms)' if x < 750
                                           else 'Late (750-2000ms)')
    col_order = list(df.grouping.unique())
    style_order = ORDERS['time_group']
    colors = sns.color_palette()[:len(col_order)]
    styles = ['-', '--', '-.', '.']
    styles = styles[:len(style_order)]
    markers = ['.', 'D', 'x', 'v']
    markers = markers[:len(style_order)]
    hues = [1, 0.4, 1.4]
    fig = plt.figure(figsize=(15,8))
    _ = add_suplabels(fig, 'Single Unit Mean Correlation to Palatability',
                      'Time (ms)', "|Spearman's R|")
    axes = []
    for i, grouping in enumerate(col_order):
        grp = df.query('grouping == @grouping')
        ax = fig.add_subplot(1, len(col_order), i+1)
        axes.append(ax)

        for j, tg in enumerate(style_order):
            tmp = grp.query('time_group == @tg').groupby('time_bin')['abs_r'].agg([np.mean, sem])
            x = np.array(tmp.index)
            x = x-x[0]
            y = tmp['mean'].to_numpy()
            y = gaussian_filter1d(y, 3)
            err = tmp['sem'].to_numpy()
            c = colors[j]
            ax.fill_between(x, y + err, y - err, color=c, alpha=0.4)
            ax.plot(x, y, color=c, linewidth=2, label=tg)
            ax.set_xlim([x[0], x[-1]])

        ax.set_title(grouping.replace('_', ' '))
        if ax.is_last_col():
            ax.legend(style_order, bbox_to_anchor=[1.2,1.2,0,0])

    plt.tight_layout()

    df['grouping'] = df.apply(lambda x: '%s_%s\n%s' % (x['exp_group'],
                                                       x['cta_group'],
                                                       x['time_group']), axis=1)
    df['comp_grouping'] = df.apply(lambda x: '%s_%s' % (x['grouping'], x['resp_time']), axis=1)
    o1 = ORDERS['exp_group']
    o2 = ORDERS['cta_group']
    o3 = ORDERS['time_group']
    o4 = ['Early (0-750ms)', 'Late (750-2000ms)']
    g_order = [f'{x}_{y}\n{z}' for x,y,z in it.product(o1,o2,o3)]
    cond_order = [f'{x}_{y}' for x,y in it.product(g_order, o4)]
    g_order = [x for x in g_order if x in df.grouping.unique()]
    cond_order = [x for x in cond_order if x in df.comp_grouping.unique()]
    fig2, ax = plt.subplots(figsize=(14,8))
    g = sns.barplot(data=df, x='grouping', y='abs_r', hue='resp_time',
                    order=g_order, hue_order=o4, ax=ax)
    kw_s, kw_p, gh_df = stats.kw_and_gh(df, 'comp_grouping', 'abs_r')
    statistics = {'KW Stat': kw_s, 'KW p-val': kw_p, 'Games-Howell posthoc': gh_df}
    # Slim down gh_df to only comparisons I care about
    valid_gh = []
    for i,row in gh_df.iterrows():
        a = row['A']
        b = row['B']
        s1 = a.split('\n')
        s1 = np.array([s1[0], *s1[1].split('_')])
        s2 = b.split('\n')
        s2 = np.array([s2[0], *s2[1].split('_')])
        if sum(s1==s2) == 2 and s1[0] == s2[0]:
            valid_gh.append(row)

    valid_gh = pd.DataFrame(valid_gh)
    plot_sig_stars(ax, valid_gh, cond_order)
    ax.set_xlabel('')
    ax.set_ylabel("|Spearman's R|")
    ax.set_title('Mean Palatability Correlation\n'
                 'only showing small subset of significant differences')
    tmp = df.groupby(['exp_name', 'exp_group', 'time_group'])['unit_num']
    tmp = tmp.agg(lambda x: len(np.unique(x)))
    tmp = tmp.groupby(['exp_group', 'time_group']).sum().reset_index()
    tmp = tmp.rename(columns={'unit_num': 'n_cells'})
    statistics['n_cells'] = tmp

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn = fn + '-comparison'
        fig2.savefig(fn + '.svg')
        agg.write_dict_to_txt(statistics, fn + '.txt')
        plt.close(fig2)
    else:
        return fig, fig2, statistics


def plot_MDS(df, group_col='exp_group', save_file=None):
    order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    col_order = ORDERS['MDS_time']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))] 
    df['group_col'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    g = sns.catplot(data=df, y='MDS_dQ_v_dN', x=group_col,
                    hue='time_group', col='time', kind='bar', order=order,
                    hue_order=hue_order, col_order=col_order)
    
    axes = g.axes[0]
    statistics = {}
    for ax, (tg, grp) in zip(axes, df.groupby('time')):
        kw_s, kw_p, gh_df = stats.kw_and_gh(grp, 'group_col', 'MDS_dQ_v_dN')
        statistics[tg] = {'kw-stat': kw_s, 'kw-p': kw_p,
                          'Games-Howell posthoc': gh_df}
        n_cells = grp.groupby('group_col').size().to_dict()
        plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)
        ax.set_title(tg)
        if ax.is_first_col():
            ax.set_ylabel('dQ/dN')

        ax.set_xlabel('')

    g.fig.set_size_inches(12,10)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Relative MDS Distances of Saccharin Trials')
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn+'.txt')
    else:
        return g, statistics

def plot_unit_firing_rates(all_units, group_col='exp_group', save_file=None):
    df = all_units.query('single_unit == True and area == "GC" and exclude==False').copy()
    ups = df.groupby(['exp_name', 'rec_group']).size().mean()
    ups_sem = df.groupby(['exp_name', 'rec_group']).size().sem()
    statistics = {'units per session': '%1.2f ± %1.2f' % (ups, ups_sem),
                  'units per group': df.groupby('exp_group').size().to_dict()}
    value_cols = ['baseline_firing', 'response_firing', 'norm_response_firing']
    id_cols = ['exp_name', 'exp_group', 'rec_group', 'area', 'unit_type', 'time_group', 'cta_group']
    df = df.melt(id_vars=id_cols, value_vars=value_cols,
                 var_name='firing_type', value_name='firing_rate')
    
    order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    col_order = ORDERS['unit_type']
    row_order = ['baseline_firing', 'response_firing', 'norm_response_firing']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))] 
    df['group_col'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    # plot baseline and response firing rates 
    g = sns.catplot(data=df, x=group_col, y='firing_rate', hue='time_group',
                    col='unit_type', row='firing_type', kind='bar', order=order,
                    hue_order=hue_order, col_order=col_order,
                    row_order=row_order, sharey=False)
    g.fig.set_size_inches((15, 12))
    for (ft, ut), group in df.groupby(['firing_type', 'unit_type']):
        row = row_order.index(ft)
        col = col_order.index(ut)
        ax = g.axes[row,col]
        if ax.is_first_row():
            ax.set_title(ut)
        else:
            ax.set_title('')

        if ax.is_first_col():
            ax.set_ylabel(' '.join(ft.split('_')[:-1]))
        else:
            ax.set_ylabel('')

        n_cells = group.groupby('group_col').size().to_dict()
        kw_s, kw_p, gh_df = stats.kw_and_gh(group, 'group_col', 'firing_rate')
        tmp = {'KW Stat': kw_s, 'KW p': kw_p, 'Games-Howell posthoc': gh_df}
        if ft in statistics.keys():
            statistics[ft][ut] = tmp
        else:
            statistics[ft] = {ut: tmp}

        plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Unit Firing Rates (Hz)')
    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn = fn + '.txt'
        g.fig.savefig(save_file)
        agg.write_dict_to_txt(statistics, fn)
        plt.close(g.fig)
    else:
        return g, statistics




