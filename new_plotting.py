#import aggregation as agg
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from scipy.stats import sem
import itertools as it
from plotting import ORDERS, add_suplabels
import analysis_stats as stats


def plot_confusion_correlations(df, save_file=None):
    """look at ID and pal confusion vs n_cells and # of trials or each tastant"""
    data_cols = ['ID_confusion', 'pal_confusion']
    comparison_vars = ['n_cells', 'nacl_trials', 'ca_trials',
                       'quinine_trials', 'sacc_trials']

    # Actually just convert string cols to numbers and look at correlation matrix
    convert_vars = ['exp_group', 'time_group', 'cta_group', 'state_group']
    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x:i for i,x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]]

    fig, ax = plt.subplots(1,1,figsize=(9,8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])
    g = sns.heatmap(df2.corr(), annot=True, vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12,8)
    g.set_title('Confusion Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)


def plot_confusion_data(df, save_file=None, group_col='exp_group', kind='bar'):
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

    print(cond_order)
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
                                        ax=id_ax, kind=kind)

        g2 = plot_box_and_paired_points(grp, group_col, 'pal_confusion',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=pal_ax, kind=kind)

        g1.axhline(50, linestyle='--', alpha=0.5, color='k')
        g2.axhline(33.3, linestyle='--', alpha=0.5, color='k')

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

        g1_y = plot_sig_stars(g1, id_gh_df, cond_order)
        g2_y = plot_sig_stars(g2, pal_gh_df, cond_order)

    if save_file:
        fn, ext = os.path.splittext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics


def plot_box_and_paired_points(df, x, y, hue, order=None, hue_order=None,
                               subjects=None, estimator=np.mean,
                               error_func=sem, kind='box', **kwargs):
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


def plot_sig_stars(ax, posthoc_df, cond_order):
    if posthoc_df is None:
        return

    truedf = posthoc_df[posthoc_df['reject']]
    if truedf.empty:
        return

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
    print(pts)
    slines = [] # x1, x2, y1, y2
    sdists = []
    max_y = 0
    for i, row in truedf.iterrows():
        g1 = row['A']
        g2 = row['B']
        p = row['pval']
        if p <0.001:
            ss = '***'
        elif p<0.01:
            ss='**'
        elif p<0.05:
            ss='*'
        else:
            continue

        x1, y1 = pts[g1]
        x2, y2 = pts[g2]
        y1 += 10
        y2 += 10
        dist = abs(x2-x1)
        sdists.append(dist)
        slines.append((x1, x2, y1, y2, ss))
        if y1 > max_y:
            max_y = y1

        if y2 > max_y:
            max_y = y2

    sdists = np.array(sdists)
    idx = list(np.argsort(sdists))
    idx.reverse()
    ytop = max_y + 5*len(truedf) + 10
    maxy = ytop
    for i in idx:
        x1, x2, y1, y2, ss = slines[i]
        mid = (x1 + x2)/2
        ax.plot([x1, x1, x2, x2], [y1, ytop, ytop, y2], linewidth=1, color='k')
        ax.text(mid, ytop, ss, horizontalalignment='center', fontsize=14, fontweight='bold')
        ytop -= 5

    return maxy+5
