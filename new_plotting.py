import aggregation as agg
from statsmodels.stats.weightstats import CompareMeans
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from scipy.stats import sem, norm, chi2_contingency, rankdata, spearmanr, chisquare, kstest
from statsmodels.stats.diagnostic import lilliefors
import itertools as it
from plotting import ORDERS, add_suplabels, change_hue
import analysis_stats as stats
from scipy.ndimage.filters import gaussian_filter1d
from blechpy import load_dataset
from blechpy.analysis import poissonHMM as ph
from blechpy.plotting import hmm_plot as hplt
import hmm_analysis as hmma


def plot_confusion_differences(df, save_file=None):
    pal_df = stats.get_diff_df(df, ['exp_group', 'state_group', 'cta_group'],
                               'time_group', 'pal_confusion')
    id_df = stats.get_diff_df(df, ['exp_group', 'state_group', 'cta_group'],
                              'time_group', 'ID_confusion')
    pal_df['grouping'] = pal_df.apply(lambda x: '%s\n%s' % (x['exp_group'], x['cta_group']), axis=1)
    id_df['grouping'] = id_df.apply(lambda x: '%s\n%s' % (x['exp_group'], x['cta_group']), axis=1)
    o1 = ORDERS['exp_group']
    o2 = ORDERS['cta_group']
    o3 = ORDERS['state_group']
    x_order = ['GFP\nCTA', 'Cre\nNo CTA', 'GFP\nNo CTA']
    cond_order = list(it.product(x_order, o3))
    
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7), sharey=False)
    sns.barplot(data=id_df, ax=axes[0], x='grouping', y='mean_diff',
                hue='state_group', order=x_order, hue_order=o3)
    sns.barplot(data=pal_df, ax=axes[1], x='grouping', y='mean_diff',
                hue='state_group', order=x_order, hue_order=o3)
    xdata = [x.get_x() + x.get_width()/2 for x in axes[0].patches]
    xdata.sort()
    tmp_pal = pal_df.set_index(['grouping', 'state_group'])[['mean_diff', 'sem_diff']].to_dict()
    tmp_id = id_df.set_index(['grouping', 'state_group'])[['mean_diff', 'sem_diff']].to_dict()
    for x, grp in zip(xdata, cond_order):
        ym = tmp_id['mean_diff'][grp]
        yd = tmp_id['sem_diff'][grp]
        axes[0].plot([x, x], [ym-yd, ym+yd], color='k', linewidth=3)
        ym = tmp_pal['mean_diff'][grp]
        yd = tmp_pal['sem_diff'][grp]
        axes[1].plot([x, x], [ym-yd, ym+yd], color='k', linewidth=3)

    for ax in axes:
        ymax = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim([-ymax, ymax])
        ax.set_xlabel('')
        ax.set_ylabel('')
        #ax.axhline(0, linestyle='--', linewidth=1, alpha=0.6, color='k')
        ax.grid(True, axis='y', linestyle=':')
        if ax.is_first_col():
            ax.set_ylabel(r'$\Delta$ % classified as NaCl')
            ax.get_legend().remove()
        else:
            ax.get_legend().set_title('HMM State')

    axes[0].set_title('ID Confusion')
    axes[1].set_title('Pal Confusion')
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Change in saccharin classification over learning')
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


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
    ncols = 3
    axes = np.array([[fig.add_subplot(nrows, ncols, j + ncols*i + 1)
                      for j in range(ncols)] for i in range(nrows)])
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[1,0])
    axes[0,1].get_shared_y_axes().join(axes[0,1], axes[1,1])
    axes[0,2].get_shared_y_axes().join(axes[0,2], axes[1,2])
    statistics = {}
    for sg, (id_ax, pal_ax, psc_ax) in zip(state_order, axes):
        grp = df.query('state_group == @sg')
        id_kw_stat, id_kw_p, id_gh_df = stats.kw_and_gh(grp, 'grouping',
                                                        'ID_confusion')
        id_aov, id_ptt = stats.anova(grp, dv='ID_confusion', between=['exp_group', 'time_group'])
        pal_kw_stat, pal_kw_p, pal_gh_df = stats.kw_and_gh(grp, 'grouping',
                                                           'pal_confusion')
        pal_aov, pal_ptt = stats.anova(grp, dv='pal_confusion', between=['exp_group', 'time_group'])
        psc_kw_stat, psc_kw_p, psc_gh_df = stats.kw_and_gh(grp, 'grouping',
                                                           'pal_confusion_score')
        psc_aov, psc_ptt = stats.anova(grp, dv='pal_confusion_score', between=['exp_group', 'time_group'])
        id_sum = grp.groupby(['exp_group', 'time_group'])['ID_confusion'].describe()
        pal_sum = grp.groupby(['exp_group', 'time_group'])['pal_confusion'].describe()
        psc_sum = grp.groupby(['exp_group', 'time_group'])['pal_confusion_score'].describe()
        statistics[sg] = {'id': {'kw_stat': id_kw_stat, 'kw_p': id_kw_p,
                                 'posthoc': id_gh_df, 'summary': id_sum,
                                 'anova': id_aov, 'ttests': id_ptt},
                          'pal': {'kw_stat': pal_kw_stat, 'kw_p': pal_kw_p,
                                  'posthoc': pal_gh_df, 'summary': pal_sum,
                                  'anova': pal_aov, 'ttests': pal_ptt},
                          'pal_score': {'kw_stat': psc_kw_stat, 'kw_p': psc_kw_p,
                                        'posthoc': psc_gh_df, 'summary': psc_sum,
                                        'anova': psc_aov, 'ttests': psc_ptt}}

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

        g3 = plot_box_and_paired_points(grp, group_col, 'pal_confusion_score',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=psc_ax, kind=kind,
                                        plot_points=plot_points)

        #g1.axhline(50, linestyle='--', alpha=0.5, color='k')
        #g2.axhline(33.3, linestyle='--', alpha=0.5, color='k')

        g1.set_ylabel(sg.replace('_', ' '))
        g1.legend_.remove()
        g1.set_xlabel('')
        g2.set_xlabel('')
        g2.set_ylabel('')
        g2.legend_.remove()
        g3.set_xlabel('')
        g3.set_ylabel('')
        if id_ax.is_first_row():
            g1.set_title('ID Confusion')
            g2.set_title('Pal Confusion')
            g3.set_title('Pal Confusion Score')

        if not pal_ax.is_last_row():
            g3.legend_.remove()
        else:
            g3.legend(bbox_to_anchor=[1.2,1.2,0,0])

        n_cells = grp.groupby('grouping').size().to_dict()
        n_cells.update({x:y/50 for x,y in n_cells.items()}) # so I can see number of session involved
        statistics[sg]['session_counts'] = n_cells
        #n_cells = None
        #g1_y = plot_sig_stars(g1, id_gh_df, cond_order, n_cells=n_cells)
        #g2_y = plot_sig_stars(g2, pal_gh_df, cond_order, n_cells=n_cells)
        #g3_y = plot_sig_stars(g3, psc_gh_df, cond_order, n_cells=n_cells)

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
                                  't_start', 't_end', 'duration'])
    df = df.query('valid == True').copy()

    # Make extra column for composite grouping
    df['duration'] = df.apply(lambda x: x['t_end'] - max(x['t_start'], 0), axis=1)
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)
    source_data_cols = ['exp_name', 'exp_group', 'cta_group', 'time_group',
                        'palatability', 'rec_group', 'n_cells', 'taste',
                        'hmm_id', 'trial', 'state_group', 'state_num',
                        't_start', 't_end', 'duration', 'pos_in_trial',
                        'n_states']

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
    plot_grps = [('early', 't_end'), ('early', 'duration')]
    titles = {'t_end': 'End Times', 't_start': 'Start Times', 'duration': 'Span'}
    axes = np.array([fig.add_subplot(1, len(plot_grps), i+1)
                     for i in range(len(plot_grps))])
    statistics = {}
    for (sg, vg), ax in zip(plot_grps, axes):
        grp = df.query('state_group == @sg')
        kw_stat, kw_p, gh_df = stats.kw_and_gh(grp, 'grouping', vg)
        aov, ptt = stats.anova(grp, dv=vg, between=[group_col, 'time_group'])
        summary = grp.groupby(['exp_group', 'time_group'])[vg].describe()
        if sg not in statistics.keys():
            statistics[sg] = {}

        statistics[sg][titles[vg]] = {'kw_stat': kw_stat, 'kw_p': kw_p,
                                      'posthoc': gh_df, 'summary': summary,
                                      'anova': aov, 'posthoc t-tests': ptt}

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
        if kind == 'bar':
            g1_y = plot_sig_stars(g1, gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        src_data_fn = fn + '_source_data.csv'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
        df.to_csv(src_data_fn, columns=source_data_cols, index=False)
    else:
        return fig, axes, statistics


def plot_timing_distributions(df, state='early', value_col='t_end', save_file=None):
    df = df[df['valid']]
    df = df.query('state_group == @state').copy()
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'], x['cta_group']), axis=1)
    df['comp_group'] = df.apply(lambda x: '%s\n%s' % (x['grouping'], x['time_group']), axis=1)

    groups = df.grouping.unique()
    time_groups = df.time_group.unique()
    groups = ['GFP_CTA', 'Cre_No CTA', 'GFP_No CTA']
    time_groups = ORDERS['time_group']
    colors = sns.color_palette()[:len(time_groups)]

    bins = np.linspace(0,2000, 12)
    bin_centers = bins[:-1] + np.diff(bins)/2
    labels = []
    dists = []

    def drop_zero_cols(x):
        return x[:, np.any(x != 0, axis=0)]

    fig, axes = plt.subplots(nrows=len(groups), figsize=(8,10))
    fit_stats = {}
    for n1, group in df.groupby('grouping'):
        idx = groups.index(n1)
        ax = axes[idx]
        for i, n2 in enumerate(time_groups):
            grp = group.query('time_group == @n2')
            data = grp[value_col]
            mu,sig = norm.fit(data)
            x = np.linspace(0, 2000, 100)
            y = norm.pdf(x, mu, sig)
            counts, _ = np.histogram(data, bins=bins)
            N = sum(counts)
            density, _ = np.histogram(data, bins=bins, density=True)
            y_fit = norm.pdf(bin_centers, mu, sig)
            labels.append('%s_%s' % (n1,n2))
            dists.append(counts)
            if ax.is_last_row():
                l1 = n2
            else:
                l1 = None

            ax.hist(data, density=True, fc=(*colors[i], 0.4), label=l1, bins=bins, edgecolor='k')
            ss_res = np.sum((density - y_fit) **2)
            ss_tot = np.sum((density - np.mean(density)) **2)
            r2 = 1 - (ss_res/ss_tot)
            scale = np.max([x/y for x,y in zip(counts,density) if y !=0])
            fit_counts = np.array([int(x*scale) for x in y_fit])
            comp_counts = drop_zero_cols(np.vstack((counts,fit_counts)))
            x2, x2p = lilliefors(data, 'norm')
#            x2, x2p = chisquare(comp_counts[0], f_exp=comp_counts[1])
#            if np.isinf(x2):
#                x2, x2p = chisquare(comp_counts[1], f_exp=comp_counts[0])

            fit_str = r'N=%i, $\mu$=%3.3g, $\sigma$=%3.3g, $r^{2}$=%0.3g' % (N, mu,sig,r2)
            #if x2p <= 0.001:
            #    fit_str += '***'
            #elif x2p <= 0.01:
            #    fit_str += '**'
            #elif x2p <= 0.05:
            #    fit_str += '*'

            ax.plot(x,y, color=colors[i], label=fit_str)
            fit_stats[n1] = {'N': N, 'mu': mu, 'sig': sig, 'r2': r2,
                             #'Lilliefors Stat': x2, 'p-value': x2p,
                             'counts': counts, 'fit_counts': fit_counts,
                             'density': density, 'fit_density': y_fit}

        ax.set_ylabel(n1)
        ax.legend()
        sns.despine(ax=ax)
        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Transition Time (ms)')

    #g = sns.displot(data=df.query('state_group == @state'), x=value_col,
    #                hue='time_group', row='grouping', hue_order = time_groups,
    #                kde=False, row_order=groups)
    #for l, ax in zip(groups, g.axes):
    #    ax.set_title('')
    #    ax.set_ylabel(l)
    #    if ax.is_last_row():
    #        ax.set_xlabel('Transition Time (ms)')

    ##g.set_titles('{row_name}')
    #g.fig.set_size_inches([7.9, 8.2])
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    title = '%s state %s times' % (state, value_col.split('_')[-1])
    fig.suptitle(title)

    # Get distributions and do stats and fits
    dists = np.vstack(dists)

    s, p, dof, exp = chi2_contingency(drop_zero_cols(dists))
    out_stats = {'omnibus': {'A': 'all', 'B': 'all', 'chi2 stat': s, 'pval': p}}
    pairs = [['%s_%s' % (x,y) for y in time_groups] for x in groups]
    npairs = len(pairs)
    for i, (A,B) in enumerate(pairs):
        i1 = labels.index(A)
        i2 = labels.index(B)
        x = drop_zero_cols(dists[[i1,i2]])
        x[1] = np.sum(x[0])*x[1]/np.sum(x[1])
        s, p = chisquare(x[0], f_exp=x[1])
        if np.isinf(s):
            s,p = chisquare(x[1], f_exp=x[0])

        # Bonferroni Correction
        p = p*npairs
        out_stats[f'{i}'] = {'A': A, 'B': B, 'chi2 stat': s, 'pval': p}
        if p <= 0.001:
            ss = '***'
        elif p <=0.01:
            ss = '**'
        elif p <= 0.05:
            ss = '*'
        else:
            ss = ''

        grp = [x for x in groups if x in A][0]
        idx = groups.index(grp)
        axes[idx].set_title(ss)

    tmp = ['%s:  %s' % (x,y) for x,y in zip(labels, dists)]
    tmp = '\n' + '\n'.join(tmp) + '\n'
    out_stats['counts'] = tmp
    out_stats['fit_stats'] = fit_stats

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(out_stats, save_file=fn+'.txt')
        return None
    else:
        return fig, axes


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
        id_aov, id_ptt = stats.anova(grp, dv='id_acc', between=['exp_group', 'time_group'])
        pal_kw_stat, pal_kw_p, pal_gh_df = stats.kw_and_gh(grp, 'grouping', 'pal_acc')
        pal_aov, pal_ptt = stats.anova(grp, dv='pal_acc', between=['exp_group', 'time_group'])
        id_sum = grp.groupby(['exp_group', 'time_group'])['id_acc'].describe()
        pal_sum = grp.groupby(['exp_group', 'time_group'])['pal_acc'].describe()
        statistics[sg] = {'id': {'kw_stat': id_kw_stat, 'kw_p': id_kw_p,
                                 'posthoc': id_gh_df, 'summary': id_sum,
                                 'anova': id_aov, 'posthoc t-tests': id_ptt},
                          'pal': {'kw_stat': pal_kw_stat, 'kw_p': pal_kw_p,
                                  'posthoc': pal_gh_df, 'summary': pal_sum,
                                  'anova': pal_aov, 'posthoc t-tests': pal_ptt}}

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
    if len(ax.patches) < len(cond_order):
        labels = ax.get_xticklabels()
        #xpts = {x.get_text(): x.get_position()[0] for x in labels}
        xpts = {x: i for i,x in enumerate(cond_order)}
        ypts = {}
        pts = {}
        for l in ax.lines:
            xd = l.get_xdata()
            yd = l.get_ydata()
            for x,y in zip(xd, yd):
                if x in ypts.keys():
                    ypts[x] = np.max((ypts[x], y))
                else:
                    ypts[x] = y

        for k,x in xpts.items():
            pts[k] = (x, ypts[x])

    else:
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
    src_cols = ['exp_name', 'exp_group', 'cta_group', 'rec_group',
                'time_group', 'taste', 'channel', 'area', 'unit_type',
                'time_start', 'time_end', 'hmm_id', 'dt', 'threshold',
                'n_trials', 'n_states', 'n_repeats', 'n_iterations', 'n_cells',
                'log_likelihood', 'cost', 'BIC']
    if save_file:
        fig.savefig(save_file)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn+'.txt')
        plt.close(fig)
        src_fn = fn + '_source_data.csv'
        ho.to_csv(src_fn, columns=src_cols, index=False)
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
    src_cols = ['exp_name', 'exp_group', 'cta_group', 'time_group',
                'rec_group', 'rec_name', 'unit_name', 'unit_num', 'electrode',
                'area', 'single_unit', 'regular_spiking', 'fast_spiking',
                'intra_J3', 'held_unit_name', 'baseline_firing',
                'response_firing', 'unit_type', 'taste', 'taste_responsive',
                'response_p', 'response_f']
    if save_file:
        fn, ext = os.path.splitext(save_file)
        tasty_df.to_csv(fn + '_source_data.csv', columns=src_cols, index=False)

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
    df = df.drop_duplicates()
    # Drop Cre - No CTA
    df = df.query('exp_group != "Cre" or cta_group != "CTA"')
    df = df.copy()
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'], x['cta_group']), axis=1)
    df['abs_r'] = df['spearman_r'].abs()
    df['r2'] = df['spearman_r']**2
    df['resp_time'] = df['time_bin'].apply(lambda x: 'Early (0-750ms)' if x < 750
                                           else 'Late (750-2000ms)')
    src_cols = ['exp_name', 'exp_group', 'cta_group', 'time_group',
                'unit_num', 'time_bin', 'resp_time', 'spearman_r',
                'r2']

    diff_df = stats.get_diff_df(df, ['exp_group', 'time_group', 'cta_group'],
                                'resp_time', 'r2')
    diff_df['grouping'] = diff_df.apply(lambda x: '%s\n%s' % (x['exp_group'],
                                                              x['cta_group']),
                                        axis=1)
    diff_df['mean_diff'] = -diff_df['mean_diff']

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
                      'Time (ms)', "Spearman's R^2")
    axes = []
    for i, grouping in enumerate(col_order):
        grp = df.query('grouping == @grouping')
        ax = fig.add_subplot(1, len(col_order), i+1)
        axes.append(ax)

        for j, tg in enumerate(style_order):
            tmp = grp.query('time_group == @tg').groupby('time_bin')['r2'].agg([np.mean, sem])
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

    df['simple_groups'] = df.apply(lambda x: '%s_%s' % (x['exp_group'],
                                                        x['cta_group']),
                                   axis=1)
    df['grouping'] = df.apply(lambda x: '%s_%s\n%s' % (x['exp_group'],
                                                       x['cta_group'],
                                                       x['time_group']), axis=1)
    df['comp_grouping'] = df.apply(lambda x: '%s_%s' % (x['grouping'], x['resp_time']), axis=1)
    o1 = ORDERS['exp_group']
    o2 = ORDERS['cta_group']
    o3 = ORDERS['time_group']
    o4 = ['Early (0-750ms)', 'Late (750-2000ms)']
    s_order = [f'{x}_{y}' for x,y in it.product(o1,o2)]
    s_order = [x for x in s_order if x in df.simple_groups.unique()]
    g_order = [f'{x}_{y}\n{z}' for x,y,z in it.product(o1,o2,o3)]
    g_order = [x for x in g_order if x in df.grouping.unique()]
    cond_order = [f'{x}_{y}' for x,y in it.product(g_order, o4)]
    cond_order = [x for x in cond_order if x in df.comp_grouping.unique()]
    fig2, ax = plt.subplots(figsize=(14,8))
    g = sns.barplot(data=df, x='grouping', y='r2', hue='resp_time',
                    order=g_order, hue_order=o4, ax=ax)
    kw_s, kw_p, gh_df = stats.kw_and_gh(df, 'comp_grouping', 'r2')
    aov, ptt = stats.anova(df, dv='r2', between=['simple_groups', 'time_group'])
    statistics = {'KW Stat': kw_s, 'KW p-val': kw_p, 'Games-Howell posthoc': gh_df,
                  'anova': aov, 'posthoc t-tests': ptt}
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
    ax.set_ylabel("Spearman's R^2")
    ax.set_title('Mean Palatability Correlation\n'
                 'only showing small subset of significant differences')
    tmp = df.groupby(['exp_name', 'exp_group', 'time_group'])['unit_num']
    tmp = tmp.agg(lambda x: len(np.unique(x)))
    tmp = tmp.groupby(['exp_group', 'time_group']).sum().reset_index()
    tmp = tmp.rename(columns={'unit_num': 'n_cells'})
    statistics['n_cells'] = tmp

    # Plot with simplified grouping
    simp_df = df.query('exclude == False')
    unique_bins = np.arange(125, 2000, 250)

    # This line will make this analysis use non-overlapping bins only
    simp_df = simp_df[simp_df.time_bin.isin(unique_bins)].copy()

    fig4, ax = plt.subplots(figsize=(10,8))
    s_order = ['GFP_CTA', 'Cre_No CTA']
    g2_order = [f'{x}\n{y}' for x,y in it.product(s_order, o3)]
    g = sns.barplot(data=simp_df, x='simple_groups', y='r2', hue='time_group',
                    order=s_order, hue_order=o3, ax=ax)
    ax.set_ylim([0,0.11])
    kw_s, kw_p, gh_df = stats.kw_and_gh(simp_df, 'grouping', 'r2')
    simp_df['cells'] = simp_df.apply(lambda x: '%s_%s_%s' % (x['exp_name'],
                                                             x['time_group'],
                                                             x['unit_num']), axis=1)
    aov, ptt = stats.anova(simp_df,
                           between=['exp_group', 'time_group'],
                           within='time_bin', dv='r2', subject='cells')
    counts = simp_df.groupby(['simple_groups', 'time_group']).size()
    tmp = simp_df.groupby(['exp_name', 'exp_group', 'time_group'])['unit_num']
    tmp = tmp.agg(lambda x: len(np.unique(x)))
    tmp = tmp.groupby(['exp_group', 'time_group']).sum().reset_index()
    tmp = tmp.rename(columns={'unit_num': 'n_cells'})
    other_stats = {'KW Stat': kw_s, 'KW p-val': kw_p, 'Games-Howell psthoc':
                   gh_df, 'anova': aov, 't-tests': ptt, 'counts': counts, 'n_cells': tmp}
    #Everything is significant, just add in illustrator
    #plot_sig_stars(ax, gh_df, g2_order)

    # Plot differences
    o1 = ['GFP\nCTA', 'Cre\nNo CTA', 'GFP\nNo CTA']
    o2 = ['preCTA', 'postCTA']
    cond_order = list(it.product(o1, o2))
    fig3, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=diff_df, ax=ax, x='grouping', y='mean_diff',
                hue='time_group', order=o1, hue_order=o2)
    xdata = [x.get_x() + x.get_width()/2 for x in ax.patches]
    xdata.sort()
    tmp = diff_df.set_index(['grouping', 'time_group'])[['mean_diff', 'sem_diff']].to_dict()
    for x, grp in zip(xdata, cond_order):
        ym = tmp['mean_diff'][grp]
        yd = tmp['sem_diff'][grp]
        ax.plot([x, x], [ym-yd, ym+yd], color='k', linewidth=3)

    ax.set_xlabel('')
    ax.grid(True, axis='y', linestyle=':')
    ax.set_ylabel(r"$\Delta$ Spearman's R^2")
    ax.get_legend().set_title('Epoch')
    ax.set_title('Change in correlation to palatability\nbetween early and late halves of response')

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn = fn + '-comparison'
        fig2.savefig(fn + '.svg')
        agg.write_dict_to_txt(statistics, fn + '.txt')
        plt.close(fig2)

        fn2 = fn.replace('comparison', 'simple_comparison')
        fig4.savefig(fn2 + '.svg')
        agg.write_dict_to_txt(other_stats, fn2 + '.txt')
        plt.close(fig4)
        src_fn = fn2 + '_source_data.csv'
        simp_df.to_csv(src_fn, columns=src_cols, index=False)

        fn = fn.replace('comparison', 'differences.svg')
        fig3.savefig(fn)
        plt.close(fig3)
    else:
        return fig, fig2, fig3, statistics


def plot_MDS(df, value_col='MDS_dQ_v_dN', group_col='exp_group',
             ylabel='dQ/dN', save_file=None, kind='bar'):
    hue_order = ORDERS[group_col]
    order = ORDERS['time_group']
    col_order = ORDERS['MDS_time']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    df['group_col'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    g = sns.catplot(data=df, y=value_col, hue=group_col,
                    x='time_group', col='time', kind=kind, order=order,
                    hue_order=hue_order, col_order=col_order, dodge=True)

    axes = g.axes[0]
    statistics = {}
    for ax, (tg, grp) in zip(axes, df.groupby('time')):
        kw_s, kw_p, gh_df = stats.kw_and_gh(grp, 'group_col', value_col)
        aov, ptt = stats.anova(grp, dv=value_col, between=[group_col, 'time_group'])
        descrip = grp.groupby([group_col, 'time_group'])[value_col].describe()
        statistics[tg] = {'kw-stat': kw_s, 'kw-p': kw_p,
                          'description': descrip,
                          'Games-Howell posthoc': gh_df,
                          'anova': aov, 'posthoc t-tests': ptt}
        n_cells = grp.groupby('group_col').size().to_dict()
        #plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)
        ax.set_title(tg)
        if ax.is_first_col():
            ax.set_ylabel(ylabel)

        ax.set_xlabel('')

    plt.tight_layout()
    g.fig.set_size_inches(12,10)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Relative MDS Distances of Saccharin Trials')
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn+'.txt')
        src_cols = ['exp_name', 'exp_group', 'cta_group', 'time_group',
                    'taste', 'trial', 'time', 'n_cells', 'MDS1', 'MDS2',
                    'MDS_dQ', 'MDS_dN', value_col]
        src_fn = fn+'.csv'
        df.to_csv(src_fn, columns=src_cols, index=False)
    else:
        return g, statistics


def plot_full_dim_MDS(df, save_file=None):
    df = df.copy()
    df['time'] = df.time.apply(lambda x: x.split(' ')[0])
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'],
                                                   x['cta_group']),
                              axis=1)
    df['plot_group'] = df.apply(lambda x: '%s\n%s' % (x['grouping'],
                                                      x['time']),
                                axis=1)
    df['comp_group'] = df.apply(lambda x: '%s_%s' % (x['plot_group'],
                                                     x['time_group']),
                                axis=1)

    df = df.query('(exp_group != "Cre" or cta_group != "CTA") '
                  'and taste != "Water"')

    o1 = ORDERS['exp_group']
    o2 = ORDERS['cta_group']
    o3 = ['Early', 'Late']
    o4 = ORDERS['time_group']
    plot_order = ['%s_%s\n%s' % (x,y,z) for z,x,y in it.product(o3,o1,o2)]
    plot_order = [x for x in plot_order if x in df['plot_group'].unique()]
    comp_order = ['%s_%s' % (x,y) for x,y in it.product(plot_order, o4)]
    row_order = ORDERS['taste'].copy()
    if 'Water' in row_order:
        _ = row_order.pop(row_order.index('Water'))

    g = sns.catplot(data=df, row='taste', x='plot_group',
                    y='dQ_v_dN_fullMDS', kind='bar', hue='time_group',
                    order=plot_order,
                    hue_order=o4,
                    row_order=row_order,
                    sharey=False)
    statistics = {}
    for tst, group in df.groupby('taste'):
        row = row_order.index(tst)
        ax = g.axes[row, 0]
        ax.set_title('')
        ax.set_ylabel(tst)
        ax.set_xlabel('')
        kw_s, kw_p, gh_df = stats.kw_and_gh(group, 'comp_group', 'dQ_v_dN_fullMDS')
        tmp = {'Kruskal-Wallis stat': kw_s, 'Kruskal-Wallis p-val': kw_p,
               'Games-Howell posthoc': gh_df}
        statistics[tst] = tmp
        valid_comp = []
        for i, row in gh_df.iterrows():
            A = row['A'].split('\n')
            B = row['B'].split('\n')
            A = [y for x in A for y in x.split('_')]
            B = [y for x in B for y in x.split('_')]
            #if (A[0] == B[0] and A[1] == B[1]) or (A[3] == B[3]):
            if A[2] == B[2]:
                valid_comp.append(row)

        valid_comp = pd.DataFrame(valid_comp)
        plot_sig_stars(ax, valid_comp, comp_order)

    g.fig.set_size_inches(16,18)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Relative MDS distances (dQ/dN)\nFull Dimension Solution')
    #g.legend.set_bbox_to_anchor([1.2,1.2,0,0])
    #plt.tight_layout()
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn+'.txt')
    else:
        return g, statistics


def plot_unit_firing_rates(all_units, group_col='exp_group', save_file=None, exclude=True):
    if exclude:
        df = all_units.query('single_unit == True and area == "GC" and exclude==False').copy()
    else:
        df = all_units.query('single_unit == True and area == "GC"').copy()

    ups = df.groupby(['exp_name', 'rec_group']).size().mean()
    ups_sem = df.groupby(['exp_name', 'rec_group']).size().sem()
    statistics = {'units per session': '%1.2f ± %1.2f' % (ups, ups_sem),
                  'units per group': df.groupby('exp_group').size().to_dict()}
    orig_df = df.copy()
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
        aov, ptt = stats.anova(group, dv='firing_rate', between=['exp_group', 'time_group'])
        tmp = {'KW Stat': kw_s, 'KW p': kw_p, 'Games-Howell posthoc': gh_df,
               'anova': aov, 't-tests': ptt}

        # Get 95% CI of difference between exp_groups
        group_names, y = zip(*group.groupby('exp_group')['firing_rate'])
        cm = CompareMeans.from_data(y[0], y[1])
        low, high = cm.tconfint_diff(alpha=0.05, usevar='unequal')
        tmp['group_diff_95CI'] = (low, high)

        if ft in statistics.keys():
            statistics[ft][ut] = tmp
        else:
            statistics[ft] = {ut: tmp}

        plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Unit Firing Rates (Hz)')
    if save_file:
        fn, ext = os.path.splitext(save_file)
        txt_fn = fn + '.txt'
        g.fig.savefig(save_file)
        agg.write_dict_to_txt(statistics, txt_fn)
        plt.close(g.fig)
        src_fn = fn + '.csv'
        src_cols = ['exp_name', 'exp_group', 'rec_name', 'rec_group',
                    'unit_name', 'unit_num', 'electrode', 'area',
                    'single_unit', 'regular_spiking', 'fast_spiking',
                    'intra_J3', 'held_unit_name', 'time_group', 'cta_group',
                    'baseline_firing', 'response_firing', 'unit_type']
        orig_df.to_csv(src_fn, columns=src_cols, index=False)
    else:
        return g, statistics


def plot_saccharin_consumption(proj, save_file=None):
    df = proj._exp_info.copy()
    df['saccharin_consumption'] = df.saccharin_consumption.apply(lambda x: 100*x)
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'], x['cta_group']), axis=1)
    o1 = ORDERS['exp_group']
    o2 = ORDERS['cta_group']
    order = ['%s_%s' % x for x in it.product(o1,o2)]
    _ = order.pop(order.index('Cre_CTA'))
    df = df.query('grouping != "Cre_CTA"')
    #df = df.query('exclude == False')
    order = [x for x in order if x in df.grouping.unique()]
    order = ['GFP_CTA', 'Cre_No CTA', 'GFP_No CTA']
    df = df.dropna()
    fig, ax = plt.subplots(figsize=(8.5,7))
    g = sns.boxplot(data=df, x='grouping', y='saccharin_consumption', order=order, ax=ax)
    n_cells = df.groupby('grouping').size().to_dict()
    kw_s, kw_p, gh_df = stats.kw_and_gh(df, 'grouping', 'saccharin_consumption')
    out_stats = {'counts': n_cells, 'Kruskal-Wallis Stat': kw_s, 'Kruskal-Wallis p-val': kw_p,
                 'Games-Howell posthoc': gh_df}
    g.set_xlabel('')
    g.set_ylabel('% Saccharin Consumption')
    g.set_title('Saccharin Consumption\nrelative to mean water consumption')
    plot_sig_stars(g, gh_df, cond_order=order, n_cells=n_cells)
    g.axhline(80, linestyle='--', color='k', alpha=0.6)
    g.set_yscale('log')
    g.set_yticklabels(g.get_yticks(minor=True), minor=True)

    if save_file:
        fig.savefig(save_file)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(out_stats, fn+'.txt')
        plt.close(fig)
    else:
        return g, out_stats


def plot_held_unit_comparison(rec1, unit1, rec2, unit2, pvals, params,
                              held_unit_name, exp_name, exp_group, taste,
                              save_file=None):
    dig1 = load_dataset(rec1).dig_in_mapping.copy().set_index('name')
    dig2 = load_dataset(rec2).dig_in_mapping.copy().set_index('name')
    ch1 = dig1.loc[taste, 'channel']
    ch2 = dig2.loc[taste, 'channel']

    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    time_start = params['response_comparison']['time_win'][0]
    time_end = params['response_comparison']['time_win'][1]
    alpha = params['response_comparison']['alpha']
    baseline_win = params['baseline_comparison']['win_size']
    smoothing = params['psth']['smoothing_win']

    t1, fr1, _ = agg.get_firing_rate_trace(rec1, unit1, ch1, bin_size=bin_size,
                                           step_size=step_size,
                                           t_start=time_start, t_end=time_end,
                                           baseline_win=baseline_win,
                                           remove_baseline=True)

    t2, fr2, _ = agg.get_firing_rate_trace(rec2, unit2, ch2, bin_size=bin_size,
                                           step_size=step_size,
                                           t_start=time_start, t_end=time_end,
                                           baseline_win=baseline_win,
                                           remove_baseline=True)

    pt1, psth1, _ = agg.get_psth(rec1, unit1, ch1, params, remove_baseline=True)

    pt2, psth2, _ = agg.get_psth(rec2, unit2, ch2, params, remove_baseline=True)

    fig, ax = plt.subplots(figsize=(10,8))

    # --------------------------------------------------------------------------------
    # Overlayed PSTH plot
    # --------------------------------------------------------------------------------
    mp1 = np.mean(psth1, axis=0)
    sp1 = sem(psth1, axis=0)
    mp2 = np.mean(psth2, axis=0)
    sp2 = sem(psth2, axis=0)
    mp1 = gaussian_filter1d(mp1, sigma=smoothing) # smooth PSTH
    mp2 = gaussian_filter1d(mp2, sigma=smoothing) # smooth PSTH
    line1 = ax.plot(pt1, mp1, linewidth=3, label='preCTA')
    ax.fill_between(pt1, mp1 - sp1, mp1 + sp1, alpha=0.4)
    line2 = ax.plot(pt2, mp2, linewidth=3, label='postCTA')
    ax.fill_between(pt2, mp2 - sp2, mp2 + sp2, alpha=0.4)
    ax.axvline(0, linewidth=2, linestyle='--', color='k')
    top = np.max((mp1 + sp1, mp2 + sp2), axis=0)
    sig_y = 1.25 * np.max(top)
    p_y = 1.1 * np.max(top)
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], 1.75*np.max(top)])
    ax.set_xlim([pt1[0], pt1[-1]])
    intervals = []
    int_ps = []
    for t, p in zip(t1, pvals):
        if p > alpha:
            continue

        start = t - bin_size/2
        end = t + bin_size/2
        if len(intervals) > 0 and intervals[-1][1] == start:
            intervals[-1][1] = end
            int_ps[-1].append(p)
        else:
            intervals.append([start, end])
            int_ps.append([p])

        ax.plot([start, end], [sig_y, sig_y], linewidth=2, color='k')
        p_str = '%0.3g' % p
        # ax.text(t, p_y, p_str, horizontalalignment='center', fontsize=12)

    for it, ip in zip(intervals, int_ps):
        mid = np.mean(it)
        max_p = np.max(ip)
        if max_p <= 0.001:
            ss = '***'
        elif max_p <= 0.01:
            ss = '**'
        elif max_p <= 0.05:
            ss = '*'
        else:
            continue

        ax.text(mid, sig_y + 0.1, ss, horizontalalignment='center')

    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Held Unit %s : %s : %s : %s\nFiring rate relative to '
                      'baseline' % (held_unit_name, exp_name, exp_group, taste))
    ax.legend()

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
    else:
        return fig, ax


def plot_PSTHs(rec, unit, params, save_file=None, ax=None):
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('name')
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10))
    else:
        fig = ax.figure

    bin_size = params['psth']['win_size']
    step_size = params['psth']['step_size']
    p_win = params['psth']['plot_window']
    smoothing = params['psth']['smoothing_win']

    rates = []
    labels = []
    time = None
    for taste, row in dim.iterrows():
        ch = row['channel']
        t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, bin_size,
                                             step_size=step_size,
                                             t_start=p_win[0], t_end=p_win[1])
        if time is None:
            time = t

        rank = agg.PAL_MAP[taste]
        # Ignore Water
        if rank > 0:
            pal = np.ones((fr.shape[0],)) * agg.PAL_MAP[taste]
            rates.append(fr)
            labels.append(pal)

        if taste != 'Water':
            pt, psth, _ = agg.get_psth(rec, unit, ch, params, remove_baseline=False)
            mp = np.mean(psth, axis=0)
            mp = gaussian_filter1d(mp, smoothing)
            sp = sem(psth, axis=0)
            ax.plot(pt, mp, linewidth=3, label=taste)
            ax.fill_between(pt, mp - sp, mp + sp, alpha=0.4)

    # Compute and plot spearman corr R^2
    if len(rates) > 0:
        rates = np.vstack(rates)
        labels = np.concatenate(labels)
        n_bins = len(time)
        s_rs = np.zeros((n_bins,))
        s_ps = np.ones((n_bins,))
        for i, t in enumerate(time):
            if all(rates[:,i] == 0):
                continue
            else:
                response_ranks = rankdata(rates[:, i])
                s_rs[i], s_ps[i] = spearmanr(response_ranks, labels)

        s_rs = s_rs**2
        s_rs = gaussian_filter1d(s_rs, smoothing)
        ax2 = ax.twinx()
        ax2.plot(time, s_rs, color='k', linestyle=':', linewidth=3)
        ax2.set_ylabel(r'Spearman $\rho^{2}')

    ax.axvline(0, color='k', linewidth=1, linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlim([pt[0], pt[-1]])
    if isinstance(unit, int):
        unit_name = 'unit%03d' % unit
    else:
        unit_name = unit_name

    ax.set_title('%s %s' % (os.path.basename(rec), unit_name))
    ax.legend()
    if not os.path.isdir(os.path.dirname(save_file)):
        os.mkdir(os.path.dirname(save_file))

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, ax


def plot_example_hmms(example_csv, best_hmms, save_file=None):
    df = pd.read_csv(example_csv, names=['exp_name', 'rec_group', 'hmm_id',
                                         'taste', 'trials', 'exp_group',
                                         'time_group'])
    def parse_trials(x):
        y = list(map(int, x[1:-1].split('.')))
        return y

    df['trials'] = df.trials.apply(parse_trials)
    rec_map = best_hmms.set_index(['exp_name', 'rec_group'])['rec_dir'].drop_duplicates()

    # Plot 1: Rows - GFP/Cre, Columns - taste, use ctaTrain for Sacc
    fig = plt.figure(figsize=(10,8))
    df1 = df.query('taste != "Saccharin" or rec_group == "ctaTrain"')
    df1 = df1.sort_values(by=['exp_group', 'taste'], ascending=False).reset_index(drop=True)
    tastes = list(df1.taste.unique())
    groups = list(df1.exp_group.unique())
    nrows = len(groups)
    ncols = len(tastes)
    assert nrows*ncols == df1.shape[0], 'Incorrect number of tastes/group'
    outer_grid = fig.add_gridspec(nrows, ncols, wspace=0.3, hspace=0.2)
    background_color = sns.color_palette('Set2')[-1]
    paired_colors = sns.color_palette('Paired')[:-2]
    paired_colors.reverse()
    for i, row in df1.iterrows():
        outer_ax = fig.add_subplot(outer_grid[i])
        for sp in outer_ax.spines.values():
            sp.set_visible(False)
        outer_ax.set_xticks([])
        outer_ax.set_yticks([])
        if outer_ax.is_first_row():
            outer_ax.set_title(row['taste'])
        if outer_ax.is_first_col():
            outer_ax.set_ylabel(row['exp_group'], labelpad=10)
        if outer_ax.is_last_row():
            outer_ax.set_xlabel('Time (ms)', labelpad=20)

        inner_grid = outer_grid[i].subgridspec(len(row['trials']), 1)
        axes = [fig.add_subplot(ig) for ig in inner_grid]
        rd = rec_map[(row['exp_name'], row['rec_group'])]
        j = tastes.index(row['taste'])
        colors = [background_color, paired_colors[2*j], paired_colors[2*j+1]]
        if row['taste'] == 'Saccharin':
            sacc_colors = colors

        plot_mini_hmm(rd, row['hmm_id'], row['taste'],
                      row['trials'], axes, colors=colors)

    # Plot 2: Row - GFP/Cre, Columms - pre/post, only sacchairn
    df2 = df.query('taste == "Saccharin"')
    df2 = df2.sort_values(by=['time_group', 'exp_group'], ascending=False).reset_index(drop=True)
    fig2 = plt.figure(figsize=(8,8))
    t_groups = list(df2.time_group.unique())
    ncols = len(t_groups)
    outer_grid = fig.add_gridspec(ncols, nrows, wspace=0.3, hspace=0.2)
    for i, row in df2.iterrows():
        outer_ax = fig2.add_subplot(outer_grid[i])
        for sp in outer_ax.spines.values():
            sp.set_visible(False)
        outer_ax.set_xticks([])
        outer_ax.set_yticks([])
        if outer_ax.is_first_row():
            outer_ax.set_title(row['exp_group'])
        if outer_ax.is_first_col():
            outer_ax.set_ylabel(row['time_group'], labelpad=10)
        if outer_ax.is_last_row():
            outer_ax.set_xlabel('Time (ms)', labelpad=20)

        inner_grid = outer_grid[i].subgridspec(len(row['trials']), 1)
        axes = [fig2.add_subplot(ig) for ig in inner_grid]
        rd = rec_map[(row['exp_name'], row['rec_group'])]
        plot_mini_hmm(rd, row['hmm_id'], row['taste'],
                      row['trials'], axes, colors=sacc_colors)



    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fig2.savefig(save_file.replace('.svg', '-2.svg'))
        plt.close(fig2)
    else:
        return fig


def plot_pal_timing_corr(timings, proj, kind='point', save_file=None):
    col_order = ORDERS['exp_group']
    hue_order = ORDERS['time_group']
    order = [1,2,3,'Sacc']
    df = agg.apply_grouping_cols(timings, proj)
    #df = df.query('exclude == False and state_group == "early" and palatability > 0')
    df = df.query('valid == True and exclude == False and state_group == "early" and taste != "Water"').copy()
    df.loc[df['taste'] == 'Saccharin', 'palatability'] = 'Sacc'
    g = sns.catplot(data=df, x='palatability', y='t_end', col='exp_group',
                    hue='time_group', col_order=col_order, hue_order=hue_order,
                    kind=kind, dodge=True, order=order, ci=95)
    g.set_ylabels('Transition Time (ms)')
    g.set_xlabels('Palatability')
    g.set_titles('{col_name}')
    g.fig.set_size_inches(15,8)
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
    else:
        return g



def plot_mini_hmm(rec_dir, hmm_id, taste, trials, axes, colors=None):
    h5_file = hmma.get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    units = ph.query_units(rec_dir, 'single', area='GC')
    dat = load_dataset(rec_dir)
    dim = dat.dig_in_mapping.set_index('name')['channel'].to_dict()
    channel = dim[taste]
    seqs = hmm.stat_arrays['best_sequences']
    gammas = hmm.stat_arrays['gamma_probabilities']
    time = hmm.stat_arrays['time']
    seqs = seqs[trials, :]
    gammas = gammas[trials, :, :]
    n_states = hmm.n_states
    spike_array, dt, s_t = ph.get_hmm_spike_data(rec_dir, 'single', channel,
                                                 time_start=time[0],
                                                 time_end=time[-1], dt=0.001,
                                                 area='GC', trials=trials)
    if colors is None:
        colors = sns.color_palette()[:n_states]

    for ax, seq, gam, sa in zip(axes, seqs, gammas, spike_array):
        axy = ax.twinx()
        hplt.plot_raster(sa, time=time, ax=axy, y_min=1, y_max=sa.shape[0])
        hplt.plot_sequence(seq, time=time, ax=ax, colors=colors)
        hplt.plot_probability_traces(gam, time=time, ax=ax, colors=colors, thresh=1.1)
        ax.set_xlim([time[0], time[-1]])
        axy.set_ylim([0, sa.shape[0]+1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for sp in axy.spines.values():
            sp.set_visible(False)

        if ax.is_first_row() and ax.is_first_col():
            ax.set_yticks([0, 1])
            axy.set_yticks([])
        else:
            ax.set_yticks([])
            axy.set_yticks([])

        if ax.is_last_row():
            ax.set_xticks([0, 1000, 2000])
        else:
            ax.set_xticks([])


def make_hmm_prob_tidy_df(best_hmms, t_start=0, t_end=1500):
    df = best_hmms.dropna(subset=['hmm_id', 'early_state', 'late_state']).copy()
    id_cols = ['exp_name', 'exp_group', 'rec_group', 'cta_group', 'time_group',
               'taste', 'palatability', 'n_cells', 'hmm_id', 'sorting', 'exclude']
    labels = []
    probs = []
    t_vec = None
    state_cols = ['early_state', 'late_state']
    for i,row in df.iterrows():
        l_row = row[id_cols].to_list()
        h5_file = agg.get_hmm_h5(row['rec_dir'])
        hmm, t, params = ph.load_hmm_from_hdf5(h5_file, int(row['hmm_id']))
        gamma_probs = hmm.stat_arrays['gamma_probabilities']
        state_seqs = hmm.stat_arrays['best_sequences']
        t_idx = np.where((t <= t_end) & (t >= t_start))[0]
        t = t[t_idx]
        if t_vec is None:
            t_vec = t.copy()
        else:
            assert np.array_equal(t_vec, t), "time vectors aren't aligned"

        valid_trials = agg.get_valid_trials(state_seqs, np.int_(row[state_cols]),
                                            min_pts=int(.05/params['dt']),
                                            time=t)
        if len(valid_trials) == 0:
            continue

        for s_col in state_cols:
            s_name = s_col.replace('_state', '')
            s_num = int(row[s_col])
            tmp_p = gamma_probs[:, s_num, t_idx]
            tmp_p = tmp_p[valid_trials,:]
            tmp_l = [(*l_row, s_name, x) for x in valid_trials]
            labels.extend(tmp_l)
            probs.append(tmp_p)

    id_cols = [*id_cols, 'state', 'trial']
    index = pd.MultiIndex.from_tuples(labels, names=id_cols)
    df2 = pd.DataFrame(np.vstack(probs), index=index, columns=t_vec)
    df2 = df2.reset_index()
    df3 = df2.melt(id_vars=id_cols, value_vars=t_vec, var_name='time',
                   value_name='probability')
    return df3

def plot_gamma_probs(best_hmms, state='late', taste='Saccharin',
                     t_start=0, t_end=1500, save_file=None):
    # Drop GFP animals that did not learn CTA and Cre animals that did learn
    # tmp = best_hmms.query('(exp_group == "Cre" and cta_group == "No CTA") or
    # (cta_group == "CTA" and exp_group == "GFP")')
    # best_hmms = tmp.copy()
    df = make_hmm_prob_tidy_df(best_hmms, t_start=t_start, t_end=t_end)
    df = df.query('exclude == False and taste == @taste and state == @state')
    df = df.dropna().copy()
    src_cols = ['exp_name', 'exp_group', 'rec_group', 'cta_group',
                'time_group', 'taste', 'palatability', 'n_cells', 'exclude',
                'state', 'trial', 'time', 'probability']

    fig,ax = plt.subplots(figsize=(15,14))
    g = sns.lineplot(data=df, x='time', y='probability', hue='exp_group',
                     style='time_group', ci=95, linewidth=3,
                     hue_order=ORDERS['exp_group'],
                     style_order=ORDERS['time_group'], ax=ax)

    g.set_title(f'Mean {state} probabilities\n{taste}')
    g.set_ylabel('Probability')
    g.set_xlabel('Time (ms)')
    g.set_ylim([0,1])
    g.set_xlim([0,1500])

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        src_fn = fn + '_source_data.csv'
        df.to_csv(src_fn, columns=src_cols, index=False)
        return None
    else:
        return fig, ax
