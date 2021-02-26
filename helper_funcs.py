import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
import analysis_stats as stats
import os


def make_tidy_response_change_data(labels, pvals, time, alpha=0.05):
    """
    labels: numpy array with columns: exp_group, exp_name, cta_group, held_unit_name, taste
    pvals: response change pvalues
    time: time vector

    make a table with columns: exp_group, exp_name, cta_group, taste, time,
    N-held_units, # diff, # not diff
    """
    bin_size = time[1] - time[0]
    shift = bin_size/2
    groups = np.unique(labels[:, (0,1,2,-1)], axis=0)
    out = []
    for (exp, anim, cta, tst) in groups:
        idx = np.where((labels[:, 0] == exp) &
                       (labels[:, 1] == anim) &
                       (labels[:, 2] == cta) &
                       (labels[:, -1] == tst))[0]
        pv = pvals[idx,:]
        n_cells = pv.shape[0]
        for i, t in enumerate(time):
            ts = t - shift
            te = t + shift
            n_diff = np.sum(pv[:,i] <= alpha)
            n_same = np.sum(pv[:,i] > alpha)
            tmp = {'exp_group': exp,
                   'cta_group': cta,
                   'exp_name': anim,
                   'taste': tst,
                   'n_cells': n_cells,
                   'time': t,
                   'time_bin': f'{ts} - {te}',
                   'n_diff': n_diff,
                   'n_same': n_same,
                   'percent_diff': 100*n_diff/n_cells}

            out.append(tmp)

    return pd.DataFrame(out)


def compare_response_changes(df, taste, plot_dir, stat_dir, file_prefix,
                             group_col='exp_group', alpha=0.05, exp_group='Cre',
                             ctrl_group='GFP'):
    df = df[df['taste'] == taste]
    stat_all, p_all, n_sig, ph_df, tables, percent_diff = \
            stats.chi2_with_posthoc(df, taste=taste, group_col=group_col,
                                    exp_group=exp_group, ctrl_group=ctrl_group,
                                    alpha=alpha)

    stat_fn = os.path.join(stat_dir, f'{file_prefix}.txt')
    plot_fn = os.path.join(plot_dir, f'{file_prefix}.svg')
    plot_held_percent_changed_new(df, group_col=group_col, taste=taste,
                                  posthoc_df=ph_df, save_file=plot_fn)
    n_sig = n_sig.to_string()
    percent_diff = percent_diff.to_string()
    ph_df = ph_df.to_string()
    with open(stat_fn, 'w') as f:
        f.write(f'{taste} Held Unit Responses Changed\n')
        f.write('-'*80 + '\n\n')
        f.write(f'Number of significantly different units:\n{n_sig}\n\n')
        f.write(f'Percent significantly different units:\n{percent_diff}\n\n')
        f.write('Chi-squared Test on number of significantly different units\n')
        f.write(f'Statistic: {stat_all}, p-value: {p_all}\n')
        f.write(f'Post-hoc comparisons:\n{ph_df}\n\nContingency Tables:\n')
        for k,v in tables.items():
            f.write(f'{k}:\n{v}\n')


def plot_held_percent_changed_new(df, group_col='exp_group', taste='Saccharin',
                                  posthoc_df=None, save_file=None):
    df = df[df['taste'] == taste]
    step = np.unique(np.diff(np.sort(df.time.unique())))[0]
    df2 = df.groupby([group_col, 'time'])[['n_diff', 'n_cells']].sum().reset_index()
    df2['percent_diff'] = df2[['n_diff', 'n_cells']].apply(lambda x: 100*x[0]/x[1], axis=1)

    # pad last time so lines plot properly
    rows = df2.loc[(df2['time'] == df['time'].max())].copy()
    rows['time'] = rows['time'] + step
    df2 = df2.append(rows, ignore_index=True)

    # shift time to be bin starts
    df2['time'] = df2['time'] - step/2

    # add sig stars to posthoc_df
    def sigstar(x):
        if x <= 0.001:
            return '***'
        elif x <= 0.01:
            return '**'
        elif x <= 0.05:
            return '*'
        else:
            return ''

    posthoc_df['sig_stars'] = posthoc_df['p-adj'].apply(sigstar)

    groups = df2[group_col].unique()
    ymax = df2.percent_diff.max() + 5
    fig, axes = plt.subplots(nrows=len(groups), figsize=(10, 12))
    for (name, group), ax in zip(df2.groupby(group_col), axes):
        tmp = group.copy()
        tmp['time'] = tmp['time'] + step
        tmp = pd.concat([group, tmp]).sort_index().reset_index(drop=True)
        ax.fill_between(tmp.time, tmp.percent_diff, alpha=0.4)
        g = sns.lineplot(data=group, x='time', y='percent_diff', hue=group_col,
                         drawstyle='steps-post', ax=ax)
        g.set_ylabel('%')
        g.get_legend().remove()
        if not ax.is_last_row():
            g.set_xlabel('')
        else:
            g.set_xlabel('Post-Stimulus Time (s)')

        ydata = g.lines[0].get_ydata()[:-1]
        for (iph, rph), y in zip(posthoc_df.iterrows(), ydata):
            ax.text(rph['time'], y+1, rph['sig_stars'],
                    horizontalalignment='center', fontsize=14)

        g.set_xlim([df2.time.min(), df2.time.max()])
        g.set_ylim([0, 45])
        x = df2.time.min() + 50
        y = 0.9*ymax
        N = group['n_cells'].unique()[0]
        ax.text(x, y, f'{name}: N = {N}', horizontalalignment='left', fontsize=14)

    fig.suptitle(f'{taste} % Held Units Changed')
    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
    else:
        return fig, axes
