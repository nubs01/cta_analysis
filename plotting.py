import os
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.decomposition import PCA
from blechpy import load_dataset, load_experiment, load_project
from blechpy.dio import h5io
from blechpy.plotting import data_plot as dplt
from blechpy.analysis import spike_analysis as sas
from scipy.stats import sem
import aggregation as agg


def plot_unit_waveforms(rec_dir, unit, ax=None, save_file=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(unit, int):
        unit_str = 'unit%03i' % unit
    else:
        unit_str = unit

    dat = load_dataset(rec_dir)
    print('Plotting %s :: %s' % (dat.data_name, unit_str))
    waves, descrip, fs = h5io.get_unit_waveforms(rec_dir, unit)
    mean_wave = np.mean(waves, axis=0)
    std_wave = np.std(waves, axis=0)
    time = np.arange(0, waves.shape[1]) / (fs/1000)
    ax.plot(time, mean_wave, color='k', linewidth=3)
    ax.plot(time, mean_wave - std_wave, color='k', linewidth=1, linestyle='--')
    ax.plot(time, mean_wave + std_wave, color='k', linewidth=1, linestyle='--')
    ax.set_title('%s -- %s' % (dat.data_name, unit_str))

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return None
    else:
        return fig, ax

def plot_held_units(all_units, save_dir):
    held_units = all_units[all_units['held_unit_name'].notnull()]
    rec_order = ['preCTA', 'ctaTrain', 'ctaTest', 'postCTA']

    for unit_name, group in held_units.groupby('held_unit_name'):
        # Sort to put in proper rec order
        tmp = group.copy()
        tmp.rec_group = tmp.rec_group.astype('category')
        tmp.rec_group.cat.set_categories(rec_order, inplace=True)
        tmp = tmp.sort_values('rec_group').reset_index(drop=True)

        # Make plot and plot each unit
        width = 9*len(tmp)
        fig = plt.figure(figsize=(width, 10))
        ax = fig.add_subplot('111')
        for i, row in tmp.iterrows():
            tmp_ax = fig.add_subplot(1, len(tmp), i+1)
            plot_unit_waveforms(row['rec_dir'], row['unit_name'], ax=tmp_ax)

        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        fig.suptitle('Held Unit %s' % unit_name)
        save_file = os.path.join(save_dir, 'held_unit_%s-waves.png' % unit_name)
        fig.savefig(save_file)
        plt.close(fig)


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

    pt1, psth1 = agg.get_psth(rec1, unit1, ch1, params)

    pt2, psth2 = agg.get_psth(rec2, unit2, ch2, params)

    fig = plt.figure(figsize=(8, 11))
    ax = fig.add_subplot('111')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('Time (ms)')
    # ax.set_ylabel('Firing Rate (Hz)')
    axes = [fig.add_subplot(4, 1, i+1) for i in range(4)]

    # --------------------------------------------------------------------------------
    # Overlayed PSTH plot
    # --------------------------------------------------------------------------------
    mp1 = np.mean(psth1, axis=0)
    sp1 = sem(psth1, axis=0)
    mp2 = np.mean(psth2, axis=0)
    sp2 = sem(psth2, axis=0)
    mp1 = gaussian_filter1d(mp1, sigma=smoothing) # smooth PSTH
    mp2 = gaussian_filter1d(mp2, sigma=smoothing) # smooth PSTH
    line1 = axes[0].plot(pt1, mp1, linewidth=3, label='preCTA')
    axes[0].fill_between(pt1, mp1 - sp1, mp1 + sp1, alpha=0.4)
    line2 = axes[0].plot(pt2, mp2, linewidth=3, label='postCTA')
    axes[0].fill_between(pt2, mp2 - sp2, mp2 + sp2, alpha=0.4)
    axes[0].axvline(0, linewidth=2, linestyle='--', color='k')
    top = np.max((mp1 + sp1, mp2 + sp2), axis=0)
    sig_y = 1.25 * np.max(top)
    p_y = 1.1 * np.max(top)
    ylim = axes[0].get_ylim()
    axes[0].set_ylim([ylim[0], 1.75*np.max(top)])
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

        axes[0].plot([start, end], [sig_y, sig_y], linewidth=2, color='k')
        p_str = '%0.3g' % p
        # axes[0].text(t, p_y, p_str, horizontalalignment='center', fontsize=12)

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

        axes[0].text(mid, sig_y + 0.1, ss, horizontalalignment='center')

    axes[0].set_ylabel('PSTH')
    axes[0].set_title('Held Unit %s : %s : %s : %s\nFiring rate relative to '
                      'baseline' % (held_unit_name, exp_name, exp_group, taste))
    axes[0].legend()

    # --------------------------------------------------------------------------------
    # Residuals
    # --------------------------------------------------------------------------------
    diff, sem_diff = sas.get_mean_difference(psth1, psth2)
    abs_diff = np.abs(diff)
    axes[1].bar(pt1, diff, yerr=sem_diff, align='center', ecolor='black', alpha=0.6, capsize=3)
    axes[1].set_ylabel('$\\Delta$')
    axes[2].plot(pt2, abs_diff, color='k', linewidth=2)
    axes[2].fill_between(pt1, abs_diff - sem_diff, abs_diff + sem_diff, color='k', alpha=0.3)
    axes[2].set_ylabel('|$\\Delta$|')


    # --------------------------------------------------------------------------------
    # Box plot to see outlier trials
    # --------------------------------------------------------------------------------
    df1 = pd.DataFrame(fr1, columns=t1)
    df1['session'] = 'preCTA'
    df2 = pd.DataFrame(fr2, columns=t2)
    df2['session'] = 'postCTA'
    df = df1.append(df2).reset_index(drop=True)
    df = pd.melt(df, id_vars=['session'], value_vars=t1, var_name='Time', value_name='Firing Rate')

    g = sns.boxplot(ax=axes[-1], data=df, x='Time', y='Firing Rate', hue='session')
    g.set_xlabel('Time (ms)')
    g.set_ylabel('Binned')
    leg = g.get_legend()
    leg.set_title('')
    labels = ['pre-CTA (N = %i)' % fr1.shape[0], 'post-CTA (N = %i)' % fr2.shape[0]]
    for t, l in zip(leg.texts, labels):
        t.set_text(l)

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return None, None
    else:
        return fig, axes


def plot_held_percent_changed(labels, time, pvals, diff_time, mean_diffs, sem_diffs, alpha, taste, save_file=None):
    groups = np.unique(labels[:, 0])
    Ngrp = len(groups)
    # Pad time for proper plotting
    t_step = np.unique(np.diff(time))[0]
    time = np.array([*time, time[-1]+t_step])
    # Shift times to be bin starts
    time = time - t_step/2
    plot_win = [time[0], time[-1]]

    fig = plt.figure(figsize=(20, 12))
    outer_ax = fig.add_subplot('121')
    outer_ax.spines['top'].set_color('none')
    outer_ax.spines['bottom'].set_color('none')
    outer_ax.spines['left'].set_color('none')
    outer_ax.spines['right'].set_color('none')
    outer_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    outer_ax.set_xlabel('Time (ms)')
    outer_ax.set_ylabel('% Held Units Changed')

    outer_ax2 = fig.add_subplot('122')
    outer_ax2.spines['top'].set_color('none')
    outer_ax2.spines['bottom'].set_color('none')
    outer_ax2.spines['left'].set_color('none')
    outer_ax2.spines['right'].set_color('none')
    outer_ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    outer_ax2.set_xlabel('Time (ms)')
    outer_ax2.set_ylabel('Avg Magnitude of Difference (Hz)')

    axes = [fig.add_subplot(Ngrp, 2, 2*i+1) for i in range(Ngrp)]
    mag_axes = [fig.add_subplot(Ngrp, 2, 2+ 2*i) for i in range(Ngrp)]

    for ax, m_ax, grp in zip(axes, mag_axes, groups):
        idx = np.where(labels[:, 0] == grp)[0]
        N = len(idx)
        p = pvals[idx, :]
        meanD = mean_diffs[idx, :]
        semD = sem_diffs[idx, :]
        n_diff = np.sum(p <= alpha, axis=0)
        if N == 0:
            print('No units for group: ' + grp)
            continue
        # pad array end for proper plotting
        n_diff = np.array([*n_diff, n_diff[-1]])
        perc_sig = 100 * n_diff / N
        ax.step(time, perc_sig, where='post')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.max(xlim) - 150
        y = np.max(ylim) - 0.1 * np.diff(ylim)
        ax.text(x, y, '%s; N = %i' % (grp, N), horizontalalignment='right', fontsize=14)
        ax.set_xlim(plot_win)

        MD = np.mean(np.abs(meanD), axis=0)
        semD = np.sqrt(np.sum(np.power(semD, 2), axis=0))
        m_ax.plot(diff_time, MD, linewidth=2, color='k')
        m_ax.fill_between(diff_time, MD - semD, MD + semD, color='k', alpha=0.4)
        m_ax.axvline(0, color='k', linestyle='--', linewidth=1)
        m_ax.set_xlim([diff_time[0], diff_time[-1]])

    mag_axes[0].set_title('Average Magnitude of Change')
    axes[0].set_title('Percent Held Unit Responses Changed')
    fig.suptitle('%s' % taste)

    if save_file is None:
        return fig, axes
    else:
        fig.savefig(save_file)
        plt.close(fig)
        return None, None


def plot_PSTHs(rec, unit, params, save_file):
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('name')
    fig, ax = plt.subplots(figsize=(15,10))
    for taste, row in dim.iterrows():
        ch = row['channel']
        pt, psth = agg.get_psth(rec, unit, ch, params)
        mp = np.mean(psth, axis=0)
        sp = sem(psth, axis=0)
        ax.plot(pt, mp, linewidth=3, label=taste)
        ax.fill_between(pt, mp - sp, mp + sp, alpha=0.4)

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

    fig.savefig(save_file)
    plt.close(fig)
    return

def plot_palatability_correlation(rec_name, unit_name, time, spearman_r, spearman_p,
                                  pearson_r, pearson_p, save_file):
    fig, axes = plt.subplots(nrows=2, figsize=(12,15))
    color = 'tab:red'
    axes[0].plot(time, spearman_r, color=color, linewidth=2)
    axes[0].set_ylabel("Spearman's R", color=color)
    axes[0].tick_params(axis='y', labelcolor=color)
    axes[0].set_xlim([time[0], time[-1]])
    tmp_ax = axes[0].twinx()
    color = 'tab:blue'
    tmp_ax.plot(time, spearman_p, color=color, linewidth=2)
    tmp_ax.set_ylabel('p-Value', color=color)
    tmp_ax.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    axes[1].plot(time, pearson_r, color=color, linewidth=2)
    axes[1].set_ylabel("Pearson's R", color=color)
    axes[1].set_xlabel('Time (ms)')
    axes[1].tick_params(axis='y', labelcolor=color)
    axes[1].set_xlim([time[0], time[-1]])
    tmp_ax = axes[1].twinx()
    color = 'tab:blue'
    tmp_ax.plot(time, pearson_p, color=color, linewidth=2)
    tmp_ax.set_ylabel('p-Value', color=color)
    tmp_ax.tick_params(axis='y', labelcolor=color)

    axes[0].set_title('Spearman Correlation')
    axes[1].set_title('Pearson Correlation')
    fig.suptitle('%s %s' % (rec_name, unit_name))
    if not os.path.isdir(os.path.dirname(save_file)):
        os.mkdir(os.path.dirname(save_file))

    fig.savefig(save_file)
    plt.close(fig)
    return


def plot_taste_responsive(resp_df, save_file):
    g = sns.catplot(data=resp_df, y='taste_responsive', x='taste',
                    col='exp_group', kind='bar', hue='time_group',
                    hue_order=['preCTA', 'postCTA'])
    g._legend.set_title('')
    g.set_titles("{col_name}")
    g.set_axis_labels('','Percent Responsive (%)')
    g.fig.set_size_inches(14,10)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Taste Responsive Cells')
    g.savefig(save_file)
    plt.close(g.fig)
    return

def plot_taste_discriminative(pal_df, save_file):
    tmp_grp = pal_df.groupby(['exp_group', 'time_group'])['taste_discriminative']
    df = tmp_grp.apply(lambda x: 100*np.sum(x)/len(x)).reset_index()
    fig, ax = plt.subplots(figsize=(12,10))
    sns.barplot(data=df, x='exp_group', y='taste_discriminative',
                hue='time_group', hue_order=['preCTA', 'postCTA'], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Percent Discriminative (%)')
    ax.set_title('Taste Discriminative Cells')
    plt.tight_layout()
    fig.savefig(save_file)
    plt.close(fig)
    return

def plot_aggregate_spearman(pal_df, save_file):
    n_grp = len(pal_df.exp_group.unique())
    fig, axes = plt.subplots(nrows=2, ncols=n_grp, figsize=(14,10))
    for i, grp in enumerate(pal_df.groupby('exp_group', sort=False)):
        name = grp[0]
        g = grp[1]
        tmp = g.groupby('rec_group')
        tmp['spearman_r'].apply(lambda x: sns.kdeplot(x, ax=axes[0,i]))
        tmp['spearman_peak'].apply(lambda x: sns.kdeplot(x, ax=axes[1,i]))
        axes[0,i].set_title(name)
        if i > 0:
            axes[0, i].get_legend().remove()
            axes[1, i].get_legend().remove()
        else:
            axes[0,i].set_ylabel('Peak Spearman R')
            axes[1,i].set_ylabel('Peak Corr Time')

    plt.tight_layout()
    fig.savefig(save_file)
    plt.close(fig)
    return


def plot_aggregate_pearson(pal_df, save_file):
    n_grp = len(pal_df.exp_group.unique())
    fig, axes = plt.subplots(nrows=2, ncols=n_grp, figsize=(14,10))
    for i, grp in enumerate(pal_df.groupby('exp_group')):
        name = grp[0]
        g = grp[1]
        tmp = g.groupby('rec_group')
        tmp['pearson_r'].apply(lambda x: sns.kdeplot(x, ax=axes[0,i]))
        tmp['pearson_peak'].apply(lambda x: sns.kdeplot(x, ax=axes[1,i]))
        axes[0,i].set_title(name)
        if i > 0:
            axes[0, i].get_legend().remove()
            axes[1, i].get_legend().remove()
        else:
            axes[0,i].set_ylabel('Peak Pearson R')
            axes[1,i].set_ylabel('Peak Corr Time')

    plt.tight_layout()
    fig.savefig(save_file)
    plt.close(fig)
    return


def plot_mean_spearman(data_file, save_file):
    data = np.load(data_file)
    labels = data['labels']
    spear_rs = data['spearman_r']
    time = data['time']
    exp_groups = np.unique(labels[:,0])
    time_groups = np.unique(labels[:,1])
    df = pd.DataFrame(spear_rs, columns=time)
    df['exp_group'] = labels[:,0]
    df['time_group'] = labels[:,1]
    df2 = pd.melt(df, id_vars=['exp_group','time_group'], value_vars=time, var_name='time', value_name='R')
    df2['R'] = df2['R'].abs()
    g = sns.catplot(data=df2, x='time', y='R', row='exp_group',
                    hue='time_group', kind='point', sharex=True, sharey=True)
    g.set_titles('{row_name}')
    g.set_xlabels('Time (ms)')
    g.set_ylabels("Spearman's R")
    g.fig.set_size_inches(15,10)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean Spearman Correlation')
    g.fig.savefig(save_file)
    plt.close(g.fig)


def plot_mean_pearson(data_file, save_file):
    data = np.load(data_file)
    labels = data['labels']
    pear_rs = data['pearson_r']
    time = data['time']
    exp_groups = np.unique(labels[:,0])
    time_groups = np.unique(labels[:,1])
    df = pd.DataFrame(pear_rs, columns=time)
    df['exp_group'] = labels[:,0]
    df['time_group'] = labels[:,1]
    df2 = pd.melt(df, id_vars=['exp_group','time_group'], value_vars=time, var_name='time', value_name='R')
    g = sns.catplot(data=df2, x='time', y='R', row='exp_group',
                    hue='time_group', kind='point', sharex=True, sharey=True)
    g.set_titles('{row_name}')
    g.set_xlabels('Time (ms)')
    g.set_ylabels("Pearson's R")
    g.fig.set_size_inches(15, 10)

    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean Pearson Correlation')
    g.fig.savefig(save_file)
    plt.close(g.fig)


def plot_taste_response_over_time(dat_file, save_file, alpha):
    data = np.load(dat_file)
    labels = data['labels']
    pvals = data['pvals']
    time = data['time']

    exp_groups = np.unique(labels[:,0])
    time_groups = np.unique(labels[:,3])
    tastes = np.unique(labels[:,-1])
    df = pd.DataFrame(pvals, columns=time)
    df['exp_group'] = labels[:,0]
    df['time_group'] = labels[:,3]
    df['taste'] = labels[:,-1]
    df2 = pd.melt(df, id_vars=['exp_group', 'time_group', 'taste'],
                  value_vars=time, var_name='time', value_name='pval')
    df2['sig'] = df2.pval <= alpha
    df3 = df2[df2.sig].groupby(['exp_group','time_group','taste', 'time'])['sig'].count().reset_index()
    def get_perc(grp):
        return 100*grp.sum()/len(grp)

    df4 = df2.groupby(['exp_group','time_group','taste', 'time'])['sig'].apply(get_perc).reset_index()
    for tst in tastes:
        fn = save_file.replace('.svg','-%s.svg' % tst)
        tmp = df4[df4.taste == tst]
        g = sns.catplot(data=tmp, x='time', y='sig', hue='time_group', row='exp_group', kind='point')
        g.set_titles('{row_name}')
        g.set_ylabels('% taste responsive cells')
        g.set_xlabels('Time (ms)')
        g._legend.set_title('')
        g.fig.set_size_inches(15,10)
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle('Taste Responsive - %s' % tst)
        g.fig.savefig(fn)
        plt.close(g.fig)


def plot_pca_traces(df, params, save_file, exp_name=None):
    bin_size = params['pca']['win_size']
    bin_step = params['pca']['step_size']
    time_start = params['pca']['time_win'][0]
    time_end = params['pca']['time_win'][1]
    smoothing = params['pca']['smoothing_win']
    n_grps = len(df.time_group.unique())

    fig_all = plt.figure(figsize=(20,15))
    fig_mean = plt.figure(figsize=(20,15))
    plt_i = 0
    colors = {'Saccharin': 'tab:purple', 'Quinine': 'tab:red', 'NaCl':
              'tab:green', 'Citric Acid': 'tab:orange', 'Water': 'tab:blue'}
    for name, group in df.groupby('time_group'):
        plt_i += 1
        rd1 = group['rec1'].unique()
        rd2 = group['rec2'].unique()
        if len(rd1) > 1 or len(rd2) > 1:
            raise ValueError('Too many recording directories')

        rd1 = rd1[0]
        rd2 = rd2[0]
        units1 = list(group['unit1'].unique())
        units2 = list(group['unit2'].unique())
        dim1 = load_dataset(rd1).dig_in_mapping.set_index('channel')
        dim2 = load_dataset(rd2).dig_in_mapping.set_index('channel')
        if len(units1) < 2:
            # No point if only 1 unit
            print('Not enough units for PCA analysis')
            continue

        time, sa = h5io.get_spike_data(rd1, units1)
        spikes = []
        labels = []
        if isinstance(sa, dict):
            for k,v in sa.items():
                ch = int(k.split('_')[-1])
                tst = dim1.loc[ch, 'name']
                l = [tst]*v.shape[0]
                labels.extend(l)
                # if there is only 1 unit
                if len(v.shape) == 2:
                    tmp = np.expand_dims(v, 1)
                else:
                     tmp = v

                spikes.append(tmp)

        else:
            if len(sa.shape) == 2:
                sa = np.expand_dims(sa, 1)

            spikes.append(sa)
            l = [dim1.loc[0,'name']]*sa.shape[0]
            labels.extend(l)

        # Again with rec2
        t, sa = h5io.get_spike_data(rd2, units2)
        if isinstance(sa, dict):
            for k,v in sa.items():
                ch = int(k.split('_')[-1])
                tst = dim2.loc[ch, 'name']
                l = [tst]*v.shape[0]
                labels.extend(l)
                # if there is only 1 unit
                if len(v.shape) == 2:
                    tmp = np.expand_dims(v, 1)
                else:
                     tmp = v

                spikes.append(tmp)

        else:
            if len(sa.shape) == 2:
                sa = np.expand_dims(sa, 1)

            spikes.append(sa)
            l = [dim2.loc[0,'name']]*sa.shape[0]
            labels.extend(l)

        spikes = np.vstack(spikes)
        labels = np.array(labels)

        # trim to size
        idx = np.where((time >= time_start) & (time <= time_end))[0]
        spikes = spikes[:, :, idx]
        time = time[idx]

        # Convert to firing rates
        bin_start = np.arange(time[0], time[-1] - bin_size + bin_step, bin_step)
        bin_time = bin_start + int(bin_size/2)
        n_trials, n_units, _ = spikes.shape
        n_bins = len(bin_start)

        firing_rate = np.zeros((n_trials, n_units, n_bins))
        for i, start in enumerate(bin_start):
            idx = np.where((time >= start) & (time <= start+bin_size))[0]
            firing_rate[:, :, i] = np.sum(spikes[:, :, idx], axis=-1) / (bin_size/1000)

        # Do PCA on all data, put in (trials*time)xcells 2D matrix
        pca_data = np.zeros((n_trials, n_units, n_bins))
        pca = PCA(n_components=2)
        fr = np.split(firing_rate, n_bins, axis=-1)
        fr = np.vstack(fr).squeeze()
        pca.fit(fr)
        pc_data = []
        pc_means = {}
        for i, t in enumerate(bin_time):
            pc = pca.transform(firing_rate[:,:,i])
            pc_tuples = [(i, x, y, z) for x,y,z in pc]
            pc_tuples = [(l, *x) for l,x in zip(labels, pc_tuples)]
            pc_data.extend(pc_tuples)

            for tst in np.unique(labels):
                idx = np.where(labels == tst)[0]
                tmp = [(x,y,z) for _,_,x,y,z in pc_tuples]
                tmp = np.array(tmp)
                tmp = tmp[idx, :]
                tmp = np.mean(tmp, axis=0)
                if tst in pc_means:
                    pc_means[tst].append(tmp)
                else:
                    pc_means[tst] = [tmp]

        # Now pc_data is a list of tuples with (taste, time_idx, x, y, z) in PC
        # space
        ax_all = fig_all.add_subplot(1, n_grps, plt_i)
        ax_mean = fig_mean.add_subplot(1, n_grps, plt_i)
        for tst, t, x, y, z in pc_data:
            ax_all.scatter(x,y,z,marker='o', color=colors[tst], alpha=0.6, label=tst)

        for tst in np.unique(labels):
            dat = np.vstack(pc_means[tst])
            ax_mean.plot(dat[:,0], dat[:,1], dat[:,2], marker='o', color=colors[tst], label=tst)

        ax_all.set_title(name)
        ax_mean.set_title(name)
        if plt_i == 0:
            ax_all.legend()
            ax_means.legend()

    fig_all.suptitle(exp_name)
    fig_mean.suptitle(exp_name)
    fig_all.savefig(save_file)
    fig_mean.savefig(save_file.replace('.svg','-mean.svg'))
    plt.close(fig_all)
    plt.close(fig_mean)


def plot_pca_distances(df, save_dir):
    '''plot NaCl-Sacc distance vs Quinine-Sacc distance, hue = time_group, row
    = exp_group
    also plot 
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    Raises
    ------
    
    '''
