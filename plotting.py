import os
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.decomposition import PCA
from blechpy import load_dataset, load_experiment, load_project
from blechpy.dio import h5io, hmmIO
from blechpy.plotting import data_plot as dplt, hmm_plot as hplt
from blechpy.analysis import spike_analysis as sas, poissonHMM as phmm
from scipy.stats import sem
import aggregation as agg
import analysis_stats as stats
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import itertools as it
import matplotlib.colors as mc
import colorsys
import glob


TASTE_COLORS = {'Saccharin': 'tab:purple', 'Quinine': 'tab:red',
                'NaCl': 'tab:green', 'Citric Acid': 'tab:orange',
                'Water': 'tab:blue'}
EXP_COLORS = {'Cre': 'tab:blue', 'GFP': 'tab:green'}
ORDERS = {'exp_group': ['GFP', 'Cre'],
          'cta_group': ['CTA', 'No CTA'],
          'taste': ['Water', 'NaCl', 'Citric Acid', 'Quinine', 'Saccharin'],
          'time_group': ['preCTA', 'postCTA'],
          'state_group': ['early', 'late']}

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

    pt1, psth1, _ = agg.get_psth(rec1, unit1, ch1, params, remove_baseline=True)

    pt2, psth2, _ = agg.get_psth(rec2, unit2, ch2, params, remove_baseline=True)

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


def plot_mean_differences_heatmap(labels, time, mean_diffs, ax=None, cbar=True,
                                  save_file=None, taste=None, t_start=None, t_end=None):
    #labels: exp_group, exp_name, held_unit_name, taste
    tastes = np.unique(labels[:,-1])
    if t_start is not None:
        idx = np.where(time >= t_start)[0]
        time = time[idx]
        mean_diffs = mean_diffs[:,idx]

    if t_end is not None:
        idx = np.where(time <= t_end)[0]
        time = time[idx]
        mean_diffs = mean_diffs[:,idx]

    if ax is not None and taste is not None:
        idx = np.where(labels[:,-1] == taste)[0]
        labels = labels[idx,:]
        mean_diffs = mean_diffs[idx,:]
        # split into groups, sort by peak time and absolute value and normalize
        # each row to max
        groups = np.unique(labels[:,0])
        plot_dat = []
        mask = []
        ylabels = []
        for gi, grp in enumerate(groups):
            idx = np.where(labels[:,0] == grp)[0]
            tmp = np.abs(mean_diffs[idx,:])
            maxes = np.max(tmp, axis=1)
            tmp = np.array([x/y for x,y in zip(tmp, maxes)])
            peaks = np.argmax(tmp, axis=1)
            sort_idx = np.argsort(peaks)
            tmp = tmp[sort_idx,:]
            plot_dat.append(tmp)
            mask.append(np.zeros_like(tmp, dtype=bool))
            yls = np.array(['']*(tmp.shape[0]+1), dtype=object)
            if gi < len(groups)-1:
                blank = np.zeros((1, len(time)))
                plot_dat.append(blank)
                mask.append(np.ones_like(blank, dtype=bool))
            else:
                yls = yls[:-1]

            mid = int(tmp.shape[0]/2)
            yls[mid] = grp
            ylabels.append(yls)

        plot_dat = np.vstack(plot_dat)
        # Trying smoothing for visualization
        plot_dat = np.array([gaussian_filter1d(x, 3) for x in plot_dat])
        mask = np.vstack(mask)
        ylabels = np.concatenate(ylabels)
        plot_df = pd.DataFrame(plot_dat, columns=time)
        g = sns.heatmap(plot_df, mask=mask, rasterized=True,
                        ax=ax, robust=True,
                        yticklabels=ylabels, cbar=cbar)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Cell #')
        ax.set_title(taste)
        fig = ax.figure
    else:
        fig, axes = plt.subplots(ncols=len(tastes), figsize=(9*len(tastes), 11))
        if len(tastes) == 1:
            axes = [axes]

        for tst, ax in zip(tastes, axes):
            if not ax.is_last_col() or cbar == False:
                cb=False
            else:
                cb=True

            plot_mean_differences_heatmap(labels, time, mean_diffs, ax=ax, taste=tst, cbar=cb)

        fig.subplots_adjust(top = 0.85)
        fig.suptitle('Held unit response changes')

    if save_file is None:
        return fig
    else:
        fig.savefig(save_file)
        plt.close(fig)


def plot_held_percent_changed(labels, time, pvals, diff_time, mean_diffs,
                              sem_diffs, alpha, taste, group_pvals=None,
                              group_col=0, save_file=None):
    # Labels: exp_group, exp_name, held_unit_name, taste
    groups = np.unique(labels[:, group_col])
    Ngrp = len(groups)
    # Pad time for proper plotting
    # time is currently bin centers
    t_step = np.unique(np.diff(time))[0]
    time = np.array([*time, time[-1]+t_step])
    # Shift times to be bin starts
    time = time - t_step/2
    plot_win = [time[0], time[-1]]
    # Truncate diffs to match time window of pvals
    if time[0] != diff_time[0] or time[-1] != diff_time[-1]:
        idx = np.where((diff_time >= time[0]) & (diff_time <= time[-1]))[0]
        diff_time = diff_time[idx]
        mean_diffs = mean_diffs[:, idx]
        sem_diffs = sem_diffs[:,idx]

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
    outer_ax2.set_ylabel('Median Magnitude of Difference (Hz)', labelpad=10)

    axes = [fig.add_subplot(Ngrp, 2, 2*i+1) for i in range(Ngrp)]
    mag_axes = [fig.add_subplot(Ngrp, 2, 2+ 2*i) for i in range(Ngrp)]

    for ax, m_ax, grp in zip(axes, mag_axes, groups):
        idx = np.where(labels[:, group_col] == grp)[0]
        N = len(idx)
        p = pvals[idx, :]
        meanD = mean_diffs[idx, :]
        semD = sem_diffs[idx, :]
        n_sig = (p <= alpha).astype('int')
        n_diff, l_diff, u_diff = stats.bootstrap(n_sig, alpha=alpha, func=np.sum)
        if N == 0:
            print('No units for group: ' + grp)
            continue

        # pad array end for proper plotting
        n_diff = np.array([*n_diff, n_diff[-1]])
        perc_sig = 100 * n_diff / N
        ax.step(time, perc_sig, where='post')
        for i, (ld, ud) in enumerate(zip(l_diff, u_diff)):
            ax.plot([time[i]+t_step/2]*2, [100*ld/N, 100*ud/N], color='k',
                    alpha=0.6, linewidth=2)

            if group_pvals is not None:
                tmp_p = group_pvals[i]
                if tmp_p > alpha:
                    continue

                if tmp_p < 0.001:
                    ss= '***'
                elif tmp_p < 0.01:
                    ss = '**'
                else:
                    ss = '*'

                ax.text(time[i]+t_step/2, 100*ud/N + 2.5, ss, horizontalalignment='center')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.max(xlim) - 150
        y = np.max(ylim) - 0.1 * np.diff(ylim)
        ax.text(x, y, '%s; N = %i' % (grp, N), horizontalalignment='right', fontsize=14)
        ax.set_xlim(plot_win)
        ax.set_ylim([np.min(ylim), np.max(ylim)+5])

        MD = np.mean(np.abs(meanD), axis=0)
        semD = np.sqrt(np.sum(np.power(semD, 2), axis=0))
        # Try plotting median magnitude of change
        #MD = np.median(np.abs(meanD), axis=0)
        #semD = 
        m_ax.plot(diff_time, MD, linewidth=2, color='k')
        m_ax.fill_between(diff_time, MD - semD, MD + semD, color='k', alpha=0.4)
        m_ax.axvline(0, color='k', linestyle='--', linewidth=1)
        m_ax.set_xlim([diff_time[0], diff_time[-1]])

    mag_axes[0].set_title('Median Magnitude of Change')
    axes[0].set_title('Percent Held Unit Responses Changed')
    fig.suptitle('%s' % taste)

    if save_file is None:
        return fig, axes
    else:
        fig.savefig(save_file)
        plt.close(fig)
        return None, None


def plot_PSTHs(rec, unit, params, save_file=None, ax=None):
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('name')
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10))
    else:
        fig = ax.figure

    for taste, row in dim.iterrows():
        ch = row['channel']
        pt, psth, _ = agg.get_psth(rec, unit, ch, params)
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

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, ax


def plot_sacc_psth_heatmap(held_df, save_dir):
    df = held_df.query('held_over == "cta" and area == "GC"').dropna(subset=['held_unit_name'])
    params = {'psth': {'win_size': 250, 'step_size': 25,
                       'smoothing_win': 0, 'plot_window': [-250, 1500]},
              'baseline_comparison': {'win_size': 1000}}

    for i, row in df.iterrows():
        rec1 = row['rec1']
        rec2 = row['rec2']
        unit1 = row['unit1']
        unit2 = row['unit2']
        unit_name = row['held_unit_name']
        exp_group= row['exp_group']
        exp_name = row['exp_name']
        fn = os.path.join(save_dir, '%s_unit-%i_heatmap.png' % (exp_group, unit_name))
        title = ('%s: %s: Held Unit #%i' % (exp_group, exp_name, unit_name))

        dat1 = load_dataset(rec1)
        dat2 = load_dataset(rec2)
        dim1 = dat1.dig_in_mapping.set_index('name')
        dim2 = dat2.dig_in_mapping.set_index('name')
        ch1 = dim1.loc['Saccharin', 'channel']
        ch2 = dim2.loc['Saccharin', 'channel']
        t1, psth1, _ = agg.get_psth(rec1, unit1, ch1, params)
        t2, psth2, _ = agg.get_psth(rec2, unit2, ch2, params)
        id1 = np.vstack(['preCTA']*psth1.shape[0])
        id2 = np.vstack(['postCTA']*psth2.shape[0])
        row_id = np.vstack([id1, id2])
        data = np.vstack([psth1, psth2])
        peak_time = np.argmax(data, axis=1)
        fig, ax = grouped_heatmap(data, row_id, peak_time, X=t1, smoothing_width=3)
        ax.set_title(title)
        fig.savefig(fn)
        plt.close(fig)


def plot_sacc_psths(held_df, save_dir):
    df = held_df.query('held_over == "cta" and area == "GC"').dropna(subset=['held_unit_name'])
    params = {'psth': {'win_size': 25, 'step_size': 25, 'smoothing_win': 0, 'plot_window': [-250, 1500]},
              'baseline_comparison': {'win_size': 1000}}
    for i, row in df.iterrows():
        rec1 = row['rec1']
        rec2 = row['rec2']
        unit1 = row['unit1']
        unit2 = row['unit2']
        unit_name = row['held_unit_name']
        exp_group= row['exp_group']
        exp_name = row['exp_name']
        fn = os.path.join(save_dir, '%s_unit-%i.png' % (exp_group, unit_name))
        title = ('%s: %s: Held Unit #%i' % (exp_group, exp_name, unit_name))

        dat1 = load_dataset(rec1)
        dat2 = load_dataset(rec2)
        dim1 = dat1.dig_in_mapping.set_index('name')
        dim2 = dat2.dig_in_mapping.set_index('name')
        ch1 = dim1.loc['Saccharin', 'channel']
        ch2 = dim2.loc['Saccharin', 'channel']
        t1, psth1, _ = agg.get_psth(rec1, unit1, ch1, params)
        t2, psth2, _ = agg.get_psth(rec2, unit2, ch2, params)
        fig, ax = plt.subplots(figsize=(15,10))
        for trace in psth1:
            ax.plot(t1, trace, color='b', linewidth=1, alpha=0.3)

        for trace in psth2:
            ax.plot(t2, trace, color='r', linewidth=1, alpha=0.3)

        mp1 = np.mean(psth1, axis=0)
        mp2 = np.mean(psth2, axis=0)
        ax.plot(t1, mp1, color='b', linewidth=3, label='preCTA')
        ax.plot(t2, mp2, color='r', linewidth=3, label='postCTA')
        ax.axvline(0, color='k', linewidth=1, linestyle='--')
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')
        fig.savefig(fn)
        plt.close(fig)


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
    spear_rs = np.abs(data['spearman_r'])
    time = data['time']
    exp_groups = np.unique(labels[:,0])
    time_groups = np.unique(labels[:,1])
    df = pd.DataFrame(spear_rs, columns=time)
    df['exp_group'] = labels[:,0]
    df['time_group'] = labels[:,1]
    df2 = pd.melt(df, id_vars=['exp_group','time_group'], value_vars=time, var_name='time', value_name='R')
    df2['R'] = df2['R'].abs()
    g = sns.FacetGrid(data=df2, row='exp_group', hue='time_group',
                      hue_order=['preCTA', 'postCTA'], sharex=True, sharey=False)
    g.map(sns.lineplot, 'time', 'R', markers=True)
    g.add_legend()
    #g = sns.catplot(data=df2, x='time', y='R', row='exp_group',
    #                hue='time_group', kind='point', sharex=True, sharey=False)
    g.set_titles('{row_name}')
    g.set_xlabels('Time (ms)')
    g.set_ylabels("Spearman's R")
    g.fig.set_size_inches(15,10)
    plt.subplots_adjust(top=0.85, left=0.1)
    g.fig.suptitle('Mean Spearman Correlation')
    g.fig.savefig(save_file)
    plt.close(g.fig)


def plot_mean_pearson(data_file, save_file):
    data = np.load(data_file)
    labels = data['labels']
    pear_rs = np.abs(data['pearson_r'])
    time = data['time']
    exp_groups = np.unique(labels[:,0])
    time_groups = np.unique(labels[:,1])
    df = pd.DataFrame(pear_rs, columns=time)
    df['exp_group'] = labels[:,0]
    df['time_group'] = labels[:,1]
    df2 = pd.melt(df, id_vars=['exp_group','time_group'], value_vars=time, var_name='time', value_name='R')
    g = sns.FacetGrid(data=df2, row='exp_group', hue='time_group',
                      hue_order=['preCTA', 'postCTA'], sharex=True, sharey=False)
    g.map(sns.lineplot, 'time', 'R', markers=True)
    g.add_legend()
    # g = sns.catplot(data=df2, x='time', y='R', row='exp_group',
    #                 hue='time_group', kind='point', sharex=True, sharey=False)
    g.set_titles('{row_name}')
    g.set_xlabels('Time (ms)')
    g.set_ylabels("Pearson's R")
    g.fig.set_size_inches(15, 10)
    plt.subplots_adjust(top=0.85, left=0.1)
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
    colors = TASTE_COLORS.copy()
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
    scatter_file = os.path.join(save_dir, 'PC_Distances.svg')
    hist_file = os.path.join(save_dir, 'PC_Distances-Histogram.svg')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    animals = df.exp_name.unique()
    n_anim = len(animals)
    time_groups = list(df.time_group.unique())
    rows = list(df.exp_group.unique())
    cols = list(df.time.unique())
    n_cols = len(rows)
    n_rows = len(cols)
    colors = {a: c for a,c in zip(animals, sns.color_palette('bright', n_anim))}
    tmp_shapes = list('o^sDpP*X8')
    shapes = {a: c for a,c in zip(time_groups, tmp_shapes)}

    s_fig = plt.figure(figsize=(16,10))
    add_suplabels(s_fig, 'PCA Distances', 'Sacc -- Quinine', 'Sacc -- NaCl')
    limits = {i: {'xlim': [df.pc_dist.max(), df.pc_dist.min()],
                  'ylim': [df.pc_dist.max(), df.pc_dist.min()]} for i in range(n_rows*n_cols)}
    axes = [s_fig.add_subplot(n_rows, n_cols, i+1) for i in range(n_rows*n_cols)]
    new_data = []

    # make custom legend from colors and shapes dictionaries
    for name, group in df.groupby(['exp_group', 'time', 'exp_name', 'time_group']):
        r_grp = name[0]
        c_grp = name[1]
        anim = name[2]
        time = name[3]
        plt_i = (cols.index(c_grp)) + (n_cols * rows.index(r_grp))
        ax = axes[plt_i]
        plt_color = colors[anim]
        plt_shape = shapes[time]
        dNaCl = group[group.taste_1.isin(['Saccharin', 'NaCl']) &
                      group.taste_2.isin(['Saccharin', 'NaCl'])]
        dQ = group[group.taste_1.isin(['Saccharin', 'Quinine']) &
                   group.taste_2.isin(['Saccharin', 'Quinine'])]
        if len(dNaCl) > 1 or len(dQ) > 1:
            raise ValueError('Too many points')

        # For each animal plot dQ vs dNaCl, dot at mean and oval for error
        xlim = limits[plt_i]['xlim']
        ylim = limits[plt_i]['ylim']
        for i, nacl in dNaCl.iterrows():
            for j, quin in dQ.iterrows():
                x = quin['pc_dist']
                dx = quin['pc_dist_sem']
                y = nacl['pc_dist']
                dy = nacl['pc_dist_sem']
                e = Ellipse((x,y), 2*dx, 2*dy, color=plt_color, alpha=0.4)
                ax.add_patch(e)
                ax.plot(x, y, color=plt_color, marker=plt_shape)
                if xlim[0] > x-dx:
                    xlim[0] = x-dx

                if xlim[1] < x+dx:
                    xlim[1] = x+dx

                if ylim[0] > y-dy:
                    ylim[0] = y-dy

                if ylim[1] < y+dy:
                    ylim[1] = y+dy

                new_data.append((r_grp, c_grp, anim, time, x/y, np.abs(x/y) * ((dx/x)**2 + (dy/y)**2)**0.5))

        if cols.index(c_grp) == 0:
            tmp = ax.get_position()
            s_fig.text(0.05, (tmp.ymax + tmp.ymin)/2, r_grp, fontsize=18, fontweight='bold')

        if rows.index(r_grp) == 0:
            ax.set_title(c_grp, pad=10)

        limits[plt_i]['xlim'] = xlim
        limits[plt_i]['ylim'] = ylim

    s_fig.subplots_adjust(left=0.15, right=0.9)
    for i, ax in enumerate(axes):
        xlim = limits[i]['xlim']
        ylim = limits[i]['ylim']
        xs = np.diff(xlim) * 0.05
        ys = np.diff(ylim) * 0.05
        xlim[0] -= xs
        xlim[1] += xs
        ylim[0] -= ys
        ylim[1] += ys
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # make Legend
    handles = []
    for k,v in shapes.items():
        item = Line2D([], [], marker=v, color='k', linewidth=0, label=k)
        handles.append(item)

    for k, v in colors.items():
        item = Patch(color=v, label=k)
        handles.append(item)

    s_fig.legend(handles=handles, loc='center right')
    s_fig.savefig(scatter_file)
    plt.close(s_fig)

    new_df = pd.DataFrame(new_data, columns=['exp_group', 'time',
                                             'exp_name', 'time_group', 'dQ_v_dN',
                                             'dQ_v_dN_std'])
    g = sns.FacetGrid(data=new_df, col='time', row='exp_group', hue='time_group', sharex=False, sharey=False)
    g.map(sns.distplot, 'dQ_v_dN')
    # g = sns.catplot(data=new_df, hue='time_group', col='exp_group', x='time',
    #                 y='dQ_v_dN', kind='violin')
    g.add_legend()
    g.fig.set_size_inches(16, 10)
    g.set_titles('{row_name}: {col_name}')
    g.set_xlabels('d$_Q$/d$_{NaCl}$')
    # g.set_xlabels('Post-stimulus response time')
    # g.fig.subplots_adjust(left=0.15)
    # for ax,_ in g.axes:
    #     tmp = ax.get_position()
    #     s_fig.text(0.05, (tmp.ymax + tmp.ymin)/2, r_grp, fontsize=18, fontweight='bold')

    g.fig.savefig(hist_file)
    plt.close(g.fig)

    # TODO: Plot histogram of distance from each saccharin trial to mean Qunine / distance to mean NaCl


def plot_pca_metric(df, save_file):
    # grouping: exp_group, exp_name, time_group, time, trial
    g = sns.catplot(data=df, x='exp_group', hue='time_group', col='time',
                    y='PC_dQ_v_dN', kind='bar',
                    hue_order=['preCTA', 'postCTA'],
                    col_order=['Early (0-750ms)', 'Late (750-1500ms)'])
    g.set_titles('{col_name}')
    g.set_ylabels('d(Sacc, Q)/d(Sacc, NaCl)')
    g.fig.set_size_inches((16,10))
    plt.close(g.fig)
    g.fig.savefig(save_file)


def plot_mds_metric(df, save_file):
    # grouping: exp_group, exp_name, time_group, time, trial
    g = sns.catplot(data=df, x='exp_group', hue='time_group', col='time',
                    y='MDS_dQ_v_dN', kind='bar',
                    hue_order=['preCTA', 'postCTA'],
                    col_order=['Early (0-750ms)', 'Late (750-1500ms)'])
    g.set_titles('{col_name}')
    g.set_ylabels('d(Sacc, Q)/d(Sacc, NaCl)')
    g.fig.set_size_inches((16,10))
    plt.close(g.fig)
    g.fig.savefig(save_file)


def plot_animal_pca(df, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for name, grp in df.groupby(['exp_name']):
        fn = os.path.join(save_dir, '%s_pca_analysis.svg' % name)
        g = sns.FacetGrid(data=grp, col='time_group', hue='taste', row='time', sharex=True, col_order=['preCTA', 'postCTA'])
        g.map(sns.scatterplot,'PC1', 'PC2')
        g.add_legend()
        g.fig.set_size_inches(16,10)
        g.set_titles('{row_name} : {col_name}')
        g.fig.savefig(fn)
        plt.close(g.fig)

        n_grps = len(grp.time_group.unique())
        fig, axes = plt.subplots(ncols=n_grps, figsize=(16,7))
        fn = os.path.join(save_dir, '%s_mean_pca.svg' % name)
        for time_group, tmp in grp.groupby('time_group'):
            if n_grps == 1:
                ax = axes
            elif time_group == 'preCTA':
                ax = axes[0]
            else:
                ax = axes[1]

            means = tmp.groupby(['taste', 'time'])[['PC1','PC2']].mean().reset_index()
            g = sns.scatterplot(data=means, hue='taste', style='time', ax=ax, x='PC1', y='PC2')
            g.set_title(time_group)

        g.legend()
        fig.savefig(fn)
        plt.close(fig)


def plot_animal_mds(df, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for name, grp in df.groupby(['exp_name']):
        fn = os.path.join(save_dir, '%s_mds_analysis.svg' % name)
        g = sns.FacetGrid(data=grp, col='time_group', hue='taste', row='time', sharex=True, col_order=['preCTA', 'postCTA'])
        g.map(sns.scatterplot,'MDS1', 'MDS2')
        g.add_legend()
        g.fig.set_size_inches(16,10)
        g.set_titles('{row_name} : {col_name}')
        g.fig.savefig(fn)
        plt.close(g.fig)

        n_grps = len(grp.time_group.unique())
        fig, axes = plt.subplots(ncols=n_grps, figsize=(16,7))
        fn = os.path.join(save_dir, '%s_mean_mds.svg' % name)
        for time_group, tmp in grp.groupby('time_group'):
            if n_grps == 1:
                ax = axes
            elif time_group == 'preCTA':
                ax = axes[0]
            else:
                ax = axes[1]

            means = tmp.groupby(['taste', 'time'])[['MDS1','MDS2']].mean().reset_index()
            g = sns.scatterplot(data=means, hue='taste', style='time', ax=ax, x='MDS1', y='MDS2')
            g.set_title(time_group)

        g.legend()
        fig.savefig(fn)
        plt.close(fig)


def add_suplabels(fig, title, xlabel, ylabel):
    ax = fig.add_subplot('111')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=40)
    ax.set_title(title, pad=40)
    return ax


def plot_state_breakdown(df, save_dir):
    plot_vars = ['t_end', 't_start', 'duration', 'cost']
    id_vars = ['rec_dir', 'exp_name', 'exp_group', 'time_group', 'rec_group',
               'rec_name', 'whole_trial', 'ordered_state',
               'trial_ordered_state', 'n_states', 'trial', 'n_cells', 'taste']
    new_df = pd.melt(df, id_vars=id_vars, value_vars=plot_vars, var_name='plot_var', value_name='value')
    # First state end times
    for v in plot_vars:
        tmp_df = new_df[new_df.plot_var == v]
        g = sns.FacetGrid(tmp_df, hue='exp_group', col='time_group', row='ordered_state')
        g.map(sns.boxplot, 'taste', 'value')
        g.map(sns.swarmplot, 'taste', 'value', color=0.25)
        g.set_titles('{row_name} : {col_name}')
        g.fig.set_size_inches(16,10)
        add_suplabels(g.fig, v, 'Taste', v)
        fn = os.path.join(save_dir, 'hmm_%s.svg' % v)
        g.fig.savefig(fn)
        plt.close(g.fig)


def plot_hmm(rec_dir, hmm_id, save_file=None, hmm=None, params=None, title_extra=None):
    # Grab data
    if rec_dir[-1] == os.sep:
        rec_dir = rec_dir[:-1]

    parsed = os.path.basename(rec_dir).split('_')
    anim = parsed[0]
    rec_group = parsed[-3]

    if hmm is None or params is None:
        handler = phmm.HmmHandler(rec_dir)
        ho = handler.get_data_overview().set_index('hmm_id')
        if hmm_id not in ho.index:
            return
        else:
            row = ho.loc[hmm_id]

        h5_file = handler.h5_file
        hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
    else:
        row = params.copy()
        row['n_iterations'] = hmm.iteration

    thresh = row['threshold']
    taste = row['taste']
    title = '%s %s\nHMM #%i: %s' % (anim, rec_group, hmm_id, taste)
    if title_extra is not None:
        title = f'{title}\n{title_extra}'

    n_trials = row['n_trials']
    n_states = row['n_states']
    colors = hplt.get_hmm_plot_colors(n_states)

    spikes, dt, time = phmm.get_hmm_spike_data(rec_dir, row['unit_type'],
                                               row['channel'],
                                               time_start=row['time_start'],
                                               time_end=row['time_end'],
                                               dt=row['dt'], trials=n_trials)
    seqs = hmm.stat_arrays['best_sequences']
    gamma_probs = hmm.stat_arrays['gamma_probabilities']
    ll_hist = hmm.stat_arrays['fit_LL'][1:]

    fig = plt.figure(figsize=(24,10), constrained_layout=False)
    gs0 = fig.add_gridspec(1,3)
    gs00 = gs0[0].subgridspec(2,1, hspace=0.4)
    gs01 = gs0[1].subgridspec(n_trials, 1)
    gs001 = gs00[1].subgridspec(1, n_states)
    gs02 = gs0[2].subgridspec(n_trials, 1)

    llax = fig.add_subplot(gs00[0])
    llax.plot(np.arange(1, len(ll_hist)+1), ll_hist, linewidth=1, color='k', alpha=0.5)
    llax.set_ylabel('Max Log Likelihood')
    llax.set_xlabel('Iteration')
    # Also mark where change in LL first dropped below threshold and the last time it dropped below thresh and stayed below
    # Also mark the maxima
    if len(ll_hist) > 0:
        llax.axvline(row['n_iterations'], color='g')
        diff_ll = np.diff(ll_hist)
        below = np.where(np.abs(diff_ll) < thresh)[0]
        if len(below) > 0:
            n1 = below[0]
            llax.axvline(n1, color='r', linestyle='--')
            changes = np.where(np.diff(below) > 1)[0]
            if len(changes) > 0:
                n2 = below[changes[-1]+1]
                llax.axvline(n2, color='m', linestyle='--')

    for i, (seq, trial, gamma) in enumerate(zip(seqs, spikes, gamma_probs)):
        ax = fig.add_subplot(gs01[i])
        gax = fig.add_subplot(gs02[i])
        hplt.plot_raster(trial, time=time, ax=ax)
        hplt.plot_sequence(seq, time=time, ax=ax, colors=colors)
        hplt.plot_probability_traces(gamma, time=time, ax=gax, colors=colors)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for spine in gax.spines.values():
            spine.set_visible(False)

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel(i)
        if time[0] < 0:
            ax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

        if ax.is_last_row():
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel('Time (ms)')

        gax.get_yaxis().set_visible(False)
        gax.get_xaxis().set_visible(False)
        gax.set_ylabel(i)
        if time[0] < 0:
            gax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

        if gax.is_last_row():
            gax.get_xaxis().set_visible(True)
            gax.set_xlabel('Time (ms)')

    tmp_ax = fig.add_subplot(gs0[1], frameon=False)
    tmp_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    tmp_ax.set_ylabel('Trials', labelpad=-25)

    tmp_ax = fig.add_subplot(gs0[2], frameon=False)
    tmp_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)

    rates = hmm.emission
    rate_axes = [fig.add_subplot(gs001[x]) for x in range(n_states)]
    hplt.plot_hmm_rates(rates, axes=rate_axes, colors=colors)
    rate_axes[-1].set_xlabel('')
    tmp_ax = fig.add_subplot(gs00[1], frameon=False)
    tmp_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    tmp_ax.set_xlabel('Firing Rate (Hz)')
    fig.subplots_adjust(top=0.85)
    fig.suptitle(title)

    if save_file is None:
        return fig
    else:
        fig.savefig(save_file)
        plt.close(fig)
        return None


def plot_hmm_sequence_heatmap(best_hmms, group_col, save_file=None):
    hmms = best_hmms.dropna(subset=['hmm_id'])
    tastes = best_hmms.taste.unique()
    fig, axes = plt.subplots(ncols=len(tastes), figsize=(5+5*len(tastes), 10))
    time_vec = None
    if len(tastes) == 1:
        axes = [axes]

    for ax, tst in zip(axes, tastes):
        df = hmms.query('taste == @tst')
        sequences = []
        row_id = []
        sort_stat = []
        for i, row in df.iterrows():
            h5_file = get_hmm_h5(row['rec_dir'])
            hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, row['hmm_id'])
            if time_vec is None:
                time_vec = time
            elif not np.array_equal(time_vec, time):
                continue

            seqs = hmm.stat_arrays['best_sequences']

            # Only include trials where both early and late states are present
            min_pts = int(50 / (1000*params['dt']))
            es, ls = row[['early_state', 'late_state']]
            # e_trials = agg.get_valid_trials(seqs, row['early_state'], min_pts=min_pts, time=time)
            # l_trials = agg.get_valid_trials(seqs, row['late_state'], min_pts=min_pts, time=time)
            # valid_trials = np.intersect1d(e_trials, l_trials)
            valid_trials = agg.get_valid_trials(seqs, [es, ls], min_pts=min_pts, time=time)
            if len(valid_trials) == 0:
                print('No valid trials found for %s %s %s' % (row['exp_name'], row['rec_group'], tst))
                continue

            seqs = seqs[valid_trials, :]

            # Convert es to 1, ls to 2, all others to negative numbers
            s2 = seqs.copy()
            tmp_i = 0
            for n in range(params['n_states']):
                if n == es:
                    m = 1
                elif n == ls:
                    m = 2
                else:
                    m = tmp_i
                    tmp_i = tmp_i - 1

                seqs[s2 == n] = m

            sequences.append(seqs)
            rid = '%s\n%s' % (row[group_col], row['time_group'])
            row_id.extend([rid]*seqs.shape[0])
            # Sort by late state start time
            for trial in seqs:
                if row['late_state'] in trial:
                    sort_stat.append(np.where(trial==row['late_state'])[0][0])
                else:
                    sort_stat.append(-1)

        sort_stat = np.array(sort_stat)
        row_id = np.array(row_id)
        sequences = np.vstack(sequences)
        if ax.is_last_col():
            cbar = True
        else:
            cbar = False

        _, ax = grouped_heatmap(sequences, row_id, sort_stat, X=time, ax=ax, cbar=cbar)
        #ax.set_xticks(np.linspace(time[0], time[-1], 10))
        if time[0] <= 0:
            ax.axvline(np.where(time==0)[0], color='b', linewidth=3, linestyle='--')

        ax.yaxis.set_tick_params(length=0, labelrotation=0)
        ax.set_title(tst)

    fig.subplots_adjust(top=0.85)
    fig.suptitle('HMM State Sequences')

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, ax


def plot_classifier_results(group, early_res, late_res,
                            label=None, save_file=None):
    rec_dir = group.rec_dir.unique()[0]
    rec_name = os.path.basename(rec_dir).split('_')
    rec_name = '_'.join(rec_name[:-2])
    h5_file = get_hmm_h5(rec_dir)
    title = rec_name
    title += ('\nEarly Acc.: %0.2f%%, Late Acc.: %0.2f%%'
              % (early_res.accuracy, late_res.accuracy))
    if label is not None:
        title += '\n' + label

    colors = TASTE_COLORS.copy()
    early_colors = {k: change_hue(c, 0.4) for k,c in colors.items()}
    late_colors = {k: change_hue(c, 1.1) for k,c in colors.items()}

    # First make sure rows align with early and late
    early_id = early_res.row_id # rec_dir, hmm_id, taste, trial_#
    late_id = late_res.row_id
    early_pred = early_res.Y_predicted
    late_pred = late_res.Y_predicted
    n_rows = early_id.shape[0]
    spikes = []
    sequences = []
    data_id = [] # taste, trial_#, early_state, late_state, n_states
    time = None
    for i, row in group:
        hmm_id = row['hmm_id']
        hmm , t, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
        channel = params['channel']
        n_trials = params['n_trials']
        t_start = params['t_start']
        t_end = params['t_end']
        dt = params['dt']
        early_state = row['early_state']
        late_state = row['late_state']
        spike_array, _, s_time = phmm.get_hmm_spike_data(rec_dir, unit_type,
                                                         channel,
                                                         time_start=t_start,
                                                         time_end=t_end, dt=dt,
                                                         trials=n_trials)
        spikes.append(spike_array)
        sequences.append(hmm.best_sequences)
        data_id.extend([(row['taste'], x, early_state, late_state, row['n_states'])
                        for x in np.arange(0, n_trials)])
        if time is None:
            time = s_time
        elif not np.array_equal(time, s_time):
            raise ValueError('Time vectors do not match')

    spikes = np.vstack(spikes)
    sequences = np.vstack(sequences)
    data_id = np.array(data_id)

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(16, 15))
    for i, (spike_arr, seq, row_id) in enumerate(zip(spikes, sequences, data_id)):
        taste, trial, es, ls, n_states = row_id
        eidx = np.where((early_id[:,2] == taste) & (early_id[:,3] == trial))[0]
        lidx = np.where((late_id[:,2] == taste) & (late_id[:,3] == trial))[0]
        e_pred = early_pred[eidx]
        l_pred = late_pred[lidx]
        colors = [plt.cm.gray(x) for x in np.linspace(0.1, 0.7, n_states-2)]
        real_colors = colors.copy()
        pred_colors = colors.copy()
        real_colors.insert(es, early_colors[taste])
        real_colors.insert(ls, late_colors[taste])
        if len(e_pred) == 0:
            pred_colors.insert(es, early_colors[e_pred])
        else:
            pred_colors.insert(es, mc.to_rgba('w'))

        if len(l_pred) == 0:
            pred_colors.insert(ls, late_colors[l_pred])
        else:
            pred_colors.insert(ls, mc.to_rgba('w'))

        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        if ax1.is_first_row():
            ax1.set_title('Actual')
            ax2.set_title('Predicted')

        hplt.plot_raster(spike_arr, time=time, ax=ax1)
        hplt.plot_raster(spike_arr, time=time, ax=ax2)
        hplt.plot_sequence(seq, time=time, ax=ax1, colors=real_colors)
        hplt.plot_sequence(seq, time=time, ax=ax2, colors=pred_colors)

        for spine in ax1.spines.values():
            spine.set_visible(False)

        for spine in ax2.spines.values():
            spine.set_visible(False)

        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        if time[0] < 0:
            ax1.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)
            ax2.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

        if ax1.is_last_row():
            ax1.get_xaxis().set_visible(True)
            ax1.set_xlabel('Time (ms)')
            ax2.get_xaxis().set_visible(True)
            ax2.set_xlabel('Time (ms)')

    fig.subplots_adjust(top=0.85)
    fig.suptitle(title)
    if save_file is None:
        return fig
    else:
        fig.savefig(save_file)
        plt.close(fig)


def plot_pal_classifier_results(group, early_res, late_res,
                                label=None, save_file=None):
    rec_dir = group.rec_dir.unique()[0]
    rec_name = os.path.basename(rec_dir).split('_')
    rec_name = '_'.join(rec_name[:-2])
    h5_file = get_hmm_h5(rec_dir)
    title = rec_name
    title += ('\nEarly Acc.: %0.2f%%, Late Acc.: %0.2f%%'
              % (early_res.accuracy, late_res.accuracy))
    if label is not None:
        title += '\n' + label

    colors = TASTE_COLORS.copy()

    # First make sure rows align with early and late
    early_id = early_res.row_id # rec_dir, hmm_id, taste, palatability, trial_#
    late_id = late_res.row_id
    early_pred = early_res.Y_predicted
    late_pred = late_res.Y_predicted
    n_rows = early_id.shape[0]

    # Make the assumption that each palatability maps to a single taste
    pal_taste_map = {p: t for t,p in np.unique(early_id[:, 2:4], axis=0)}

    fig, axes = plt.subplots(nrows=2, ncols=2)
    early_real_colors = [colors[x[2]] for x in early_id]
    late_real_colors = [colors[x[2]] for x in late_id]
    early_pred_colors = [colors[pal_taste_map[x]] for x in early_pred]
    late_pred_colors = [colors[pal_taste_map[x]] for x in late_pred]
    eX = early_res.X
    lX = late_res.X
    axes[0,0].scatter(eX[:,0], eX[:,1], c=early_real_colors, alpha=0.6)
    axes[1,0].scatter(eX[:,0], eX[:,1], c=early_pred_colors, alpha=0.6)
    axes[0,1].scatter(lX[:,0], lX[:,1], c=late_real_colors, alpha=0.6)
    axes[1,1].scatter(lX[:,0], lX[:,1], c=late_pred_colors, alpha=0.6)

    axes[0,0].set_ylabel('Actual\nLDA #2')
    axes[0,0].set_title('Early State')
    axes[1,0].set_ylabel('Predicted\nLDA #2')
    axes[1,0].set_xlabel('LDA #1')
    axes[0,1].set_title('Late State')
    axes[1,1].set_xlabel('LDA #1')
    fig.subplots_adjust(top=0.85)
    fig.suptitle(title)
    if save_file is None:
        return fig
    else:
        fig.savefig(save_file)
        plt.close(fig)


def plot_regression_results(group, state_set, early_pal=None, late_pal=None,
                            label=None, save_file=None):
    # Plot Residuals
    # Run PCA, plot 
    # Actually should be doing LDA classification instead of linear regression
    pass


def change_hue(color, amount):
    '''lightens (amount < 1) or darkens (amount > 1) an rgb color.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    '''
    try:
        c = mc.cnames[color]
    except:
        c = color
        
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    return tmp[0]


def plot_hmm_sequences(hmm, time, n_baseline=None, row_id=None, save_file=None):
    '''row_id should be a 1-D vector that groups trials
    '''
    paths = hmm.stat_arrays['best_sequences']
    gamma = hmm.stat_arrays['gamma_probabilities']
    n_states = hmm.n_states
    if row_id is None:
        # Single column of plots
        row_id = np.repeat(0, paths.shape[0])

    if n_baseline is None:
        colors = hplt.get_hmm_plot_colors(n_states)
    else:
        colors = [plt.cm.gray(x) for x in np.linspace(0.1, 0.8, n_baseline)]
        n_tastes = int((n_states - n_baseline)/2)
        t_colors = [plt.cm.tab10(x) for x in np.linspace(0,1, n_tastes)]
        for col in t_colors:
            colors.append(change_hue(col, 0.4))
            colors.append(change_hue(col, 1.1))


    groups = np.unique(row_id)
    n_col = len(groups)
    n_row = np.max([np.sum(row_id == g) for g in groups])
    fig, axes = plt.subplots(ncols=n_col, nrows=n_row, figsize=(8*n_col, n_row))
    if n_col == 1:
        axes = np.expand_dims(axes, 1)

    if n_row == 1:
        axes = np.expand_dims(axes, 0)

    handles = []
    labels = []
    for col, grp in enumerate(groups):
        idx = np.where(row_id == grp)[0]
        for row in range(len(idx)):
            seq = paths[idx[row], :]
            _, tmp_h, tmp_l = hplt.plot_sequence(seq, time=time, colors=colors,
                                                 ax=axes[row, col])
            for l, h in zip(tmp_l, tmp_h):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)

            if time[0] != 0:
                axes[row, col].axvline(0, color='red', linestyle='--',
                                       linewidth=3, alpha=0.8)

        axes[0, col].set_title(grp)
        axes[-1, col].set_xlabel('Time (ms)')

    fig.subplots_adjust(right=0.9)
    mid = int(n_row/2)
    axes[mid, -1].legend(handles, labels, loc='upper center',
                         bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                         fontsize=14)
    if save_file is None:
        return fig, axes
    else:
        fig.savefig(save_file)
        plt.close(fig)
        return None, None


def plot_hmm_coding_accuracy(df, save_file=None):
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18,24))
    time_order = ['preCTA', 'postCTA']
    exp_order = ['Cre', 'GFP']
    x_group = 'exp_group'
    hue_group = 'time_group'
    x_order = exp_order
    hue_order = time_order

    df = df.dropna(subset=['early_ID_acc', 'late_ID_acc', 'early_pal_acc', 'late_pal_acc'])
    g = sns.barplot(data=df, x=x_group, hue=hue_group,
                    hue_order=hue_order, order=x_order,
                    ax=axes[0,0], y='early_ID_acc')
    g.axhline(100/3, color='k', linestyle='--', alpha=0.6)
    g.legend_.remove()
    g.set_ylabel('Early\nAcc')
    g.set_xlabel('')
    g.set_title('ID Classification')

    g = sns.barplot(data=df, x=x_group, hue=hue_group,
                    hue_order=hue_order, order=x_order,
                    ax=axes[1,0], y='late_ID_acc')
    g.axhline(100/3, color='k', linestyle='--', alpha=0.6)
    g.legend_.remove()
    g.set_ylabel('Late\nAcc')
    g.set_xlabel('')
    #g.set_title('Late-State ID Classification')

    g = sns.barplot(data=df, x=x_group, hue=hue_group,
                    hue_order=hue_order, order=x_order,
                    ax=axes[0,1], y='early_pal_acc')
    g.axhline(100/3, color='k', linestyle='--', alpha=0.6)
    g.legend_.remove()
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_title('Palatability Classification')

    g = sns.barplot(data=df, x=x_group, hue=hue_group,
                    hue_order=hue_order, order=x_order,
                    ax=axes[1,1], y='late_pal_acc')
    g.axhline(100/3, color='k', linestyle='--', alpha=0.6)
    g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    g.set_ylabel('')
    g.set_xlabel('')
    #g.set_title('Late-State Palatability Classification')

    # Scatter plots to compare early & late classifier accuracy
    g = sns.scatterplot(data=df, x='early_ID_acc', y='late_ID_acc',
                        hue='exp_group', hue_order=exp_order, ax=axes[4,0])
    min_v = df[['early_ID_acc', 'late_ID_acc']].min().min()
    max_v = df[['early_ID_acc', 'late_ID_acc']].max().max()
    g.plot([min_v, max_v], [min_v, max_v], color='k', linestyle='--', alpha=0.6)
    g.legend_.remove()
    g.set_ylabel('Late ID\nAcc')
    g.set_xlabel('Early ID ACC')

    g = sns.scatterplot(data=df, x='early_pal_acc', y='late_pal_acc',
                        hue='exp_group', hue_order=exp_order, ax=axes[4,1])
    min_v = df[['early_pal_acc', 'late_pal_acc']].min().min()
    max_v = df[['early_pal_acc', 'late_pal_acc']].max().max()
    g.plot([min_v, max_v], [min_v, max_v], color='k', linestyle='--', alpha=0.6)
    g.legend_.remove()
    g.set_ylabel('')
    g.set_xlabel('Early Pal ACC')

    # Bar plot of confusion
    def get_confusion(tup):
        # compute confusion value from tuple with (n_NaCl, n_Quinine)
        # right now I'm outputing % nacl classifications
        return 100  * tup[0] / np.sum(tup)

    # df['eid_con'] = df['early_ID_confusion'].apply(get_confusion)
    # df['lid_con'] = df['late_ID_confusion'].apply(get_confusion)
    # df['epal_con'] = df['early_pal_confusion'].apply(get_confusion)
    # df['lpal_con'] = df['late_pal_confusion'].apply(get_confusion)
    df = df.copy()
    df['eid_con'] = df['early_ID_confusion']
    df['lid_con'] = df['late_ID_confusion']
    df['epal_con'] = df['early_pal_confusion']
    df['lpal_con'] = df['late_pal_confusion']

    g = sns.barplot(data=df, x=x_group, order=x_order, hue=hue_group,
                    hue_order=hue_order, ax=axes[2,0], y='eid_con')
    g.legend_.remove()
    g.set_ylabel('Early Con\n% NaCl')
    g.set_xlabel('')

    g = sns.barplot(data=df, x=x_group, order=x_order, hue=hue_group,
                    hue_order=hue_order, ax=axes[3,0], y='lid_con')
    g.legend_.remove()
    g.set_ylabel('Late Con\n% NaCl')
    g.set_xlabel('')

    g = sns.barplot(data=df, x=x_group, order=x_order, hue=hue_group,
                    hue_order=hue_order, ax=axes[2,1], y='epal_con')
    g.legend_.remove()
    g.set_ylabel('')
    g.set_xlabel('')

    g = sns.barplot(data=df, x=x_group, order=x_order, hue=hue_group,
                    hue_order=hue_order, ax=axes[3,1], y='lpal_con')
    g.legend_.remove()
    g.set_ylabel('')
    g.set_xlabel('')

    fig.subplots_adjust(top=0.9)
    fig.suptitle('HMM Population Coding')
    #fig.tight_layout()

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return None
    else:
        return fig


def plot_hmm_timings(df, save_file=None):
    # time_group- col, state_group - row, exp_group - hue, taste - x
    # row 1: early state end time
    # row 2: late state start time
    # row 3: early state durations
    # row 4: late state durations
    df = df.query('valid == True').copy()
    exp_order = ['Cre', 'GFP']
    #taste_order = ['Water', 'Citric Acid', 'Quinine', 'NaCl', 'Saccharin']
    taste_order = ['Saccharin']
    time_order = ['preCTA', 'postCTA']
    # _ = taste_order.pop(0)
    late_df = df.query('state_group == "late"')
    early_df = df.query('state_group == "early"')
    n_tastes = len(taste_order)
    fig, axes = plt.subplots(nrows=4, ncols=n_tastes, figsize=(5*n_tastes+5,18))
    if axes.ndim == 1:
        axes = np.expand_dims(axes, 1)

    # Row 1
    for ci, taste in enumerate(taste_order):
        edf = early_df.query('taste == @taste')
        ldf = late_df.query('taste == @taste')
    # df1 = early_df.query('time_group == "preCTA"')
    # df2 = early_df.query('time_group == "postCTA"')
        g = sns.boxplot(data=edf, x='exp_group', order=exp_order, hue='time_group',
                        hue_order=time_order, ax = axes[0,ci], y='t_end')
        g.set_xlabel('')
        if axes[0, ci].is_first_col():
            g.set_ylabel('Early\nEnd')
        else:
            g.set_ylabel('')

        g.legend_.remove()
        g.set_title(taste)

    # g = sns.boxplot(data=df2, x='exp_group', order=exp_order, hue='time_group',
    #                 hue_order=time_order, ax = axes[0,1], y='t_end')
    # g.set_xlabel('')
    # g.set_ylabel('')
    # g.legend_.remove()
    # g.set_title('Post-CTA')

        # Row 2
    # df1 = late_df.query('time_group == "preCTA"')
    # df2 = late_df.query('time_group == "postCTA"')
        g = sns.boxplot(data=ldf, x='exp_group', order=exp_order, hue='time_group',
                        hue_order=time_order, ax = axes[1,ci], y='t_start')
        g.set_xlabel('')
        if axes[0, ci].is_first_col():
            g.set_ylabel('Late\nStart')
        else:
            g.set_ylabel('')

        g.legend_.remove()

    # g = sns.boxplot(data=df2, x='taste', order=taste_order, hue='exp_group',
    #                 hue_order=exp_order, ax = axes[1,1], y='t_start')
    # g.set_xlabel('')
    # g.set_ylabel('')
    # g.legend_.remove()

    # Row 3
    # df1 = early_df.query('time_group == "preCTA"')
    # df2 = early_df.query('time_group == "postCTA"')
        g = sns.boxplot(data=edf, x='exp_group', order=exp_order, hue='time_group',
                        hue_order=time_order, ax = axes[2,ci], y='duration')
        g.set_xlabel('')
        if axes[0, ci].is_first_col():
            g.set_ylabel('Early\nDuration')
        else:
            g.set_ylabel('')

        if axes[2, ci].is_last_col():
            g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            g.legend_.remove()

    # g = sns.boxplot(data=df2, x='taste', order=taste_order, hue='exp_group',
    #                 hue_order=exp_order, ax = axes[2,1], y='duration')
    # g.set_xlabel('')
    # g.set_ylabel('')
    # g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Row 3
    # df1 = late_df.query('time_group == "preCTA"')
    # df2 = late_df.query('time_group == "postCTA"')
        g = sns.boxplot(data=ldf, x='exp_group', order=exp_order, hue='time_group',
                        hue_order=time_order, ax = axes[3,ci], y='duration')
        g.set_xlabel('')
        if axes[0, ci].is_first_col():
            g.set_ylabel('Late\nDuration')
        else:
            g.set_ylabel('')

        g.legend_.remove()

    # g = sns.boxplot(data=df2, x='taste', order=taste_order, hue='exp_group',
    #                 hue_order=exp_order, ax = axes[3,1], y='duration')
    # g.set_xlabel('')
    # g.set_ylabel('')
    # g.legend_.remove()

    fig.subplots_adjust(top=0.9, right=0.85, left=0.15)
    fig.suptitle('HMM State Timing')
    #fig.tight_layout()

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return None
    else:
        return fig


def plot_median_gamma_probs(best_hmms, save_file=None):
    # Drop GFP animals that did not learn CTA and Cre animals that did learn
    # tmp = best_hmms.query('(exp_group == "Cre" and cta_group == "No CTA") or
    # (cta_group == "CTA" and exp_group == "GFP")')
    # best_hmms = tmp.copy()
    best_hmms = best_hmms.dropna()

    # diff line style for each time_group
    # diff color for each exp_group
    # Not dropping single state trials
    fig, ax = plt.subplots(nrows=2, figsize=(8,11))
    time_groups = best_hmms.time_group.unique()
    exp_groups = best_hmms.exp_group.unique()
    colors = EXP_COLORS
    hues = {'preCTA': 0.4, 'postCTA': 1.1}
    line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
    line_styles = {k:v for k,v in zip(time_groups, line_styles)}
    t_start = 0
    t_end = 1500
    dt = None
    for name, group in best_hmms.groupby(['exp_group', 'time_group']):
        eg = name[0]
        tg = name[1]
        early_traces = []
        late_traces = []
        time = None
        for i, row in group.iterrows():
            if np.isnan(row['early_state']) or np.isnan(row['late_state']):
                continue

            anim, rec_group, taste = row[['exp_name', 'rec_group', 'taste']]
            h5_file = get_hmm_h5(row['rec_dir'])
            hmm, t, params = phmm.load_hmm_from_hdf5(h5_file, int(row['hmm_id']))
            es = int(row['early_state'])
            ls = int(row['late_state'])
            gamma_probs = hmm.stat_arrays['gamma_probabilities']
            state_seqs = hmm.stat_arrays['best_sequences']
            t_idx = np.where((t <= t_end) & (t >= t_start))[0]
            eprobs = gamma_probs[:, es, t_idx]
            lprobs = gamma_probs[:, ls, t_idx]
            t = t[t_idx]
            state_seqs = state_seqs[:, t_idx]

            # Drop any trials where the state in question is not present in the
            # decoded sequence, or the duration of the state is less than 50ms
            valid_trials = agg.get_valid_trials(state_seqs, [es, ls],
                                            min_pts=int(.05/params['dt']),
                                            time=t)
            # e_trials = agg.get_valid_trials(state_seqs, es, min_pts=int(.05/params['dt']), time=t)
            # l_trials = agg.get_valid_trials(state_seqs, ls, min_pts=int(.05/params['dt']), time=t)
            # Only plot trials where both states are present
            #valid_trials = np.intersect1d(e_trials, l_trials)
            if len(valid_trials) == 0:
                print(f'No valid trials for {anim} - {rec_group} - {taste}')
                continue

            eprobs = eprobs[valid_trials, :]
            lprobs = lprobs[valid_trials, :]


            emed = np.median(eprobs, axis=0)
            lmed = np.median(lprobs, axis=0)
            early_traces.append(emed)
            late_traces.append(lmed)
            if time is None:
                time = t
                dt = params['dt']
            elif dt != params['dt']:
                raise ValueError('Non-matching time steps')

        pcol = change_hue(colors[eg], hues[tg])
        if early_traces != []:
            early_traces = np.vstack(early_traces)
            etrace = np.median(early_traces, axis=0)
            ax[0].plot(time, etrace, color=pcol, linestyle=line_styles[tg],
                       label='%s %s' % (eg, tg))

        if late_traces != []:
            late_traces = np.vstack(late_traces)
            ltrace = np.median(late_traces, axis=0)
            ax[1].plot(time, ltrace, color=pcol, linestyle=line_styles[tg])

    fig.subplots_adjust(top=0.9, left=0.15)
    fig.suptitle('Median Gamma Probabilities\nSaccharin')
    ax[0].set_ylabel('Early\nState')
    ax[0].legend()
    ax[1].set_ylabel('Late\nState')
    ax[1].set_xlabel('Time (ms)')

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return None
    else:
        return fig


def plot_mean_gamma_probs(best_hmms, save_file=None):
    # diff line style for each time_group
    # diff color for each exp_group
    # Not dropping single state trials
    fig, ax = plt.subplots(nrows=2, figsize=(8,11))
    time_groups = best_hmms.time_group.unique()
    exp_groups = best_hmms.exp_group.unique()
    colors = EXP_COLORS
    hues = {'preCTA': 0.4, 'postCTA': 1.1}
    line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
    line_styles = {k:v for k,v in zip(time_groups, line_styles)}
    t_start = 0
    t_end = 1500
    dt = None
    for name, group in best_hmms.groupby(['exp_group', 'time_group']):
        eg = name[0]
        tg = name[1]
        early_traces = []
        late_traces = []
        time = None
        for i, row in group.iterrows():
            if np.isnan(row['early_state']) or np.isnan(row['late_state']):
                continue

            anim, rec_group, taste = row[['exp_name', 'rec_group', 'taste']]
            h5_file = get_hmm_h5(row['rec_dir'])
            hmm, t, params = phmm.load_hmm_from_hdf5(h5_file, int(row['hmm_id']))
            es = int(row['early_state'])
            ls = int(row['late_state'])
            gamma_probs = hmm.stat_arrays['gamma_probabilities']
            state_seqs = hmm.stat_arrays['best_sequences']
            t_idx = np.where((t <= t_end) & (t >= t_start))[0]
            eprobs = gamma_probs[:, es, t_idx]
            lprobs = gamma_probs[:, ls, t_idx]
            t = t[t_idx]
            state_seqs = state_seqs[:, t_idx]

            # Drop any trials where the state in question is not present in the
            # decoded sequence, or the duration of the state is less than 50ms
            valid_trials =  agg.get_valid_trials(state_seqs, [es, ls],
                                             min_pts=int(.05/params['dt']),
                                             time=t)
            #e_trials = agg.get_valid_trials(state_seqs, es, min_pts=int(.05/params['dt']), time=t)
            #l_trials = agg.get_valid_trials(state_seqs, ls, min_pts=int(.05/params['dt']), time=t)
            if len(valid_trials) == 0:
                print(f'No valid trials for {anim} - {rec_group} - {taste}')
                continue

            eprobs = eprobs[valid_trials, :]
            lprobs = lprobs[valid_trials, :]
            emed = np.mean(eprobs, axis=0)
            lmed = np.mean(lprobs, axis=0)
            early_traces.append(emed)
            late_traces.append(lmed)
            if time is None:
                time = t
                dt = params['dt']
            elif dt != params['dt']:
                raise ValueError('Non-matching time steps')

        pcol = change_hue(colors[eg], hues[tg])
        if early_traces != []:
            early_traces = np.vstack(early_traces)
            etrace = np.mean(early_traces, axis=0)
            esd = np.std(early_traces, axis=0)
            ax[0].plot(time, etrace, color=pcol, linestyle=line_styles[tg],
                       label='%s %s' % (eg, tg))
            # ax[0].fill_between(time, etrace - esd, etrace+esd, color=pcol, alpha=0.5)

        if late_traces != []:
            late_traces = np.vstack(late_traces)
            ltrace = np.mean(late_traces, axis=0)
            lsd = np.std(late_traces, axis=0)
            ax[1].plot(time, ltrace, color=pcol, linestyle=line_styles[tg])
            # ax[1].fill_between(time, ltrace - lsd, ltrace+lsd, color=pcol, alpha=0.5)

    fig.subplots_adjust(top=0.9, left=0.15)
    fig.suptitle('Mean Gamma Probabilities\nSaccharin')
    ax[0].set_ylabel('Early\nState')
    ax[0].legend()
    ax[1].set_ylabel('Late\nState')
    ax[1].set_xlabel('Time (ms)')

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return None
    else:
        return fig


def grouped_heatmap(data, group_var, sort_var, X=None, ax=None,
                    smoothing_width=None, cbar=True, cmap='rocket'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10))
    else:
        fig = ax.figure

    if X is None:
        X = np.arange(data.shape[1])

    groups = np.unique(group_var)
    plot_dat = []
    ylabels = []
    mask = []
    for gi, grp in enumerate(groups):
        idx = np.where(group_var == grp)[0]
        tmp = data[idx, :]
        tmp_sort = sort_var[idx]

        sort_idx = np.argsort(tmp_sort)
        tmp = tmp[sort_idx,:]
        plot_dat.append(tmp)
        mask.append(np.zeros_like(tmp, dtype=bool))
        yls = np.array(['']*(tmp.shape[0]+1), dtype=object)
        if gi < len(groups)-1:
            blank = np.zeros((1, tmp.shape[1]))
            plot_dat.append(blank)
            mask.append(np.ones_like(blank, dtype=bool))
        else:
            yls = yls[:-1]

        mid = int(tmp.shape[0]/2)
        yls[mid] = grp
        ylabels.append(yls)

    plot_dat = np.vstack(plot_dat)
    if smoothing_width is not None:
        # Trying smoothing for visualization
        plot_dat = np.array([gaussian_filter1d(x, smoothing_width) for x in plot_dat])

    mask = np.vstack(mask)
    ylabels = np.concatenate(ylabels)
    plot_df = pd.DataFrame(plot_dat, columns=X)
    g = sns.heatmap(plot_df, mask=mask, rasterized=True,
                    ax=ax, robust=True,
                    yticklabels=ylabels, cbar=cbar, cmap=cmap)
    return fig, ax


def plot_state_timing_comparison(df, group_col='exp_group', taste='Saccharin'):
    '''plot early state end time and late state start time, box plot with
    animal means plotted with lines showing change from pre to post
    '''
    df = df.query('taste == "Saccharin"')
    ldf = df.query('state_group == "late"')
    edf = df.query('state_group == "early"')
    fig, axes = plt.subplots(nrows=2)

    assert group_col in ORDERS.keys(), f'No group ordering found for {group_col}'
    hue_order = ORDERS['time_group']
    groups = df[group_col].unique()
    group_order = [x for x in ORDERS[group_col] if x in groups]
