# Single Cell Analysis

## Non-held units
### % taste responsive neurons (pre vs post CTA) (data readout per animal) [Bar plot GFP vs Cre]
### % palatability responsive neurons (pre vs post CTA) (data readout per animal) [Bar plot GFP vs Cre]
### % Taste discriminative neurons

## Held-units (baseline removed: subtract mean firing rate when t<0)
### Use palatability responsive units held over 2 recordings to deduce palatability rank order of saccharin pre and post
### % neurons with significant change in response over CTA [data readout per animal] (Bar plot GFP vs Cre)
### Initial time of significant change [histogram per animal, average and deviation in data readout] (Bar plot or dot & box plot GFP vs Cre)
### % units changed at each time point of response [step plot per animal and GFP vs Cre]
### Average change in magnitude of response at each time point [plot per animal and GFP vs Cre]
### % units with change in baseline firing rate [data readout per animal] (bar plot GFP vs Cre)

# Population analysis
## Semi-held units: Units held across pre-CTA compared to units held across post-CTA
### Run HMM and get time of switch on each trial (make plots of HMM results for each trial) [HMM probabilities in data file]
### Plot MDS of each trial's ID phase and PAL phase seperately [per animal and GFP vs CTA]
### Compute relative distances between tastants (mean MDS) pre-CTA and post-CTA in ID and PAL [data readout: table per animal](bar plot GFP vs Cre]
### Plot PCA trajectory of each trial [per animal, pre vs post CTA, use paired color scheme] 

# So for each dataset, get:
# - Number of taste responsive neurons
# - Number of palatability responsive units

# For each experiment, compute:


import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import mannwhitneyu, spearmanr, sem
from scipy.ndimage.filters import gaussian_filter1d
import itertools as it
from blechpy import load_experiment, load_dataset, load_project
from blechpy.dio import h5io
from blechpy.analysis import stat_tests as stt, spike_analysis as sas
from blechpy.utils import write_tools as wt, userIO
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt

plot_params = {'xtick.labelsize': 14, 'ytick.labelsize': 14,
               'axes.titlesize': 26, 'figure.titlesize': 28,
               'axes.labelsize': 24}
matplotlib.rcParams.update(plot_params)


def get_taste_mapping(rec_dirs):
    tastants = set()
    for rd in rec_dirs:
        dat = load_dataset(rd)
        tmp = dat.dig_in_mapping
        tastants.update(tmp['name'].to_list())

    taste_map = {}
    for rd in rec_dirs:
        dat = load_dataset(rd)
        rn = dat.data_name
        din = dat.dig_in_mapping
        for t in tastants:
            if taste_map.get(t) is None:
                taste_map[t] = {}

            tmp = din['channel'][din['name'] == t]
            if not tmp.empty:
                taste_map[t][rn] = tmp.values[0]

    return taste_map, tastants


def analyze_held_unit(unit_info, rec_key, norm_func=None, params=None, plot_dir=None):
    '''Check all changes in baselines firing and taste response between two or more groups

    Parameters
    ----------
    unit_info: pd.DataFrame
        chunk of DataFrame containing info for a single held unit in tidy data
        format
        columns: unit, electrode, area, recording, rec_unit, rec_group
    rec_key: dict,
        keys are number dictating order of directories.
        values are dicts with keys: name, dir, group and values as strings
    norm_func: function
        function to use to normalize firing rate traces before comparison
        take args: time (1xN np.ndarray) and fr (MxN np.ndarray) and returns a
        transformed fr of same shape
    params: dict
    '''
    groups = unit_info['rec_group'].unique()
    unit_name = unit_info['held_unit_name'].unique()[0]

    group_pairs = list(it.combinations(groups, 2))
    dir_key = {x['name']: x['dir'] for x in  rec_key.values()}
    group_key = {v['group']: v['name'] for v in rec_key.values()}
    tastants = unit_info['tastant'].unique()

    # Un-pack params
    bin_size = params['response_comparison']['win_size']
    bin_step = params['response_comparison']['step_size']
    time_win = params['response_comparison']['time_win']
    alpha = params['response_comparison']['alpha']
    baseline_win = params['baseline_comparison']['win_size']
    baseline_alpha = params['baseline_comparison']['alpha']
    tasty_win = params['taste_responsive']['win_size']
    tasty_alpha = params['taste_responsive']['alpha']
    psth_win = params['psth']['win_size']
    psth_step = params['psth']['step_size']
    psth_time_win = params['psth']['plot_window']
    smoothing_win = params['psth']['smoothing_win']

    data = {}
    psth_fn = os.path.join(plot_dir, 'Unit-%s_PSTH.png' % unit_name)
    pfig, pax = plt.subplots(ncols=len(groups), figsize=(20,10))  # overlay PSTH per group
    if not isinstance(pax, np.ndarray):
        pax = [pax]

    pax = {g : pax[i] for i,g in enumerate(groups)}
    cfigs = {}  # per taste, norm psth overlay (groups) and response change
    cax = {}
    cfig_fn = {}
    for i, row in unit_info.iterrows():
        g = row['rec_group']
        unit = row['rec_unit_name']
        rn = row['recording']
        rd = dir_key[rn]
        t = row['tastant']
        channel = row['dig_in_ch']

        if g not in data:
            data[g] = {}
            data[g]['tasty_ps'] = []

        if t not in cfigs:
            cfigs[t], cax[t] = plt.subplots(ncols=len(group_pairs)+1, figsize=(20,10))
            if not isinstance(cax[t], np.ndarray):
                cax[t] = np.array([cax[t]])

            cfig_fn[t] = os.path.join(plot_dir, t, 'Unit-%s_Comparison.png' % unit_name)
            if not os.path.isdir(os.path.join(plot_dir, t)):
                os.mkdir(os.path.join(plot_dir,t))

        # aggregate baseline firing before CTA (even over days)
        times, spikes = h5io.get_spike_data(rd, unit, channel)
        idx = np.where((times >= -baseline_win) & (times < 0))[0]
        tmp = np.sum(spikes[:, idx], axis=1) / abs(baseline_win)
        tmp = tmp / 1000  # convert to Hz
        if 'baseline' in data[g]:
            data[g]['baseline'] = np.hstack((data[g]['baseline'], tmp))
        else:
            data[g]['baseline'] = tmp

        del tmp, idx

        if t not in data[g]:
            data[g][t] = {}
        else:
            raise KeyError('Already existing data for group %s and tastant %s' % (g,t))

        bin_time, fr = sas.get_binned_firing_rate(times, spikes, bin_size, bin_step)
        psth_time, psth = sas.get_binned_firing_rate(times, spikes, psth_win, psth_step)
        psth = gaussian_filter1d(psth, sigma=smoothing_win) # smooth PSTH
        tmp_idx = np.where((psth_time >= psth_time_win[0]) & (psth_time <= psth_time_win[1]))[0]
        psth_time = psth_time[tmp_idx]
        psth = psth[:, tmp_idx]
        bin_time = bin_time - bin_size/2  # to get bin starts rather than centers
        data[g][t]['raw_response'] = fr
        data[g][t]['time'] = bin_time
        data[g][t]['psth_time'] = psth_time
        data[g][t]['psth_fr'] = psth

        mpsth = np.mean(psth, axis=0)
        spsth = sem(psth, axis=0)
        pax[g].fill_between(psth_time, mpsth-spsth, mpsth+spsth, alpha=0.4)
        pax[g].plot(psth_time, mpsth, linewidth=3, label=t)

        if norm_func:
            fr = norm_func(bin_time, fr)
            data[g][t]['norm_response'] = fr
            norm_psth = norm_func(psth_time, psth)
            data[g][t]['norm_psth_fr'] = norm_psth
            mp = np.mean(norm_psth, axis=0)
            sp = sem(norm_psth, axis=0)
            cax[t][0].fill_between(psth_time, mp-sp, mp+sp, alpha=0.4)
            cax[t][0].plot(psth_time, mp, linewidth=3, label=g)

        del times, spikes, bin_time, fr


    out = {}
    for i, ax in enumerate(pax.items()):
        ax[1].set_title(ax[0])
        ax[1].set_xlabel('Time (ms)')
        ax[1].legend()
        ax[1].axvline(0, color='black', linestyle='--', linewidth=1)
        if i == 0:
            ax[1].set_ylabel('Firing Rate (Hz)')

    pfig.suptitle('Peri-stimulus time histograms: Unit %s' % unit_name)

    for t, fig in cfigs.items():
        fig.subplots_adjust(top=0.8)
        fig.suptitle('Taste response change\nUnit: %s, Tastant: %s' % (unit_name, t))
        cax[t][0].set_xlabel('Time (ms)')
        cax[t][0].set_ylabel('Firing Rate (Hz)')
        cax[t][0].legend()
        cax[t][0].set_title('Normalized PSTH')
        cax[t][0].axvline(0, color='black', linewidth=1, linestyle='--')

    # Loop through group pairs and compare responses
    # Store delta, p-values, u-stats, 
    # Mean response for each tastant for each group
    if len(groups) == 1:
        for g in groups:
            out[g] = {'mean_baseline': np.mean(data[g]['baseline']),
                      'sem_baseline': sem(data[g]['baseline'])}

        pfig.savefig(psth_fn)
        plt.close(pfig)
        for t, fig in cfigs.items():
            fig.savefig(cfig_fn[t])
            plt.close(fig)

        return out

    for gidx, pair in enumerate(group_pairs):
        g1 = pair[0]
        g2 = pair[1]
        k = '%s_vs_%s' % (g1, g2)

        # Compare baselines
        baseline1 = data[g1]['baseline']
        baseline2 = data[g2]['baseline']
        base_u, base_p = compare_responses(baseline1, baseline2)

        # Store baseline data
        if g1 not in out:
            out[g1] = {}
            out[g1]['mean_baseline'] = np.mean(baseline1)
            out[g1]['sem_baseline'] = sem(baseline1)

        if g2 not in out:
            out[g2] = {}
            out[g2]['mean_baseline'] = np.mean(baseline2)
            out[g2]['sem_baseline'] = sem(baseline2)

        mean_baseline_change, sem_baseline_change = sas.get_mean_difference(baseline1, baseline2)

        if k not in out:
            out[k] = {}

        # Store Baseline stats
        out[k]['baseline_shift'] = False
        out[k]['baseline_p'] = base_p
        out[k]['baseline_u'] = base_u
        out[k]['mean_baseline_change'] = mean_baseline_change
        out[k]['sem_baseline_change'] = sem_baseline_change
        if base_p <= baseline_alpha:
            out[k]['baseline_shift'] = True

        for t in tastants:
            if t not in data[g1] or t not in data[g2]:
                continue

            normalize = False
            raw1 = data[g1][t]['raw_response']
            t1 = data[g1][t]['time']
            raw2 = data[g2][t]['raw_response']
            t2 = data[g2][t]['time']
            psth_time = data[g1][t]['psth_time']
            psth1 = data[g1][t]['psth_fr']
            psth2 = data[g2][t]['psth_fr']

            idx1 = np.where((t1>=time_win[0]) & (t1<=time_win[1]))[0]
            idx2 = np.where((t2>=time_win[0]) & (t2<=time_win[1]))[0]
            if not np.array_equal(t1, t2):
                raise ValueError('Uncomparable time vectors')

            raw_u = np.zeros((len(idx1),))
            raw_p = np.ones(raw_u.shape)

            if 'norm_response' in data[g1][t]:
                norm1 = data[g1][t]['norm_response']
                norm2 = data[g2][t]['norm_response']
                n_psth1 = data[g1][t]['norm_psth_fr']
                n_psth2 = data[g2][t]['norm_psth_fr']
                norm_u = raw_u.copy()
                norm_p = raw_p.copy()
                normalize = True

            # Store mean responses and delta
            # raw_change = sas.get_mean_difference(raw1, raw2)
            raw_change = sas.get_mean_difference(psth1, psth2)
            if t not in out[k]:
                out[k][t] = {}

            out[k][t]['raw_mean_change'] = raw_change[0]
            out[k][t]['raw_sem_change'] = raw_change[1]
            out[k][t]['time'] = t1

            if normalize:
                # norm_change = sas.get_mean_difference(norm1, norm2)
                norm_change = sas.get_mean_difference(n_psth1, n_psth2)
                cax[t][gidx+1].fill_between(psth_time, norm_change[0] - norm_change[1],
                                            norm_change[0] + norm_change[1], alpha=0.4)
                cax[t][gidx+1].plot(psth_time, norm_change[0],
                                    linewidth=3, label='norm_change')
                cax[t][gidx+1].set_title('Normalized Response Change\n%s' % k)
                out[k][t]['norm_mean_change'] = norm_change[0]
                out[k][t]['norm_sem_change'] = norm_change[1]
            else:
                cax[t][gidx+1].fill_between(psth_time, raw_change[0] - raw_change[1],
                                            raw_change[0] + raw_change[1], alpha=0.4)
                cax[t][gidx+1].plot(psth_time, raw_change[0],
                                    linewidth=3, label='raw_change')
                cax[t][gidx+1].set_title('Raw Response Change\n%s' % k)

            cax[t][gidx+1].set_xlabel('Time (ms)')
            cax[t][gidx+1].axvline(0, color='black', linewidth=1, linestyle='--')
            if i==0:
                cax[t][gidx+1].set_ylabel('Response Change (Hz)')


            if t not in out[g1]:
                out[g1][t] = {}
                out[g1][t]['raw_response'] = np.mean(raw1, axis=0)
                out[g1][t]['raw_sem'] = sem(raw1, axis=0)
                out[g1][t]['time'] = t1
                if normalize:
                    out[g1][t]['norm_response'] = np.mean(norm1, axis=0)
                    out[g1][t]['norm_sem'] = sem(norm1, axis=0)

            if t not in out[g2]:
                out[g2][t] = {}
                out[g2][t]['raw_response'] = np.mean(raw2, axis=0)
                out[g2][t]['raw_sem'] = sem(raw2, axis=0)
                out[g2][t]['time'] = t2
                if normalize:
                    out[g2][t]['norm_response'] = np.mean(norm2, axis=0)
                    out[g2][t]['norm_sem'] = sem(norm2, axis=0)


            out[k][t]['p_time'] = t1[idx1]
            for i, idx in enumerate(zip(idx1,idx2)):
                raw_u[i], raw_p[i] = compare_responses(raw1[:, idx[0]],
                                                       raw2[:, idx[1]])
                if normalize:
                    norm_u[i], norm_p[i] = compare_responses(norm1[:, idx[0]],
                                                             norm2[:, idx[1]])

            # Bonferroni correction
            raw_p = raw_p * len(idx1)
            out[k][t]['raw_p'] = raw_p
            out[k][t]['raw_u'] = raw_u
            raw_sig = np.where(raw_p <= alpha)[0]
            if len(raw_sig) > 0:
                out[k][t]['raw_change'] = True
                raw_sig = np.sort(raw_sig)
                out[k][t]['raw_earliest_change'] = out[k][t]['p_time'][raw_sig[0]]
                out[k][t]['raw_latest_change'] = out[k][t]['p_time'][raw_sig[-1]]
            else:
                out[k][t]['raw_change'] = False
                raw_sig = np.sort(raw_sig)
                out[k][t]['raw_earliest_change'] = None
                out[k][t]['raw_latest_change'] = None

            if normalize:
                # Bonferroni correction
                norm_p = norm_p * len(idx1)
                norm_sig = np.where(norm_p <= alpha)[0]
                if len(norm_sig) > 0:
                    out[k][t]['norm_change'] = True
                    norm_sig = np.sort(norm_sig)
                    out[k][t]['norm_earliest_change'] = out[k][t]['p_time'][norm_sig[0]]
                    out[k][t]['norm_latest_change'] = out[k][t]['p_time'][norm_sig[-1]]
                else:
                    out[k][t]['norm_change'] = False
                    out[k][t]['norm_earliest_change'] = None
                    out[k][t]['norm_latest_change'] = None

                out[k][t]['norm_p'] = norm_p
                out[k][t]['norm_u'] = norm_u

    pfig.savefig(psth_fn)
    plt.close(pfig)
    for t, fig in cfigs.items():
        fig.savefig(cfig_fn[t])
        plt.close(fig)

    if out == {}:
        raise ValueError()

    return out


def compare_responses(resp1, resp2):
    try:
        u, p = mannwhitneyu(resp1, resp2, alternative='two-sided')
    except ValueError:
        u = 0
        p = 1

    return u, p


def get_baseline_firing(rec, unit, win_size=1500):
    '''Returns a vectors of baseline firing rates for the unit aggregated from
    win_size ms before every stimulus delivery

    Parameters
    ----------
    rec : str, path to recording directory
    unit: str or int, unit name or number
    win_size: int (optional), ms before stimulus delivery to use (default=1500)

    Returns
    -------
    numpy.ndarray, vector of baseline firing rates in Hz
    '''
    dat = load_dataset(rec)
    if dat is None:
        raise FileNotFoundError('Dataset not found for %s' % rec)

    if isinstance(unit, str):
        unit_num = h5io.parse_unit_number(unit)
    elif isinstance(unit, int):
        unit_num = unit
        unit = 'unit%03i' % unit_num

    dim = dat.dig_in_mapping.copy()
    dins = dim.query('exclude==False and spike_array==True')['channel'].tolist()
    baselines = []
    for channel in dins:
        times, spikes = h5io.get_spike_data(rec, unit_num, channel)
        idx = np.where((times >= -win_size) & (times < 0))[0]
        tmp = np.sum(spikes[:, idx], axis=1) / abs(win_size)
        tmp = tmp / 1000  # convert to Hz
        baselines.append(tmp)

    return np.hstack(baselines)


def compare_baseline_firing(rd1, u1, rd2, u2, win_size=1500):
    '''Compares baseline firing rates between 2 unit using the Mann-Whitney
    U-test. Baseline firing is taken from all trials from all digital inputs
    that are not excluded and have spike arrays.

    Parameters
    ----------
    rd1: str, path to recording directory 1
    u1: str or int, unit from recording directory 1
    rd2: str, path to recording directory 2
    u2: str or int, unit from recording directory 2
    win_size: int (optional), size of window to use, in ms (default=1500)

    Returns
    -------
    dict, float
    stats, p-value
    '''
    baseline1 = get_baseline_firing(rd1, u1, win_size=win_size)
    baseline2 = get_baseline_firing(rd2, u2, win_size=win_size)

    base_u, base_p = mannwhitneyu(baseline1, baseline2,
                                  alternative='two-sided')
    stats = {'u-stat': base_u, 'p-val': base_p,
             'baseline1': (np.mean(baseline1), sem(baseline1)),
             'baseline2': (np.mean(baseline2), sem(baseline2))}

    return stats, base_p


def check_taste_responsiveness(rec, unit, win_size=1500, alpha=0.05):
    '''Runs through all digital inputs (non-excluded) and determines if neuron
    is taste responsive and to which tastant. Compares win_size ms before
    stimulus to win_size ms after stimulus with Mann-Whitney U-test.
    
    Parameters
    ----------
    rec: str, path to recording directory
    unit: int or str, unit name or number
    win_size: int (optional), window size in ms (default=1500)
    alpha: float (optional), significance level for tests (default=0.5)
    
    Returns
    -------
    bool: whether unit is taste responsive at all
    dict: computed statistics from each tastant

    Raises
    ------
    FileNotFoundError: if not dataset file in recording dir
    '''
    dat = load_dataset(rec)
    if dat is None:
        raise FileNotFoundError('No dataset found in %s' % rec)

    dim = dat.dig_in_mapping.copy()
    dins = dim.query('exclude==False and spike_array==True')['channel'].tolist()
    stats = {}
    taste_responsive = False
    p_values = {}

    # Bonferroni correction
    # TODO: Is this actually needed/applicable?
    # alpha = alpha/len(dins)

    for i in dins:
        p, s = stt.check_taste_response(rec, unit, i, win_size=win_size)
        stats[i] = s
        p_values[i] = p

    # Apply Bonferroni correction and test for taste responsiveness of cell as a whole
    # Leave p-values unaltered for use in checking if responsive to single tastant
    if any([p < alpha/len(p_values) for p in p_values]):
        taste_responsive = True

    return taste_responsive, stats


def check_palatability_responsiveness(rec, unit, win_size=250, step_size=25):
    # TODO: make this
    pass


def deduce_palatability_rank_order(rec, unit, dins, window):
    # TODO: make this
    pass





ANALYSIS_PARAMS = {'taste_responsive': {'win_size': 1500, 'alpha': 0.05},
                   'pal_responsive': {'win_size': 250, 'step_size': 25,
                                      'time_win': [0, 2000], 'alpha': 0.05},
                   'baseline_comparison': {'win_size': 1500, 'alpha': 0.01},
                   'response_comparison': {'win_size': 250, 'step_size': 125,
                                           'time_win': [0, 2000], 'alpha': 0.05},
                   'psth': {'win_size': 250, 'step_size': 25, 'smoothing_win': 3,
                            'plot_window': [-1500, 2000]}}

class ProjectAnalysis(object):
    def __init__(self, project):
        if isinstance(project, str):
            project = load_project(project)

        if project is None:
            raise FileNotFoundError('No _project.p file found')

        self._project = project
        self.proj_name = project.data_name
        self.root_dir = project.root_dir
        self._analysis_dir = os.path.join(self.root_dir,
                                          '%s_analysis' % self.proj_name)
        self._data_dir = os.path.join(self._analysis_dir, 'data')
        self._plot_dir = os.path.join(self._analysis_dir, 'plots')

        if not os.path.isdir(self._analysis_dir):
            os.mkdir(self._analysis_dir)

        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)

        if not os.path.isdir(self._plot_dir):
            os.mkdir(self._plot_dir)

        self._params = {'response_change_alpha': 0.05,
                        'mag_change_window': [-1500, 2000]}

    def run(self, overwrite=False, new_params=None):
        print('Analyzing Project %s...' % self.proj_name)
        params = self._params
        alpha = params['response_change_alpha']
        t_win = params['mag_change_window']
        exp_info = self._project._exp_info
        exp_groups = exp_info['exp_group'].unique()
        data, tastes = self._get_plot_data(alpha=alpha, overwrite=overwrite,
                                           params=new_params)
        plot_dir = self._plot_dir

        for t in tastes:
            mfn = os.path.join(plot_dir, 'Magnitude_Change-%s.png' % t)
            pfn = os.path.join(plot_dir, 'Responses_Changed-%s.png' % t)
            mfig, m_ax = plt.subplots(ncols=len(exp_groups),
                                      figsize=(15,10))
            pfig, p_ax = plt.subplots(ncols=len(exp_groups),
                                      figsize=(15,10))

            for i, eg in enumerate(exp_groups):
                n_units = data[eg][t]['units']
                # Plot avg mag change for each group per tastant
                time = data[eg][t]['mag_time']
                idx = np.where((time >= t_win[0]) & (time <= t_win[1]))[0]
                mmc = data[eg][t]['mean_mag_change'][idx]
                smc = data[eg][t]['sem_mag_change'][idx]
                time = time[idx]
                m_ax[i].fill_between(time, mmc - smc, mmc + smc, alpha=0.5)
                m_ax[i].plot(time, mmc, linewidth=3)
                m_ax[i].set_title('%s : %s, N=%i' % (eg, t, n_units))
                m_ax[i].set_xlabel('Time (ms)')
                m_ax[i].axvline(0, color='red', linestyle='--', linewidth=2)
                if i == 0:
                    m_ax[i].set_ylabel('Magnitude of reponse change (Hz)')

                # Plot % units change at each time point for each group per tastant
                n_sig  = data[eg][t]['n_changed']
                n_t = data[eg][t]['change_time']
                # pad array end for proper plotting
                step = np.unique(np.diff(n_t))[0]
                n_sig =np.array([*n_sig, n_sig[-1]])
                n_t = [*n_t, n_t[-1]]
                print(t)
                print(n_sig)
                n_sig = 100 * n_sig / n_units
                p_ax[i].step(n_t, n_sig, where='post')
                p_ax[i].set_xlabel('Post-stimulus Time (ms)')
                p_ax[i].set_title('%s : %s, N=%i' % (eg, t, n_units))
                if i == 0:
                    p_ax[i].set_ylabel('% units with reponse changed')

            mfig.suptitle('Average Magnitude of Reponse Change')
            mfig.savefig(mfn)

            pfig.suptitle('Percent units with significant change in response')
            pfig.savefig(pfn)
            plt.close('all')

    def _aggregate_experiment_arrays(self, overwrite=False, params=None):
        exp_info = self._project._exp_info
        exp_groups = exp_info['exp_group'].unique()

        out = dict.fromkeys(exp_groups, None)
        for i, row in exp_info.iterrows():
            ea = CtaExperimentAnalysis(row['exp_dir'], params=params)
            if not ea.complete or overwrite:
                ea.run(overwrite=overwrite)

            arrays = ea.get_analysis_data_arrays()
            eg = row['exp_group']
            if out[eg] is None:
                out[eg] = arrays
            else:
                for k, v in arrays.items():
                    out[eg][k] = np.vstack((out[eg][k], v))

        return out

    def _get_plot_data(self, alpha=0.05, overwrite=False, params=None):
        exp_info = self._project._exp_info
        exp_groups = exp_info['exp_group'].unique()

        data = self._aggregate_experiment_arrays(overwrite=overwrite, params=params)
        tastes = set()
        for eg in exp_groups:
            labels = data[eg]['row_labels']
            headers = data[eg].pop('label_headers')[0,:]
            tidx = np.where(headers == 'tastant')[0][0]
            tastes.update(np.unique(labels[:, tidx]))

        tidx = np.where(headers == 'tastant')[0][0]
        new_data = {}
        for eg, t in it.product(exp_groups, tastes):
            labels = data[eg]['row_labels']
            idx = np.where(labels[:, tidx] == t)[0]
            tmp_dat = {k:v[idx, :] for k, v in data[eg].items()}

            if eg not in new_data:
                new_data[eg] = {}

            if t not in new_data[eg]:
                new_data[eg][t] = {}

            mean_mag = np.mean(abs(tmp_dat['mean_response_change']), axis=0)
            sem_mag = np.sqrt(np.sum(np.power(tmp_dat['sem_response_change'],2), axis=0))
            mag_time = tmp_dat['response_time'][0, :]
            p_time = tmp_dat['p_time'][0, :]
            n_sig = np.sum(tmp_dat['response_change_p'] <= alpha, axis=0)

            new_data[eg][t]['units'] = len(idx)
            new_data[eg][t]['n_changed'] = n_sig
            new_data[eg][t]['change_time'] = p_time
            new_data[eg][t]['mean_mag_change'] = mean_mag
            new_data[eg][t]['sem_mag_change'] = sem_mag
            new_data[eg][t]['mag_time'] = mag_time

        return new_data, tastes





class CtaExperimentAnalysis(object):
    def __init__(self, experiment=None, params=None):
        if experiment is None or isinstance(experiment, str):
            experiment = load_experiment(experiment)

        self._root_dir = experiment.root_dir
        self._experiment = experiment
        save_dir = self.analysis_dir = experiment.analysis_dir
        exp_name = experiment.data_name


        # Setup directories
        data_dir = self._data_dir = os.path.join(save_dir, 'data')
        plot_dir = self._plot_dir = os.path.join(save_dir, 'plots')
        held_unit_dir = os.path.join(data_dir, 'held_unit_analysis')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        if not os.path.isdir(held_unit_dir):
            os.makedirs(held_unit_dir)

        # List files
        self._files = {'recording_info':
                       os.path.join(save_dir, 'recording_key.json'),
                       'params': os.path.join(save_dir, 'analysis_params.json'),
                       'single_unit_summary':
                       os.path.join(data_dir, 'single_unit_summary.tsv'),
                       'all_unit_dataframe':
                       os.path.join(data_dir, '%s_unit_data.p' % exp_name),
                       #'population_data':
                       #os.path.join(data_dir, 'population_data.json'),
                       'data_readout':
                       os.path.join(data_dir, 'data_readout.txt'),
                       'held_unit_arrays':
                       os.path.join(held_unit_dir, 'held_unit_arrays.npz')}

        # Check files
        file_check = self._file_check()
        self._read_existing_data(params=params)

        if all(file_check.values()):
            self.complete = True
        else:
            self.complete = False

    def _melt_held_units(self):
        df = self._experiment.held_units.copy().drop(columns=['J3'])
        df2 = pd.melt(df, ['unit','electrode','single_unit','unit_type','area'],
                      var_name='recording', value_name='rec_unit')
        tmp_key = {v['name']: v['group'] for v in self._rec_key.values()}
        df2.loc[:,'rec_group'] = df2['recording'].map(tmp_key)
        df2 = df2.dropna()
        return df2

    def _gather_all_units(self):
        df = None
        rec_key = self._rec_key
        held = self._melt_held_units()
        for k, v in rec_key.items():
            da = DatasetAnalysis(v['dir'], params=self._params)
            if not da._complete:
                da.uun()

            ut = da._data.copy()
            ut = ut.drop(columns=['mean_taste_response', 'sem_taste_response'])
            ut = ut.rename(columns={'unit':'rec_unit_name',
                                    'taste_p_value':'taste_responsive_p',
                                    'taste_u_stat': 'taste_responsive_u',
                                    'mean_taste_change': 'mean_taste_response_mag',
                                    'sem_taste_change': 'sem_taste_response_mag'})
            ut['experiment'] = self._experiment.data_name
            ut['rec_group'] = v['group']
            ut['held_unit_name'] = ut.apply(lambda x: apply_held_unit_name(held, x), axis=1)
            if df is None:
                df = ut.copy()
            else:
                df = df.append(ut, ignore_index=True).reset_index(drop=True)

        return df

    def _file_check(self):
        out = dict.fromkeys(self._files.keys(), False)
        for k, v in self._files.items():
            if os.path.isfile(v):
                out[k] = True

        return out

    def _read_existing_data(self, params=None):
        status = self._file_check()
        # Load, update or create params
        if status['params']:
            self._params = wt.read_dict_from_json(self._files['params'])
        else:
            self._params = deepcopy(ANALYSIS_PARAMS)

        if params is not None:
            for k, v in params.items():
                if self._params.get(k) and isinstance(v, dict):
                    self._params[k].update(v)
                else:
                    self._params[k] = v

        wt.write_dict_to_json(self._params, self._files['params'])

        # Load single unid summary
        if status['single_unit_summary']:
            self._single_unit_summary = wt.read_pandas_from_table(self._files['single_unit_summary'])
        else:
            self._single_unit_summary = None

        # load recording info (name, dir, group)
        if status['recording_info']:
            self._rec_key = wt.read_dict_from_json(self._files['recording_info'])
        else:
            # Assign Analysis groups
            # TODO: make less brute force
            groups = {'Pre-CTA': ['preCTA', 'ctaTrain'],
                      'Post-CTA': ['postCTA', 'ctaTest']}

            # Key to maintain recording order and information
            rec_key = {}
            for i, rec in enumerate(self._experiment.rec_labels.items()):
                tmp = [k for k, v in groups.items() for x in v if x in rec[0]]
                if len(tmp) == 0:
                    raise ValueError('No group matches for %s' % rec[0])

                rec_key[i] = {'name': rec[0],
                              'dir': rec[1],
                              'group': tmp[0]}

            self._rec_key = rec_key
            wt.write_dict_to_json(rec_key, self._files['recording_info'])

    def run(self, norm_func=sas.remove_baseline, overwrite=True):
        if self.complete and not overwrite:
            print('Analysis already complete. Run with overwrite=True to re-run')
            return
        else:
            print('Analyzing Experiment %s...' % self._experiment.data_name)

        status = self._file_check()
        # Single unit analysis
        if not status['single_unit_summary'] or overwrite:
            self._get_single_unit_summary(overwrite=overwrite)

        if not status['held_unit_arrays'] or overwrite:
            self._process_all_units(norm_func=norm_func, overwrite=overwrite)
            self._make_plots()

    def _process_all_units(self, norm_func=None, overwrite=False):
        status = self._file_check()
        if status['held_unit_arrays'] and status['all_unit_dataframe'] and not overwrite:
            print('Analyzed data already exists. Pass overwrite=True to '
                  're-analyze')
            return

        units = self._gather_all_units()
        rec_key = self._rec_key
        exp_name = self._experiment.data_name
        single_units = units[units['held_unit_name'].isnull()]
        single_unit_groups = single_units.groupby(['recording', 'rec_unit_name'])

        tasty_alpha = self._params['taste_responsive']['alpha']

        def apply_taste_responsiveness(group):
            return any([p < tasty_alpha/len(group) for p in group['taste_responsive_p']])

        # Compute and attach taste_responsive_all using Bonferroni correction
        # NOTE: taste responsive p-values ARE NOT bonferroni corrected
        tra = single_unit_groups.apply(apply_taste_responsiveness)
        tra = tra.rename('taste_responsive_all')
        single_units = single_units.merge(tra, left_on=['recording',
                                                        'rec_unit_name'],
                                          right_index=True)


        held_units = units[units['held_unit_name'].notnull()]
        held_unit_groups = held_units.groupby('held_unit_name')
        tra = held_unit_groups.apply(apply_taste_responsiveness)
        tra = tra.rename('taste_responsive_all')
        held_units = held_units.merge(tra, left_on='held_unit_name', right_index=True)

        # NOTE: response change data saved is computed from taste response with
        # the baseline subtracted
        # NOTE: stored resopnse_change p-values ARE bonferroni corrected
        arrays = dict.fromkeys(['mean_response_change', 'sem_response_change',
                                'response_change_p', 'response_change_u',
                                'row_labels', 'response_time', 'p_time',
                                'label_headers'])

        arr_key = {'norm_mean_change': 'mean_response_change',
                   'norm_sem_change': 'sem_response_change',
                   'norm_p': 'response_change_p',
                   'norm_u':  'response_change_u',
                   'time' : 'response_time',
                   'p_time': 'p_time'}

        label_headers = np.array(['experiment', 'held_unit_name', 'tastant',
                                  'comparison'])
        arrays['label_headers'] = label_headers

        def add_to_array(a_name, arr):
            if a_name not in arrays:
                return

            if arrays[a_name] is None:
                arrays[a_name] = arr
            else:
                arrays[a_name] = np.row_stack((arrays[a_name], arr))

        new_df_cols = ['held_unit_name', 'rec_group', 'tastant',
                       'baseline_shift', 'mean_baseline_change',
                       'sem_baseline_change', 'baseline_p', 'baseline_u',
                       'mean_baseline', 'sem_baseline', 'response_change',
                       'earliest_response_change', 'latest_response_change',
                       'min_response_p']

        new_df_starter = dict.fromkeys(new_df_cols)
        output_dataframe = None
        for hu_name, unit_group in held_units.groupby('held_unit_name'):
            out = analyze_held_unit(unit_group, rec_key, norm_func=norm_func,
                                    params=self._params, plot_dir=self._plot_dir)

            group_dat = {k: out[k] for k in out.keys() if '_vs_' not in k}
            comp_dat = {k: out[k] for k in out.keys() if '_vs_' in k}
            out_df = []
            tastes = unit_group['tastant'].unique()
            rgroups = unit_group['rec_group'].unique()

            for i, row in unit_group.iterrows():
                rg = row['rec_group']
                t = row['tastant']
                tmp = new_df_starter.copy()
                tmp['held_unit_name'] = hu_name
                tmp['rec_group'] = rg
                tmp['tastant'] = t
                gdat = group_dat[rg]
                tmp['mean_baseline'] = gdat['mean_baseline']
                tmp['sem_baseline'] = gdat['sem_baseline']
                comp_keys = [k for k in comp_dat.keys() if rg in k]
                if len(comp_keys) == 1:
                    cdat = comp_dat[comp_keys[0]]
                    for ck, cv in cdat.items():
                        if ck in new_df_cols:
                            tmp[ck] = cv

                    if t in cdat:
                        tmp['response_change'] = cdat[t]['norm_change']
                        tmp['earliest_response_change'] = cdat[t]['norm_earliest_change']
                        tmp['latest_response_change'] = cdat[t]['norm_latest_change']
                        tmp['min_response_p'] = np.min(cdat[t]['norm_p'])

                elif len(comp_keys) > 1:
                    raise ValueError('I havent thought of how to deal with this')

                out_df.append(tmp)

            # Save arrays
            for cname, cdat in comp_dat.items():
                for ck, cv in cdat.items():
                    if ck in tastes:
                        l_row = np.array([exp_name, hu_name, ck, cname])
                        add_to_array('row_labels', l_row)
                        for ak, av in arr_key.items():
                            add_to_array(av, cv[ak])

            if output_dataframe is None:
                output_dataframe = pd.DataFrame.from_dict(out_df)
            else:
                output_dataframe = output_dataframe.append(out_df,
                                                           ignore_index=True)
                output_dataframe = output_dataframe.reset_index(drop=True)

        tmp = held_units.merge(output_dataframe, on=['held_unit_name', 'rec_group',
                                                     'tastant'])
        if len(tmp) != len(held_units):
            raise ValueError('Something fucked up')

        tmp = tmp.drop(columns=['mean_baseline_x', 'sem_baseline_x'])
        tmp = tmp.rename(columns={'mean_baseline_y':'mean_baseline',
                                  'sem_baseline_y':'sem_baseline'})
        held_units = tmp
        all_unit_data = pd.concat((single_units, held_units), sort=False)
        all_unit_data = all_unit_data.reset_index(drop=True)

        # Save data
        all_unit_data.to_pickle(self._files['all_unit_dataframe'])
        np.savez(self._files['held_unit_arrays'], **arrays)

    def get_unit_data(self):
        status = self._file_check()
        if status['all_unit_dataframe']:
            return pd.read_pickle(self._files['all_unit_dataframe'])
        else:
            return None

    def get_analysis_data_arrays(self):
        status = self._file_check()
        if status['held_unit_arrays']:
            npz = np.load(self._files['held_unit_arrays'])
            out = {}
            for k in npz.files:
                out[k] = npz[k]

            npz.close()
            return out
        else:
            return None

    def _get_single_unit_summary(self, overwrite=False):
        rec_key = self._rec_key
        summary = None
        for i in  sorted(rec_key.keys()):
            rn = rec_key[i]['name']
            rd = rec_key[i]['dir']
            dat = load_dataset(rd)

            # Analyze single unit responses
            analyzer = DatasetAnalysis(rd)
            if not analyzer._complete or overwrite:
                analyzer.run(overwrite=overwrite)

            # Grab summary of single unit stats
            if summary is None:
                summary = analyzer._summary.copy()
            else:
                summary = summary.append(analyzer._summary,
                                         ignore_index=True).reset_index(drop=True)

        summary = summary.dropna(axis=1)
        totals = summary.groupby(['recording', 'area']).count()
        totals = totals['unit'].rename('n_cells')
        counts = totals.to_frame()

        if 'taste_responsive' in summary:
            tasty = summary.loc[summary['taste_responsive']]
            tasty = tasty.groupby(['recording', 'area']).count()
            tasty = tasty['unit'].rename('n_taste_responsive')
            counts = pd.merge(counts, tasty, left_index=True, right_index=True)
            counts['%_taste_responsive'] = 100*counts['n_taste_responsive'] / counts['n_cells']

        if 'pal_responsive' in summary:
            pally = summary.loc[summary['pal_responsive']].groupby(['recording', 'area']).count()
            pally = pally['unit'].rename('n_pal_responsive')
            counts = pd.merge(counts, pally, left_index=True, right_index=True)
            counts['%_pal_responsive'] = 100*counts['n_pal_responsive'] / counts['n_cells']

        counts = counts.reset_index()
        self._single_unit_summary = counts.copy()
        wt.write_pandas_to_table(counts, self._files['single_unit_summary'],
                                 overwrite=overwrite)

    def get_avg_mag_response_change(self):
        status = self._file_check()
        if not status['held_unit_arrays']:
            return None

        npz = np.load(self._files['held_unit_arrays'])
        mrc = npz['mean_response_change']
        labels = npz['row_labels']
        time  = npz['reponse_time']
        headers = npz['label_headers']
        npz.close()

        tidx = np.where(headers == 'tastant')[0][0]
        tastes = np.unique(labels[:, tidx])
        out = {}
        out['time'] = time
        for t in tastes:
            idx = np.where(labels[:, tidx] == t)[0]
            out[t]['mean'] = np.mean(abs(mrc[idx, :]), axis=0)
            out[t]['sem'] = sem(abs(mrc[idx, :]), axis=0)

        return out

    def _make_plots(self):
        # Plot:
        #   - Average magnitude of response change (line plot)
        #       - Plot per tastant per rec_group
        #   - % units changed at each time point (step line plot)
        #       - Plot per tastant
        #   - % units with baseline change, with response change (bar plot)
        #       - bar for baseline and bar per tastant
        #   - Overlayed PSTHs
        #       - Plot per unit, subplot per rec_group
        #   - Joint plot (Scatter x hist) of earliest response change time and
        #     latest response change time
        #       - Plot per tastant

        unit_df = self.get_unit_data()
        if unit_df is None:
            raise FileNotFoundError('Analysis has not yet been run.')

        tastes = unit_df['tastant'].unique()
        rec_groups = unit_df['rec_group'].unique()

        all_units = unit_df.drop_duplicates(subset=['recording', 'area', 'rec_unit_name'])
        held_units = unit_df.dropna(subset=['held_unit_name'])

        unit_count = all_units.groupby(['recording','area'])
        unit_count = unit_count.agg({'taste_responsive_all':sum, 'rec_unit_name': 'count'})
        unit_count = unit_count.rename(columns={'taste_responsive_all':'taste_responsive',
                                                'rec_unit_name':'n_units'})

        held_totals = held_units.drop_duplicates(subset=['area','held_unit_name'])
        held_totals = held_totals.groupby('area').agg({'baseline_shift':'sum',
                                                       'held_unit_name': 'count'})
        held_totals = held_totals.rename(columns={'held_unit_name':'n_units'})

        resp_totals = held_units.dropna(subset=['response_change'])
        resp_totals = resp_totals.drop_duplicates(subset=['tastant', 'held_unit_name'])
        resp_totals = resp_totals.groupby(['area','tastant']).agg({'response_change':'sum',
                                                                   'held_unit_name':'count'})
        resp_totals = resp_totals.rename(columns={'held_unit_name':'n_units'})

        # Make data readout
        with open(self._files['data_readout'], 'w') as f:
            print(unit_count, file=f)
            print('\n----------\n', file=f)
            print(held_totals, file=f)
            print('\n----------\n', file=f)
            print(resp_totals, file=f)

        # TODO: make plots



class DatasetAnalysis(object):
    def __init__(self, dataset, params=None, shell=False):
        if dataset is None:
            dataset = userIO.get_filedirs('Select dataset', shell=shell)

        if isinstance(dataset, str):
            data_dir = dataset
            dataset = load_dataset(data_dir)

        if dataset is None:
            raise FileNotFoundError('dataset.p object not found in %s' % data_dir)

        self._dataset = dataset
        self.root_dir = data_dir
        del data_dir

        # Create directory structure for analysis data
        save_dir = os.path.join(self.root_dir, 'rn_analysis')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir
        self._data_dir = data_dir = os.path.join(save_dir, 'data')
        self._plot_dir = os.path.join(save_dir, 'plots')
        self._shell = shell
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        if not os.path.isdir(self._plot_dir):
            os.mkdir(self._plot_dir)

        self._files = {'params': os.path.join(save_dir, 'analysis_params.json'),
                       'summary': os.path.join(data_dir, 'analysis_summary.tsv'),
                       'data': os.path.join(data_dir, 'analysis_data.tsv')}

        self._data = None
        self._params = None
        self._summary = None
        status = self._file_check()
        if status['params']:
            self._params = wt.read_dict_from_json(self._files['params'])
        else:
            self._params = deepcopy(ANALYSIS_PARAMS)

        if params is not None:
            for k, v in params.items():
                if self._params.get(k) and isinstance(v, dict):
                    self._params[k].update(v)
                else:
                    self._params[k] = v

        wt.write_dict_to_json(self._params, self._files['params'])

        if status['summary']:
            self._summary = wt.read_pandas_from_table(self._files['summary'])

        if status['data']:
            self._data = wt.read_pandas_from_table(self._files['data'])

        if all(status.values()):
            self._complete = True
        else:
            self._complete = False

    def _file_check(self):
        out = dict.fromkeys(self._files.keys(), False)
        for k, v in self._files.items():
            if os.path.isfile(v):
                out[k] = True

        return out

    def run(self, overwrite=False, shell=None):
        if self._complete and not overwrite:
            return
        else:
            print('Analyzing dataset %s...' % self._dataset.data_name)

        if shell is not None:
            self._shell = shell
        else:
            shell = self._shell

        params = self._params
        dat = load_dataset(self.root_dir)
        rec_name = dat.data_name
        data_columns=['recording', 'area', 'unit', 'single_unit',
                      'unit_type', 'electrode', 'tastant', 'dig_in_ch',
                      'taste_responsive', 'taste_p_value', 'taste_u_stat',
                      'mean_baseline', 'sem_baseline',
                      'mean_taste_response', 'sem_taste_response',
                      'mean_taste_change', 'sem_taste_change',
                      'pal_responsive', 'pal_p_value','pal_u_stat']

        summary_columns = ['recording', 'area', 'unit',
                           'taste_responsive', 'pal_responsive',
                           'min_taste_p', 'tastant']

        unit_table = dat.get_unit_table().query('single_unit == True')  # Restrict to single units
        em = dat.electrode_mapping
        dim = dat.dig_in_mapping
        details = []
        summary = []
        for i, row in unit_table.iterrows():

            # Grab relevant values from tables
            unit_name = row['unit_name']
            unit_num = row['unit_num']
            el = row['electrode']
            single_unit = row['single_unit']
            area = em.query('Electrode == @el')['area'].values[0]
            if row['regular_spiking']:
                unit_type = 'pyramidal'
            elif row['fast_spiking']:
                unit_type = 'interneuron'
            else:
                unit_type = 'unclassified'

            taste_responsive, stats = check_taste_responsiveness(self.root_dir, unit_name,
                                                                 **params['taste_responsive'])

            min_p = 1
            min_taste = None
            for channel in stats.keys():
                dig_in = dim.loc[dim['channel'] == channel].squeeze()
                taste = dig_in['name']

                if dig_in['exclude'] or not dig_in['spike_array']:
                    continue


                tasty = False
                if stats[channel]['p-val'] <= params['taste_responsive']['alpha']:
                    tasty = True

                if stats[channel]['p-val'] < min_p:
                    min_p = stats[channel]['p-val']
                    min_taste = taste

                out = {'recording': rec_name, 'area': area, 'unit': unit_name,
                       'single_unit': single_unit, 'unit_type': unit_type,
                       'electrode': el, 'tastant': taste, 'dig_in_ch': channel,
                       'taste_responsive': tasty,
                       'taste_p_value': stats[channel]['p-val'],
                       'taste_u_stat': stats[channel]['u-stat'],
                       'mean_baseline': stats[channel]['baseline'][0],
                       'sem_baseline': stats[channel]['baseline'][1],
                       'mean_taste_response': stats[channel]['response'][0],
                       'sem_taste_response': stats[channel]['response'][1],
                       'mean_taste_change': stats[channel]['delta'][0],
                       'sem_taste_change': stats[channel]['delta'][1]}

                # TODO: Palatability analysis and append to out dict
                details.append(out.copy())

            sum_out = {'recording': rec_name, 'area': area, 'unit': unit_name,
                       'taste_responsive': taste_responsive,
                       'min_taste_p': min_p, 'tastant': min_taste}
            summary.append(sum_out.copy())

        # Store data in DataFrame
        single_unit_data = pd.DataFrame(details, columns=data_columns)
        single_unit_summary = pd.DataFrame(summary, columns=summary_columns)

        self._data = single_unit_data
        self._summary = single_unit_summary
        wt.write_pandas_to_table(single_unit_data, self._files['data'],
                                 overwrite=True)
        wt.write_pandas_to_table(single_unit_summary, self._files['summary'],
                                 overwrite=True)

        # TODO: Plot overlayed PSTHs with sig stars next to significant tastes on legend


def _convert_exp_output_data(data):
    df_cols = ['experiment', 'exp_group', 'held_unit_name', 'electrode',
               'single_unit', 'unit_type', 'recording', 'rec_group',
               'area', 'rec_unit_name', 'tastant', 'mean_baseline', 'sem_baseline',
               'baseline_shift', 'baseline_p',
               'mean_baseline_change', 'sem_baseline_change',
               'taste_responsive_all', 'taste_responsive',
               'taste_response_p', 'mean_taste_response',
               'sem_taste_response', 'response_change', 'earliest_response_divergence',
               'lastest_response_divergence']
    # Except for baseline info, use normalized response change


## Apply functions
def apply_unit_type(row):
    rsu = row['regular_spiking']
    fsu = row['fast_spiking']
    if rsu and fsu:
        out = 'mis-labelled'
    elif rsu:
        out = 'pyramidal'
    elif fsu:
        out = 'interneuron'
    else:
        out = 'un-labelled'

    return out

def apply_held_unit_name(held_df, row):
    rn = row['recording']
    un = row['rec_unit_name']
    tmp = held_df.query('recording==@rn and rec_unit==@un')['unit'].values
    if len(tmp) == 0:
        return None
    else:
        return tmp[0]

def flatten_dict(d):
    out = {}
    for k,v in d.items():
        if isinstance(v, dict):
            tmp = flatten_dict(v)
            for i,j in tmp.items():
                if isinstance(i, str):
                    new_key = (k, i)
                else:
                    new_key = (k, *i)

                out[new_key] = j

        else:
            out[(k)] = v

    return out
