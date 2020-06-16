import os
import pdb
import pandas as pd
import numpy as np
import feather
import aggregation as agg
import plotting as plt
import statistics as stats
import population_analysis as pop
from scipy.stats import mannwhitneyu, spearmanr, sem, f_oneway, rankdata, pearsonr, ttest_ind
from blechpy import load_project, load_dataset, load_experiment
from blechpy.plotting import data_plot as dplt
from copy import deepcopy
from blechpy.analysis import spike_analysis as sas, poissonHMM as phmm
from blechpy.dio import h5io
from scipy.stats import sem
from blechpy.utils import write_tools as wt
import pylab as pyplt

ANALYSIS_PARAMS = {'taste_responsive': {'win_size': 750, 'alpha': 0.05},
                   'pal_responsive': {'win_size': 250, 'step_size': 25,
                                      'time_win': [0, 2000], 'alpha': 0.05},
                   'baseline_comparison': {'win_size': 1500, 'alpha': 0.01},
                   'response_comparison': {'win_size': 250, 'step_size': 250,
                                           'time_win': [0, 2000], 'alpha': 0.05},
                   'psth': {'win_size': 250, 'step_size': 25, 'smoothing_win': 3,
                            'plot_window': [-1500, 2000]},
                   'pca': {'win_size': 250, 'step_size': 25,
                           'smoothing_win': 3,
                           'plot_window': [-500, 2000], 'time_win': [-1000, 2000]}}

def update_params(new, old):
    out = deepcopy(old)
    for k,v in new.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v

    return out


class ProjectAnalysis(object):
    def __init__(self, proj):
        self.root_dir = os.path.join(proj.root_dir, proj.data_name + '_analysis')
        self.project = proj
        save_dir = os.path.join(self.root_dir, 'single_unit_analysis')
        self.save_dir = save_dir
        self.files = {'all_units': os.path.join(save_dir, 'all_units.feather'),
                      'held_units': os.path.join(save_dir, 'held_units.feather'),
                      'params': os.path.join(save_dir, 'analysis_params.json')}

    def detect_held_units(self, percent_criterion=95, raw_waves=False, overwrite=False):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if os.path.isfile(all_units_file) and os.path.isfile(held_units_file) and not overwrite:
            all_units = feather.read_dataframe(all_units_file)
            held_df = feather.read_dataframe(held_units_file)
        else:
            all_units, held_df = agg.find_held_units(self.project, percent_criterion, raw_waves)
            feather.write_dataframe(all_units, all_units_file)
            feather.write_dataframe(held_df, held_units_file)

            # Plot waveforms and J3 distribution
            plot_dir = os.path.join(save_dir, 'held_unit_waveforms')
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            plt.plot_held_units(all_units, plot_dir)
            dplt.plot_J3s(all_units['intra_J3'].dropna().to_numpy(),
                          held_df['inter_J3'].dropna().to_numpy(),
                          save_dir, percent_criterion)

        return all_units, held_df

    def get_unit_info(self):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isfile(all_units_file) or not os.path.isfile(held_units_file):
            raise ValueError('Please run get_held_units first')

        all_units = feather.read_dataframe(all_units_file)
        held_df = feather.read_dataframe(held_units_file)
        if 'time_group' not in all_units.columns:
            time_map = {'preCTA' : 'preCTA', 'ctaTrain': 'preCTA', 'ctaTest': 'postCTA', 'postCTA': 'postCTA'}
            all_units['time_group'] = all_units.rec_group.map(time_map)
            self.write_unit_info(all_units=all_units)

        return all_units, held_df

    def write_unit_info(self, all_units=None, held_df=None):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if all_units is not None:
            feather.write_dataframe(all_units, all_units_file)

        if held_df is not None:
            feather.write_dataframe(held_df, held_units_file)

    def get_params(self, params=None):
        params_file = self.files['params']
        if os.path.isfile(params_file):
            base_params = wt.read_dict_from_json(params_file)
        else:
            base_params = deepcopy(ANALYSIS_PARAMS)

        if params is not None:
            params = update_params(params, base_params)
        else:
            params = base_params

        wt.write_dict_to_json(params, params_file)
        return params

    def analyze_response_changes(self, params=None, overwrite=False):
        all_units, held_df = self.get_unit_info()
        save_dir = os.path.join(self.save_dir, 'held_unit_response_changes')
        save_file = os.path.join(save_dir, 'response_change_data.npz')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        params = self.get_params(params)

        if os.path.isfile(save_file) and not overwrite:
            data = np.load(save_file)
            return data
        else:
            rec_map = {'preCTA': 'postCTA', 'ctaTrain': 'ctaTest'}
            all_units = all_units.dropna(subset=['held_unit_name'])
            all_units = all_units[all_units['area'] == 'GC']

            # Output structures
            alpha = params['response_comparison']['alpha']
            labels = [] # List of tuples: (exp_group, held_unit_name, taste)
            pvals = []
            differences = []
            sem_diffs = []
            test_stats = []
            comp_time = None
            diff_time = None

            for held_unit_name, group in all_units.groupby('held_unit_name'):
                if not any(group.time_group == 'preCTA') or not any(group.time_group == 'postCTA'):
                    continue

                pre_grp = group.query('time_group == "preCTA"')
                for unit_id1, row in pre_grp.iterrows():
                    post_row = group.loc[group.rec_group == rec_map[row['rec_group']]]
                    if post_row.empty:
                        continue
                    else:
                        unit_id2 = post_row.index[0]
                        post_row = post_row.to_dict(orient='records')[0]

                    rec1 = row['rec_dir']
                    rec2 = post_row['rec_dir']
                    unit1 = row['unit_num']
                    unit2= post_row['unit_num']
                    exp_group = row['exp_group']
                    exp_name = row['exp_name']
                    print('Comparing Held Unit %s, %s vs %s' % (held_unit_name, row['rec_name'], post_row['rec_name']))

                    tastes, pvs, tstat, md, md_sem, ctime, dtime = compare_taste_responses(rec1, unit1, rec2, unit2, params)
                    l = [(exp_group, held_unit_name, t) for t in tastes]
                    labels.extend(l)
                    pvals.extend(pvs)
                    test_stats.extend(tstat)
                    differences.extend(md)
                    sem_diffs.extend(md_sem)
                    if comp_time is None:
                        comp_time = ctime
                    elif not np.array_equal(ctime, comp_time):
                        raise ValueError('Times dont match')

                    if diff_time is None:
                        diff_time = dtime
                    elif not np.array_equal(dtime, diff_time):
                        raise ValueError('Times dont match')

                    for tst in tastes:
                        plot_dir = os.path.join(save_dir, 'Held_Unit_Plots', tst)
                        if not os.path.isdir(plot_dir):
                            os.makedirs(plot_dir)

                        fig_file = os.path.join(plot_dir, 'Held_Unit_%s-%s.svg' % (held_unit_name, tst))

                        plt.plot_held_unit_comparison(rec1, unit1, rec2, unit2,
                                                      pvs[0], params, held_unit_name,
                                                      exp_name, exp_group, tst,
                                                      save_file=fig_file)

            labels = np.vstack(labels) # exp_group, held_unit_name, taste
            pvals = np.vstack(pvals)
            test_stats = np.vstack(test_stats)
            differences = np.vstack(differences)
            sem_diffs = np.vstack(sem_diffs)

            # Stick it all in an npz file
            np.savez(save_file, labels=labels, pvals=pvals, test_stats=test_stats,
                     mean_diff=differences, sem_diff=sem_diffs,
                     comp_time=comp_time, diff_time=diff_time)
            data = np.load(save_file)
            return data

    def make_aggregate_held_unit_plots(self):
        save_dir = os.path.join(self.root_dir, 'single_unit_analysis', 'held_unit_response_changes')
        save_file = os.path.join(save_dir, 'response_change_data.npz')
        params_file = os.path.join(save_dir, 'analysis_params.json')

        if os.path.isfile(params_file):
            params = wt.read_dict_from_json(params_file)
        else:
            raise FileNotFoundError('No params found')

        if os.path.isfile(save_file):
            data = np.load(save_file)
        else:
            raise FileNotFoundError('No data file found')

        alpha = params['response_comparison']['alpha']
        labels = data['labels']
        pvals = data['pvals']
        comp_time = data['comp_time']
        mean_diff = data['mean_diff']
        sem_diff = data['sem_diff']
        diff_time = data['diff_time']
        tastes = np.unique(labels[:, -1])
        # Make aggregate plots
        plot_dir = os.path.join(save_dir, 'Held_Unit_Plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        for tst in tastes:
            idx = np.where(labels[:,-1] == tst)[0]
            l = labels[idx, :]
            p = pvals[idx, :]
            md = mean_diff[idx, :]
            sd = sem_diff[idx, :]
            save_file = os.path.join(plot_dir, '%s_responses_changed.svg' % tst)
            plt.plot_held_percent_changed(l, comp_time, p, diff_time, md, sd, alpha, tst, save_file=save_file)

        return

    def process_single_units(self, params=None, overwrite=False):
        save_dir = os.path.join(self.save_dir, 'single_unit_responses')
        pal_file = os.path.join(save_dir, 'palatability_data.npz')
        resp_file = os.path.join(save_dir, 'taste_responsive_pvals.npz')
        tasty_unit_file = os.path.join(save_dir, 'unit_taste_responsivity.feather')
        pal_unit_file = os.path.join(save_dir, 'unit_pal_discrim.feather')
        params = self.get_params(params)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if os.path.isfile(tasty_unit_file) and os.path.isfile(pal_unit_file) and not overwrite:
            resp_units = feather.read_dataframe(tasty_unit_file)
            pal_units = feather.read_dataframe(pal_unit_file)
            return resp_units, pal_units

        if overwrite and os.path.isfile(pal_file):
            os.remove(pal_file)

        if overwrite and os.path.isfile(resp_file):
            os.remove(resp_file)

        all_units, _ = self.get_unit_info()
        all_units = all_units[all_units['single_unit']]
        resp_units = all_units.groupby('rec_dir', group_keys=False).apply(apply_tastes)
        print('-' * 80)
        print('Processing taste resposiveness')
        print('-' * 80)
        resp_units = resp_units.apply(lambda x: apply_taste_responsive(x, params, resp_file), axis=1)
        feather.write_dataframe(resp_units, tasty_unit_file)

        pal_units = all_units[all_units.rec_name.str.contains('4taste')].copy()
        plot_dir = os.path.join(save_dir, 'single_unit_plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        def foo(x):
            return apply_discrim_and_pal(x, params, pal_file, plot_dir)

        print('-' * 80)
        print('Processing taste discrimination and palatability')
        print('-' * 80)
        pal_units = pal_units.apply(foo, axis=1)
        pal_units['taste_discriminative'] = pal_units['taste_discriminative'].astype('bool')
        feather.write_dataframe(pal_units, pal_unit_file)
        return resp_units, pal_units

    def make_aggregate_single_unit_plots(self):
        resp_units, pal_units = self.process_single_units()
        save_dir = os.path.join(self.save_dir, 'single_unit_responses')
        resp_file = os.path.join(save_dir, 'taste_responsive.svg')
        discrim_file = os.path.join(save_dir, 'taste_discriminative.svg')
        spearman_file = os.path.join(save_dir, 'palatability_spearman.svg')
        pearson_file = os.path.join(save_dir, 'palatability_pearson.svg')
        params = self.get_params()
        # For responsive, plot 
        if 'time_group' not in resp_units.columns or 'time_group' not in pal_units.columns:
            time_map = {'preCTA' : 'preCTA', 'ctaTrain': 'preCTA', 'ctaTest':
                        'postCTA', 'postCTA': 'postCTA'}
            resp_units['time_group'] = resp_units.rec_group.map(time_map)
            pal_units['time_group'] = pal_units.rec_group.map(time_map)

        tmp_grp = resp_units.groupby(['exp_group', 'time_group', 'taste'])['taste_responsive']
        resp_df = tmp_grp.apply(lambda x: 100 * np.sum(x) / len(x)).reset_index()
        plt.plot_taste_responsive(resp_df, resp_file)
        plt.plot_taste_discriminative(pal_units, discrim_file)
        plt.plot_aggregate_spearman(pal_units, spearman_file)
        plt.plot_aggregate_pearson(pal_units, pearson_file)

        spear_mean = os.path.join(save_dir, 'Mean_Spearmann.svg')
        pear_mean = os.path.join(save_dir, 'Mean_Pearson.svg')
        resp_time = os.path.join(save_dir, 'Taste_responsive_over_time.svg')
        resp_data = os.path.join(save_dir, 'taste_responsive_pvals.npz')
        pal_data = os.path.join(save_dir, 'palatability_data.npz')
        plt.plot_mean_spearman(pal_data, spear_mean)
        plt.plot_mean_pearson(pal_data, pear_mean)
        alpha = params['taste_responsive']['alpha']
        plt.plot_taste_response_over_time(resp_data, resp_time, alpha)

    def pca_analysis(self, overwrite=False):
        '''Grab units held across pre OR post. For each animal do pca on firing
        rate traces, then plot for individual traces and mean trace for each
        taste. 1 plot per animal, pre & post subplot
        '''
        save_dir = os.path.join(self.save_dir, 'pca_analysis')
        pc_data_file = os.path.join(save_dir, 'pc_data.feather')
        dist_data_file = os.path.join(save_dir, 'pc_dist_data.feather')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if os.path.isfile(pc_data_file) and os.path.isfile(dist_data_file) and not overwrite:
            pc_data = feather.read_dataframe(pc_data_file)
            dist_data = feather.read_dataframe(dist_data_file)
            return pc_data, dist_data

        params = self.get_params()
        all_units, held_units = self.get_unit_info()
        all_units = all_units.dropna(subset=['held_unit_name'])
        all_units = all_units.query('area == "GC"')
        unit_names = all_units.held_unit_name.unique()
        held_units = held_units.dropna(subset=['held_unit_name'])
        held_units = held_units[held_units['held_unit_name'].isin(unit_names)]
        held_units = held_units.apply(apply_info_from_rec_dir, axis=1)
        held_units = held_units.dropna(subset=['time_group'])
        unit_names = held_units['held_unit_name'].unique()
        all_units = all_units[all_units.held_unit_name.isin(unit_names)]
        # Now all_units and held_units have only units that are held over one
        # half of the experiment and are in GC
        grp = held_units.groupby(['exp_name', 'time_group'])
        pc_data = grp.apply(lambda x: pop.apply_pca_analysis(x, params)).reset_index()
        grp = pc_data.groupby(['exp_name', 'time_group', 'time'])
        dist_data = grp.apply(pop.apply_pc_distances).reset_index()
        feather.write_dataframe(pc_data, pc_data_file)
        feather.write_dataframe(dist_data, dist_data_file)
        return pc_data, dist_data

    def plot_pca_data(self):
        pass


def apply_info_from_rec_dir(row):
    rd1 = row['rec1']
    rd2 = row['rec2']
    if 'pre' in rd1 and 'Train' in rd2:
        row['held_over'] = 'pre'
        row['time_group'] = 'preCTA'
    elif 'Test' in rd1 and 'post' in rd2:
        row['held_over'] = 'post'
        row['time_group'] = 'postCTA'
    elif 'Train' in rd1 and 'Test' in rd2:
        row['held_over'] = 'cta'
        row['time_group'] = None
    else:
        raise ValueError('Doesnt fit into group')

    rec = os.path.basename(rd1).split('_')
    row['exp_name'] = rec[0]
    row['rec_group'] = rec[-3]
    return row


def apply_discrim_and_pal(row, params, save_file, plot_dir):
    d_win = params['taste_responsive']['win_size']
    d_alpha = params['taste_responsive']['alpha']
    p_bin_size = params['pal_responsive']['win_size']
    p_step = params['pal_responsive']['step_size']
    p_alpha = params['pal_responsive']['alpha']
    p_win = params['pal_responsive']['time_win']

    rec = row['rec_dir']
    rec_name = row['rec_name']
    unit = row['unit_num']
    unit_name = row['unit_name']
    exp_group = row['exp_group']
    time_group = row['time_group']
    print('Analyzing %s %s...' % (rec_name, unit_name))

    # Check taste dicriminability
    # Grab taste responses for each taste
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('channel')
    dim = dim[dim.exclude == False]
    channels = dim.index.tolist()
    tastes = dim.name.tolist()
    responses = []
    for ch in channels:
        t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, d_win, t_start=0,
                                             t_end=d_win)
        responses.append(fr)

    if len(responses) > 1:
        f, p = f_oneway(*responses)
        row['taste_discriminative'] = p <= d_alpha
        row['discrim_p'] = p
        row['discrim_f'] = f
    else:
        row['taste_discriminative'] = False
        row['discrim_p'] = np.NaN
        row['discrim_f'] = np.NaN

    # Check palatability
    responses = []
    palatability = []
    time = None
    for ch in channels:
        rank = dim.loc[ch, 'palatability_rank']
        if rank <= 0:
            continue

        t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, p_bin_size,
                                             step_size=p_step,
                                             t_start=p_win[0], t_end=p_win[1])
        if time is None:
            time = t
        elif not np.array_equal(time, t):
            raise ValueError('Time vectors dont match')

        pal = np.ones((fr.shape[0],)) * rank
        responses.append(fr)
        palatability.append(pal)

    if len(responses) < 3:
        row['spearman_r'] = np.NaN
        row['spearman_p'] = np.NaN
        row['spearman_peak'] = np.NaN
        row['pearson_r'] = np.NaN
        row['pearson_p'] = np.NaN
        row['pearson_peak'] = np.NaN
    else:
        responses = np.vstack(responses)
        palatability = np.concatenate(palatability)
        n_bins = len(time)
        s_rs = np.zeros((n_bins,))
        s_ps = np.ones((n_bins,))
        p_rs = np.zeros((n_bins,))
        p_ps = np.ones((n_bins,))
        for i, t in enumerate(time):
            if all(responses[:,i] == 0):
                continue
            else:
                response_ranks = rankdata(responses[:, i])
                s_rs[i], s_ps[i] = spearmanr(response_ranks, palatability)
                p_rs[i], p_ps[i] = pearsonr(responses[:, i], palatability)

        sidx = np.where(s_ps <= p_alpha)[0]
        pidx = np.where(p_ps <= p_alpha)[0]
        if len(sidx) == 0:
            sidx = np.arange(0, n_bins)

        if len(pidx) == 0:
            pidx = np.arange(0, n_bins)

        smax = np.argmax(np.abs(s_rs[sidx]))
        smax = sidx[smax]
        pmax = np.argmax(np.abs(p_rs[pidx]))
        pmax = pidx[pmax]

        row['spearman_r'] = s_rs[smax]
        row['spearman_p'] = p_rs[smax]
        row['spearman_peak'] = time[smax]
        row['pearson_r'] = p_rs[pmax]
        row['pearson_p'] = p_ps[pmax]
        row['pearson_peak'] = time[pmax]

        # Save data array
        label = (exp_group, time_group, rec, unit)
        if not os.path.isfile(save_file):
            np.savez(save_file, labels=np.array([label]), time=time,
                     spearman_r=s_rs, spearman_p=s_ps, pearson_r=p_rs,
                     pearson_p=p_ps)
        else:
            data = np.load(save_file)
            labels = np.vstack((data['labels'], label))
            if not np.array_equal(time, data['time']):
                raise ValueError('Time doesnt match')

            spearman_r = np.vstack((data['spearman_r'], s_rs))
            spearman_p = np.vstack((data['spearman_p'], s_ps))
            pearson_r = np.vstack((data['pearson_r'], p_rs))
            pearson_p = np.vstack((data['pearson_p'], p_ps))
            np.savez(save_file, labels=labels, time=time,
                     spearman_r=spearman_r, spearman_p=spearman_p, pearson_r=pearson_r,
                     pearson_p=pearson_p)

        # Plot PSTHs
        # Plot spearman r & p
        # Plot pearson r & p
        psth_fn = '%s_%s_psth.svg' % (rec_name, unit_name)
        psth_file = os.path.join(plot_dir, 'PSTHs', psth_fn)
        corr_file = os.path.join(plot_dir, 'Palatability', psth_fn.replace('psth', 'corr'))
        plt.plot_PSTHs(rec, unit, params, psth_file)
        plt.plot_palatability_correlation(rec_name, unit_name, time, s_rs, s_ps,
                                          p_rs, p_ps, corr_file)

    return row


def apply_taste_responsive(row, params, data_file):
    bin_size = params['taste_responsive']['win_size']
    alpha = params['taste_responsive']['alpha']
    rec = row['rec_dir']
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('name')
    taste = row['taste']
    ch = dim.loc[taste, 'channel']
    unit = row['unit_num']
    unit_name = row['unit_name']
    print('Analyzing %s %s...' % (row['rec_name'], unit_name))

    t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, bin_size,
                                         t_start=-bin_size, t_end=bin_size)
    baseline = fr[:, 0]
    response = fr[:, 1]
    if all(baseline == 0) and all(response == 0):
        f = 0
        p = 1
    else:
        f, p = ttest_ind(baseline, response)

    row['taste_responsive'] = (p <= alpha)
    row['reponse_p'] = p
    row['response_f'] = f

    # Break it up by time and save array
    # Use one way anova and dunnett's post hoc to compare all time bins to baseline
    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    t_end = params['response_comparison']['time_win'][1]
    time, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, bin_size,
                                            t_start=-bin_size, t_end=t_end)
    f, p = f_oneway(*fr.T)
    if p > alpha:
        return row

    baseline = fr[:,0]
    fr = fr[:,1:]
    fr = [fr[:,i] for i in range(fr.shape[1])]
    time = time[1:]
    n_bins = len(time)
    # Now use Dunnett's to compare each time bin to baseline
    CIs, pvals = stats.dunnetts_post_hoc(baseline, fr, alpha)
    # Open npz file and append to arrays
    pvals = np.array(pvals)
    this_label = (row['exp_group'], row['rec_dir'], row['unit_num'],
                  row['time_group'], row['taste'])

    if os.path.isfile(data_file):
        data = np.load(data_file)
        labels = data['labels']
        PV = data['pvals']
        labels = np.vstack((labels, this_label))
        PV = np.vstack((PV, pvals))
        if not np.array_equal(data['time'], time):
            raise ValueError('Time vectors dont match')

    else:
        labels = np.array(this_label)
        PV = pvals

    np.savez(data_file, labels=labels, pvals=PV, time=time)
    return row


def compare_taste_responses(rec1, unit1, rec2, unit2, params):
    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    time_start = params['response_comparison']['time_win'][0]
    time_end = params['response_comparison']['time_win'][1]
    baseline_win = params['baseline_comparison']['win_size']

    dat1 = load_dataset(rec1)
    dat2 = load_dataset(rec2)

    dig1 = dat1.dig_in_mapping.copy().set_index('name')
    dig2 = dat2.dig_in_mapping.copy().set_index('name')
    out_labels = []
    out_pvals = []
    out_ustats = []
    out_diff = []
    out_diff_sem = []
    bin_time = None
    for taste, row in dig1.iterrows():
        ch1 = row['channel']
        ch2 = dig2.loc[taste, 'channel']
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

        if bin_time is None:
            bin_time = t1

        if not np.array_equal(bin_time, t1) or not np.array_equal(t1, t2):
            raise ValueError('Unqueal time vectors')

        nt = len(t1)
        pvals = np.ones((nt,))
        ustats = np.zeros((nt,))
        for i, y in enumerate(zip(fr1.T, fr2.T)):
            # u, p = mann_whitney_u(y[0], y[1])
            # Mann-Whitney U gave odd results, trying anova
            u, p = f_oneway(y[0], y[1])
            pvals[i] = p
            ustats[i] = u

        # Apply bonferroni correction
        pvals = pvals * nt

        # Compute mean difference using psth parameters
        pt1, psth1 = agg.get_psth(rec1, unit1, ch1, params)
        pt2, psth2 = agg.get_psth(rec2, unit2, ch2, params)
        # diff, sem_diff = sas.get_mean_difference(psth1, psth2)
        # Mag diff plot looked odd, trying using same binning as comparison
        diff, sem_diff = sas.get_mean_difference(fr1, fr2)

        #Store stuff
        out_pvals.append(pvals)
        out_ustats.append(ustats)
        out_diff.append(diff)
        out_diff_sem.append(sem_diff)
        out_labels.append(taste)

    return out_labels, out_pvals, out_ustats, out_diff, out_diff_sem, bin_time, bin_time # pt1


def mann_whitney_u(resp1, resp2):
    try:
        u, p = mannwhitneyu(resp1, resp2, alternative='two-sided')
    except ValueError:
        u = 0
        p = 1

    return u, p


def apply_tastes(rec_group):
    rec_dir = rec_group.rec_dir.unique()[0]
    dat = load_dataset(rec_dir)
    tastes = dat.dig_in_mapping.name.tolist()
    tmp = rec_group.to_dict(orient='records')
    out = []
    for t in tastes:
        for item in tmp:
            j = item.copy()
            j['taste'] = t
            out.append(j)

    return pd.DataFrame(out)


def fix_area(proj):
    exp_areas = {'RN5': 'right', 'RN10': 'both', 'RN11': 'right',
                 'RN15': 'both', 'RN16': 'both', 'RN17': 'both',
                 'RN18': 'both', 'RN19': 'right', 'RN20': 'right',
                 'RN21': 'right', 'RN22': 'both', 'RN23': 'right',
                 'RN24': 'both', 'RN25': 'both'}
    exp_info = proj._exp_info
    for i, row in exp_info.iterrows():
        exp = load_experiment(row['exp_dir'])
        name = row['exp_name']
        ingc = exp_areas[name]
        if ingc is 'right':
            el = np.arange(8, 24)
        elif ingc is 'left':
            el = np.concatenate([np.arange(0,8), np.arange(24, 32)])
        elif ingc is 'none':
            el = np.arange(0,32)
        else:
            el = None

        for rec in exp.recording_dirs:
            dat = load_dataset(rec)
            print('Fixing %s...' % dat.data_name)
            em = dat.electrode_mapping
            em['area'] = 'GC'
            if el is not None:
                em.loc[em['Channel'].isin(el), 'area'] = 'STR'

            h5io.write_electrode_map_to_h5(dat.h5_file, em)
            dat.save()

    return


class HmmAnalysis(object):
    def __init__(self, proj):
        self.root_dir = os.path.join(proj.root_dir, proj.data_name + '_analysis')
        self.project = proj
        save_dir = os.path.join(self.root_dir, 'hmm_analysis')
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.files = {'params': os.path.join(save_dir, 'hmm_params.json'),
                      'hmm_overview': os.path.join(save_dir, 'hmm_overview.feather')}

    def fit(self):
        tmp = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
               'max_iter': 300, 'n_repeats': 10, 'time_start': 0,
               'time_end': 2000}
        params = [{'n_states': i+2, **tmp.copy()} for i in range(4)]
        save_file = self.files['hmm_overview']
        fit_df = None
        for i, row in self.project._exp_info.iterrows():
            exp = load_experiment(row['exp_dir'])
            for rec_dir in exp.recording_dirs:
                dat = load_dataset(rec_dir)
                units = dat.get_unit_table()
                units = units[units.single_unit]
                if len(units) < 4:
                    continue
                else:
                    handler = phmm.HmmHandler(rec_dir)
                    handler.add_params(params)
                    handler.run()
                    df = handler.get_data_overview().copy()
                    df['rec_dir'] = rec_dir
                    if fit_df is None:
                        fit_df = df
                    else:
                        fit_df = fit_df.append(df, ignore_index=True)

                feather.write_dataframe(fit_df, save_file)
                pyplt.close('all')


if __name__=="__main__":
    print('Hello World')
