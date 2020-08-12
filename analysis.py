import os
import shutil
import pdb
import pandas as pd
import numpy as np
import feather
import pickle
import aggregation as agg
import plotting as plt
import analysis_stats as stats
import population_analysis as pop
import hmm_analysis as hmma
from scipy.stats import mannwhitneyu, spearmanr, sem, f_oneway, rankdata, pearsonr, ttest_ind
from blechpy import load_project, load_dataset, load_experiment
from blechpy.plotting import data_plot as dplt
from copy import deepcopy
from blechpy.analysis import spike_analysis as sas, poissonHMM as phmm
from blechpy.dio import h5io
from scipy.stats import sem
from blechpy.utils import write_tools as wt
import pylab as pyplt
from datetime import datetime

ANALYSIS_PARAMS = {'taste_responsive': {'win_size': 750, 'alpha': 0.05},
                   'pal_responsive': {'win_size': 250, 'step_size': 25,
                                      'time_win': [0, 2000], 'alpha': 0.05},
                   'baseline_comparison': {'win_size': 1500, 'alpha': 0.01},
                   'response_comparison': {'win_size': 250, 'step_size': 250,
                                           'time_win': [0, 1500], 'alpha': 0.05,
                                           'n_boot':10000},
                   'psth': {'win_size': 250, 'step_size': 25, 'smoothing_win': 3,
                            'plot_window': [-1000, 1500]},
                   'pca': {'win_size': 750, 'step_size': 750,
                           'smoothing_win': 3,
                           'plot_window': [-500, 2000], 'time_win': [0, 1500]}}

ELF_DIR = '/data/Katz_Data/Stk11_Project/'
MONO_DIR = '/media/roshan/Gizmo/Katz_Data/Stk11_Project/'
if os.path.isdir(MONO_DIR):
    LOCAL_MACHINE = 'mononoke'
elif os.path.isdir(ELF_DIR):
    LOCAL_MACHINE = 'StealthElf'
else:
    LOCAL_MACHINE = ''


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
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.files = {'all_units': os.path.join(save_dir, 'all_units.feather'),
                      'held_units': os.path.join(save_dir, 'held_units.feather'),
                      'params': os.path.join(save_dir, 'analysis_params.json')}

    def detect_held_units(self, percent_criterion=95, raw_waves=True, overwrite=False):
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
            time_map = {'preCTA' : 'preCTA', 'ctaTrain': 'preCTA', 'ctaTest':
                        'postCTA', 'postCTA': 'postCTA'}
            all_units['time_group'] = all_units.rec_group.map(time_map)
            self.write_unit_info(all_units=all_units)

        if 'time_group' not in held_df.columns:
            held_df = held_df.apply(apply_info_from_rec_dir, axis=1)
            self.write_unit_info(held_df=held_df)

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
            labels = [] # List of tuples: (exp_group, exp_name, held_unit_name, taste)
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

                    tastes, pvs, tstat, md, md_sem, ctime, dtime = \
                            compare_taste_responses(rec1, unit1, rec2, unit2,
                                                    params, method='anova')
                    l = [(exp_group, exp_name, held_unit_name, t) for t in tastes]
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

        params = self.get_params()

        if os.path.isfile(save_file):
            data = np.load(save_file)
        else:
            raise FileNotFoundError('No data file found')

        alpha = params['response_comparison']['alpha']
        n_boot = params['response_comparison']['n_boot']
        labels = data['labels'] # exp_group, exp_name, held_unit_name, taste
        pvals = data['pvals']
        comp_time = data['comp_time']
        mean_diff = data['mean_diff']
        sem_diff = data['sem_diff']
        diff_time = data['diff_time']
        tastes = np.unique(labels[:, -1])
        # Make new labels for plotting learned CTA vs didn't learn
        learning_labels = labels.copy()
        learn_map = self.project._exp_info[['exp_name', 'CTA_learned']].copy().set_index('exp_name')
        for i, l in enumerate(labels):
            if learn_map.loc[l[1]]['CTA_learned']:
                learning_labels[i,0] = 'CTA'
            else:
                learning_labels[i,0] = 'No CTA'

        # Make aggregate plots
        plot_dir = os.path.join(save_dir, 'Held_Unit_Plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        heatmap_file = os.path.join(plot_dir, 'Held_unit_response_changes.svg')
        plt.plot_mean_differences_heatmap(labels, diff_time, mean_diff,
                                          save_file=heatmap_file, t_start=0)
        for tst in tastes:
            idx = np.where(labels[:,-1] == tst)[0]
            l = labels[idx, :]
            learn_l = learning_labels[idx, :]
            p = pvals[idx, :]
            md = mean_diff[idx, :]
            sd = sem_diff[idx, :]
            save_file = os.path.join(plot_dir, '%s_responses_changed.svg' % tst)
            learn_file = os.path.join(plot_dir, '%s_responses_changed-CTA_groups.svg' % tst)
            stat_file = os.path.join(save_dir, '%s_reponse_change_stats.txt' % tst)
            learn_stat_file = os.path.join(save_dir, '%s_responses_changed-CTA_groups.txt' % tst)
            tmp_data = (p <= alpha).astype('int')
            # Compare Cre vs GFP
            comp_p, test_stat, n_sigs = stats.permutation_test(l, tmp_data,
                                                               alpha=alpha,
                                                               n_boot=n_boot)
            plt.plot_held_percent_changed(l, comp_time, p, diff_time, md, sd,
                                          alpha, tst, group_pvals=comp_p, save_file=save_file)
            with open(stat_file, 'w') as f:
                print('%s %% held units changed' % tst, file=f)
                print('-'*80, file=f)
                print('time (ms): %s' % comp_time, file=f)
                if n_sigs is not None:
                    for k,v in n_sigs.items():
                        ix = np.where(l[:,0] == k)[0]
                        print('%s (# units changed): %s' % (k, v), file=f)
                        print('%s (# of units): %i' % (k, len(ix)), file=f)

                print('mean_difference (%%): %s' % test_stat, file=f)
                print('p-vals: %s' % comp_p, file=f)
                print('n_boot: %i' % n_boot, file=f)

            # Compare CTA vs No CTA
            comp_p, test_stat, n_sigs = stats.permutation_test(learn_l, tmp_data,
                                                               alpha=alpha,
                                                               n_boot=n_boot)
            plt.plot_held_percent_changed(learn_l, comp_time, p, diff_time, md, sd,
                                          alpha, tst, group_pvals=comp_p, save_file=learn_file)
            with open(learn_stat_file, 'w') as f:
                print('%s %% held units changed' % tst, file=f)
                print('-'*80, file=f)
                print('time (ms): %s' % comp_time, file=f)
                if n_sigs is not None:
                    for k,v in n_sigs.items():
                        ix = np.where(learn_l[:,0] == k)[0]
                        print('%s (# units changed): %s' % (k, v), file=f)
                        print('%s (# of units): %i' % (k, len(ix)), file=f)

                print('mean_difference (%%): %s' % test_stat, file=f)
                print('p-vals: %s' % comp_p, file=f)
                print('n_boot: %i' % n_boot, file=f)

            anim_dir = os.path.join(plot_dir, 'Per_Animal', tst)
            if not os.path.isdir(anim_dir):
                os.makedirs(anim_dir)

            animals = np.unique(l[:,1])
            for anim in animals:
                fn = os.path.join(anim_dir, '%s_%s_responses_changed.svg' % (anim,tst))
                a_idx = np.where(l[:,1] == anim)[0]
                a_l = l[a_idx,:]
                a_p = p[a_idx, :]
                a_md = md[a_idx, :]
                a_sd = sd[a_idx, :]
                plt.plot_held_percent_changed(a_l, comp_time, a_p, diff_time,
                                              a_md, a_sd, alpha, anim + ': ' + tst,
                                              save_file=fn)


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
        other_dist_file = os.path.join(save_dir, 'pc_dQ_v_dN_data.feather')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if os.path.isfile(pc_data_file) and os.path.isfile(dist_data_file) and os.path.isfile(other_dist_file) and not overwrite:
            pc_data = feather.read_dataframe(pc_data_file)
            dist_data = feather.read_dataframe(dist_data_file)
            other_dist_data = feather.read_dataframe(other_dist_file)
            return pc_data, dist_data, other_dist_data

        params = self.get_params()
        all_units, held_units = self.get_unit_info()
        all_units = all_units.dropna(subset=['held_unit_name'])
        all_units = all_units.query('area == "GC"')
        unit_names = all_units.held_unit_name.unique()
        held_units = held_units.dropna(subset=['held_unit_name'])
        held_units = held_units[held_units['held_unit_name'].isin(unit_names)]
        held_units = held_units.apply(apply_info_from_rec_dir, axis=1)
        if 'exp_group' not in held_units.columns:
            exp_map = self.project._exp_info.set_index('exp_name')['exp_group'].to_dict()
            held_units['exp_group'] = held_units.exp_name.map(exp_map)

        held_units = held_units.dropna(subset=['time_group'])
        unit_names = held_units['held_unit_name'].unique()
        all_units = all_units[all_units.held_unit_name.isin(unit_names)]
        # Now all_units and held_units have only units that are held over one
        # half of the experiment and are in GC
        grp = held_units.groupby(['exp_name', 'exp_group', 'time_group'])
        pc_data = grp.apply(lambda x: pop.apply_pca_analysis(x, params)).reset_index().drop(columns=['level_3'])
        pc_data['time'] = pc_data['time'].astype('int')
        pc_data['time'] = pc_data.time.apply(lambda x: 'Early (0-750ms)' if x <=750 else 'Late (750-1500ms)')
        grp = pc_data.groupby(['exp_name', 'exp_group', 'time_group', 'time'])
        dist_data = grp.apply(pop.apply_pc_distances).reset_index()
        pc_dist_metrics = grp.apply(pop.apply_pc_dist_metric).reset_index(drop=True)
        mds_dist_metrics = grp.apply(pop.apply_mds_dist_metric).reset_index(drop=True)
        dist_metrics = pd.merge(pc_dist_metrics, mds_dist_metrics,
                                on=['exp_name', 'exp_group', 'time_group',
                                    'time', 'taste','trial', 'n_cells', 'PC1',
                                    'PC2', 'MDS1', 'MDS2'])
        feather.write_dataframe(pc_data, pc_data_file)
        feather.write_dataframe(dist_data, dist_data_file)
        feather.write_dataframe(dist_metrics, other_dist_file)
        return pc_data, dist_data, dist_metrics

    def plot_pca_data(self):
        pc_data, dist_data, metric_data = self.pca_analysis()
        save_dir = os.path.join(self.save_dir, 'pca_analysis')
        mds_dir = os.path.join(self.save_dir, 'mds_analysis')
        if not os.path.isdir(mds_dir):
            os.mkdir(mds_dir)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        plt.plot_pca_distances(dist_data, os.path.join(save_dir, 'distances'))
        plt.plot_pca_metric(metric_data, os.path.join(save_dir, 'relative_PCA_distances.svg'))
        plt.plot_mds_metric(metric_data, os.path.join(mds_dir, 'relative_MDS_distances.svg'))
        plt.plot_animal_pca(pc_data, os.path.join(save_dir, 'animal_pca'))
        plt.plot_animal_mds(pc_data, os.path.join(mds_dir, 'animal_mds'))

        # Change exp group to CTA learning and re-plot
        learn_map = self.project._exp_info.set_index('exp_name')
        def foo(x):
            if learn_map.loc[x]['CTA_learned']:
                return 'CTA'
            else:
                return 'No CTA'

        metric_data['exp_group'] = metric_data['exp_name'].apply(foo)
        plt.plot_pca_metric(metric_data, os.path.join(save_dir, 'relative_PCA_distances-CTA.svg'))
        plt.plot_mds_metric(metric_data, os.path.join(mds_dir, 'relative_MDS_distances-CTA.svg'))


    def run(self, overwrite=False):
        self.detect_held_units(overwrite=overwrite, raw_waves=True)
        self.analyze_response_changes(overwrite=overwrite)
        self.make_aggregate_held_unit_plots()
        self.process_single_units(overwrite=overwrite)
        self.make_aggregate_single_unit_plots()
        self.pca_analysis(overwrite=overwrite)
        self.plot_pca_data()


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


def _deprecated_compare_taste_responses(rec1, unit1, rec2, unit2, params):
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
        pt1, psth1, baseline1 = agg.get_psth(rec1, unit1, ch1, params)
        pt2, psth2, baseline2 = agg.get_psth(rec2, unit2, ch2, params)
        diff, sem_diff = sas.get_mean_difference(psth1, psth2)
        # Mag diff plot looked odd, trying using same binning as comparison
        # diff, sem_diff = sas.get_mean_difference(fr1, fr2)
        # ^This looked worse, going back

        #Store stuff
        out_pvals.append(pvals)
        out_ustats.append(ustats)
        out_diff.append(diff)
        out_diff_sem.append(sem_diff)
        out_labels.append(taste)

    return out_labels, out_pvals, out_ustats, out_diff, out_diff_sem, bin_time, pt1


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

        self.base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                            'max_iter': 500, 'n_repeats': 25, 'time_start': 0,
                            'time_end': 1750, 'n_states': 2, 'area': 'GC'}
        # base params changed 8/4/20: max_iter 1000 -> 500, time_end 2000 -> 1500, n_states=2
        # changed 8/6/20: n_states->3
        # Changed 8/10/20: time_end-> 1750 & n_states -> 2

    def fit(self):
        tmp = self.base_params
        params = [{'n_states': i+2, **tmp.copy()} for i in range(2)]
        save_file = self.files['hmm_overview']
        fit_df = None
        for i, row in self.project._exp_info.iterrows():
            exp = load_experiment(row['exp_dir'])
            for rec_dir in exp.recording_dirs:
                dat = load_dataset(rec_dir)
                units = dat.get_unit_table()
                units = units[units.single_unit]
                if len(units) < 3:
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

    def check_hmm_fitting(self):
        df = self.get_sorted_hmms()
        if df is None:
            return None

        return df.groupby(['rec_dir', 'taste'])['sorting'].apply(lambda x: any(x == 'best'))

    def refit_rejected(self, common_log=None):
        base_params = self.base_params
        sorted_df = self.get_sorted_hmms()
        PA = ProjectAnalysis(self.project)
        all_units, held_units = PA.get_unit_info()
        refit_hmms(sorted_df, base_params, all_units, log_file=common_log)
        #ho = self.get_hmm_overview(overwrite=True)
        return

    def get_hmm_overview(self, overwrite=False):
        if not os.path.isfile(self.files['hmm_overview']):
            overwrite = True

        if not overwrite:
            ho = feather.read_dataframe(self.files['hmm_overview'])
        else:
            ho = None
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
                        df = handler.get_data_overview().copy()
                        df['rec_dir'] = rec_dir
                        if ho is None:
                            ho = df
                        else:
                            ho = ho.append(df, ignore_index=True)

            feather.write_dataframe(ho, self.files['hmm_overview'])

        if not 'exp_name' in ho.columns:
            ho[['exp_name','rec_group']] = ho['rec_dir'].apply(parse_rec)
            feather.write_dataframe(ho, self.files['hmm_overview'])

        if not 'll_check' in ho.columns:
            ho['ll_check'] = ho.apply(hmma.check_ll_asymptote, axis=1)
            feather.write_dataframe(ho, self.files['hmm_overview'])

        if not 'single_state_trials' in ho.columns:
            ho['single_state_trials'] = ho.apply(hmma.check_single_state_trials, axis=1)
            feather.write_dataframe(ho, self.files['hmm_overview'])

        return ho

    def sort_hmms(self, overwrite=False):
        sorted_hmms = self.get_sorted_hmms()
        if sorted_hmms is not None and not overwrite:
            return sorted_hmms

        ho = self.get_hmm_overview(overwrite=overwrite)
        new_sorting = hmma.sort_hmms(ho)
        if sorted_hmms is not None:
            for i, row in sorted_hmms.iterrows():
                if row['sort_method'] == 'manual':
                    j = ((new_sorting['rec_dir'] == row['rec_dir']) &
                         (new_sorting['hmm_id'] == row['hmm_id']))
                    new_sorting.loc[j, 'sorting'] = row['sorting']
                    new_sorting.loc[j, 'sort_method'] = row['sort_method']
                    k = ((new_sorting['rec_dir'] == row['rec_dir']) &
                         (new_sorting['taste'] == row['taste']) &
                         (new_sorting['hmm_id'] != row['hmm_id']))
                    new_sorting.loc[k, 'sorting'] = 'rejected'
                    new_sorting.loc[k, 'sort_method'] = 'manual'

        self.write_sorted_hmms(new_sorting)
        self.plot_sorted_hmms(overwrite=overwrite)
        return new_sorting

    def write_sorted_hmms(self, sorted_hmms):
        sorted_file = os.path.join(self.save_dir, 'sorted_hmms.feather')
        feather.write_dataframe(sorted_hmms, sorted_file)

    def get_sorted_hmms(self):
        sorted_file = os.path.join(self.save_dir, 'sorted_hmms.feather')
        if os.path.isfile(sorted_file):
            return feather.read_dataframe(sorted_file)
        else:
            return None

    def get_best_hmms(self):
        df = self.get_sorted_hmms()
        df = df[df['sorting'] == 'best']
        return df

    def mark_hmm_as(self, sorting, **kwargs):
        '''kwargs should be column & value and will be used to manually re-sort
        HMMs and mark then as "best", "rejected" or "refit"
        '''
        qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in kwargs.items()])
        nqry = ' or '.join(['{} != "{}"'.format(k,v) for k,v in kwargs.items()])
        sorted_hmms = self.get_sorted_hmms()
        j = None
        for k,v in kwargs.items():
            tmp = (sorted_hmms[k] == v)
            if j is None:
                j = tmp
            else:
                j = j & tmp

        old_sorting = sorted_hmms.loc[j, 'sorting'].unique()
        print('-'*80)
        print('Marking HMMs as %s' % sorting)
        print(sorted_hmms.loc[j][['exp_name', 'rec_group', 'taste', 'hmm_id',
                                  'sorting', 'sort_method']])
        print('-'*80)
        sorted_hmms.loc[j, 'sorting'] = sorting
        sorted_hmms.loc[j, 'sort_method'] = 'manual'
        # make sure there is only 1 best for each rec_dir & taste
        best_df = sorted_hmms[sorted_hmms['sorting'] == 'best']
        multiple_bests = []
        for name, group in best_df.groupby(['rec_dir', 'taste']):
            if len(group) != 1:
                multiple_bests.append(group.copy())

        self.write_sorted_hmms(sorted_hmms)

    def mark_hmm_state(self, exp_name, rec_group, hmm_id, early_state=None, late_state=None):
        '''Set the HMM state number that identifies the stat to be used as the
        early state or late state in analysis
        '''
        if early_state is None and late_state is None:
            return

        hmm_df = self.get_sorted_hmms()
        i = ((hmm_df['exp_name'] == exp_name) &
             (hmm_df['rec_group'] == rec_group) &
             (hmm_df['taste'] == taste))

        print('-'*80)
        print('Setting state for HMMs: %s %s %s')
        if early_state is not None:
            print('    - setting early_state to state #%i' % early_state)
            hmm_df.loc[i, 'early_state'] = early_state

        if late_state is not None:
            print('    - setting late_state to state #%i' % early_state)
            hmm_df.loc[i, 'late_state'] = late_state

        print('Saving dataframe...')
        self.write_sorted_hmms(hmm_df)
        print('-'*80)

    def plot_sorted_hmms(self, overwrite=False, skip_rejected=True):
        plot_dirs = {'best': os.path.join(self.save_dir, 'Best_HMMs'),
                     'rejected': os.path.join(self.save_dir, 'Rejected_HMMs'),
                     'refit': os.path.join(self.save_dir, 'Refit_HMMs')}
        sorted_hmms = self.get_sorted_hmms()
        for k,v in plot_dirs.items():
            if os.path.isdir(v) and overwrite:
                shutil.rmtree(v)

            if not os.path.isdir(v):
                os.mkdir(v)

        for i, row in sorted_hmms.iterrows():
            fn = '%s_%s_HMM%i-%s.svg' % (row['exp_name'], row['rec_group'],
                                         row['hmm_id'], row['taste'])
            if row['sorting'] not in plot_dirs.keys():
                continue

            if skip_rejected and row['sorting'] == 'rejected':
                continue

            fn = os.path.join(plot_dirs[row['sorting']], fn)
            if os.path.isfile(fn) and not overwrite:
                continue

            plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn)

    def plot_hmms(self, overwrite=False):
        hmm_df = self.get_hmm_overview()
        grp_keys = list(self.base_params.keys())
        grp_keys.append('n_states')
        exp_map = self.project._exp_info.set_index('exp_name')['exp_group'].to_dict()

        for grp_i, (name, group) in enumerate(hmm_df.groupby(grp_keys)):
            if len(group) < 3:
                continue

            save_dir = os.path.join(self.save_dir, 'All_HMMs', 'Paramters_%i' % grp_i)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            params = group[grp_keys].to_dict(orient='records')[0]
            wt.write_dict_to_json(params, os.path.join(save_dir, 'params.json'))
            plot_dir = os.path.join(save_dir, 'Plots')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            state_breakdown = None
            for row_i, (_, row) in enumerate(group.iterrows()):
                rec_dir = row['rec_dir']
                hmm_id = row['hmm_id']
                fn = os.path.join(plot_dir, '%s_%s_HMM%i-%s.svg' % (row['exp_name'],
                                                                    row['rec_group'],
                                                                    hmm_id,
                                                                    row['taste']))

                if os.path.isfile(fn) and not overwrite:
                    continue

                plt.plot_hmm(rec_dir, hmm_id, save_file=fn)

    def analyze_state_durations(self, overwrite=False):
        hmm_df = self.get_sorted_hmms()
        hmm_df = hmm_df[hmm_df['sorting'] == 'best']
        grp_keys = list(self.base_params.keys())
        grp_keys.append('n_states')
        exp_map = self.project._exp_info.set_index('exp_name')['exp_group'].to_dict()

        for grp_i, (name, group) in enumerate(hmm_df.groupby(grp_keys)):
            save_dir = os.path.join(self.save_dir, 'Paramters_%i' % grp_i)
            if os.path.isdir(save_dir):
                if overwrite:
                    shutil.rmtree(save_dir)
                else:
                    continue

            os.mkdir(save_dir)
            params = group[grp_keys].to_dict(orient='records')[0]
            wt.write_dict_to_json(params, os.path.join(save_dir, 'params.json'))
            plot_dir = os.path.join(save_dir, 'Plots')
            os.mkdir(plot_dir)

            state_breakdown = None
            for row_i, (_, row) in enumerate(group.iterrows()):
                rec_dir = row['rec_dir']
                hmm_id = row['hmm_id']
                tmp = hmma.get_state_breakdown(rec_dir, hmm_id)
                tmp['exp_group'] = tmp['exp_name'].map(exp_map)
                if state_breakdown is None:
                    state_breakdown = tmp.copy()
                else:
                    state_breakdown = state_breakdown.append(tmp).reset_index(drop=True)

            dat_file = os.path.join(save_dir, 'state_breakdown.feather')
            feather.write_dataframe(state_breakdown, dat_file)


def get_saccharin_consumption(anim_dir):
    '''greabs animal metadata from anim-dir and returns
    mean_saccharin_consumption/mean_water_consumption
    drops CTA Training day. 
    '''
    ld = [os.path.join(anim_dir, x) for x in os.listdir(anim_dir) if 'metadata.p' in x]
    if len(ld) == 0:
        return None
    if len(ld) != 1:
        raise ValueError('%i metadata files found. Expected 1.' % len(ld))

    ld = ld[0]
    with open(ld, 'rb') as f:
        dat = pickle.load(f)

    def fix(x):
        if 'Saccharin' in x:
            return 'Saccharin'
        else:
            return 'Water'

    drinks = dat.bottle_tests.iloc[1:].copy() # Drop first day of water dep
    drinks['Substance'] = drinks['Substance'].apply(fix)
    if not any(drinks.Substance == 'Saccharin'):
        return None

    ctaTrain = [x for x in dat.ioc_tests if 'Train' in x['Test Type']]
    if len(ctaTrain) == 0:
        mean_water = drinks[drinks.Substance == 'Water']['Change (g)'].astype('float').mean()
        mean_sacc = drinks[drinks.Substance == 'Saccharin']['Change (g)'].astype('float').mean()
        return mean_sacc/mean_water


    ctaDay = ctaTrain[0]['Test Time']
    a = ctaDay.replace(hour=0,minute=0,second=0)
    b = ctaDay.replace(hour=23,minute=59,second=59)
    tmp2 = drinks.truncate(after=a).append(drinks.truncate(before=b))
    mean_water = tmp2[tmp2.Substance == 'Water']['Change (g)'].astype('float').mean()
    mean_sacc = tmp2[tmp2.Substance == 'Saccharin']['Change (g)'].astype('float').mean()
    # tmp2['Norm Change'] = (tmp2['Change (g)'] / mean(water))
    return mean_sacc/mean_water


def apply_consumption_to_project(proj):
    #if 'saccharin_consumption' not in proj._exp_info.columns:
    tmp = proj._exp_info['exp_dir'].apply(get_saccharin_consumption)
    print(tmp)
    proj._exp_info['saccharin_consumption'] = tmp
    #else:
    #    tmp = proj._exp_info['saccharin_consumption']

    proj._exp_info['CTA_learned'] = (tmp < 0.8)
    proj.save()


class Analysis(object):
    def __init__(self, data_dir, analysis_name, analysis_dir=None):
        pass

    def _check_files(self):
        pass

    def _load_params(self):
        pass

    def _write_params(self):
        pass

    def _update_params(self):
        pass

    def run(self):
        pass


def compare_taste_responses(rec1, unit1, rec2, unit2, params, method='bootstrap'):
    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    time_start = params['response_comparison']['time_win'][0]
    time_end = params['response_comparison']['time_win'][1]
    baseline_win = params['baseline_comparison']['win_size']
    n_boot = params['response_comparison']['n_boot']
    alpha = params['response_comparison']['alpha']

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

        if method.lower() == 'anova':
            nt = len(t1)
            pvals = np.ones((nt,))
            ustats = np.zeros((nt,))
            for i, y in enumerate(zip(fr1.T, fr2.T)):
                # u, p = mann_whitney_u(y[0], y[1])
                # Mann-Whitney U gave odd results, trying anova
                u, p = f_oneway(y[0], y[1])
                pvals[i] = p
                ustats[i] = u
        elif method.lower() == 'bootstrap':
            nt = len(t1)
            tmp_lbls = np.vstack(fr1.shape[0]*['u1'] + fr2.shape[0]*['u2'])
            tmp_data = np.vstack((fr1,fr2))
            pvals, ustats, _ = stats.permutation_test(tmp_lbls, tmp_data,
                                                      alpha)

        # apply bonferroni correction
        pvals = pvals*nt

        # Compute mean difference using psth parameters
        pt1, psth1, base1 = agg.get_psth(rec1, unit1, ch1, params)
        pt2, psth2, base2 = agg.get_psth(rec2, unit2, ch2, params)
        # Original
        # diff, sem_diff = sas.get_mean_difference(psth1, psth2)
        # Mag diff plot looked odd, trying using same binning as comparison
        # diff, sem_diff = sas.get_mean_difference(fr1, fr2)
        # ^This looked worse, going back
        # Now computing and plotting the difference in zscored firing rates
        zpsth1 = (psth1-base1[0])/base1[1]
        zpsth2 = (psth2-base2[0])/base2[1]
        diff, sem_diff = sas.get_mean_difference(psth1, psth2)

        #Store stuff
        out_pvals.append(pvals)
        out_ustats.append(ustats)
        out_diff.append(diff)
        out_diff_sem.append(sem_diff)
        out_labels.append(taste)

    return out_labels, out_pvals, out_ustats, out_diff, out_diff_sem, bin_time, pt1


def parse_rec(rd):
    if rd[-1] == os.sep:
        rd = rd[:-1]

    parsed = os.path.basename(rd).split('_')
    return pd.Series([parsed[0], parsed[-3]], index=['exp_name', 'rec_group'])


# def refit_hmms(refit_df, base_params, log_file=None):
#     if not os.path.isfile(log_file):
#         f = open(log_file, 'w')
#         f.close()
# 
#     id_cols = ['taste', 'n_states', 'dt', 'time_start', 'time_end']
#     df = refit_df[['rec_dir', 'hmm_id', *id_cols]]
#     for rec_dir, group in df.groupby('rec_dir'):
#         if log_file is not None:
#             with open(log_file, 'r+') as f:
#                 processed = f.read().split('\n')
#                 if rec_dir in processed:
#                     print('Skipping %s\nAlready Processed' % rec_dir)
#                     continue
#                 else:
#                     f.write(rec_dir + '\n')
# 
#         if ELF_DIR in rec_dir and not os.path.isdir(rec_dir) and os.path.isdir(MONO_DIR):
#             rd = rec_dir.replace(ELF_DIR, MONO_DIR)
#         else:
#             rd = rec_dir
# 
#         print('Processing HMMs for %s' % rd)
#         handler = phmm.HmmHandler(rd)
#         for i, row in group.iterrows():
#             if row['taste'] == 'Water':
#                 continue
# 
#             handler.delete_hmm(**row[id_cols])
#             params = {'n_states': row['n_states'], 'taste': row['taste'],
#                       **base_params.copy()}
#             handler.add_params(params)
# 
#         handler.run()
# 
#     pyplt.close('all')

def refit_hmms(sorted_df, base_params, all_units,log_file=None, rec_params={}):
    '''re-wrtitten 8/4/20
    edited 8/10/20
    '''
    if not os.path.isfile(log_file):
        f = open(log_file, 'w')
        f.close()

    all_units = all_units[all_units['single_unit']]
    id_cols = ['taste', 'n_states', 'dt', 'time_start', 'time_end']
    df = sorted_df[['rec_dir', 'hmm_id', 'sorting', 'sort_method', *id_cols]]
    # refit = [] # rec_dir, local_rec_dir, taste, channel
    # for name, group in df.groupby(['rec_dir', 'taste']):
    #     if any(group['sorting'] == 'best'):
    #         continue

    #     rec_dir, taste = name
    #     rd = get_local_path(rec_dir)

    #     dat = load_dataset(rd)
    #     dim = dat.dig_in_mapping.set_index('name')
    #     channel = dim.loc[taste]
    #     refit.append((rec_dir, rd, taste, channel))

    # refit = np.array(refit)
    # rec_dirs = np.unique(refit[:,0])
    rec_dirs = all_units.rec_dir.unique()
    for rec in rec_dirs:
        if log_file is not None:
            with open(log_file, 'r+') as f:
                processed = f.read().split('\n')
                if rec in processed:
                    print('Skipping %s\nAlready Processed' % rec)
                    continue
                else:
                    f.write(rec + '\n')
                    f.write(LOCAL_MACHINE + '\n')
                    f.write(datetime.now().strftime('%m/%d/%Y %H:%M') + '\n\n')

        print('Processing %s' % rec)
        units = all_units[all_units['rec_dir'] == rec]
        rd = get_local_path(rec)
        if not all(units['area'] == 'GC'):
            # Previous HMMs were fitting using all cells, must refit if STR units were included
            h5_file = hmma.get_hmm_h5(rd)
            print('Deleting hdf5 since STR cells were included')
            os.remove(h5_file)



        # idx = np.where(refit[:,0] == rec)[0]
        # group = refit[idx, :]
        params = base_params.copy()
        if rec in rec_params.keys():
            for k,v in rec_params[rec].items():
                params[k] = v

        handler = phmm.HmmHandler(rd)
        units = units[units['area'] == 'GC']
        if len(units) < 3:
            print('Not enough single units in GC. skipping...')
            continue

        if len(units) > 20:
            params['unit_type'] = 'pyramidal'

        dat = load_dataset(rd)
        tastes = dat.dig_in_mapping['name'].to_list()
        dim = dat.dig_in_mapping.copy().set_index('name')
        new_params = []
        for tst in tastes:
            tmp_i = ((sorted_df['rec_dir'] == rec) &
                     (sorted_df['taste'] == tst))
            sortings = sorted_df.loc[tmp_i, 'sorting']
            if any(sortings == 'best'):
                continue

            handler.delete_hmm(taste=tst, n_states=params['n_states'],
                               unit_type=params['unit_type'])
            if tst != 'Water':
                p = params.copy()
                p['taste'] = tst
                p['channel'] = dim.loc[tst, 'channel']
                handler.add_params(p)

        # overview = handler.get_data_overview()
        # for _, _, taste, channel in group:
        #     # delete all hmms for taste
        #     handler.delete_hmm(taste=taste, n_states=params['n_states'])
        #     p = params.copy()
        #     p['channel'] = channel
        #     p['taste'] = taste
        #     handler.add_params(p)

        handler.run()


def get_local_path(path):
    if ELF_DIR in path and not os.path.isdir(path) and os.path.isdir(MONO_DIR):
        out = path.replace(ELF_DIR, MONO_DIR)
    else:
        out = path

    return out


def analyze_saccharin_confusion(held_df, best_hmms):
    held_df = held_df.copy()
    held_df = held_df.dropna(subset=['held_unit_name', 'time_group'])
    training_tastes = ['NaCl', 'Quinine']
    test_tastes = ['Saccharin']
    # For each exp_name, time_group, make early firing rate array & late firing rate array
    # also make early and late array for saccharin
    # Use NaCl + Quinine array to train a NB Classifier and Saccharin to test it
    # Record Number classified as NaCl / Number Quinine
    # out_df has columns: exp_name, time_group, exp_group, N_cells, N_NaCl, N_Quinine, relative_confusion
    # Use best hmm for rec/taste to get windows for firing rate
    # Use ANOVA to look at classifier accuracy, group and N_cells
    # For now allow 1 held unit to be enough but later may restrict to 2+ cells
    template = {'exp_name': None, 'time_group': None, 'exp_group': None,
                'n_cells': None, 'n_trials': None, 'n_nacl_early': None,
                'n_quinine_early': None, 'n_nacl_late': None, 'n_quinine_late': None,
                'relative_confusion_early': None, 'relative_confusion_late': None}
    out = []
    for name, group in held_df.groupby(['exp_name', 'time_group']):
        rec1 = group.rec1.unique()[0]
        units1 = list(group.unit1)
        rec2 = group.rec2.unique()[0]
        units2 = list(group.unit2)
        # Make sure rec1 has NaCl and Quinine and rec2 has Saccharin
        if 'preCTA' in rec2 or 'postCTA' in rec2:
            tmp = rec1
            rec1 = rec2
            rec2 = tmp
            tmp = units1
            units1 = units2
            units2 = tmp

        early_train = []
        late_train = []
        train_labels = []
        h5_file1 = hmma.get_hmm_h5(rec1)
        h5_file2 = hmma.get_hmm_h5(rec2)
        for taste in training_tastes:
            i = ((best_hmms['rec_dir'] == rec1) &
                 (best_hmms['taste'] == taste) &
                 (best_hmms['sorting'] == 'best'))
            tmp = best_hmms.loc[i]
            if len(tmp) > 1:
                raise ValueError('Too many best HMMs for %s %s' % (rec1, taste))
            elif len(tmp) == 0:
                continue

            tmp = tmp.iloc[0]
            if np.isnan(tmp['early_state']) or np.isnan(tmp['late_state']):
                continue

            tmp_l, tmp_er, tmp_lr = hmma.get_early_and_late_firing_rates(rec1,
                                                                         tmp['hmm_id'],
                                                                         tmp['early_state'],
                                                                         tmp['late_state'],
                                                                         units=units1)
            tmp_l = [(taste, *x) for x in tmp_l]
            early_train.append(tmp_er)
            late_train.append(tmp_lr)
            train_labels.append(tmp_l)

        if len(early_train) < len(training_tastes):
            continue

        # Grab Saccharin Data
        early_test = []
        late_test = []
        test_labels = []
        for taste in test_tastes:
            i = ((best_hmms['rec_dir'] == rec2) &
                 (best_hmms['taste'] == taste) &
                 (best_hmms['sorting'] == 'best'))
            tmp = best_hmms.loc[i]
            if len(tmp) > 1:
                raise ValueError('Too many best HMMs for %s %s' % (rec2, taste))
            elif len(tmp) == 0:
                continue

            tmp = tmp.iloc[0]
            if np.isnan(tmp['early_state']) or np.isnan(tmp['late_state']):
                continue

            tmp_l, tmp_er, tmp_lr = hmma.get_early_and_late_firing_rates(rec2,
                                                                         tmp['hmm_id'],
                                                                         tmp['early_state'],
                                                                         tmp['late_state'],
                                                                         units=units2)
            tmp_l = [(taste, *x) for x in tmp_l]
            early_test.append(tmp_er)
            late_test.append(tmp_lr)
            test_labels.append(tmp_l)

        if len(early_test) < len(test_tastes):
            continue

        # Now train calssifier and test
        early_train = np.vstack(early_train)
        early_test = np.vstack(early_test)
        late_train = np.vstack(late_train)
        late_test = np.vstack(late_test)
        train_labels = np.vstack(train_labels)
        test_labels = np.vstack(test_labels)

        early_gnb = GaussianNB()
        early_pred = early_gnb.fit(early_train, train_labels[:, 0]).predict(early_test)
        late_gnb = GaussianNB()
        late_pred = late_gnb.fit(late_train, train_labels[:,0]).predict(late_test)
        n_trials = test_labels.shape[0]
        e_nacl = np.sum(early_pred == 'NaCl')
        l_nacl = np.sum(late_pred == 'NaCl')
        e_quin = np.sum(early_pred == 'Quinine')
        l_quin = np.sum(late_pred == 'Quinine')
        row = tempalte.copy()
        row['exp_name'] = name[0]
        row['time_group'] = name[1]
        row['exp_group'] = group.exp_group.unique()[0]
        row['n_cells'] = len(group)
        row['n_trials'] = n_trials
        row['n_nacl_early'] = e_nacl
        row['n_nacl_late'] = l_nacl
        row['n_quinine_early'] = e_quin
        row['n_quinine_late'] = l_quin
        row['relative_confusion_early'] = e_nacl / e_quin
        row['relative_confusion_late'] = l_nacl / l_quin
        out.append(row)

    return pd.DataFrame(out)




if __name__=="__main__":
    print('Hello World')
