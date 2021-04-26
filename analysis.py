from pushbullet_tools import push_message as pm
import os
import shutil
import pdb
import pandas as pd
import numpy as np
import feather
import pickle
import aggregation as agg
import plotting as plt
import new_plotting as nplt
import analysis_stats as stats
import helper_funcs as hf
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
from tqdm import tqdm
from statsmodels.stats.weightstats import CompareMeans
from joblib import parallel_backend

ANALYSIS_PARAMS = {'taste_responsive': {'win_size': 750, 'alpha': 0.05},
                   'pal_responsive': {'win_size': 250, 'step_size': 25,
                                      'time_win': [0, 2000], 'alpha': 0.05},
                   'baseline_comparison': {'win_size': 1500, 'alpha': 0.01},
                   'response_comparison': {'win_size': 250, 'step_size': 250,
                                           'time_win': [0, 1500], 'alpha': 0.05,
                                           'n_boot':1000},
                   'psth': {'win_size': 250, 'step_size': 25, 'smoothing_win': 3,
                            'plot_window': [-1000, 2000]},
                   'pca': {'win_size': 750, 'step_size': 750,
                           'smoothing_win': 3,
                           'plot_window': [-500, 2000], 'time_win': [0, 1500]}}

DIRECTORIES = {'StealthElf': '/data/Katz_Data/Stk11_Project/',
               'Mononoke': '/media/roshan/Gizmo/Katz_Data/Stk11_Project/'}

LOCAL_MACHINE = os.uname()[1]
DATA_DIR = DIRECTORIES.get(LOCAL_MACHINE)


def get_file_dirs(animID):
    anim_dir = os.path.join(DATA_DIR, animID)
    fd = [os.path.join(anim_dir, x) for x in os.listdir(anim_dir)]
    file_dirs = [x for x in fd if os.path.isdir(x)]
    out = []
    for f in file_dirs:
        fl = os.listdir(f)
        if any([x.endswith('.dat') for x in fl]):
            out.append(f)

    return out, anim_dir


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
        '''Gathers all single units from all recordings. For each cells, the
        intra-recording J3 value is computed and then for each pair of units on
        the same electrode from consecutive recordings the inter-recording J3
        values if computed. For each experimental group (exp_group), the
        inter-rec J3 is compared to the group's intra_recording J3
        distribution, and units are labelled are held if the inter-rec J3 is
        less than the 95th percentile of the group intra-rec J3 distribution
        '''
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if os.path.isfile(all_units_file) and os.path.isfile(held_units_file) and not overwrite:
            all_units = feather.read_dataframe(all_units_file)
            held_df = feather.read_dataframe(held_units_file)
        else:
            all_units, held_df = agg.find_held_units(self.project,
                                                     percent_criterion,
                                                     raw_waves)
            self.write_unit_info(all_units=all_units, held_df=held_df)

            # Plot waveforms and J3 distribution
            plot_dir = os.path.join(save_dir, 'held_unit_waveforms')
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            plt.plot_held_units(all_units, plot_dir)
            dplt.plot_J3s(all_units['intra_J3'].dropna().to_numpy(),
                          held_df['inter_J3'].dropna().to_numpy(),
                          save_dir, percent_criterion)

        return self.get_unit_info()

    def get_unit_info(self):
        '''Grabs unit info dataframes and makes sure they have all of the
        appropriate columns
        '''
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isfile(all_units_file) or not os.path.isfile(held_units_file):
            raise ValueError('Please run detect_held_units first')

        all_units = feather.read_dataframe(all_units_file)
        held_df = feather.read_dataframe(held_units_file)
        grouping_cols = ['cta_group', 'exp_group', 'time_group', 'exclude']

        if any([x not in all_units.columns for x in grouping_cols]):
            all_units = agg.apply_grouping_cols(all_units, self.project)
            self.write_unit_info(all_units=all_units)

        if any([x not in held_df.columns for x in grouping_cols]):
            held_df = agg.apply_grouping_cols(held_df, self.project)
            self.write_unit_info(held_df=held_df)

        if 'area' not in held_df.columns:
            for i, row in held_df.iterrows():
                r1 = row['rec1']
                u1 = row['unit1']
                held_df.loc[i, 'area'] = all_units.query('rec_dir == @r1 and unit_name == @u1').iloc[0]['area']
            self.write_unit_info(held_df=held_df)

        if 'response_firing' not in all_units.columns:
            all_units = apply_unit_firing_rates(all_units)
            self.write_unit_info(all_units=all_units)

        if 'unit_type' not in all_units.columns:
            def foo(x):
                if not x['single_unit']:
                    return 'multi'
                elif x['regular_spiking']:
                    return 'pyramidal'
                else:
                    return 'interneuron'

            all_units['unit_type'] = all_units.apply(foo, axis=1)
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
        '''returns analysis params that are stored in json file, and updates
        them with new params if given
        '''
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
        '''loops through all held cells and compares pre- to post- CTA
        responses for each avaliable tastant. Stores data in npz file
        '''
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
            learn_map = self.project._exp_info.set_index('exp_name')['CTA_learned']
            learn_map = learn_map.apply(lambda x: 'CTA' if x else 'No CTA').to_dict()

            # Output structures
            alpha = params['response_comparison']['alpha']
            labels = [] # List of tuples: (exp_group, exp_name, cta_group, held_unit_name, taste)
            pvals = []
            ci_low = []
            ci_high = []
            differences = []
            sem_diffs = []
            test_stats = []
            comp_time = None
            diff_time = None

            for held_unit_name, group in all_units.groupby('held_unit_name'):
                #Skip units that aren't held from pre to post CTA
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
                    print('Comparing Held Unit %s, %s vs %s' % (held_unit_name,
                                                                row['rec_name'],
                                                                post_row['rec_name']))

                    # Compare taste responses
                    tastes, pvs, tstat, md, md_sem, ctime, dtime, low, high = \
                            compare_taste_responses(rec1, unit1, rec2, unit2,
                                                    params, method='anova')
                    l = [(exp_group, exp_name, learn_map[exp_name],
                          held_unit_name, t) for t in tastes]
                    labels.extend(l)
                    pvals.extend(pvs)
                    ci_low.extend(low)
                    ci_high.extend(high)
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

                    for i, tst in enumerate(tastes):
                        plot_dir = os.path.join(save_dir, 'Held_Unit_Plots', tst)
                        new_plot_dir = os.path.join(plot_dir, 'clean_plots')
                        if not os.path.isdir(plot_dir):
                            os.makedirs(plot_dir)

                        if not os.path.isdir(new_plot_dir):
                            os.makedirs(new_plot_dir)

                        fig_file = os.path.join(plot_dir, 'Held_Unit_%s-%s.svg' % (held_unit_name, tst))
                        new_fn = os.path.join(new_plot_dir,
                                              'Held_Unit_%s-%s.svg' %
                                              (held_unit_name, tst))

                        plt.plot_held_unit_comparison(rec1, unit1, rec2, unit2,
                                                      pvs[i], params, held_unit_name,
                                                      exp_name, exp_group, tst,
                                                      save_file=fig_file)
                        nplt.plot_held_unit_comparison(rec1, unit1, rec2, unit2,
                                                       pvs[i], params, held_unit_name,
                                                       exp_name, exp_group, tst,
                                                       save_file=new_fn)

            labels = np.vstack(labels) # exp_group, exp_name, cta_group, held_unit_name, taste
            pvals = np.vstack(pvals)
            ci_low = np.vstack(ci_low)
            ci_high = np.vstack(ci_high)
            test_stats = np.vstack(test_stats)
            differences = np.vstack(differences)
            sem_diffs = np.vstack(sem_diffs)

            # Stick it all in an npz file
            np.savez(save_file, labels=labels, pvals=pvals, test_stats=test_stats,
                     mean_diff=differences, sem_diff=sem_diffs,
                     comp_time=comp_time, diff_time=diff_time, ci_low=ci_low,
                     ci_high=ci_high)
            data = np.load(save_file)
            return data

    def make_aggregate_held_unit_plots(self):
        save_dir = os.path.join(self.root_dir, 'single_unit_analysis',
                                'held_unit_response_changes')
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
        all_df = hf.make_tidy_response_change_data(labels, pvals, comp_time, alpha=alpha)
        all_df = agg.apply_grouping_cols(all_df, self.project)
        save_cols = ['exp_name', 'exp_group', 'cta_group', 'taste', 'time',
                     'time_bin', 'n_diff', 'n_same', 'exclude']
        all_df.to_csv(os.path.join(save_dir,
                                   'held_unit_response_change_data.csv'),
                      columns=save_cols,
                      index=False)

        # Make aggregate plots
        plot_dir = os.path.join(save_dir, 'Held_Unit_Plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        heatmap_file = os.path.join(plot_dir, 'Held_unit_response_changes.svg')
        plt.plot_mean_differences_heatmap(labels, diff_time, mean_diff,
                                          save_file=heatmap_file, t_start=0)
        for tst in tastes:
            df = all_df[all_df['taste'] == tst]
            idx = np.where(labels[:,-1] == tst)[0]
            l = labels[idx, :]
            p = pvals[idx, :]
            md = mean_diff[idx, :]
            sd = sem_diff[idx, :]

            # Compare all Cre to GFP
            hf.compare_response_changes(df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed',
                                        group_col='exp_group', alpha=alpha,
                                        exp_group='Cre', ctrl_group='GFP')

            # Compare all CTA to No CTA
            hf.compare_response_changes(df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-CTA_groups',
                                        group_col='cta_group', alpha=alpha,
                                        exp_group='No CTA', ctrl_group='CTA')

            # Compare Cre-No CTA to GFP-CTA
            sub_df = df.query('(exp_group == "Cre" & cta_group == "No CTA") '
                              '| (exp_group == "GFP" & cta_group == "CTA")')
            hf.compare_response_changes(sub_df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-exclude',
                                        group_col='exp_group', alpha=alpha,
                                        exp_group='Cre', ctrl_group='GFP')

            # Compare Cre-No CTA to GFP-No CTA
            sub_df = df.query('(exp_group == "Cre" & cta_group == "No CTA") '
                              '| (exp_group == "GFP" & cta_group == "No CTA")')
            hf.compare_response_changes(sub_df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-Cre_v_BadGFP',
                                        group_col='exp_group', alpha=alpha,
                                        exp_group='Cre', ctrl_group='GFP')

            # Compare GFP CTA vs No CTA
            sub_df = df.query('(exp_group == "GFP" & cta_group == "No CTA") '
                              '| (exp_group == "GFP" & cta_group == "CTA")')
            hf.compare_response_changes(sub_df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-GFP_v_GFP',
                                        group_col='cta_group', alpha=alpha,
                                        exp_group='No CTA', ctrl_group='CTA')

            # Plot changes per Animals
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

    def plot_figure_2(self):
        '''Plot and compare unit firing rates. Paper Figure 2.
        '''
        save_dir = os.path.join(self.save_dir, 'single_unit_responses')
        all_units, _ = self.get_unit_info()
        fn = os.path.join(save_dir, 'unit_firing_rates.svg')
        nplt.plot_unit_firing_rates(all_units, save_file=fn)

    def process_single_units(self, params=None, overwrite=False):
        '''For all single units, analyze and plots taste_responsivity,
        taste_discrimination, and palatability_resposivity
        '''
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

        # Plot and compare firing rates: Cre-NoCTA vs GFP-CTA
        all_units, _ = self.get_unit_info()
        fn = os.path.join(save_dir, 'unit_firing_rates.svg')
        nplt.plot_unit_firing_rates(all_units, save_file=fn)

        # Compute taste responsiveness
        all_units = all_units[all_units['single_unit']]
        resp_units = all_units.groupby('rec_dir', group_keys=False).apply(apply_tastes)
        print('-' * 80)
        print('Processing taste resposiveness')
        print('-' * 80)
        resp_units = resp_units.apply(lambda x: apply_taste_responsive(x, params, resp_file), axis=1)
        resp_units = agg.apply_grouping_cols(resp_units, self.project)
        feather.write_dataframe(resp_units, tasty_unit_file)
        # Plot and compare taste responsive units: Cre-NoCTA vs GFP-CTA
        fn = os.path.join(save_dir, 'taste_responsive.svg')
        df = resp_units.query('exclude == False and taste != "Water"')
        nplt.plot_taste_responsive_units(df, fn)

        # Use units from 4taste sessions to compute and compare taste
        # discrimination and palatability correlation
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
        pal_units = agg.apply_grouping_cols(pal_units, self.project)
        feather.write_dataframe(pal_units, pal_unit_file)
        df = pal_units[pal_units['exclude'] == False]
        nplt.plot_pal_responsive_units(df, save_dir)

        return resp_units, pal_units

    def make_aggregate_single_unit_plots(self):
        '''Plot palatability correlations and taste_responsive over post-stimulus time
        '''
        resp_units, pal_units = self.process_single_units()
        save_dir = os.path.join(self.save_dir, 'single_unit_responses')
        params = self.get_params()

        spearman_file = os.path.join(save_dir, 'palatability_spearman.svg')
        plt.plot_aggregate_spearman(pal_units, spearman_file)
        spear_mean = os.path.join(save_dir, 'Mean_Spearman.svg')
        pal_data = os.path.join(save_dir, 'palatability_data.npz')
        nplt.plot_mean_spearman_correlation(pal_data, self.project, spear_mean)

        #pearson_file = os.path.join(save_dir, 'palatability_pearson.svg')
        #plt.plot_aggregate_pearson(pal_units, pearson_file)
        #pear_mean = os.path.join(save_dir, 'Mean_Pearson.svg')
        #plt.plot_mean_pearson(pal_data, pear_mean)

        alpha = params['taste_responsive']['alpha']
        resp_time = os.path.join(save_dir, 'Taste_responsive_over_time.svg')
        resp_data = os.path.join(save_dir, 'taste_responsive_pvals.npz')
        plt.plot_taste_response_over_time(resp_data, resp_time, alpha)

    def pca_analysis(self, overwrite=False):
        '''Grab units held across pre OR post. For each animal do pca & MDS on firing
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
        dist_metrics = agg.apply_grouping_cols(dist_metrics, self.project)
        pc_data = agg.apply_grouping_cols(pc_data, self.project)
        feather.write_dataframe(pc_data, pc_data_file)
        feather.write_dataframe(dist_data, dist_data_file)
        feather.write_dataframe(dist_metrics, other_dist_file)
        return pc_data, dist_data, dist_metrics

    def plot_MDS_data(self):
        '''Plot and compare MDS distances for Saccharin
        '''
        pc_data, dist_data, metric_data = self.pca_analysis()
        save_dir = os.path.join(self.save_dir, 'pca_analysis')
        mds_dir = os.path.join(self.save_dir, 'mds_analysis')
        if not os.path.isdir(mds_dir):
            os.mkdir(mds_dir)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        metric_data = agg.apply_grouping_cols(metric_data, self.project)
        metric_data = metric_data.query('exclude == False')
        fn = os.path.join(mds_dir, 'Saccharin_MDS_distances.svg')
        nplt.plot_MDS(metric_data, save_file=fn, ylabel='dQ-dN',
                      value_col='MDS_dQ_minus_dN', kind='point')


    def fix_palatability(self):
        '''Make sure all recordings have taste palatability properly labelled
        to agree with agg.PAL_MAP
        '''
        agg.fix_palatability(self.project, pal_map=agg.PAL_MAP)

    def fix_areas(self):
        '''Make sure all electrode area are properly labelled to agree with
        agg.ELECTRODES_IN_GC as determined by visual confirmation in slices
        '''
        agg.set_electrode_areas(self.project, el_in_gc=agg.ELECTRODES_IN_GC)

    def plot_saccharin_consumption(self):
        '''Plot and compare Saccharin free consumption relative to previous average
        water consumption.
        '''
        save_dir = self.save_dir
        fn = os.path.join(save_dir, 'Saccharin_consumption.svg')
        nplt.plot_saccharin_consumption(self.project, fn)

    def run(self, overwrite=False):
        '''Runs all analyses for single and held units. Comment out earlier
        lines when not needed. Be sure to run all when adding new animals
        '''
        #self.fix_areas()
        #self.fix_palatability()
        #self.detect_held_units(overwrite=overwrite, raw_waves=True)
        self.analyze_response_changes(overwrite=overwrite)
        self.make_aggregate_held_unit_plots()
        self.process_single_units(overwrite=overwrite)
        self.make_aggregate_single_unit_plots()
        self.pca_analysis(overwrite=overwrite)
        self.plot_pca_data()
        self.plot_saccharin_consumption()


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
        nplt.plot_PSTHs(rec, unit, params, save_file=psth_file)
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
    row['response_p'] = p
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


def apply_tastes(rec_group):
    '''expands dataframe to have rows for every tastant in each recording
    '''
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


class HmmAnalysis(object):
    def __init__(self, proj):
        self.root_dir = os.path.join(proj.root_dir, proj.data_name + '_analysis')
        self.project = proj
        save_dir = os.path.join(self.root_dir, 'hmm_analysis')
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.files = {'params': os.path.join(save_dir, 'hmm_params.json'),
                      'hmm_overview': os.path.join(save_dir, 'hmm_overview.feather'),
                      'sorted_hmms': os.path.join(save_dir, 'sorted_hmms.feather'),
                      'best_hmms': os.path.join(save_dir, 'best_hmms.feather'),
                      'hmm_coding': os.path.join(save_dir, 'hmm_coding.feather'),
                      'hmm_confusion': os.path.join(save_dir, 'hmm_confusion.feather'),
                      'hmm_mds': os.path.join(save_dir, 'hmm_mds.feather'),
                      'hmm_timings': os.path.join(save_dir, 'hmm_timings.feather')}

        self.base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                            'max_iter': 100, 'n_repeats': 50, 'time_start': -250,
                            'time_end': 2000, 'n_states': 3, 'area': 'GC',
                            'hmm_class':'PoissonHMM', 'threshold': 1e-10,
                            'notes': 'sequential constraint'}

    def fit(self):
        '''Run HMM fitting using base params for all recordings with 3+ cells
        '''
        params = self.base_params
        save_file = self.files['hmm_overview']
        fit_df = None
        for i, row in self.project._exp_info.iterrows():
            exp = load_experiment(row['exp_dir'])
            for rec_dir in exp.recording_dirs:
                units = phmm.query_units(rec_dir, params['unit_type'], area=params['area'])
                dat = load_dataset(rec_dir)
                if len(units) < 3:
                    continue
                else:
                    handler = phmm.HmmHandler(rec_dir)
                    handler.add_params(params)
                    handler.run(constraint_func=phmm.sequential_contraint)
                    df = handler.get_data_overview().copy()
                    df['rec_dir'] = rec_dir
                    if fit_df is None:
                        fit_df = df
                    else:
                        fit_df = fit_df.append(df, ignore_index=True)

                feather.write_dataframe(fit_df, save_file)
                pyplt.close('all')

    def get_hmm_overview(self, overwrite=False):
        '''Aggregate paramters and info for all fitted HMMs into single dataframe
        '''
        if not os.path.isfile(self.files['hmm_overview']):
            overwrite = True

        if not overwrite:
            ho = feather.read_dataframe(self.files['hmm_overview'])
        else:
            ho = None
            print('aggregating hmm data...')
            for i, row in self.project._exp_info.iterrows():
                exp = load_experiment(row['exp_dir'])
                for rec_dir in exp.recording_dirs:
                    print('processing %s' % os.path.basename(rec_dir))
                    h5_file = hmma.get_hmm_h5(rec_dir)
                    if h5_file is None:
                        continue

                    handler = phmm.HmmHandler(rec_dir)
                    df = handler.get_data_overview().copy()
                    df['rec_dir'] = rec_dir
                    if ho is None:
                        ho = df
                    else:
                        ho = ho.append(df, ignore_index=True)

            ho = agg.apply_grouping_cols(ho, self.project)
            feather.write_dataframe(ho, self.files['hmm_overview'])

        if not 'single_state_trials' in ho.columns:
            print('counting single state trials...')
            ho['single_state_trials'] = ho.apply(hmma.check_single_state_trials, axis=1)
            feather.write_dataframe(ho, self.files['hmm_overview'])

        return ho

    def write_sorted_hmms(self, sorted_hmms):
        '''write sorted HMM dataframe to file
        '''
        sorted_file = self.files['sorted_hmms']
        feather.write_dataframe(sorted_hmms, sorted_file)

    def get_sorted_hmms(self):
        '''get sorted HMM dataframe
        '''
        sorted_file = self.files['sorted_hmms']
        if os.path.isfile(sorted_file):
            sorted_hmms = feather.read_dataframe(sorted_file)
            if 'early_state' not in sorted_hmms.columns:
                sorted_hmms['early_state'] = np.nan

            if 'late_state' not in sorted_hmms.columns:
                sorted_hmms['late_state'] = np.nan

            if 'sorting' not in sorted_hmms.columns:
                sorted_hmms['sorting'] = 'rejected'

            if 'sort_method' not in sorted_hmms.columns:
                sorted_hmms['sort_method'] = 'auto'

            if 'palatability' not in sorted_hmms.columns:
                pal_map = agg.PAL_MAP
                sorted_hmms['palatability'] = sorted_hmms.taste.map(pal_map)
                self.write_sorted_hmms(sorted_hmms)

            if 'exclude' not in sorted_hmms.columns:
                sorted_hmms = agg.apply_grouping_cols(sorted_hmms, self.project)
                self.write_sorted_hmms(sorted_hmms)

            return sorted_hmms
        else:
            return None

    def mark_hmm_state(self, exp_name, rec_group, hmm_id, early_state=None, late_state=None):
        '''Set the HMM state numbers that identifies the states to be used as the
        early state or late state in analysis
        '''
        if early_state is None and late_state is None:
            return

        hmm_df = self.get_sorted_hmms()
        i = ((hmm_df['exp_name'] == exp_name) &
             (hmm_df['rec_group'] == rec_group) &
             (hmm_df['hmm_id'] == hmm_id))

        print('-'*80)
        print('Setting state for HMMs: %s %s %s' % (exp_name, rec_group, hmm_id))
        if early_state is not None:
            print('    - setting early_state to state #%i' % early_state)
            hmm_df.loc[i, 'early_state'] = early_state

        if late_state is not None:
            print('    - setting late_state to state #%i' % late_state)
            hmm_df.loc[i, 'late_state'] = late_state

        print('Saving dataframe...')
        self.write_sorted_hmms(hmm_df)
        print('-'*80)

    def plot_sorted_hmms(self, overwrite=False, skip_tags=[],
                         sorting_tag=None, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
            if sorting_tag is not None:
                save_dir = os.path.join(save_dir, sorting_tag)

        plot_dirs = {'best': os.path.join(self.save_dir, 'Best_HMMs'),
                     'rejected': os.path.join(self.save_dir, 'Rejected_HMMs'),
                     'refit': os.path.join(self.save_dir, 'Refit_HMMs')}
        sorted_hmms = self.get_sorted_hmms()
        if sorting_tag is not None:
            sorted_hmms = sorted_hmms.query('sorting == @sorting_tag')
            tags = [sorting_tag]
            plot_dirs = {sorting_tag: save_dir}
        else:
            tags = sorted_hmm.sorting.unique()
            plot_dirs = {x: os.path.join(save_dir, x) for x in tags}

        for k,v in plot_dirs.items():
            if os.path.isdir(v) and overwrite:
                shutil.rmtree(v)

            if not os.path.isdir(v):
                os.mkdir(v)

        pbar = tqdm(total=sorted_hmms.shape[0])
        for i, row in sorted_hmms.iterrows():
            fn = '%s_%s_HMM%i-%s.svg' % (row['exp_name'], row['rec_group'],
                                         row['hmm_id'], row['taste'])

            #print(f'plotting {fn}...')
            if row['sorting'] not in plot_dirs.keys():
                pbar.update()
                continue

            if row['sorting'] in skip_tags:
                pbar.update()
                continue

            fn = os.path.join(plot_dirs[row['sorting']], fn)
            if os.path.isfile(fn) and not overwrite:
                pbar.update()
                continue

            es = row['early_state']
            ls = row['late_state']
            title_extra = f'Early: {es}, Late: {ls}'
            plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn,
                         title_extra=title_extra)
            pbar.update()

        pbar.close()

    def plot_all_hmms(self, overwrite=False):
        sorted_df = self.get_sorted_hmms()
        plot_dir = os.path.join(self.save_dir, 'Fitted_HMMs')
        grpby = sorted_df.groupby(['exp_name', 'rec_group', 'taste'])
        counter = 0
        total = sorted_df.shape[0]
        for name, group in grpby:
            anim_dir = os.path.join(plot_dir, '_'.join(name))
            if not os.path.isdir(anim_dir):
                os.makedirs(anim_dir)

            for i, row in group.iterrows():
                counter += 1
                print('Plotting %i of %i' % (counter, total))
                fn = '_'.join(name) + '_HMM#%s.svg' % row['hmm_id']
                fn = os.path.join(anim_dir, fn)
                if os.path.isfile(fn) and not overwrite:
                    continue

                plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn)

    def plot_BIC_comparison(self):
        ho = self.get_hmm_overview()
        BIC_file = os.path.join(self.save_dir, 'bic_comparison.svg')
        nplt.plot_BIC(ho, self.project, save_file=BIC_file)

    def analyze_hmms(self, overwrite=False, save_dir=None):
        all_units, _  = ProjectAnalysis(self.project).get_unit_info()
        coding_file = self.files['hmm_coding']
        timing_file = self.files['hmm_timings']
        confusion_file = self.files['hmm_confusion']
        mds_file = self.files['hmm_mds']
        if save_dir is not None:
            cfn = os.path.basename(coding_file)
            cnfn = os.path.basename(confusion_file)
            tfn = os.path.basename(timing_file)
            mfn = os.path.basename(mds_file)
            coding_file = os.path.join(save_dir, cfn)
            timing_file = os.path.join(save_dir, tfn)
            confusion_file = os.path.join(save_dir, cnfn)
            mds_file = os.path.join(save_dir, mfn)
        else:
            save_dir = self.save_dir


        best_hmms = self.get_best_hmms(save_dir=save_dir)
        timing_stats = timing_file.replace('feather', 'txt')
        if os.path.isfile(coding_file) and not overwrite:
            coding = feather.read_dataframe(coding_file)
        else:
            coding = hmma.analyze_hmm_state_coding(best_hmms, all_units)
            feather.write_dataframe(coding, coding_file)

        if os.path.isfile(timing_file) and not overwrite:
            timings = feather.read_dataframe(timing_file)
        else:
            timings = hmma.analyze_hmm_state_timing(best_hmms)
            feather.write_dataframe(timings, timing_file)

        if os.path.isfile(confusion_file) and not overwrite:
            confusion = feather.read_dataframe(confusion_file)
        else:
            confusion = hmma.saccharin_confusion_analysis(best_hmms, all_units,
                                                          area='GC',
                                                          single_unit=True,
                                                          repeats=50)
            feather.write_dataframe(confusion, confusion_file)

        if os.path.isfile(mds_file) and not overwrite:
            mds = feather.read_dataframe(mds_file)
        else:
            mds = hmma.hmm_mds_analysis(best_hmms, all_units, area='GC',
                                        single_unit=True)
            mds = agg.apply_grouping_cols(mds, self.project)
            feather.write_dataframe(mds, mds_file)

        if not os.path.isfile(timing_stats) or overwrite:
            descrip = hmma.describe_hmm_state_timings(timings)
            with open(timing_stats, 'w') as f:
                print(descrip, file=f)

        return coding, timings, confusion, mds

    def plot_hmm_timing(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        _, timing, _, _ = self.analyze_hmms(save_dir=save_dir)
        df = timing.copy()
        if 'exclude' not in df.columns:
            df = agg.apply_grouping_cols(df, self.project)

        plot_dir = os.path.join(save_dir, 'timing_analysis')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)

        corr_file = os.path.join(plot_dir, 'timing_correlations.svg')
        nplt.plot_timing_correlations(df, save_file=corr_file)
        corr_file = os.path.join(plot_dir, 'timing_correlations-exclude.svg')
        nplt.plot_timing_correlations(df[df['exclude'] == False],
                                         save_file=corr_file)

        dist_file = os.path.join(plot_dir, 'early_end_distributions.svg')
        nplt.plot_timing_distributions(df, state='early', value_col='t_end', save_file=dist_file)

        dist_file = os.path.join(plot_dir, 'late_start_distributions.svg')
        nplt.plot_timing_distributions(df, state='late', value_col='t_start', save_file=dist_file)

        dist_file = os.path.join(plot_dir, 'Saccharin_late_start_distributions.svg')
        nplt.plot_timing_distributions(df.query('taste == "Saccharin"'),
                                       state='late', value_col='t_start',
                                       save_file=dist_file)

        # Plot mean late state gamma probabilities
        print('Plotting Saccharin late state probabilities...')
        best_hmms = self.get_best_hmms(save_dir=save_dir)
        fn = os.path.join(plot_dir, 'Saccharin_Mean_Late_State_Probabilities.svg')
        nplt.plot_gamma_probs(best_hmms, state='late', taste='Saccharin', save_file=fn)

        for taste, grp in df.groupby('taste'):
            # plot distributions
            fn1 = os.path.join(plot_dir, f'{taste}_early_end_distributions.svg')
            nplt.plot_timing_distributions(grp, state='early',
                                           value_col='t_end', save_file=fn1)

            exc_df = grp[grp['exclude'] == False]
            # exp_group
            comp_file = os.path.join(plot_dir, f'{taste}_timing_comparison.svg')
            nplt.plot_timing_data(grp, save_file=comp_file, group_col='exp_group')

            # exp_group excluding GFP-NoCTA
            comp_file = os.path.join(plot_dir, f'{taste}_timing_comparison-exclude.svg')
            nplt.plot_timing_data(exc_df, save_file=comp_file, group_col='exp_group')

            # cta_group
            comp_file = os.path.join(plot_dir, f'{taste}_timing_comparison-CTA.svg')
            nplt.plot_timing_data(grp, save_file=comp_file, group_col='cta_group')

            # cta_group excluding GFP-NoCTA
            comp_file = os.path.join(plot_dir, f'{taste}_timing_comparison-CTA-exclude.svg')
            nplt.plot_timing_data(exc_df, save_file=comp_file, group_col='cta_group')

    def plot_hmm_confusion(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        _, _, confusion, _ = self.analyze_hmms(save_dir=save_dir)
        confusion['exclude'] = confusion.apply(lambda x: True if
                                               (x['exp_group'] == 'GFP' and
                                                x['cta_group'] == 'No CTA')
                                               else False, axis=1)
        plot_dir = os.path.join(save_dir, 'confusion_analysis')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)

        exc_df = confusion[confusion['exclude'] == False]

        # Make confusion plots using exp_group
        corr_file = os.path.join(plot_dir, 'confusion_correlations.svg')
        comp_file = os.path.join(plot_dir, 'confusion_comparison.svg')
        diff_file = os.path.join(plot_dir, 'confusion_differences.svg')
        nplt.plot_confusion_correlations(confusion, save_file=corr_file)
        nplt.plot_confusion_data(confusion, save_file=comp_file, group_col='exp_group')
        nplt.plot_confusion_differences(confusion, save_file=diff_file)

        # Make confusion plots using exp_group and excluding GFP-NoCTA
        corr_file = os.path.join(plot_dir, 'confusion_correlations-exclude.svg')
        comp_file = os.path.join(plot_dir, 'confusion_comparison-exclude.svg')
        nplt.plot_confusion_correlations(exc_df, save_file=corr_file)
        nplt.plot_confusion_data(exc_df, save_file=comp_file, group_col='exp_group')

        # Make confusion plots using cta_group
        comp_file = os.path.join(plot_dir, 'confusion_comparison-CTA.svg')
        nplt.plot_confusion_data(confusion, save_file=comp_file, group_col='cta_group')

        # Make confusion plots using cta_group and excluding GFP-NoCTA
        comp_file = os.path.join(plot_dir, 'confusion_comparison-CTA-exclude.svg')
        nplt.plot_confusion_data(exc_df, save_file=comp_file, group_col='cta_group')

    def plot_hmm_coding(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        coding, _, _, _ = self.analyze_hmms(save_dir=save_dir)
        coding['exclude'] = coding.apply(lambda x: True if
                                         (x['exp_group'] == 'GFP' and
                                          x['cta_group'] == 'No CTA')
                                         else False, axis=1)
        plot_dir = os.path.join(save_dir, 'coding_analysis')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)

        exc_df = coding[coding['exclude'] == False]

        # Make coding plots using exp_group
        corr_file = os.path.join(plot_dir, 'coding_correlations.svg')
        comp_file = os.path.join(plot_dir, 'coding_comparison.svg')
        nplt.plot_coding_correlations(coding, save_file=corr_file)
        nplt.plot_coding_data(coding, save_file=comp_file, group_col='exp_group')

        # Make coding plots using exp_group and excluding GFP-NoCTA
        corr_file = os.path.join(plot_dir, 'coding_correlations-exclude.svg')
        comp_file = os.path.join(plot_dir, 'coding_comparison-exclude.svg')
        nplt.plot_coding_correlations(exc_df, save_file=corr_file)
        nplt.plot_coding_data(exc_df, save_file=comp_file, group_col='exp_group')

        # Make coding plots using cta_group
        comp_file = os.path.join(plot_dir, 'coding_comparison-CTA.svg')
        nplt.plot_coding_data(coding, save_file=comp_file, group_col='cta_group')

        # Make coding plots using cta_group and excluding GFP-NoCTA
        comp_file = os.path.join(plot_dir, 'coding_comparison-CTA-exclude.svg')
        nplt.plot_coding_data(exc_df, save_file=comp_file, group_col='cta_group')

    def plot_hmm_seqs_and_probs(self, save_dir=None):
        '''Plot median and mean gamma probs and HMM sequences
        '''
        if save_dir is None:
            save_dir = self.save_dir

        best_hmms = self.get_best_hmms(save_dir=save_dir)
        coding, timings, confusion, mds = self.analyze_hmms(save_dir=save_dir)
        prob_fn = os.path.join(save_dir, 'HMM_Median_Gamma_Probs.png')
        mean_prob_fn = os.path.join(save_dir, 'HMM_Mean_Gamma_Probs.png')
        seq_fn = os.path.join(save_dir, 'All_Sequences.png')
        sacc_seq_fn = os.path.join(save_dir, 'Saccharin_Sequences.svg')
        seq_cta_fn = os.path.join(save_dir, 'All_Sequences-CTA.png')
        sacc_seq_cta_fn = os.path.join(save_dir, 'Saccharin_Sequences-CTA.png')

        plt.plot_median_gamma_probs(best_hmms.query('taste == "Saccharin"'), prob_fn)
        plt.plot_mean_gamma_probs(best_hmms.query('taste == "Saccharin"'), mean_prob_fn)

        df = best_hmms.query('taste == "Saccharin" and exclude == False')
        plt.plot_median_gamma_probs(df, prob_fn.replace('.png', '-exclude.svg'))

        plt.plot_hmm_sequence_heatmap(best_hmms, 'exp_group', seq_fn)
        plt.plot_hmm_sequence_heatmap(best_hmms.query('taste=="Saccharin" and exclude == False'), 'exp_group', sacc_seq_fn)
        plt.plot_hmm_sequence_heatmap(best_hmms, 'cta_group', seq_cta_fn)
        plt.plot_hmm_sequence_heatmap(best_hmms.query('taste=="Saccharin"'), 'cta_group', sacc_seq_cta_fn)

        # Write extra info to txt file
        unit_fn = os.path.join(save_dir, 'Unit_Info.txt')
        with open(unit_fn, 'w') as f:
            all_units, held_units = ProjectAnalysis(self.project).get_unit_info()
            all_units = all_units.query('single_unit == True and area == "GC"')
            held_units = held_units.query('area == "GC"')
            out = ['All Unit Counts']
            out.append('-'*80)
            a_count = all_units.groupby(['exp_group', 'exp_name',
                                         'rec_group'])['unit_num'].count()
            held_units = held_units.dropna(subset=['held_unit_name'])
            h_count = held_units.groupby(['exp_group', 'exp_name', 'held_over'])['held'].count()
            out.append(repr(a_count))
            out.append('='*80)
            out.append('')
            out.append('Held Unit Counts')
            out.append('-'*80)
            out.append(repr(h_count))
            print('\n'.join(out), file=f)

    def process_fitted_hmms(self, overwrite=False, hmm_plots=False):
        '''Go through all HMMs, sort them by similar parameter sets, and run
        all analyses and plots for each set of parameters containing at least
        30 HMMs (this is just some number I chose so that it didn't try to run
                 analyses on sets of HMMs that weren't fit for all recordings)
        '''
        with pm.push_alert(success_msg='HMM Processing Complete! :D'):
            ho = self.get_hmm_overview(overwrite=overwrite)
            ho = self.get_hmm_overview(overwrite=False)
            self.sort_hmms_by_params(overwrite=overwrite)
            self.mark_early_and_late_states()
            sorted_df = self.get_sorted_hmms()
            param_sets = sorted_df.sorting.unique()

            self.plot_BIC_comparison()

            for set_name in param_sets:
                if set_name == 'rejected':
                    continue

                save_dir = os.path.join(self.save_dir, set_name)
                if os.path.isdir(save_dir) and overwrite:
                    shutil.rmtree(save_dir)

                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)

                self.write_hmm_sorting_params(sorting=set_name, save_dir=save_dir)
                best_hmms = self.get_best_hmms(overwrite=True, save_dir=save_dir,
                                               sorting=set_name)
                notes = best_hmms.dropna().notes.unique()[0]
                if 'BIC test' in notes:
                    continue

                if sum(sorted_df['sorting'] == set_name) > 30:
                    try:
                        self.analyze_hmms(overwrite=True, save_dir=save_dir)
                        self.plot_hmm_seqs_and_probs(save_dir=save_dir)
                        self.plot_hmm_coding(save_dir=save_dir)
                        self.plot_hmm_confusion(save_dir=save_dir)
                        self.plot_hmm_timing(save_dir=save_dir)
                        self.plot_hmm_trial_breakdown(save_dir=save_dir)
                    except:
                        print(f'Failed to analyze {set_name}')

                if hmm_plots:
                    plot_dir = os.path.join(save_dir, 'HMM Plots')
                    if os.path.isdir(plot_dir) and overwrite:
                        shutil.rmtree(plot_dir)

                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)

                    self.plot_sorted_hmms(overwrite=overwrite, save_dir=plot_dir,
                                          sorting_tag=set_name)

        return sorted_df

    def plot_hmm_trial_breakdown(self, save_dir=None):
        '''Plot breakdown of all trials to which which trials were used in
        analyses and which were excldued due to poor fits
        '''
        if save_dir is None:
            save_dir = self.save_dir

        best_hmms = self.get_best_hmms(save_dir=save_dir)
        sorting = best_hmms.sorting.unique()
        if len(sorting) != 1:
            raise ValueError('Multiple or Zero sortings found in best_hmms')
        else:
            sorting = sorting[0]

        print('Gathering hmm trial data ... ')
        trial_df = get_hmm_trial_info(self, sorting=sorting)
        feather.write_dataframe(trial_df,
                                os.path.join(save_dir, 'hmm_trial_breakdown.feather'))

        fn = os.path.join(save_dir, 'hmm_trial_breakdown.svg')
        nplt.plot_hmm_trial_breakdown(trial_df, self.project, save_file=fn)

    @pm.push_alert(success_msg='HMM processing complete!')
    def process_sorted_hmms(self, sorting='params #5', save_dir=None,
                            overwrite=False, hmm_plots=False):
        '''run all analyses for a single parameter set. You should manually
        inspect the sorted_hmms dataframe to decide which parameter set to use,
        and change the default "sorting" accordingly
        '''
        if save_dir is None:
            save_dir = self.save_dir

        sorted_df = self.get_sorted_hmms()
        if not sorting in sorted_df.sorting.unique():
            raise ValueError(f'Sorting not found: {sorting}')

        self.write_hmm_sorting_params(sorting=sorting, save_dir=save_dir)
        self.plot_BIC_comparison()
        best_hmms = self.get_best_hmms(overwrite=overwrite, sorting=sorting, save_dir=save_dir)
        self.analyze_hmms(overwrite=overwrite, save_dir=save_dir)
        self.plot_hmm_seqs_and_probs(save_dir=save_dir)
        self.plot_hmm_coding(save_dir=save_dir)
        self.plot_hmm_confusion(save_dir=save_dir)
        self.plot_hmm_timing(save_dir=save_dir)
        self.plot_hmm_trial_breakdown(save_dir=save_dir)
        if hmm_plots:
            plot_dir = os.path.join(save_dir, 'HMM Plots')
            if os.path.isdir(plot_dir):
                shutil.rmtree(plot_dir)

            os.mkdir(plot_dir)

            self.plot_sorted_hmms(overwrite=True, save_dir=plot_dir,
                                  sorting_tag=sorting)

    def write_hmm_sorting_params(self, sorting='params #5', save_dir=None):
        '''write parameters to a pretty text file
        '''
        if save_dir is None:
            save_dir = self.save_dir

        param_cols = ['n_states', 'time_start', 'time_end', 'area',
                      'unit_type', 'dt', 'n_trials', 'notes']
        sorted_df = self.get_sorted_hmms()
        df = sorted_df.query('sorting == @sorting')
        name, group = next(iter(df.groupby(param_cols)))
        summary = ['-'*80,
                   'parameter set: %s' % sorting,
                   '# of states: %i' % name[0],
                   'time: %i to %i' % (name[1], name[2]),
                   'area: %s' % name[3],
                   'unit type: %s' % name[4],
                   'dt: %g' % name[5],
                   'n_trials: %i' % name[6],
                   'notes: %s' % name[7],
                   '# of HMMs: %i' % len(group),
                   '-'*80]
        with open(os.path.join(save_dir, 'HMM_parameters.txt'), 'w') as f:
            f.write('\n'.join(summary))

    def reset_hmm_sorting(self):
        sorted_df = self.get_sorted_hmms()
        sorted_df['sorting'] = 'rejected'
        sorted_df['sort_method'] = 'auto'
        self.write_sorted_hmms(sorted_df)

    def mark_early_and_late_states(self):
        '''For each HMM, mark the early and late states as the pair of states
        most likely to be present in the 200-700ms and 750-1500ms time bins
        respectively.
        '''
        ho = self.get_sorted_hmms()
        def apply_states(row):
            h5 = hmma.get_hmm_h5(row['rec_dir'])
            hmm, _, _ = phmm.load_hmm_from_hdf5(h5, row['hmm_id'])
            tmp = hmma.choose_early_late_states(hmm)
            return pd.Series({'early_state': tmp[0], 'late_state': tmp[1]})

        ho[['early_state', 'late_state']] = ho.apply(apply_states, axis=1)
        self.write_sorted_hmms(ho)

    def iterhmms(self, **kwargs):
        '''Iterate through all HMM models in sorted_hmms. kwargs can be passed
        as column=value to query and iterate through only hmms matching the
        specified criteria
        '''
        ho = self.get_sorted_hmms()
        qstr = ' and '.join([f'{k} == "{v}"' for k,v in kwargs.items()])
        if qstr != '':
            ho = ho.query(qstr)

        for i, row in ho.iterrows():
            h5 = hmma.get_hmm_h5(row['rec_dir'])
            hmm, _, params = phmm.load_hmm_from_hdf5(h5, row['hmm_id'])
            yield i, hmm, params, row

    def sort_hmms_by_params(self, overwrite=False):
        '''sorts HMMs as those with common parameters
        '''
        sorted_df = self.get_sorted_hmms()
        if sorted_df is None or overwrite:
            sorted_df = self.get_hmm_overview()

        sorted_df['sorting'] = 'rejected'
        sorted_df['sort_method'] = 'auto'
        met_params = sorted_df.query('(notes == "sequential - final") |'
                                     ' (notes == "sequential - low thresh") |'
                                     ' (notes == "sequential - BIC test")')
        param_cols = ['n_states', 'time_start', 'time_end', 'area',
                      'unit_type', 'dt', 'n_trials', 'notes']
        param_num = 0
        for name, group in met_params.groupby(param_cols):
            param_label = ('%i states, %i to %i ms, %s %s units, '
                           'dt = %g, n_trials = %i, notes: %s' % name)
            summary = ['-'*80,
                       'parameter #: %i' % param_num,
                       '# of states: %i' % name[0],
                       'time: %i to %i' % (name[1], name[2]),
                       'area: %s' % name[3],
                       'unit type: %s' % name[4],
                       'dt: %g' % name[5],
                       'n_trials: %i' % name[6],
                       'notes: %s' % name[7],
                       '# of HMMs: %i' % len(group),
                       '-'*80]
            print('\n'.join(summary))

            idx = np.array((group.index))
            sorted_df.loc[idx, 'sorting'] = 'params #%i' % param_num
            sorted_df.loc[idx, 'sorting_notes'] = param_label
            param_num += 1

        self.write_sorted_hmms(sorted_df)

    def sort_hmms_by_BIC(self):
        '''Looks through HMMs designated as for the BIC test and determines
        which of those have the best_BIC (lowest)
        '''
        sorted_df = self.get_sorted_hmms()
        if sorted_df is None:
            sorted_df = self.get_hmm_overview()
            sorted_df['sorting'] = 'rejected'
            sorted_df['sort_method'] = 'auto'

        df = sorted_df[sorted_df.notes.str.contains('BIC test')]
        minima = []
        for name, group in df.groupby(['exp_name', 'rec_group', 'taste']):
            minima.append(group.BIC.idxmin())

        sorted_df.loc[minima, 'sorting'] = 'best_BIC'
        self.write_sorted_hmms(sorted_df)


def get_saccharin_consumption(anim_dir):
    '''grabs animal metadata from anim-dir and returns
    mean_saccharin_consumption/mean_water_consumption
    drops CTA Training day because water consumption is messed the night after
    LiCl injection.
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
    proj._exp_info['cta_group'] = proj._exp_info.CTA_learned.map({True: 'CTA', False: 'No CTA'})
    proj.save()


def compare_taste_responses(rec1, unit1, rec2, unit2, params, method='anova'):
    '''Compares responses for a single unit from rec1 to rec2 for each taste in
    the recordings
    '''
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
    out_diff_low = []
    out_diff_high = []
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
            diff_low = np.zeros((nt,))
            diff_high = np.zeros((nt,))
            for i, y in enumerate(zip(fr1.T, fr2.T)):
                # u, p = mann_whitney_u(y[0], y[1])
                # Mann-Whitney U gave odd results, trying anova
                u, p = f_oneway(y[0], y[1])
                pvals[i] = p
                ustats[i] = u
                cm = CompareMeans.from_data(y[0], y[1])
                low, high = cm.tconfint_diff(alpha=0.05, usevar='unequal')
                diff_low[i] = low
                diff_high[i] = high
        elif method.lower() == 'bootstrap':
            raise ValueError('bootstrap comparison method is deprecated')

        # apply bonferroni correction
        pvals = pvals*nt

        # Compute mean difference using psth parameters
        pt1, psth1, base1 = agg.get_psth(rec1, unit1, ch1, params, remove_baseline=True)
        pt2, psth2, base2 = agg.get_psth(rec2, unit2, ch2, params, remove_baseline=True)
        diff, sem_diff = sas.get_mean_difference(psth1, psth2)

        #Store stuff
        out_pvals.append(pvals)
        out_ustats.append(ustats)
        out_diff.append(diff)
        out_diff_sem.append(sem_diff)
        out_labels.append(taste)
        out_diff_low.append(diff_low)
        out_diff_high.append(diff_high)

    return out_labels, out_pvals, out_ustats, out_diff, out_diff_sem, bin_time, pt1, out_diff_low, out_diff_high


def get_local_path(path):
    if ELF_DIR in path and not os.path.isdir(path) and os.path.isdir(MONO_DIR):
        out = path.replace(ELF_DIR, MONO_DIR)
    elif MONO_DIR in path and not os.path.isdir(path) and os.path.isdir(ELF_DIR):
        out = path.replace(MONO_DIR, ELF_DIR)
    else:
        out = path

    return out


def fit_anim_hmms(anim):
    '''Fit more HMMs for a single animal. Change params as needed.
    '''
    if anim == 'RN5':
        anim = 'RN5b'

    file_dirs, anim_dir = get_file_dirs(anim)
    base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                   'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
                   'time_end': 2000, 'n_states': 3, 'area': 'GC',
                   'hmm_class': 'PoissonHMM', 'threshold':1e-10,
                   'notes': 'sequential - low thresh'}

    for rec_dir in file_dirs:
        units = phmm.query_units(rec_dir, 'single', area='GC')
        if len(units) < 2:
            continue

        handler = phmm.HmmHandler(rec_dir)
        p = base_params.copy()
        handler.add_params(p)
        # p2 = p.copy()
        # p2['n_states'] = 2
        # p2['time_start'] = 0
        # handler.add_params(p2)
        # p3 = p.copy()
        # p3['n_states'] = 2
        # p3['time_start'] = 200
        # handler.add_params(p3)
        # p4 = p.copy()
        # p4['time_start'] = -1000
        # p4['n_states'] = 4
        # handler.add_params(p4)
        # if 'ctaTrain' in rec_dir:
        #     p5 = p.copy()
        #     _ = p5.pop('n_trials')
        #     handler.add_params(p5)

        dataname = os.path.basename(rec_dir)
        with pm.push_alert(success_msg=f'Done fitting for {dataname}'):
            print('Fitting %s' % os.path.basename(rec_dir))
            if LOCAL_MACHINE == 'StealthElf':
                handler.run(constraint_func=phmm.sequential_constraint)
            elif LOCAL_MACHINE == 'Mononoke':
                with parallel_backend('multiprocessing'):
                    handler.run(constraint_func=phmm.sequential_constraint)


@pm.push_alert(success_msg='Finished fitting HMMs')
def fit_hmms_for_BIC_test(proj, all_units, min_states=2, max_states=6):
    '''fit HMMs to compare BIC for multiple states
    '''
    base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                   'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
                   'time_end': 2000, 'area': 'GC',
                   'hmm_class': 'PoissonHMM', 'threshold':1e-6,
                   'notes': 'sequential - BIC test'}

    for i, row in proj._exp_info.iterrows():
        anim = row['exp_name']
        with pm.push_alert(success_msg=f'Done fiiting for {anim}'):
            if anim == 'RN5':
                anim = 'RN5b'

            file_dirs, anim_dir = get_file_dirs(anim)
            for rec_dir in file_dirs:
                units = all_units.query('area == "GC" and rec_dir == @rec_dir and single_unit == True')
                if len(units) < 3:
                    print('Only %i units for %s. Skipping...' % (len(units), os.path.basename(rec_dir)))
                    continue

                handler = phmm.HmmHandler(rec_dir)
                for N in range(min_states, max_states):
                    p = base_params.copy()
                    p['n_states'] = N
                    handler.add_params(p)

                print(f'Fitting {rec_dir}')
                if LOCAL_MACHINE == 'StealthElf':
                    handler.run(constraint_func=hmma.sequential_constrained)
                elif LOCAL_MACHINE == 'Mononoke':
                    with parallel_backend('multiprocessing'):
                        handler.run(constraint_func=hmma.sequential_constrained)


def get_hmm_trial_info(HA, sorting='params #5'):
    out = []
    for i, hmm, params, row in HA.iterhmms(sorting=sorting):
        seqs = hmm.stat_arrays['best_sequences']
        time = hmm.stat_arrays['time']

        # Only looking at the presence of states past t=0
        tidx = np.where(time > 0)[0]
        seqs = seqs[:, tidx]

        es = row['early_state']
        ls = row['late_state']
        if np.isnan(es) or np.isnan(ls):
            continue

        for j, trial in enumerate(seqs):
            tmp = row.copy()
            tmp['trial'] = j
            if es in trial and ls in trial:
                tmp['state_presence'] = 'both'
            elif es in trial:
                tmp['state_presence'] = 'early_only'
            elif ls in trial:
                tmp['state_presence'] = 'late_only'
            else:
                tmp['state_presence'] = 'neither'

            out.append(tmp)

    df = pd.DataFrame(out)
    df = agg.apply_grouping_cols(df, HA.project)
    return df


def apply_unit_firing_rates(df):
    def get_firing_rates(row):
        rec = row['rec_dir']
        unit = row['unit_num']
        t, sa = h5io.get_spike_data(rec, unit)
        dim = h5io.get_digital_mapping(rec, 'in')
        if isinstance(sa, dict):
            spikes = np.vstack(list(sa.values()))
        else:
            spikes = sa

        bidx = np.where((t < 0) & (t>-1000))[0]
        ridx = np.where((t > 0) & (t<1500))[0]
        baseline = np.mean(np.sum(spikes[:,bidx], axis=1))
        response = np.mean(np.sum(spikes[:,ridx], axis=1))
        return pd.Series({'baseline_firing': baseline, 'response_firing': response})

    df[['baseline_firing', 'response_firing']] = df.apply(get_firing_rates, axis=1)
    df['norm_response_firing'] = df.eval('response_firing - baseline_firing')
    return df

def consolidate_results():
    '''Copy to important files to dropbox
    '''
    save_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'Harmonia',
                            'Share', 'Stk11_Results')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)
    sds = {'data': os.path.join(save_dir, 'data'),
           'plots': os.path.join(save_dir, 'plots'),
           'stats': os.path.join(save_dir, 'stats'),
           'source_data': os.path.join(save_dir, 'source_data')}

    for k,v in sds.items():
        os.mkdir(v)

    proj_dir = DATA_DIR
    proj = load_project(proj_dir)
    PA = ProjectAnalysis(proj)
    HA = HmmAnalysis(proj)

    d1 = os.path.join(PA.save_dir,'single_unit_responses')
    d2 = os.path.join(PA.save_dir, 'mds_analysis')
    d3 = os.path.join(HA.save_dir, 'coding_analysis')
    d4 = os.path.join(HA.save_dir, 'confusion_analysis')
    d5 = os.path.join(HA.save_dir, 'timing_analysis')
    d6 = os.path.join(PA.save_dir, 'held_unit_response_changes')
    d7 = os.path.join(d6, 'Held_Unit_Plots')
    d8 = os.path.join(PA.save_dir, 'pca_analysis')
    files = [
             os.path.join(d1, 'unit_taste_responsivity.feather'),
             os.path.join(d1, 'unit_pal_discrim.feather'),
             os.path.join(d1, 'palatability_data.npz'),
             os.path.join(d8, 'pc_data.feather'),
             os.path.join(d8, 'pc_dQ_v_dN_data.feather'),
             # Taste responsive
             os.path.join(d1, 'unit_firing_rates.svg'),
             os.path.join(d1, 'unit_firing_rates.csv'),
             os.path.join(d1, 'unit_firing_rates.txt'),
             os.path.join(d1, 'taste_responsive.svg'),
             os.path.join(d1, 'taste_responsive_source_data.csv'),
             os.path.join(d1, 'taste_responsive-stats.svg'),
             os.path.join(d1, 'taste_responsive-stats.txt'),
             # Pal responsive
             os.path.join(d1, 'Mean_Spearman.svg'),
             os.path.join(d1, 'Mean_Spearman-comparison.svg'),
             os.path.join(d1, 'Mean_Spearman-comparison.txt'),
             os.path.join(d1, 'Mean_Spearman-simple_comparison.svg'),
             os.path.join(d1, 'Mean_Spearman-simple_comparison.txt'),
             os.path.join(d1, 'Mean_Spearman-simple_comparison_source_data.csv'),
             os.path.join(d1, 'Mean_Spearman-differences.svg'),
             os.path.join(d1, 'palatability_spearman_corr.svg'),
             os.path.join(d1, 'palatability_spearman_corr.txt'),
             # Taste discriminative
             os.path.join(d1, 'taste_discriminative.svg'),
             os.path.join(d1, 'taste_discriminative_comparison.txt'),
             os.path.join(d1, 'taste_discriminative_comparison.svg'),
             os.path.join(d1, 'Taste_responsive_over_time-Saccharin.svg'),
             # MDS analysis
             os.path.join(d2, 'Saccharin_MDS_distances.svg'),
             os.path.join(d2, 'Saccharin_MDS_distances.txt'),
             os.path.join(d2, 'Saccharin_MDS_distances-alternate.svg'),
             os.path.join(d2, 'Saccharin_MDS_distances-alternate.csv'),
             os.path.join(d2, 'Saccharin_MDS_distances-alternate.txt'),
             os.path.join(d2, 'FullDim_MDS_distances.svg'),
             os.path.join(d2, 'FullDim_MDS_distances.txt'),
             # HMM identity and palatability coding
             os.path.join(d3, 'coding_correlations-exclude.svg'),
             os.path.join(d3, 'coding_comparison-exclude.svg'),
             os.path.join(d3, 'coding_comparison-exclude.txt'),
             # HMM identity and palatability confusion
             os.path.join(d4, 'confusion_correlations-exclude.svg'),
             os.path.join(d4, 'confusion_comparison-exclude.svg'),
             os.path.join(d4, 'confusion_comparison-exclude.txt'),
             os.path.join(d4, 'confusion_differences.svg'),
             # HMM transition timing
             os.path.join(d5, 'timing_correlations-exclude.svg'),
             os.path.join(d5, 'early_end_distributions.svg'),
             os.path.join(d5, 'early_end_distributions.txt'),
             os.path.join(d5, 'late_start_distributions.svg'),
             os.path.join(d5, 'late_start_distributions.txt'),
             os.path.join(d5, 'Saccharin_early_end_distributions.svg'),
             os.path.join(d5, 'Saccharin_early_end_distributions.txt'),
             os.path.join(d5, 'Citric Acid_early_end_distributions.svg'),
             os.path.join(d5, 'Citric Acid_early_end_distributions.txt'),
             os.path.join(d5, 'Quinine_early_end_distributions.svg'),
             os.path.join(d5, 'Quinine_early_end_distributions.txt'),
             os.path.join(d5, 'NaCl_early_end_distributions.svg'),
             os.path.join(d5, 'NaCl_early_end_distributions.txt'),
             os.path.join(d5, 'Saccharin_late_start_distributions.svg'),
             os.path.join(d5, 'Saccharin_late_start_distributions.txt'),
             os.path.join(d5, 'Saccharin_timing_comparison-exclude.svg'),
             os.path.join(d5, 'Saccharin_timing_comparison-exclude.txt'),
             os.path.join(d5, 'Saccharin_timing_comparison-exclude_source_data.csv'),
             os.path.join(d5, 'Saccharin_Mean_Late_State_Probabilities.svg'),
             os.path.join(d5, 'Saccharin_Mean_Late_State_Probabilities_source_data.csv'),
             os.path.join(HA.save_dir, 'HMM_Median_Gamma_Probs.png'),
             os.path.join(HA.save_dir, 'HMM_Median_Gamma_Probs-exclude.svg'),
             os.path.join(HA.save_dir, 'HMM_parameters.txt'),
             os.path.join(HA.save_dir, 'hmm_trial_breakdown.svg'),
             os.path.join(HA.save_dir, 'hmm_trial_breakdown.txt'),
             os.path.join(proj_dir, 'Stk11_Project_project.p'),
             *HA.files.values(),
             *PA.files.values(),
             HA.files['hmm_timings'].replace('.feather', '.txt'),
             os.path.join(HA.save_dir, 'bic_comparison.svg'),
             os.path.join(HA.save_dir, 'bic_comparison.txt'),
             os.path.join(HA.save_dir, 'bic_comparison_source_data.csv'),
             os.path.join(HA.save_dir, 'Saccharin_Sequences.svg'),
             # Held Unit Response changes
             os.path.join(d6, 'held_unit_response_change_data.csv'),
             os.path.join(d6, 'response_change_data.npz'),
             os.path.join(d6, 'complete_response_change_data.csv'),
             os.path.join(d6, 'Saccharin_responses_changed-exclude.txt'),
             os.path.join(d6, 'Saccharin_responses_changed-Cre_v_BadGFP.txt'),
             os.path.join(d6, 'Saccharin_responses_changed-GFP_v_GFP.txt'),
             os.path.join(d7, 'Saccharin_responses_changed-exclude.svg'),
             os.path.join(d7, 'Saccharin_responses_changed-Cre_v_BadGFP.svg'),
             os.path.join(d7, 'Saccharin_responses_changed-GFP_v_GFP.svg'),
             os.path.join(HA.save_dir, 'hmm_trial_breakdown.feather'),
             os.path.join(PA.save_dir, 'Saccharin_consumption.svg'),
             os.path.join(PA.save_dir, 'Saccharin_consumption.txt'),
             os.path.join(HA.root_dir, 'All_spike_and_HMM_data.hdf5'),
            ]

    ext_map = {'feather': 'data', 'npy': 'data', 'npz': 'data', 'p': 'data',
               'svg': 'plots', 'png': 'plots', 'txt': 'stats', 'json': 'data',
               'csv': 'source_data', 'hdf5': 'source_data'}
    missing = []
    for f in files:
        fn, ext = os.path.splitext(f)
        dest = sds[ext_map[ext[1:]]]
        if os.path.isfile(f):
            fn2 = shutil.copy(f, dest)
        else:
            missing.append(f)

        print(f'copied {f} --> {fn2}\n')

    for f in missing:
        print(f'Missing file: {f}\n')

def package_hmm_data(HA, sorting='params #5'):
    '''Package HMM fits and raw Data for publication
    '''
    sorted_df = HA.get_sorted_hmms()
    df = sorted_df.query('sorting == @sorting').copy()
    df = df.query('n_cells >= 3').copy()
    df = agg.apply_grouping_cols(df, HA.project)
    cols = ['exp_name', 'rec_dir', 'rec_group', 'exp_group', 'time_group',
            'cta_group', 'taste', 'channel', 'palatability', 'hmm_id',
            'hmm_class', 'unit_type', 'area', 'n_states', 'n_cells',
            'n_repeats', 'n_iterations', 'n_trials', 'threshold', 'time_start',
            'time_end', 'dt', 'BIC', 'log_likelihood', 'max_log_prob',
            'cost', 'early_state', 'late_state', 'exclude', 'notes']
    df = df[cols].copy()

    proj_cols = ['exp_name', 'exp_group', 'saccharin_consumption',
                 'CTA_learned', 'cta_group', 'exclude']
    proj_info = HA.project._exp_info.dropna()[proj_cols]
    fn = os.path.join(HA.root_dir, 'All_spike_and_HMM_data.hdf5')
    phmm.package_project_data(df, fn, experiment_info=proj_info)

def package_response_change_data(PA):
    '''Package response change data for publication
    '''
    fn = os.path.join(PA.root_dir, 'single_unit_analysis',
                      'held_unit_response_changes', 'response_change_data.npz')
    save_fn = os.path.join(PA.root_dir, 'single_unit_analysis',
                           'held_unit_response_changes',
                           'complete_response_change_data.csv')
    data = np.load(fn)
    time = data['comp_time']
    pvals = data['pvals']
    F = data['test_stats']
    ci_low = data['ci_low']
    ci_high = data['ci_high']
    labels = data['labels']
    label_names = ['exp_group', 'exp_name', 'cta_group', 'held_unit_name', 'taste']
    mi = pd.MultiIndex.from_frame(pd.DataFrame(labels, columns=label_names))
    p_df = pd.DataFrame(pvals, index=mi, columns=time)
    f_df = pd.DataFrame(F, index=mi, columns=time)
    l_df = pd.DataFrame(ci_low, index=mi, columns=time)
    h_df = pd.DataFrame(ci_high, index=mi, columns=time)

    p_df = p_df.reset_index().melt(id_vars=label_names, value_vars=time,
                                   var_name='time', value_name='p')
    f_df = f_df.reset_index().melt(id_vars=label_names, value_vars=time,
                                   var_name='time', value_name='F')
    l_df = l_df.reset_index().melt(id_vars=label_names, value_vars=time,
                                   var_name='time', value_name='95CI_low')
    h_df = h_df.reset_index().melt(id_vars=label_names, value_vars=time,
                                   var_name='time', value_name='95CI_high')
    merge_cols = [*label_names, 'time']
    df = pd.merge(p_df, f_df, on=merge_cols)
    df = pd.merge(df, l_df, on=merge_cols)
    df = pd.merge(df, h_df, on=merge_cols)
    #df = agg.apply_grouping_cols(df, PA.project)

    # only saccharin
    #df = df.query('taste == "Saccharin"')

    df.to_csv(save_fn, index=False)

if __name__=="__main__":
    print('Hello World')
