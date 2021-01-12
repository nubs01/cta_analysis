import os
import glob
from scipy.ndimage.filters import gaussian_filter1d
from blechpy.analysis import poissonHMM as ph
from blechpy import load_dataset
from blechpy.dio import hmmIO
from blechpy.utils.particles import AnonHMMInfoParticle
from scipy.stats import mode
import numpy as np
import pandas as pd
from itertools import permutations, product
import statsmodels.api as sm
import analysis_stats as stats
from tqdm import tqdm
import pingouin
from joblib import Parallel, delayed, cpu_count
from collections import Counter
import aggregation as agg


def deduce_state_order(best_paths):
    '''Looks at best paths and determines the most common ordering of states by
    getting the mode at each position in the sequence order. Return dict that
    has states as keys and order as value
    '''
    n_states = len(np.unique(best_paths))
    trial_orders = [get_simple_order(trial) for trial in best_paths]
    i = 0
    out = {}
    tmp = trial_orders.copy()
    while len(tmp) > 0:
        # get first state in each sequence, unless state has already been assigned an order
        a = [x[0] for x in tmp if x[0] not in out.values()]
        if len(a) == 0:
            tmp = a
            continue

        common = mode(a).mode[0]  # get most common state in this position
        out[i] = common
        # Remove first position
        tmp = [x[1:] for x in tmp if len(x)>1]
        i += 1

    # check that every state is in out
    states = np.unique(best_paths)
    for x in states:
        if x not in out.values():
            out[i] = x
            i += 1

    # Actually I want it flipped so it maps state -> order
    out = {v:k for k,v in out.items()}
    return out


def get_simple_order(seq):
    '''returns order of items in a sequence without repeats, so [1,1,2,2,1,3]
    gives [1,2,3]
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_absolute_order(seq):
    '''returns orders of items in a sequence, so [1,1,2,2,1,3] gives [1,2,1,3]
    '''
    out = [seq[0]]
    for x in seq:
        if x != out[-1]:
            out.append(x)

    return out


def get_state_breakdown(rec_dir, hmm_id, h5_file=None):
    if rec_dir[-1] == os.sep:
        rec_dir = rec_dir[:-1]

    if h5_file is None:
        handler = ph.HmmHandler(rec_dir)
        hmm, hmm_time, params = handler.get_hmm(hmm_id)
    else:
        hmm, hmm_time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)

    hmm_id = params['hmm_id']
    n_states = params['n_states']
    dt = params['dt']
    time_start = params['time_start']
    time_end = params['time_end']
    max_iter = params['max_iter']
    threshold = params['threshold']
    unit_type = params['unit_type']
    channel = params['channel']
    n_trials = params['n_trials']
    spikes, dt, time = ph.get_hmm_spike_data(rec_dir, unit_type, channel,
                                             time_start=time_start,
                                             time_end=time_end, dt=dt,
                                             trials=n_trials)
    best_paths = hmm.stat_arrays['best_sequences'].astype('int')
    state_order = deduce_state_order(best_paths)
    taste = params['taste']
    n_cells = params['n_cells']
    n_iterations = hmm.iteration

    # Parse out rec_group and time_group
    tmp = os.path.basename(rec_dir).split('_')
    rec_group = tmp[-3]
    exp = tmp[0]
    if 'Train' in rec_group or 'pre' in rec_group:
        time_group = 'preCTA'
    elif 'Test' in rec_group or 'post' in rec_group:
        time_group = 'postCTA'

    dat = load_dataset(rec_dir)
    row = {'exp_name': exp, 'rec_dir': rec_dir, 'rec_name': dat.data_name,
           'hmm_id': hmm_id, 'taste': taste, 'channel': channel, 'trial': None,
           'n_states': n_states, 'hmm_state': None, 'ordered_state': None,
           'trial_ordered_state': None, 't_start': None, 't_end': None,
           'duration': None, 'cost': None, 'time_group': time_group,
           'rec_group': rec_group, 'whole_trial': False, 'n_cells': n_cells,
           'recurrence_in_trial': None}

    out = []
    for trial, (trial_path, trial_spikes) in enumerate(zip(best_paths, spikes)):
        trial_order = get_absolute_order(trial_path)
        tmp_path = trial_path.copy()
        tmp_time = hmm_time.copy()
        multiplicity = {x:0 for x in np.unique(trial_path)}
        for i, state in enumerate(trial_order):
            tmp_row = row.copy()
            tmp_row['trial'] = trial
            tmp_row['hmm_state'] = state
            tmp_row['trial_ordered_state'] = i
            tmp_row['ordered_state'] = state_order[state]
            tmp_row['t_start'] = tmp_time[0]
            multiplicity[state] += 1
            tmp_row['recurrence_in_trial'] = multiplicity[state]
            if i == len(trial_order) - 1:
                tmp_row['t_end'] = tmp_time[-1]
                if i == 0:
                    tmp_row['whole_trial'] = True
            else:
                end_idx = np.min(np.where(tmp_path != state))
                tmp_row['t_end'] = tmp_time[end_idx-1]
                # Trim tmp_path and tmp_time
                tmp_path = tmp_path[end_idx:]
                tmp_time = tmp_time[end_idx:]

            tmp_row['duration'] = tmp_row['t_end'] - tmp_row['t_start']

            # Compute cost
            idx = np.where((time >= tmp_row['t_start']) & (time <= tmp_row['t_end']))[0]
            fr = np.sum(trial_spikes[:, idx], axis=1) / (tmp_row['duration']/1000)
            est = hmm.emission[:, state]
            tmp_row['cost'] = np.sum((fr-est)**2)**0.5
            out.append(tmp_row)

    return pd.DataFrame(out)


def check_hmms(hmm_df):
    hmm_df['asymptotic_ll'] = hmm_df.apply(check_ll_asymptote)


def check_ll_asymptote(row):
    thresh = 1e-3
    rec_dir = row['rec_dir']
    hmm_id = row['hmm_id']
    n_iter = row['n_iterations']-1
    h5_file = get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    ll_hist = hmm.stat_arrays['fit_LL']
    filt_ll = gaussian_filter1d(ll_hist, 4)
    # TODO: Finish this
    diff_ll = np.diff(filt_ll)
    if len(ll_hist) == 0:
        return 'no_hist'

    # Linear fit, if overall trend is decreasing, it fails
    z = np.polyfit(range(len(ll_hist)), filt_ll, 1)
    if z[0] <= 0:
        return 'decreasing'

    # Check if it has plateaued
    if all(np.abs(diff_ll[n_iter-5:n_iter]) <= thresh):
        return 'plateau'

    # if its a maxima and hasn't plateaued it needs to continue fitting
    if np.max(filt_ll) == filt_ll[n_iter]:
        return 'increasing'

    return 'flux'


def _deprecated_get_best_hmms(df):
    '''Take the hmm_overview dataframe and returns (best_df, refit_df) of which
    are good HMMs and which need to be refit
    '''
    required_params = {'n_trials': 15, 'dt': 0.001, 'unit_type':'single'}
    qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in required_params.items()])
    df = df.query(qry)
    # One HMM for each animal rec and taste
    df = df.query('n_states == 3 or n_states == 2')
    refit = []
    best_hmms = []
    for name, group in df.groupby(['exp_name', 'rec_group', 'taste']):
        grp = group.query('ll_check == "plateau" and single_state_trials < 7')
        # If some HMMs plateaued then pick the one with the best log likelihood
        if len(grp) > 0:
            best = grp.loc[grp.max_log_prob.idxmax()]
            best_hmms.append(best)
            continue

        # otherwise pick an increasing to refit
        grp = group[group['ll_check'] == 'increasing']
        if len(grp) > 0:
            refit.append(grp.loc[grp.max_log_prob.idxmax()])
            continue

        # otherwise put all in refit
        for i, row in group.iterrows():
            refit.append(row)

    # Now get all the best and put them in a single hdf5 and make plots
    # Make best and refit dataframes
    best_df = pd.DataFrame(best_hmms).reset_index(drop=True)
    refit_df = pd.DataFrame(refit).reset_index(drop=True)
    return best_df, refit_df


def run_state_classification_analysis_old(best_df):
    '''Grab 4taste sessions and compute taste ID classification accuracy with early and late states
    Test all combinations of early and late states across HMMs in rec_group
    2-state HMMs are already split into early and late 
    For 3-state HMMs, ordered state 0 or 1 could be early and 1 or 2 could be Late
    test all posibilities and returns and plots pairings that give the best
    early state accuracy and the pairing with the best Late state acuracy
    Only uses CA, NaCl, and Quinine
    
    Parameters
    ----------
    best_df : dataframe of best HMMs, one per rec_dir per taste
    
    
    Returns
    -------
    
    '''
    tastes = ['Citric Acid', 'Quinine', 'NaCl']
    qry = ' and '.join(['taste == "%s"' % x for x in tastes])
    df = best_df.query('rec_group == "preCTA" or rec_group == "postCTA"')
    df = df.query(qry)

    for name, group in df.groupby('rec_dir'):
        rec_name = os.path.basename(name).split('_')
        rec_name = '_'.join(rec_name[:-2])
        data_dir = os.path.join(save_dir, rec_name)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        if len(group) < 3:
            print('%s is missing HMMs. Cannot do classification' % name)
            continue

        orderings = []
        for i, row in group.iterrows():
            tmp = [(x, y) for x,y in permutations(range(row['n_states']), 2)
                    if y > x]
            orderings.append(tmp)

        state_sets = list(product(*orderings))
        best_eid = None # (state_set_idx, accuracy)
        best_lid = None
        best_epal = None # (state_set_idx, RegressionResults)
        best_lpal = None
        all_data = {}
        for i, states in enumerate(state_sets):
            early_states = [x[0] for x in states]
            late_states = [x[1] for x in states]
            early_id = run_NB_ID_classifier(group, early_states)
            late_id = run_NB_ID_classifier(group, late_states)
            early_pal = run_pal_classifier(group, early_states)
            late_pal = run_pal_classifier(group, late_states)
            if best_eid is None or early_id.accuracy > best_eid[1].accuracy:
                best_eid = (i, early_id)

            if best_lid is None or late_id.accuracy > best_lid[1].accuracy:
                best_lid = (i, late_id)

            if best_epal is None or early_pal.accuracy > best_epal[1].accuracy:
                best_epal = (i, early_pal)

            if best_lpal is None or late_pal.accuracy > best_lpal[1].accuracy:
                best_lpal = (i, late_pal)

            all_data[i] = {'early_id': early_id, 'late_id': late_id,
                           'early_pal': early_pal, 'late_pal': late_pal}

        tags = np.array(['best_early_id', 'best_late_id', 'best_early_pal',
                         'best_late_pal'])
        best_sets = np.array([best_eid[0], best_lid[0], best_epal[0], best_lpal[0]])
        best_ID
        unique_sets = np.unique(best_sets)
        # Make plots of all the best
        for i in unique_sets:
            ordered_set = state_sets[i]
            hmm_set = []
            for (i, row), states in zip(group.iterrows(), ordered_set):
                rec_dir = row['rec_dir']
                hmm_id = row['hmm_id']
                h5_file = get_hmm_h5(rec_dir)
                hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
                state_map = deduce_state_order(hmm.stat_arrays['best_sequences'])
                state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
                hmm_set.append((state_map[x] for x in states))

            # now hmm_set is a list of tuples denoting the early and late state
            # of each hmm in terms of the hmm's state numbers
            idx = np.where(best_sets == i)[0]
            label = 'Set #%i: ' % i + ', '.join(tags[idx])
            fn = os.path.join(data_dir, '%s-Set%i-ID_Classifier.svg' % (rec_name, i))
            mplt.plot_classifier_results(group, hmm_set,
                                         early=all_data[i]['early_id'],
                                         late=all_data[i]['late_id'],
                                         label=label, save_file=fn)
            fn = fn.replace('ID_Classifier', 'Pal_Classifier')
            mplt.plot_pal_classifier_results(group, hmm_set,
                                             early=all_data[i]['early_pal'],
                                             late=all_data[i]['late_pal'],
                                             label=label, save_file=fn)
            fn = fn.replace('Pal_Regression.svg', '.txt')
            write_id_pal_to_text(fn, group, state_sets[i],
                                 early_id=all_data[i]['early_id'],
                                 late_id=all_data[i]['late_id'],
                                 early_pal=all_data[i]['early_pal'],
                                 late_pal=all_data[i]['late_pal'],
                                 label=label)


    pass


def ID_pal_hmm_analysis(best_df, plot_dir=None):
    '''needs best_hmm dataframe from make_best_hmm_list
    '''
    # important columns: rec_dir, taste, exp_name, rec_group, time_group,
    # cta_group, n_cells, best_hmm, early_state, late_state, exp_group, channel, palatability
    #TODO: Add palatability to best_df

    # out columns: rec_dir, exp_name, exp_group, rec_group,
    # time_group, cta_group, n_cells, n_tastes
    # early_ID_acc, late_ID_acc, early_pal_acc, late_pal_acc
    id_cols = ['rec_dir', 'exp_name', 'rec_group', 'time_group', 'cta_group', 'n_cells']
    confusion_tastes = ['NaCl', 'Quinine', 'Saccharin']
    out = []
    for name, group in best_df.groupby(id_cols):
        tmp = {k:v for k,v in zip(id_cols, name)}
        eid, lid, epal, lpal, econ, lcon = [None]*6
        tmp['n_tastes'] = len(group)
        if len(group) >= 2:
            eid = get_NB_classifier_accuracy(group, 'taste', 'early_state')
            lid  = get_NB_classifier_accuracy(group, 'taste', 'late_state')
            epal = get_LDA_classifier_accuracy(group, 'palatability',
                                               'early_state')
            lpal = get_LDA_classifier_accuracy(group, 'palatability',
                                               'late_state')

            if plot_dir is None:
                continue

            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            fn = os.path.join(plot_dir, '%s_%s-ID_Classifier.svg' % (exp_name, rec_group))
            mplt.plot_classifier_results(group, early=eid, late=lid,
                                         label=label, save_file=fn)
            fn = fn.replace('ID_Classifier', 'Pal_Classifier')
            mplt.plot_pal_classifier_results(group, early=eid, late=lid,
                                             label=label, save_file=fn)
            fn = fn.replace('Pal_Regression.svg', '.txt')
            write_id_pal_to_text(fn, group, early_id=eid, late_id=lid,
                                 early_pal=epal, late_pal=lpal, label=label)

        tmp['early_ID_acc'] = eid.accuracy
        tmp['late_ID_acc'] = lid.accuracy
        tmp['early_pal_acc'] = epal
        tmp['late_pal_acc'] = lpal
        out.append(tmp)

    return pd.DataFrame(out)


def run_NB_ID_classifier_deprecated(group, states):
    '''
    
    Parameters
    ----------
    group : pd.DataFrame, must have columns: rec_dir, hmm_id
    states: list of int, which state of each HMM to use for classification
    
    Returns
    -------
    float : accuracy [0,1]
    '''
    labels = []
    rates = []
    identifiers = []
    for (i, row), state in zip(group.iterrows(), states):
        rec_dir = row['rec_dir']
        hmm_id = row['hmm_id']
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.stat_arrays['best_sequences'])
        state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
        tmp_r, tmp_trials = get_state_firing_rates(rec_dir, hmm_id, state_map[state])
        # drop single state trials
        tmp_l = np.repeat(row['taste'], tmp_r.shape[0])
        tmp_id = [(rec_dir, hmm_id, row['taste'], x) for x in tmp_trials]
        labels.append(tmp_l)
        rates.append(tmp_r)
        identifiers.extend(tmp_n)

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    model = stats.NBClassifier(labels, rates, row_id=identifiers)
    results = model.fit()
    return results


def get_early_and_late_firing_rates(rec_dir, hmm_id, early_state, late_state, units=None):
    '''Early state gives the firing rate during the first occurence of that
    state in each trial. Late state gives the instance of that state that
    occurs after the early state in each trial. Trials that are all 1 state are
    dropped and trials where the late state does not occur at all after the
    early state
    '''
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = params['t_start']
    t_end = params['t_end']
    dt = params['dt']
    if units is None:
        units = params['unit_type']

    spike_array, dt, s_time = ph.get_hmm_spike_data(rec_dir, units,
                                                      channel,
                                                      time_start=t_start,
                                                      time_end=t_end, dt=dt,
                                                      trials=n_trials)
    # spike_array is trial x neuron x time
    n_trials, n_cells, n_steps = spike_array.shape
    early_rates = []
    late_rates = []
    dropped_trials = []
    labels = [] # trial, early_start, early_end, late_start, late_end
    for trial, (spikes, path) in enumerate(zip(spike_array, hmm.stat_arrays['best_sequences'])):
        if not early_state in path or not late_state in path:
            dropped_trials.append(trial)
            continue

        if len(np.unique(path)) == 1:
            dropped_trials.append(trial)
            continue

        # only grab first instance of early state
        eidx = np.where(path == early_state)[0]
        lidx = np.where(path == late_state)[0]
        ei1 = eidx[0]  # First instance of early state
        if not any(lidx > ei1):
            # if not late state after early state, drop trial
            dropped_trial.append(trial)
            continue

        idx2 = np.where(path != early_state)[0]
        if len(idx2) == 0 or not any(idx2 > ei1):
            ei2 = len(path)-1
        else:
            ei2 = np.min(idx2[idx2>idx1])

        li1 = np.min(lidx[lidx > ei2])
        idx2 = np.where(path != late_state)[0]
        if len(idx2) == 0 or not any(idx2 > li1):
            li2 = len(path) - 1
        else:
            li2 = np.min(idx2[idx2 > li1])

        et1 = time[ei1]
        et2 = time[ei2]
        lt1 = time[li1]
        lt2 = time[li2]
        labels.append((trial, et1, et2, lt1, lt2))
        e_si = np.where((s_time >= et1) & (s_time < et2))[0]
        l_si = np.where((s_time >= lt1) & (s_time < lt2))[0]
        e_tmp = np.sum(spikes[:, e_si], axis=-1) / (dt*len(e_si))
        l_tmp = np.sum(spikes[:, l_si], axis=-1) / (dt*len(l_si))
        early_rates.append(e_tmp)
        late_rates.append(l_tmp)

    out_labels = np.array(labels)
    early_out = np.array(early_rates)
    late_out = np.array(late_rates)
    return out_labels, early_out, late_out


def run_pal_classifier_deprecated(group, states):
    '''
    
    Parameters
    ----------
    group : pd.DataFrame, must have columns: rec_dir, hmm_id
    states: list of int, which state (orderd state number) of each HMM to use for classification
    
    Returns
    -------
    analysis_stats.ClassifierResult
    '''
    labels = []
    rates = []
    identifiers = []
    for (i, row), state in zip(group.iterrows(), states):
        rec_dir = row['rec_dir']
        hmm_id = row['hmm_id']
        dim = load_dataset(rec_dir).dig_in_mapping.set_index('name')
        pal = dim.loc[row['taste'], 'palatability_rank']
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.stat_arrays['best_sequences'])
        state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
        tmp_r, tmp_trial = get_state_firing_rates(rec_dir, hmm_id, state_map[state])
        # single states already dropped
        tmp_l = np.repeat(pal, tmp_r.shape[0])
        tmp_id = [(rec_dir, hmm_id, row['taste'], pal, x) for x in tmp_trials]
        labels.append(tmp_l)
        rates.append(tmp_r)
        identifiers.extend(tmp_n)

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    model = stats.LDAClassifier(labels, rates, row_id=identifiers)
    results = model.leave1out_fit()
    return results


def get_LDA_classifier_accuracy_deprecated(group, label_col, state_col):
    labels = []
    rates = []
    identifiers = []
    for i, row in group.iterrows():
        rec_dir = row['rec_dir']
        hmm_id = row['hmm_id']
        pal = row[label_col]
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        tmp_r, tmp_trial = get_state_firing_rates(rec_dir, hmm_id, row[state_col])
        # single states already dropped
        tmp_l = np.repeat(pal, tmp_r.shape[0])
        tmp_id = [(rec_dir, hmm_id, row[label_col], pal, x) for x in tmp_trials]
        labels.append(tmp_l)
        rates.append(tmp_r)
        identifiers.extend(tmp_n)

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    model = stats.LDAClassifier(labels, rates, row_id=identifiers)
    results = model.leave1out_fit()
    return results


def run_pal_regression(group, states):
    '''
    
    Parameters
    ----------
    group : pd.DataFrame, must have columns: rec_dir, hmm_id
    states: list of int, which state (orderd state number) of each HMM to use for classification
    
    Returns
    -------
    statsmodels.regression.linear_model.ResgressionResults
    '''
    labels = []
    rates = []
    for (i, row), state in zip(group.iterrows(), states):
        rec_dir = row['rec_dir']
        hmm_id = row['hmm_id']
        dim = load_dataset(rec_dir).dig_in_mapping.set_index('name')
        pal = dim.loc[row['taste'], 'palatability_rank']
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.stat_arrays['best_sequences'])
        state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
        tmp_r, tmp_ss = get_state_firing_rates(rec_dir, hmm_id, state_map[state])
        keep_idx = np.where(tmp_ss == False)[0]
        tmp_r = tmp_r[keep_idx]
        tmp_l = np.repeat(pal, tmp_r.shape[0])
        labels.append(tmp_l)
        rates.append(tmp_r)

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    rates = sm.add_constant(rates)
    model = sm.OLS(labels, rates)
    results = model.fit()
    return results


def write_id_pal_to_text(save_file, group, early_id=None,
                         late_id=None, early_pal=None, late_pal=None,
                         label=None):
    rec_dir = group.rec_dir.unique()[0]
    rec_name = os.path.basename(rec_dir).split('_')
    rec_name = '_'.join(rec_name[:-2])
    info_table = []
    for i, row in group.iterrows():
        rec_dir = row['rec_dir']
        hmm_id = row['hmm_id']
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.stat_arrays['best_sequences'])
        # hmm_id, early_state, late_state, early_hmm_state, late_hmm_state
        early_state = row['early_state']
        late_state = row['late_state']
        info_table.append((hmm_id, early_state, late_state))

    info_df = pd.DataFrame(info_table, columns=['hmm_id', 'early_state',
                                                'late_state'])
    out = []
    out.append(rec_name)
    out.append(rec_dir)
    out.append(label)
    out.append('='*80)
    out.append(info_df.to_string(index=False))
    out.append('-'*80)
    out.append('')
    out.append('Naive Bayes Taste Identity Classifier Accuracy')
    out.append('Computed with Leave 1 Out training and testing')
    out.append('Tastes classified: %s' % str(group.taste.to_list()))
    out.append('Early State ID Classification Accuracy: %0.2f' % early_id.accuracy)
    out.append('Late State ID Classification Accuracy: %0.2f' % late_id.accuracy)
    out.append('-'*80)
    out.append('')
    out.append('LDA Palatability Classification')
    out.append('Trained and Tested with all data points')
    out.append('*'*80)
    out.append('')
    out.append('Early State Pal Classification Accuracy: %0.2f' % early_pal.accuracy)
    out.append('Late State Pal Classification Accuracy: %0.2f' % late_pal.accuracy)
    # out.append('Early State Regression')
    # out.append(early_pal.summary('palatability'))
    # out.append('')
    # out.append('*'*80)
    # out.append('')
    # out.append('Late State Regression')
    # out.append(late_pal.summary('palatability'))
    with open(save_file, 'w') as f:
        f.write('\n'.join(out))


## Helper Functions ##

def get_state_firing_rates(rec_dir, hmm_id, state, units=None, min_dur=50,
                           remove_baseline=False, other_state=None):
    '''returns an Trials x Neurons array of firing rates giving the mean firing
    rate of each neuron in the first instance of state in each trial

    Parameters
    ----------
    rec_dir : str, recording directory
    hmm_id : int, id number of the hmm to get states from
    state: int, identity of the state
    units: list of str, optional
        which unit names to use. if not provided units are queried based on hmm
        params
    min_dur: int, optional
        minimum duration in ms of a state for it to be used. Default is 50ms.

    Returns
    -------
    np.ndarray : Trials x Neuron matrix of firing rates


    Raises
    ------

    '''
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = params['time_start']
    t_end = params['time_end']
    dt = params['dt']
    unit_type = params['unit_type']
    area = params['area']
    seqs = hmm.stat_arrays['best_sequences']
    if units is not None:
        unit_type = units

    spike_array, s_dt, s_time = ph.get_hmm_spike_data(rec_dir, unit_type,
                                                      channel,
                                                      time_start=t_start,
                                                      time_end=t_end, dt=dt,
                                                      trials=n_trials, area=area)
    if s_time[0] < 0:
        idx = np.where(s_time < 0)[0]
        prestim = spike_array[:,:, idx]
        prestim = np.sum(prestim, axis=-1) / (s_dt*len(idx))
        baseline = np.mean(prestim, axis=0)
    else:
        baseline = 0


    # both states must be present for at least 50ms each post-stimulus
    check_states = [state, other_state] if other_state else [state]
    valid_trials = agg.get_valid_trials(seqs, check_states, min_pts= 50/(dt*1000), time=time)

    # spike_array is trial x neuron x time
    n_trials, n_cells, n_steps = spike_array.shape
    rates = []
    trial_nums = []
    for trial, (spikes, path) in enumerate(zip(spike_array, seqs)):
        # Skip if state is not in trial
        if trial not in valid_trials:
            continue

        # Skip trial if there is only 1 state
        # if len(np.unique(path)) == 1:
        #     continue

        # Skip trials with only 1 state longer than the min_dur
        summary = summarize_sequence(path)
        # if len(np.where(summary[:,-1] >= min_dur)[0]) < 2:
        #     continue

        # only grab first instance of state
        idx1 = np.where(path == state)[0][0]
        idx2 = np.where(path != state)[0]
        if len(idx2) == 0 or not any(idx2 > idx1):
            idx2 = len(path)-1
        else:
            idx2 = idx2[idx2 > idx1][0]

        t1 = time[idx1]
        t2 = time[idx2]
        # Skip trial if this particular state is shorter than min_dur
        # This is because a state this short a) can't provide a good firing
        # rate and b) is probably forced by the constraints and not real
        if t2 - t1 < min_dur:
            continue

        si = np.where((s_time >= t1) & (s_time < t2))[0]
        tmp = np.sum(spikes[:, si], axis=-1) / (dt*len(si))
        if remove_baseline and s_time[0] < 0:
            tmp = tmp - baseline

        trial_nums.append(trial)
        rates.append(tmp)

    return np.array(rates), np.array(trial_nums)


def get_classifier_data(group, label_col, state_col, all_units,
                        remove_baseline=False, other_state_col=None):
    units = get_common_units(group, all_units)
    if units == {}:
        return None, None, None

    labels = []
    rates = []
    identifiers = []
    for i, row in group.iterrows():
        rec_dir = row['rec_dir']
        hmm_id = int(row['hmm_id'])
        state = row[state_col]
        if other_state_col:
            state2 = row[other_state_col]
        else:
            state2 = None

        un = units[rec_dir]
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        tmp_r, tmp_trials = get_state_firing_rates(rec_dir, hmm_id, state,
                                                   units=un,
                                                   remove_baseline=remove_baseline,
                                                   other_state=state2)
        tmp_l = np.repeat(row[label_col], tmp_r.shape[0])
        tmp_id = [(rec_dir, hmm_id, row[label_col], x) for x in tmp_trials]
        if len(tmp_r) == 0:
            # This should trigger when all trials are single state
            continue

        labels.append(tmp_l)
        rates.append(tmp_r)
        identifiers.extend(tmp_id)

    # if no valid trials were found for any taste
    if len(rates) == 0:
        return None, None, None

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    return labels, rates, identifiers


def get_common_units(group, all_units):
    held = np.array(all_units.held_unit_name.unique())
    rec_dirs = group.rec_dir.unique()
    if len(rec_dirs) == 1:
        rd = rec_dirs[0]
        out = {rd: all_units.query('rec_dir == @rd')['unit_name'].to_list()}
        return out

    for rd in group.rec_dir.unique():
        tmp = all_units.query('rec_dir == @rd').dropna(subset=['held_unit_name'])
        units = np.array(tmp['held_unit_name'])
        held = np.intersect1d(held, units)

    out = {}
    if len(held) == 0:
        return out

    for rd in group.rec_dir.unique():
        tmp = all_units[all_units['held_unit_name'].isin(held) &
                        (all_units['rec_dir'] == rd)]
        out[rd] = tmp['unit_name'].to_list()

    return out


def check_single_state_trials(row, min_dur=50):
    '''takes a row from hmm_overview and determines the number of single state decoded paths
    min_dur signifies the minimum time in ms that a state must be present to
    count
    '''
    rec_dir = row['rec_dir']
    hmm_id = row['hmm_id']
    h5_file = get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    dt = params['dt'] * 1000  # convert from sec to ms
    min_idx = int(min_dur/dt)
    paths = hmm.stat_arrays['best_sequences']
    single_state_trials = 0
    for path in paths:
        info = summarize_sequence(path)
        idx = np.where(info[:,-1] >= min_dur)[0]
        if len(idx) < 2:
            single_state_trials += 1

    return single_state_trials


def summarize_sequence(path):
    '''takes a 1-D sequences of categorical info and returns a matrix with
    columns: state, start_idx, end_idx, duration in samples
    '''
    tmp_path = path.copy()
    out = []
    a = np.where(np.diff(path) != 0)[0]
    starts = np.insert(a+1,0,0)
    ends = np.insert(a, len(a), len(path)-1)
    for st, en in zip(starts, ends):
        out.append((path[st], st, en, en-st+1))

    return np.array(out)


def is_sequence_valid(seq, time, early_state, late_state, min_dur=50):
    early_state = int(early_state)
    late_state = int(late_state)
    seq = seq.astype('int')
    dt = np.unique(np.diff(time))[0]
    min_idx = int(min_dur / dt)
    info = summarize_sequence(seq)
    good_seg = np.where(info[:,-1] >= min_idx)[0]
    good_info = info[good_seg, :]
    n_early = np.sum(info[:, 0] == early_state)
    n_late = np.sum(info[:, 0] == late_state)


    # if entrire sequence is one state, reject it
    if len(np.unique(seq)) == 1:
        return False

    # if late state is not in trial, reject it
    if late_state not in seq or late_state not in good_info[:,0]:
        return False

    # if the only instance of the early state is the first state and its
    # duration is less than min_dur, reject it
    if (seq[0] == 'early_state' and
        info[0,-1] < min_idx and
        n_early == 1):
        return False

    # if after the early state there is no late state >min_dur, reject it
    if n_early > 0:
        e1 = np.where(info[:,0] == early_state)[0][0]
        t1 = np.where(good_info[:,1] >= info[e1,2])[0]
        if len(t1) == 0:
            return False

        if late_state not in good_info[t1, 0]:
            return False

    # if after time 0, only the early state is present or neither state is present
    # then reject trial
    idx = np.where(time > 0)[0]
    if (all(seq[idx] == early_state) or
        (early_state not in seq[idx] and late_state not in seq[idx])):
        return False


def is_state_in_seq(seq, state, min_pts=1, time=None):
    '''returns True if given state is present and
    has more than min_pts consecutive points in that state. If time is given,
    this will only consider t>0
    '''
    if time is not None:
        tidx = np.where(time > 0)[0]
        seq = seq.copy()[tidx]

    if state not in seq:
        return False

    summary = summarize_sequence(seq)
    idx = np.where(summary[:,0] == state)[0]
    summary = summary[idx, :]
    if not any(summary[:,-1] >= min_pts):
        return False

    return True

## HMM Organization/Sorting ##

def make_necessary_hmm_list(all_units, min_cells=3, area='GC'):
    df = all_units.query('single_unit == True and area == @area')
    id_cols = ['rec_dir', 'exp_name', 'exp_group', 'rec_group', 'time_group', 'cta_group']
    out = []
    for name, group in df.groupby(id_cols):
        if len(group) < min_cells:
            continue

        dat = load_dataset(name[0])
        dim = dat.dig_in_mapping
        n_cells = len(group)
        for i, row in dim.iterrows():
            if (row['exclude']): # or row['name'].lower() == 'water'):
                continue

            tmp = {k: v for k,v in zip(id_cols, name)}
            tmp['taste'] = row['name']
            tmp['channel'] = row['channel']
            tmp['palatability'] = row['palatability_rank']
            tmp['n_cells'] = n_cells
            if tmp['palatability'] < 1:
                # Fix palatability so that Water and Saccharin are -1
                tmp['palatability'] = -1

            out.append(tmp)

    return pd.DataFrame(out)


def make_best_hmm_list(all_units, sorted_hmms, min_cells=3, area='GC', sorting='best'):
    df = make_necessary_hmm_list(all_units, min_cells=min_cells, area=area)
    sorted_hmms = sorted_hmms.query('sorting == @sorting')
    hmm_df = sorted_hmms.set_index(['exp_name', 'rec_group', 'taste'])
    def apply_info(row):
        exp, rec, tst = row[['exp_name', 'rec_group', 'taste']]
        hid, srt, es, ls = None, None, None, None
        if (exp, rec, tst) in hmm_df.index:
            hr = hmm_df.loc[(exp, rec, tst)]
            hid, srt, es, ls = hr[['hmm_id', 'sorting', 'early_state', 'late_state']]

        return pd.Series({'hmm_id': hid, 'sorting': srt, 'early_state': es, 'late_state': ls})

    df[['hmm_id', 'sorting', 'early_state', 'late_state']] = df.apply(apply_info, axis=1)
    return df


def sort_hmms(df, required_params=None):
    '''Adds four columns to hmm_overview dataframe, [sorting, sort_type, early_state, late_state].
    sorting can be best, reject, or refit. sort_type is "params" if params
    failed to meet requirements or "auto" if rest of this algo sorts them, if
    HMMs are sorted by user this will be "manual"
    '''
    out_df = df.copy()
    if required_params is not None:
        qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in required_params.items()])
        df = df.query(qry)

    # One HMM for each animal rec and taste
    df = df.query('n_states == 3 or n_states == 2')
    met_params = np.array((df.index))
    out_df['sorting'] = 'rejected'
    out_df['sort_method'] = 'params'
    out_df.loc[met_params, 'sort_method'] = 'auto'
    print('sorting hmms...')
    dfgrp = df.groupby(['exp_name', 'rec_group', 'taste'])
    for name, group in tqdm(dfgrp, total=len(dfgrp)):
        grp = group.query('single_state_trials < 7')
        if len(grp) > 0:
            best_idx = grp.log_likelihood.idxmax()
            out_df.loc[best_idx, 'sorting'] = 'best'
            continue

        grp2 = group.query('single_state_trials >= 7')
        if len(grp2) > 0:
            best_idx = grp2.log_likelihood.idxmax()
            out_df.loc[best_idx, 'sorting'] = 'refit'
        # grp = group.query('ll_check == "plateau" and single_state_trials < 7')
        # # If some HMMs plateaued then pick the one with the best log likelihood
        # if len(grp) > 0:
        #     best_idx = grp.max_log_prob.idxmax()
        #     out_df.loc[best_idx, 'sorting'] = 'best'
        #     continue

        # # otherwise pick an increasing to refit
        # grp = group[group['ll_check'] == 'increasing']
        # if len(grp) > 0:
        #     refit_idx = grp.max_log_prob.idxmax()
        #     out_df.loc[refit_idx, 'sorting'] = 'refit'
        #     continue

        # # otherwise put all in refit
        # out_df.loc[group.index, 'sorting'] = 'refit'

    out_df['early_state'] = np.nan
    out_df['late_state'] = np.nan
    #TODO: Way to choose early and late state
    return out_df


def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    if len(tmp) == 0:
        return None

    return tmp[0]


def sort_hmms_by_rec(df, required_params=None):
    '''Finds best HMMs but uses same parameter set for each recording. Attempts
    tom minimize single state trials and maximize log likelihood
    '''
    out_df = df.copy()
    if required_params is not None:
        qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in required_params.items()])
        df = df.query(qry)

    df = df[df.notes.str.contains('fix')]
    # One HMM for each animal rec and taste
    met_params = np.array((df.index))
    out_df['sorting'] = 'rejected'
    out_df['sort_method'] = 'params'
    out_df.loc[met_params, 'sort_method'] = 'auto'
    print('sorting hmms...')
    dfgrp = df.groupby(['exp_name', 'rec_group'])
    key_params = ['dt', 'n_states', 'time_start', 'time_end', 'notes', 'unit_type']
    for name, group in tqdm(dfgrp, total=len(dfgrp)):
        tastes = group.taste.unique()
        good_params = []
        ok_params = []
        for pset, subgroup in group.groupby(key_params):
            if not all([x in tastes for x in subgroup.taste]):
                continue

            tmp = {k:v for k,v in zip(key_params, pset)}
            tmp['LL'] = subgroup.log_likelihood.sum()
            if all([x < 7 for x in subgroup.single_state_trials]):
                good_params.append(tmp)
            else:
                ok_params.append(tmp)

        print('Found %i good params for %s' % (len(good_params), '_'.join(name)))
        if len(good_params) > 0:
            idx = np.argmax([x['LL'] for x in good_params])
            best_params = good_params[idx]
        elif len(ok_params) > 0:
            idx = np.argmax([x['LL'] for x in ok_params])
            best_params = ok_params[idx]
        else:
            continue

        _ = best_params.pop('LL')
        qstr = ' and '.join(['{} == "{}"'.format(k,v) for k,v in best_params.items()])
        tmp_df = group.query(qstr)
        best_idx = tmp_df.index
        out_df.loc[best_idx, 'sorting'] = 'best'

    out_df['early_state'] = np.nan
    out_df['late_state'] = np.nan
    #TODO: Way to choose early and late state
    return out_df


## HMM Constraints ##

def PI_A_constrained(PI, A, B):
    '''Constrains HMM to always start in state 0 then move into any other
    state. States are only allowed to transition into higher number states
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    A[-1, :-1] = 0.0
    A[-1, -1] = 1.0
    # This will make states consecutive
    if n_states > 2:
        for i in np.arange(1,n_states-1):
            A[i, :i] = 0.0
            A[i,:] = A[i,:]/np.sum(A[i,:])

    return PI, A, B


def A_contrained(PI, A, B):
    '''Constrains HMM to always start in state 0 then move into any other
    state. States are only allowed to transition into higher number states
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    A[-1, :-1] = 0.0
    A[-1, -1] = 1.0
    # This will make states consecutive
    if n_states > 2:
        for i in np.arange(1,n_states-1):
            A[i, :i] = 0.0
            A[i,:] = A[i,:]/np.sum(A[i,:])

    return PI, A, B


def sequential_constrained(PI, A, B):
    '''Forces all state to occur sequentially
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    for i in np.arange(n_states):
        if i > 0:
            A[i, :i] = 0.0

        if i < n_states-2:
            A[i, i+2:] = 0.0

        A[i, :] = A[i,:]/np.sum(A[i,:])

    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    return PI, A, B

## HMM decoding ##

def analyze_hmm_state_coding(best_hmms, all_units):
    '''create output dataframe with columns: exp_name, time_group, exp_group,
    cta_group, early_ID_acc, early_pal_acc, early_ID_confusion,
    early_pal_confusion, late_ID_acc, late_pal_acc, late_ID_confusion,
    late_pal_confusion
    '''
    # TODO: Remove unit hard-coding
    all_units = all_units.query('area == "GC" and single_unit == True')

    best_hmms = best_hmms.dropna(subset=['hmm_id', 'early_state', 'late_state'])
    out_keys = ['exp_name', 'exp_group', 'time_group', 'cta_group',
                'early_ID_acc', 'early_ID_confusion', 'early_pal_acc',
                'early_pal_confusion', 'late_ID_acc', 'late_ID_confusion',
                'late_pal_acc', 'late_pal_confusion', 'n_cells', 'n_held_cells']
    template = dict.fromkeys(out_keys)
    id_cols = ['exp_name', 'exp_group', 'time_group', 'cta_group']
    id_tastes = ['Citric Acid', 'NaCl', 'Quinine']
    pal_tastes = ['Citric Acid', 'NaCl', 'Quinine']
    confusion_tastes = ['NaCl', 'Quinine', 'Saccharin']
    confusion_pal = [3, 1, -1]
    out = []
    for name, group in best_hmms.groupby(id_cols):
        tmp = template.copy()
        for k,v in zip(id_cols, name):
            tmp[k] = v

        # ID Classification
        if group.taste.isin(id_tastes).sum() == len(id_tastes):
            eia = NB_classifier_accuracy(group[group.taste.isin(id_tastes)],
                                         'taste', 'early_state', all_units,
                                         other_state_col='late_state')
            lia = NB_classifier_accuracy(group[group.taste.isin(id_tastes)],
                                         'taste', 'late_state', all_units,
                                         other_state_col='early_state')
            if eia is not None:
                tmp['early_ID_acc'] = eia.accuracy

            if lia is not None:
                tmp['late_ID_acc'] = lia.accuracy

        # Palatability Classification
        if group.taste.isin(pal_tastes).sum() == len(pal_tastes):
            epa = LDA_classifier_accuracy(group[group.taste.isin(pal_tastes)],
                                          'palatability', 'early_state', all_units,
                                          other_state_col='late_state')
            lpa = LDA_classifier_accuracy(group[group.taste.isin(pal_tastes)],
                                          'palatability', 'late_state', all_units,
                                          other_state_col='early_state')
            if epa is not None:
                tmp['early_pal_acc'] = epa.accuracy

            if lpa is not None:
                tmp['late_pal_acc'] = lpa.accuracy

        # ID Confusion
        if group.taste.isin(confusion_tastes).sum() == len(confusion_tastes):
            j = group.taste.isin(confusion_tastes)
            eic = NB_classifier_confusion(group[j], 'taste', 'early_state', all_units,
                                          train_labels=confusion_tastes[:-1],
                                          test_labels=[confusion_tastes[-1]],
                                          other_state_col='late_state')
            lic = NB_classifier_confusion(group[j], 'taste', 'late_state', all_units,
                                          train_labels=confusion_tastes[:-1],
                                          test_labels=[confusion_tastes[-1]],
                                          other_state_col='early_state')
            # output is a tuple with (n_nacl, n_quinine), that is number of
            # saccharin trials classified as each
            # UPDATE: now just returns % nacl
            if eic is not None:
                tmp['early_ID_confusion'] = eic

            if lic is not None:
                tmp['late_ID_confusion'] = lic

        # Pal Confusion
        if group.taste.isin(confusion_tastes).sum() == len(confusion_tastes):
            j = group.taste.isin(confusion_tastes)
            epc = LDA_classifier_confusion(group[j], 'palatability', 'early_state', all_units,
                                           train_labels=confusion_pal[:-1],
                                           test_labels=[confusion_pal[-1]],
                                           other_state_col='late_state')
            lpc = LDA_classifier_confusion(group[j], 'palatability', 'late_state', all_units,
                                           train_labels=confusion_pal[:-1],
                                           test_labels=[confusion_pal[-1]],
                                           other_state_col='early_state')
            if epc is not None:
                tmp['early_pal_confusion'] = epc

            if lpc is not None:
                tmp['late_pal_confusion'] = lpc

        # n_cells in 4taste rec (single units in GC, which are used for classfier, not for HMM fitting
        # therefore if HMM fitted with pyrmadial this will use all single units
        rec_dirs = group.rec_dir.unique()
        rd = [x for x in rec_dirs if '4taste' in x]
        if len(rd) != 0:
            rd = rd[0]
            tmp_units = all_units.query('rec_dir == @rd')
            n_cells = len(tmp_units)
            tmp['n_cells'] = n_cells

        # n_held_cells, number of common units between 2 recordings used for confusion computation
        if len(rec_dirs) > 1:
            j = group.taste.isin(confusion_tastes)
            units = get_common_units(group[j], all_units)
            if units == {}:
                tmp['n_held_cells'] = 0
            else:
                tmp['n_held_cells'] = len(list(units.values())[0])

        out.append(tmp)

    return pd.DataFrame(out)


def NB_classifier_accuracy(group, label_col, state_col, all_units, other_state_col=None):
    '''uses rec_dir and hmm_id to creating firing rate array (trials x cells)
    label_col is the column used to label trials, state_col is used to identify
    which hmm state to use for classification

    Parameters
    ----------
    group : pd.DataFrame
        must have columns: rec_dir, hmm_id and columns that provide the labels
        for classification and the hmm state to be used
    label_col: str, column of dataframe that provides the classification labels
    state_col: str, column of dataframe that provides the hmm state to use
    all_units: pd.DataFrame
        dataframe of all units with columns rec_dir, area, single_unit, held_unit_name

    Returns
    -------
    float : accuracy [0,1]
    '''
    labels, rates, identifiers = get_classifier_data(group, label_col,
                                                     state_col, all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None

    n_cells = rates.shape[1]
    if n_cells < 2:
        return None

    # Check to make sure all expected tastes are present in the acquired data,
    # since trials are dropped if the first appearance of the state in the
    # trial is less than min_dur (50 ms)
    expected_tastes = group.taste.to_list()
    if not all([x in labels for x in expected_tastes]):
        return None

    model = stats.NBClassifier(labels, rates, row_id=identifiers)
    results = model.leave1out_fit()
    return results


def LDA_classifier_accuracy(group, label_col, state_col, all_units, other_state_col=None):
    '''uses rec_dir and hmm_id to creating firing rate array (trials x cells)
    label_col is the column used to label trials, state_col is used to identify
    which hmm state to use for classification

    Parameters
    ----------
    group : pd.DataFrame
        must have columns: rec_dir, hmm_id and columns that provide the labels
        for classification and the hmm state to be used
    label_col: str, column of dataframe that provides the classification labels
    state_col: str, column of dataframe that provides the hmm state to use
    all_units: pd.DataFrame
        dataframe of all units with columns rec_dir, area, single_unit, held_unit_name

    Returns
    -------
    float : accuracy [0,1]
    '''
    labels, rates, identifiers = get_classifier_data(group, label_col,
                                                     state_col, all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None

    n_cells = rates.shape[1]
    if n_cells < 2:
        return None

    model = stats.LDAClassifier(labels, rates, row_id=identifiers)
    results = model.leave1out_fit()
    return results


def NB_classifier_confusion(group, label_col, state_col, all_units,
                            train_labels=None, test_labels=None, other_state_col=None):
    if len(train_labels) != 2:
        raise ValueError('2 training labels are required for confusion calculations')

    if len(test_labels) != 1:
        raise ValueError('Too many test labels')

    labels, rates, identifiers = get_classifier_data(group,  label_col,
                                                     state_col, all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None

    n_cells = rates.shape[1]
    if n_cells < 2:
        return None

    train_idx = np.where([x in train_labels for x in labels])[0]
    test_idx = np.where([x in test_labels for x in labels])[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    model = stats.NBClassifier(labels[train_idx], rates[train_idx, :],
                               row_id=identifiers[train_idx, :])
    model.fit()
    predictions = model.predict(rates[test_idx, :])
    counts = [len(np.where(predictions == x)[0]) for x in train_labels]
    #return counts
    return 100 * counts[0] / np.sum(counts)  ## returns % nacl
    #return counts[1]/np.sum(counts)
    #q_count = len(np.where(predictions == 'Quinine')[0])
    #return 100 * q_count / len(predictions)


def LDA_classifier_confusion(group, label_col, state_col, all_units,
                            train_labels=None, test_labels=None, other_state_col=None):
    if len(train_labels) != 2:
        raise ValueError('2 training labels are required for confusion calculations')

    if len(test_labels) != 1:
        raise ValueError('Too many test labels')

    labels, rates, identifiers = get_classifier_data(group,  label_col,
                                                     state_col, all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None

    n_cells = rates.shape[1]
    if n_cells < 2:
        return None

    train_idx = np.where([x in train_labels for x in labels])[0]
    test_idx = np.where([x in test_labels for x in labels])[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    model = stats.LDAClassifier(labels[train_idx], rates[train_idx, :],
                                row_id=identifiers[train_idx, :])
    model.fit()
    predictions = model.predict(rates[test_idx, :])
    #counts = [np.sum(predictions == x) for x in train_labels]
    #q_count = len(np.where(predictions == 'Quinine')[0])
    #return 100 * q_count / len(predictions)
    #return counts[1]/np.sum(counts)
    counts = [len(np.where(predictions == x)[0]) for x in train_labels]
    #return counts
    return 100 * counts[0] / np.sum(counts)  ## returns % nacl


def choose_early_late_states(hmm, early_window=[200,700], late_window=[750, 1500]):
    '''picks state that most often appears in the late_window as the late
    state, and the state that most commonly appears in the early window
    (excluding the late state) as the early state.

    Parameters
    ----------
    hmm: phmm.PoissonHMM
    early_window: list of int, time window in ms [start, end]
    late_window: list of int, time window in ms [start, end]

    Returns
    -------
    int, int : early_state, late_state
        early_state = None if it cannout choose one
    '''
    seqs = hmm.stat_arrays['best_sequences']
    time = hmm.stat_arrays['time']
    trial_win = np.where(time > 0)[0]
    eidx = np.where((time >= early_window[0]) & (time < early_window[1]))[0]
    lidx = np.where((time >= late_window[0]) & (time < late_window[1]))[0]
    #drop single trial states
    good_trials = []
    for i, s in enumerate(seqs):
        if len(np.unique(s[trial_win])) != 1:
            good_trials.append(i)

    good_trials = np.array(good_trials)
    if len(good_trials) == 0:
        return None, None

    seqs = seqs[good_trials, :]
    n_trials = seqs.shape[0]

    lbins = list(np.arange(1, hmm.n_states))
    ebins = list(np.arange(0, hmm.n_states-1))
    lcount = []
    ecount = []
    for i,j in zip(ebins, lbins):
        ecount.append(np.sum(seqs[:, eidx] == i))
        lcount.append(np.sum(seqs[:, lidx] == j))

    #lcount, lbins = np.histogram(seqs[:, lidx], np.arange(hmm.n_states+1))
    #ecount, ebins = np.histogram(seqs[:, eidx], np.arange(hmm.n_states+1))
    pairs = [(x,y) for x,y in product(ebins,lbins) if x!=y]
    probs = []
    for x,y in pairs:
        i1 = ebins.index(x)
        i2 = lbins.index(y)
        p1 = ecount[i1] / n_trials
        p2 = lcount[i2] / n_trials
        probs.append(p1 * p2)

    best_idx = np.argmax(probs)
    early_state, late_state = pairs[best_idx]

    # early_state = ebins[np.argmax(ecount)]
    # if early_state in lbins:
    #     idx = list(lbins).index(early_state)
    #     lbins = list(lbins)
    #     lbins.pop(idx)
    #     lcount = list(lcount)
    #     lcount.pop(idx)

    # late_state = lbins[np.argmax(lcount)]

    # tmp = 0
    # early_state = None
    # for count, idx in zip(ecount, ebins):
    #     if count > tmp and idx != late_state:
    #         early_state = idx
    #         tmp = count

    return early_state, late_state


## State timing analysis ##
def analyze_hmm_state_timing(best_hmms, min_dur=50):
    '''create output array with columns: exp_name, exp_group, rec_dir,
    rec_group, time_group, cta_group, taste, hmm_id, trial, state_group,
    state_num, t_start, t_end, duration
    ignore trials with only 1 state > min_dur ms
    '''
    out_keys = ['exp_name', 'exp_group', 'cta_group', 'time_group', 'palatability',
                'rec_group', 'rec_dir', 'n_cells', 'taste', 'hmm_id', 'trial',
                'state_group', 'state_num', 't_start', 't_end', 'duration', 'pos_in_trial',
                'unit_type', 'area', 'dt', 'n_states', 'notes', 'valid']
    best_hmms = best_hmms.dropna(subset=['hmm_id', 'early_state', 'late_state']).copy()
    best_hmms.loc[:,'hmm_id'] = best_hmms['hmm_id'].astype('int')
    id_cols = ['exp_name', 'exp_group', 'time_group', 'cta_group', 'rec_group',
               'rec_dir', 'taste', 'hmm_id', 'palatability']
    param_cols = ['n_cells', 'n_states', 'dt', 'area', 'unit_type', 'notes']
    # State group is early or late
    out = []
    for i, row in best_hmms.iterrows():
        template = dict.fromkeys(out_keys)
        for k in id_cols:
            template[k] = row[k]

        h5_file = get_hmm_h5(row['rec_dir'])
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, row['hmm_id'])
        for k in param_cols:
            template[k] = params[k]

        dt = params['dt'] * 1000  # dt in ms
        min_pts = int(min_dur/dt)
        row_id = hmm.stat_arrays['row_id'] # hmm_id, channel, taste, trial, all string
        best_seqs = hmm.stat_arrays['best_sequences']
        for ids, path in zip(row_id, best_seqs):
            tmp = template.copy()
            tmp['trial'] = int(ids[-1])

            # Mark valid as whether both early and late state are present and longer than min_dur
            e_valid = is_state_in_seq(path, row['early_state'], min_pts=min_pts, time=time)
            l_valid = is_state_in_seq(path, row['late_state'], min_pts=min_pts, time=time)
            tmp['valid'] = (e_valid and l_valid)

            summary = summarize_sequence(path).astype('int')
            # summary = summary[np.where(summary[:,-1] >= min_dur/dt)[0], :]
            # Skip single state trials
            if summary.shape[0] < 2:
                #continue
                # Instead mark single state trials
                tmp['single_state'] = True
            else:
                tmp['single_state'] = False

            early_flag = False
            late_flag = False
            for j, s_row in enumerate(summary):
                s_tmp = tmp.copy()
                s_tmp['state_num'] = s_row[0]
                s_tmp['t_start'] = time[s_row[1]]
                s_tmp['t_end'] = time[s_row[2]]
                s_tmp['duration'] = s_row[3]/dt
                # only first appearance of state is marked as early or late
                if s_row[0] == row['early_state'] and not early_flag:
                    s_tmp['state_group'] = 'early'
                    early_flag = True
                elif s_row[0] == row['late_state'] and not late_flag:
                    s_tmp['state_group'] = 'late'
                    late_flag = True

                s_tmp['pos_in_trial'] = j
                out.append(s_tmp)

    return pd.DataFrame(out)


def describe_hmm_state_timings(timing):
    def header(txt):
        tmp = '-'*80 + '\n' + txt + '\n' + '-'*80
        return tmp

    timing = timing.query('valid == True').copy()
    # First look at Saccharin
    sdf = timing.query('taste == "Saccharin"')
    esdf = sdf.query('state_group == "early"')
    lsdf = sdf.query('state_group == "late"')
    out = []

    out.append(header('Saccharin Early State Analysis'))
    out.append('Animals in data & trial counts')
    out.append(esdf.groupby(['exp_name', 'exp_group', 'cta_group',
                             'time_group'])['trial'].count().to_string())
    out.append('')
    out.append('Single State Trials: %i' % esdf.single_state.sum())
    out.append('Single state trials removed for analysis')
    out.append('')
    out.append('Early State End Times')
    out.append('='*80)
    esdf = esdf.query('single_state == False') # drop single state trials
    out.append(esdf.groupby(['exp_group',
                             'time_group'])['t_end'].describe().to_string())
    out.append('')
    out.append(esdf.groupby(['cta_group',
                             'time_group'])['t_end'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = esdf.mixed_anova(dv='t_end', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    aov = esdf.mixed_anova(dv='t_end', between='cta_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    out.append('*'*80)

    out.append('Now after dropping GFP animals that did not learn')
    tmp = esdf.query('(exp_group == "Cre") or '
                     '(exp_group == "GFP" and cta_group == "CTA")')
    out.append(tmp.groupby(['exp_group',
                             'time_group'])['t_end'].describe().to_string())
    out.append('')
    out.append(tmp.groupby(['cta_group',
                             'time_group'])['t_end'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = tmp.mixed_anova(dv='t_end', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')

    out.append('Early State Durations')
    out.append('='*80)
    out.append(esdf.groupby(['exp_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append(esdf.groupby(['cta_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = esdf.mixed_anova(dv='duration', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    aov = esdf.mixed_anova(dv='duration', between='cta_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    out.append('*'*80)

    out.append('Now after dropping GFP animals that did not learn')
    out.append(tmp.groupby(['exp_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append(tmp.groupby(['cta_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = tmp.mixed_anova(dv='duration', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')

    out.append(header('Saccharin Late State Analysis'))
    out.append('Animals in data & trial counts')
    out.append(lsdf.groupby(['exp_name', 'exp_group', 'cta_group',
                             'time_group'])['trial'].count().to_string())
    out.append('')
    out.append('Single State Trials: %i' % lsdf.single_state.sum())
    out.append('Single state trials removed for analysis')
    out.append('')
    out.append('Late State Start Times')
    out.append('='*80)
    lsdf = lsdf.query('single_state == False') # drop single state trials
    out.append(lsdf.groupby(['exp_group',
                             'time_group'])['t_start'].describe().to_string())
    out.append('')
    out.append(lsdf.groupby(['cta_group',
                             'time_group'])['t_start'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = lsdf.mixed_anova(dv='t_start', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    aov = lsdf.mixed_anova(dv='t_start', between='cta_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    out.append('*'*80)

    out.append('Now after dropping GFP animals that did not learn')
    tmp = lsdf.query('(exp_group == "Cre") or '
                     '(exp_group == "GFP" and cta_group == "CTA")')
    out.append(tmp.groupby(['exp_group',
                             'time_group'])['t_start'].describe().to_string())
    out.append('')
    out.append(tmp.groupby(['cta_group',
                             'time_group'])['t_start'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = tmp.mixed_anova(dv='t_start', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')

    out.append('Late State Durations')
    out.append('='*80)
    out.append(lsdf.groupby(['exp_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append(lsdf.groupby(['cta_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = lsdf.mixed_anova(dv='duration', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    aov = lsdf.mixed_anova(dv='duration', between='cta_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    out.append('*'*80)

    out.append('Now after dropping GFP animals that did not learn')
    out.append(tmp.groupby(['exp_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append(tmp.groupby(['cta_group',
                             'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = tmp.mixed_anova(dv='duration', between='exp_group',
                           within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    return '\n'.join(out)


## For fitting Anon's HMMS ##
class CustomHandler(ph.HmmHandler):
    def __init__(self, h5_file):
        self.root_dir = os.path.dirname(h5_file)
        self.save_dir = self.root_dir
        self.h5_file = h5_file

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.plot_dir = os.path.join(self.save_dir, 'HMM_Plots')
        if not os.path.isdir(self.plot_dir):
            os.mkdir(self.plot_dir)

        if not os.path.isfile(h5_file):
            hmmIO.setup_hmm_hdf5(h5_file, infoParticle=AnonHMMInfoParticle)

        self.load_params()

    def load_params(self):
        self._data_params = []
        self._fit_params = []
        h5_file = self.h5_file
        if not os.path.isfile(h5_file):
            return

        overview = self.get_data_overview()
        if overview.empty:
            return

        keep_keys = list(ph.HMM_PARAMS.keys())
        keep_keys.append('rec_dir')
        for i in overview.hmm_id:
            _, _, _, _, p = hmmIO.read_hmm_from_hdf5(h5_file, i)
            for k in list(p.keys()):
                if k not in keep_keys:
                    _ = p.pop(k)

            self.add_params(p)

    def add_params(self, params):
        if isinstance(params, list):
            for p in params:
                self.add_params(p)

            return
        elif not isinstance(params, dict):
            raise ValueError('Input must be a dict or list of dicts')

        # Fill in blanks with defaults
        for k, v in ph.HMM_PARAMS.items():
            if k not in params.keys():
                params[k] = v
                print('Parameter %s not provided. Using default value: %s'
                      % (k, repr(v)))

        # Grab existing parameters
        data_params = self._data_params
        fit_params = self._fit_params

        # require additional rec_dir parameter
        if 'rec_dir' not in params.keys():
            raise ValueError('recording directory must be provided')

        # Get taste and trial info from dataset
        dat = load_dataset(params['rec_dir'])
        dat._change_root(params['rec_dir'])
        dim = dat.dig_in_mapping.query('exclude == False and spike_array == True')

        if params['taste'] is None:
            tastes = dim['name'].tolist()
            single_taste = True
        elif isinstance(params['taste'], list):
            tastes = [t for t in params['taste'] if any(dim['name'] == t)]
            single_taste = False
        elif params['taste'] == 'all':
            tastes = dim['name'].tolist()
            single_taste = False
        else:
            tastes = [params['taste']]
            single_taste = True

        dim = dim.set_index('name')
        if not hasattr(dat, 'dig_in_trials'):
            dat.create_trial_list()

        trials = dat.dig_in_trials
        hmm_ids = [x['hmm_id'] for x in data_params]
        if single_taste:
            for t in tastes:
                p = params.copy()

                p['taste'] = t
                # Skip if parameter is already in parameter set
                if any([hmmIO.compare_hmm_params(p, dp) for dp in data_params]):
                    print('Parameter set already in data_params, '
                          'to re-fit run with overwrite=True')
                    continue

                if t not in dim.index:
                    print('Taste %s not found in dig_in_mapping or marked to exclude. Skipping...' % t)
                    continue

                if p['hmm_id'] is None:
                    hid = get_new_id(hmm_ids)
                    p['hmm_id'] = hid
                    hmm_ids.append(hid)

                p['channel'] = dim.loc[t, 'channel']
                unit_names = query_units(dat, p['unit_type'], area=p['area'])
                p['n_cells'] = len(unit_names)
                if p['n_trials'] is None:
                    p['n_trials'] = len(trials.query('name == @t'))

                data_params.append(p)
                for i in range(p['n_repeats']):
                    fit_params.append(p.copy())

        else:
            if any([hmmIO.compare_hmm_params(params, dp) for dp in data_params]):
                print('Parameter set already in data_params, '
                      'to re-fit run with overwrite=True')
                return

            channels = [dim.loc[x,'channel'] for x in tastes]
            params['taste'] = tastes
            params['channel'] = channels

            # this is basically meaningless right now, since this if clause
            # should only be used with ConstrainedHMM which will fit 5
            # baseline states and 2 states per taste
            params['n_states'] = params['n_states']*len(tastes)

            if params['hmm_id'] is None:
                hid = ph.get_new_id(hmm_ids)
                params['hmm_id'] = hid
                hmm_ids.append(hid)

            unit_names = ph.query_units(dat, params['unit_type'],
                                        area=params['area'])
            params['n_cells'] = len(unit_names)
            if params['n_trials'] is None:
                params['n_trials'] = len(trials.query('name == @t'))

            data_params.append(params)
            for i in range(params['n_repeats']):
                fit_params.append(params.copy())

        self._data_params = data_params
        self._fit_params = fit_params

    def run(self, parallel=True, overwrite=False, constraint_func=None):
        h5_file = self.h5_file
        if overwrite:
            fit_params = self._fit_params
        else:
            fit_params = [x for x in self._fit_params if not x['fitted']]

        if len(fit_params) == 0:
            return

        print('Running fittings')
        if parallel:
            n_cpu = np.min((cpu_count()-1, len(fit_params)))
        else:
            n_cpu = 1

        results = Parallel(n_jobs=n_cpu, verbose=100)(delayed(ph.fit_hmm_mp)
                                                     (p['rec_dir'], p, h5_file,
                                                      constraint_func)
                                                     for p in fit_params)


        ph.memory.clear(warn=False)
        print('='*80)
        print('Fitting Complete')
        print('='*80)
        print('HMMs written to hdf5:')
        for hmm_id, written in results:
            print('%s : %s' % (hmm_id, written))

        #self.plot_saved_models()
        self.load_params()


## New confusion Analysis ##
def stratified_shuffle_split(labels, data, repeats, test_label):
    '''generator to split data by unique labels and sample with replacement
    from each to generate new groups of same size. 
    rows of data are observations.

    Returns
    -------
    train_data, train_labels, test_data, test_labels
    '''
    groups = np.unique(labels)
    counts = {}
    datasets = {}
    for grp in groups:
        idx = np.where(labels == grp)[0]
        counts[grp] = idx.shape[0]
        datasets[grp] = data[idx, :]

    rng = np.random.default_rng()
    for i in range(repeats):
        tmp_lbls = []
        tmp_data = []
        for grp in groups:
            N = counts[grp]
            idx = rng.choice(N, N, replace=True)
            tmp_data.append(datasets[grp][idx, :])
            tmp_lbls.extend(np.repeat(grp, N))

        tmp_lbls = np.array(tmp_lbls)
        tmp_data = np.vstack(tmp_data)
        train_idx = np.where(tmp_lbls != test_label)[0]
        test_idx = np.where(tmp_lbls == test_label)[0]
        train = tmp_data[train_idx, :]
        train_lbls = tmp_lbls[train_idx]
        test = tmp_data[test_idx, :]
        test_lbls = tmp_lbls[test_idx]
        yield train, train_lbls, test, test_lbls


def run_classifier(train, train_labels, test, test_labels, classifier=stats.NBClassifier):
    model = classifier(train_labels, train)
    model.fit()
    predictions = model.predict(test)
    accuracy = 100 * sum((x == y for x,y in zip(predictions, test_labels))) / len(test_labels)
    return accuracy, predictions


def saccharin_confusion_analysis(best_hmms, all_units, area='GC',
                                 single_unit=True, repeats=20):
    '''create output dataframe with columns: exp_name, time_group, exp_group,
    cta_group, state_group, ID_confusion, pal_confusion, n_cells, 
    '''
    all_units = all_units.query('(area == @area) and (single_unit == @single_unit)')
    best_hmms = best_hmms.dropna(subset=['hmm_id', 'early_state', 'late_state'])
    out_keys = ['exp_name', 'exp_group', 'time_group', 'cta_group',
                'state_group', 'ID_confusion', 'pal_confusion',
                'pal_counts_nacl', 'pal_counts_ca', 'pal_counts_quinine',
                'n_cells', 'nacl_trials', 'ca_trials', 'quinine_trials', 'sacc_trials']
    template = dict.fromkeys(out_keys)
    id_cols = ['exp_name', 'exp_group', 'time_group', 'cta_group']
    state_columns = ['early_state', 'late_state']
    other_state = {'early_state': 'late_state', 'late_state': 'early_state'}
    id_tastes = ['NaCl', 'Quinine', 'Saccharin']
    pal_tastes = ['NaCl', 'Citric Acid', 'Quinine', 'Saccharin']
    pal_map = {'NaCl': 3, 'Citric Acid': 2, 'Quinine': 1, 'Saccharin': -1}
    out = []
    for name, group in best_hmms.groupby(id_cols):
        for state_col in state_columns:
            tmp = template.copy()
            for k,v in zip(id_cols, name):
                tmp[k] = v

            tmp['state_group'] = state_col.replace('_state', '')

            if group.taste.isin(id_tastes).sum() != len(id_tastes):
                continue

            if group.taste.isin(pal_tastes).sum() == len(pal_tastes):
                run_pal = True
            else:
                run_pal = False

            group = group[group.taste.isin(pal_tastes)]

            labels, rates, identifiers = get_classifier_data(group, 'taste',
                                                             state_col,
                                                             all_units,
                                                             remove_baseline=True,
                                                             other_state_col=other_state[state_col])
            trials = Counter(labels)
            if rates is None or any([trials[x] < 2 for x in id_tastes]):
                # if no valid trials were found for NaCl, Quinine or Saccharin
                continue

            tmp['n_cells'] = rates.shape[1]
            tmp['nacl_trials'] = trials['NaCl']
            tmp['ca_trials'] = trials['Citric Acid']
            tmp['quinine_trials'] = trials['Quinine']
            tmp['sacc_trials'] = trials['Saccharin']
            for train, train_lbls, test, test_lbls \
                    in stratified_shuffle_split(labels, rates, repeats, 'Saccharin'):
                row = tmp.copy()
                tst_l = test_lbls.copy()
                tst_l[:] = 'NaCl'
                id_acc, _ = run_classifier(train, train_lbls, test, tst_l,
                                           classifier=stats.NBClassifier)
                row['ID_confusion'] = id_acc
                if run_pal:
                    train_l = np.fromiter(map(pal_map.get, train_lbls), int)
                    tst_l = np.fromiter(map(pal_map.get, tst_l), int)
                    pal_acc, pred = run_classifier(train, train_l, test, tst_l,
                                                   classifier=stats.LDAClassifier)
                    row['pal_confusion'] = pal_acc
                    counts = Counter(pred)
                    row['pal_counts_nacl'] = counts[pal_map['NaCl']]
                    row['pal_counts_ca'] = counts[pal_map['Citric Acid']]
                    row['pal_counts_quinine'] = counts[pal_map['Quinine']]

                out.append(row)

    return pd.DataFrame(out)


