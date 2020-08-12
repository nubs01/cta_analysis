import os
import glob
from scipy.ndimage.filters import gaussian_filter1d
from blechpy.analysis import poissonHMM as ph
from blechpy import load_dataset
from blechpy.dio import hmmIO
from scipy.stats import mode
import numpy as np
import pandas as pd
from itertools import permutations, product
import statsmodels.api as sm
import analysis_stats as stats


def analyze_hmm(hmm, rec_dir, params):
    pass

# determine overall state order
# get starts and end times for each state

def analyze_state_correlation(hmm, rec_dir, params):
    pass
# if 4taste, determine correlation to identity and to palatability of each state
# 


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
    best_paths = hmm.best_sequences.astype('int')
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
    ll_hist = hmm.ll_hist
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


def check_single_state_trials(row):
    rec_dir = row['rec_dir']
    hmm_id = row['hmm_id']
    h5_file = get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    paths = hmm.best_sequences
    single_state_trials = 0
    for path in paths:
        if len(np.unique(path)) == 1:
            single_state_trials += 1

    return single_state_trials


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


def sort_hmms(df, required_params=None):
    '''Adds two columns to hmm_overview dataframe, sorting and sort_type.
    sorting can be best, reject, or refit. sort_type is "params" if params
    failed to meet requirements or "auto" if rest of this algo sorts them, if
    HMMs are sorted by user this will be "manual"
    '''
    out_df = df.copy()
    if required_params is None:
        required_params = {'n_trials': 15, 'dt': 0.001, 'unit_type': 'single'}

    qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in required_params.items()])
    df = df.query(qry)
    # One HMM for each animal rec and taste
    df = df.query('n_states == 3 or n_states == 2')
    met_params = np.array((df.index))
    out_df['sorting'] = 'rejected'
    out_df['sort_method'] = 'params'
    out_df.loc[met_params, 'sort_method'] = 'auto'
    for name, group in df.groupby(['exp_name', 'rec_group', 'taste']):
        grp = group.query('ll_check == "plateau" and single_state_trials < 7')
        # If some HMMs plateaued then pick the one with the best log likelihood
        if len(grp) > 0:
            best_idx = grp.max_log_prob.idxmax()
            out_df.loc[best_idx, 'sorting'] = 'best'
            continue

        # otherwise pick an increasing to refit
        grp = group[group['ll_check'] == 'increasing']
        if len(grp) > 0:
            refit_idx = grp.max_log_prob.idxmax()
            out_df.loc[refit_idx, 'sorting'] = 'refit'
            continue

        # otherwise put all in refit
        out_df.loc[group.index, 'sorting'] = 'refit'

    return out_df



def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    return tmp[0]


def run_state_classification_analysis(best_df):
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
                hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
                state_map = deduce_state_order(hmm.best_sequences)
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


def get_early_and_late_firing_rates(rec_dir, hmm_id, early_state, late_state, units=None):
    '''Early state gives the firing rate during the first occurence of that
    state in each trial. Late state gives the instance of that state that
    occurs after the early state in each trial. Trials that are all 1 state are
    dropped and trials where the late state does not occur at all after the
    early state
    '''
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = params['t_start']
    t_end = params['t_end']
    dt = params['dt']
    if units is None:
        units = params['unit_type']

    spike_array, dt, s_time = phmm.get_hmm_spike_data(rec_dir, units,
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
    for trial, (spikes, path) in enumerate(zip(spike_array, hmm.best_sequences)):
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

def get_state_firing_rates(rec_dir, hmm_id, state):
    '''returns an Trials x Neurons array of firing rates giving the mean firing
    rate of each neuron in the first instance of state in each trial
    
    Parameters
    ----------
    rec_dir : str, recording directory
    hmm_id : int, id number of the hmm to get states from
    state: int, identity of the state
    
    Returns
    -------
    np.ndarray : Trials x Neuron matrix of firing rates
    
    
    Raises
    ------
    
    '''
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = params['t_start']
    t_end = params['t_end']
    dt = params['dt']
    unit_type = params['unit_type']
    spike_array, dt, s_time = phmm.get_hmm_spike_data(rec_dir, unit_type,
                                                      channel,
                                                      time_start=t_start,
                                                      time_end=t_end, dt=dt,
                                                      trials=n_trials)
    # spike_array is trial x neuron x time
    n_trials, n_cells, n_steps = spike_array.shape
    rates = []
    single_state_trials = []
    for trial, (spikes, path) in enumerate(zip(spike_array, hmm.best_sequences)):
        if not state in path:
            continue

        if len(np.unique(path)) == 1:
            single_state_trials.append(True)
        else:
            single_state_trials.append(False)

        # only grab first instance of state
        idx1 = np.where(path == state)[0][0]
        idx2 = np.where(path != state)[0]
        if len(idx2) == 0 or not any(idx2 > idx1):
            idx2 = len(path)-1
        else:
            idx2 = idx2[idx2 > idx1][0]

        t1 = time[idx1]
        t2 = time[idx2]
        si = np.where((s_time >= t1) & (s_time < t2))[0]
        tmp = np.sum(spikes[:, si], axis=-1) / (dt*len(si))
        rates.append(tmp)

    return np.array(rates), np.array(single_state_trials)


def run_NB_ID_classifier(group, states):
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
        hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.best_sequences)
        state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
        tmp_r, tmp_ss = get_state_firing_rates(rec_dir, hmm_id, state_map[state])
        # drop single state trials
        keep_idx = np.where(tmp_ss == False)[0]
        tmp_r = tmp_r[keep_idx]
        tmp_l = np.repeat(row['taste'], tmp_r.shape[0])
        tmp_id = [(rec_dir, hmm_id, row['taste'], x) for x in np.arange(0, tmp_r.shape[0])]
        labels.append(tmp_l)
        rates.append(tmp_r)
        identifiers.extend(tmp_n)

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    model = stats.NBClassifier(labels, rates, row_id=identifiers)
    results = model.fit()
    return results


def run_pal_classifier(group, states):
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
        hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.best_sequences)
        state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
        tmp_r, tmp_ss = get_state_firing_rates(rec_dir, hmm_id, state_map[state])
        # drop single state trials
        keep_idx = np.where(tmp_ss == False)[0]
        tmp_r = tmp_r[keep_idx]
        tmp_l = np.repeat(pal, tmp_r.shape[0])
        tmp_id = [(rec_dir, hmm_id, row['taste'], pal, x) for x in np.arange(0, tmp_r.shape[0])]
        labels.append(tmp_l)
        rates.append(tmp_r)
        identifiers.extend(tmp_n)

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    model = stats.LDAClassifier(labels, rates, row_id=identifiers)
    results = model.fit()
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
        hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.best_sequences)
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


def write_id_pal_to_text(save_file, group, state_set, early_id=None,
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
        hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.best_sequences)
        state_map = {v:k for k,v in state_map.items()}  # Now state_map maps from order -> state
        # hmm_id, early_state, late_state, early_hmm_state, late_hmm_state
        early_state = state_set[i][0]
        late_state = state_set[i][1]
        info_table.append((hmm_id, early_state, late_state,
                           state_map[early_state], state_map[late_state]))

    info_df = pd.DataFrame(info_table, columns=['hmm_id', 'early_state',
                                                'late_state', 'early_hmm_state',
                                                'late_hmm_state'])
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







