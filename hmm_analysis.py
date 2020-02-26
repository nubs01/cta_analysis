import tables
import blechpy
import os
import numpy as np
import pandas as pd
from blechpy.analysis import poissonHMM as phmm
from blechpy.utils import write_tools as wt
import pickle
import pylab as plt
import multiprocessing as mp


HMM_PARAMS = {'time_window': [-1000, 2000], 'convergence_thresh': 1e-4,
              'max_iter': 2000, 'n_repeats': 3, 'unit_type': 'single',
              'bin_size': 0.001}


class InfoParticle(tables.IsDescription):
    taste = tables.StringCol(20)
    channel = tables.Int16Col()
    n_cells = tables.Int32Col()
    n_trials = tables.Int32Col()
    n_states = tables.Int32Col()
    dt = tables.Float64Col()
    BIC = tables.Float64Col()
    converged = tables.BoolCol()
    threshold = tables.Float64Col()
    cost = tables.Float64Col()


class HmmAnalysis(object):
    def __init__(self, dat, n_states, save_dir=None, params=None):
        if isinstance(dat, str):
            dat = blechpy.load_dataset(dat)
            if dat is None:
                raise FileNotFoundError('No dataset.p file found given directory')

        if save_dir is None:
            save_dir = os.path.join(dat.root_dir,
                                    '%s_analysis' % dat.data_name)

        self._dataset = dat
        self.root_dir = dat.root_dir
        self.save_dir = save_dir
        self.n_states = n_states

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        plot_dir = os.path.join(save_dir, '%i_states' % n_states, 'plots')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        self._plot_dir = plot_dir

        self._files = {'hmm_data': os.path.join(save_dir, 'hmm_data.hdf5'),
                       'params' : os.path.join(save_dir, 'hmm_params.json')}
        file_check = self._file_check()
        self.update_params(params)

        dig_in_map = dat.dig_in_mapping.query('spike_array == True and exclude == False')
        self._dig_ins = dig_in_map.set_index('name')
        self._fitted_models = {}
        self._setup_hdf5()

    def _file_check(self):
        out = {}
        for k, v in self._files.items():
            if os.path.isfile(v):
                out[k] = True
            else:
                out[k] = False

        return out

    def update_params(self, params=None):
        file_check = self._file_check()
        if file_check['params']:
            old_params = wt.read_dict_from_json(self._files['params'])
        else:
            old_params = HMM_PARAMS.copy()

        if params:
            for k, v in params.items():
                if isinstance(v, dict):
                    old_params.update(v)
                else:
                    old_params[k] = v

        self.params = old_params
        wt.write_dict_to_json(self.params, self._files['params'])

    def _setup_hdf5(self):
        dig_ins = self._dig_ins
        h5_file = self._files['hmm_data']
        n_states = self.n_states
        state_str = '%i_states' % n_states

        with tables.open_file(h5_file, 'a') as hf5:
            # Taste -> PI, A, B, BIC, state_sequences, nStates, nCells, dt
            if not 'data_overview' in hf5.root:
                # Contains taste, channel, n_cells, n_trials, n_states, dt, BIC
                table = hf5.create_table('/', 'data_overview', InfoParticle,
                                         'Basic info for each digital_input')
                table.flush()


            if not state_str in hf5.root:
                hf5.create_group('/', state_str, '%i State HMM Solutions' % n_states)

            for taste in dig_ins.index:
                t = taste.replace(' ', '_')
                if t not in hf5.root[state_str]:
                    # Contains initial_distribution, transition, emission, state_sequences
                    # time vector
                    hf5.create_group('/%s' % state_str, t, '%s HMM Data' % taste)

            hf5.flush()


    def get_unit_names(self):
        dat = self._dataset
        unit_type = self.params['unit_type']
        unit_table = dat.get_unit_table()

        if unit_type == 'single':
            ut = unit_table.query('single_unit == True')
            units = ut['unit_name'].tolist()
        elif unit_type == 'all':
            units = unit_table['unit_name'].tolist()
        elif unit_type == 'pyramidal':
            ut = unit_table.query('single_unit == True and regular_spiking == True')
            units = ut['unit_name'].tolist()
        elif unit_type == 'interneuron':
            ut = unit_table.query('single_unit == True and fast_spiking == True')
            units = ut['unit_name'].tolist()
        else:
            raise ValueError('unit_type must be single, all, pyramidal or interneuron')

        return units

    def get_spike_data(self, taste):
        bin_size = self.params['bin_size']
        dig_ins = self._dig_ins
        channel = dig_ins.loc[taste]['channel']
        rec_dir = self.root_dir
        window = self.params['time_window']
        units = self.get_unit_names()
        time, spike_array = blechpy.dio.h5io.get_spike_data(rec_dir, units, channel)
        idx = np.where((time >= window[0]) & (time < window[1]))[0]
        time = time[idx]
        spike_array = spike_array[:, :, idx].astype('int32')
        dt = np.unique(np.diff(time))[0] / 1000
        if bin_size > dt:
            spike_array, dt, time = rebin_spike_array(spike_array, dt, time, bin_size)

        return spike_array, dt, time

    def run(self, overwrite=False, parallel=True):
        params = self.params
        rec_dir = self.root_dir
        n_states = self.n_states
        window = params['time_window']
        unit_type = params['unit_type']
        n_repeats = params['n_repeats']
        max_iter = params['max_iter']
        thresh = params['convergence_thresh']
        dig_ins = self._dig_ins
        h5_file = self._files['hmm_data']


        for taste, row in dig_ins.iterrows():
            # Check if trained HMM already exists
            # Add overwrite flag
            if HMM_exists(h5_file, taste, n_states) and not overwrite:
                self._fitted_models[taste] = self.load_hmm_from_h5(taste)
                continue

            channel = row['channel']
            spike_array, dt, time = self.get_spike_data(taste)
            n_trials, n_cells, n_steps = spike_array.shape

            print('='*80 + '\nTraining HMMs for %s\n' % taste + '='*80)
            convergence = []
            BIC = []
            hmms = []

            def update(ans):
                hmms.append(ans)

            if parallel:
                pool = mp.get_context('spawn').Pool(mp.cpu_count()-1)
                for i in range(n_repeats):
                    pool.apply_async(phmm.fit_hmm_mp,
                                     (n_states, spike_array, dt, max_iter, thresh),
                                     callback=update)

                pool.close()
                pool.join()
            else:
                tmp_hmm = phmm.fit_hmm_mp(n_states, spike_array, dt,
                                          max_iter, thresh)
                update(tmp_hmm)

            print('Fitting Complete!')
            print('Picking Best HMM and Saving...')
            for hmm in hmms:
                convergence.append(hmm.isConverged(thresh))
                tmp_bic, _ = hmm.get_BIC()
                BIC.append(tmp_bic)

            if all(convergence):
                print('All HMMs converged...picking best...')
                idx = np.argmin(BIC)
                bestHMM = hmms[idx]
                bestBIC = BIC[idx]
            elif any(convergence):
                print('Some HMMs converged...picking best...')
                bestHMM = None
                while bestHMM is None:
                    idx = np.argmin(BIC)
                    if hmms[idx].isConverged(thresh):
                        bestHMM = hmms[idx]
                        bestBIC = BIC[idx]
                    else:
                        hmms.pop(idx)
                        BIC.pop(idx)

            else:
                print('No HMMs converged...picking best...')
                idx = np.argmin(BIC)
                minHMM = None
                minBIC = None
                for h in hmms:
                    mats, iteration, BICs = h.find_best_in_history()
                    if minBIC is None:
                        minBIC = np.min(BICs)
                        h.set_matrices(mats)
                        minHMM = h
                    elif np.min(BICs) < minBIC:
                        minBIC = np.min(BICs)
                        h.set_matrices(mats)
                        minHMM = h

                bestHMM = minHMM
                bestBIC = minBIC


            # Save HMM to h5
            self._fitted_models[taste] = bestHMM
            self.add_hmm_to_h5(taste, bestHMM, overwrite=overwrite)


    def add_hmm_to_h5(self, taste, hmm, overwrite=False):
        print('Writing best HMM for %s to hdf5...' % taste)
        plot_dir = self._plot_dir
        thresh = self.params['convergence_thresh']
        converged = hmm.isConverged(thresh)
        n_states = hmm.n_states
        n_trials, n_cells, n_steps = hmm.data.shape
        channel = self._dig_ins.loc[taste]['channel']
        state_str = '%i_states' % n_states

        dt = hmm.dt
        PI = hmm.initial_distribution
        A = hmm.transition
        B = hmm.emission
        BIC, bestPaths = hmm.get_BIC()
        cost = hmm.cost

        window = self.params['time_window']
        time = np.arange(window[0], window[1], dt*1000)

        h5_file = self._files['hmm_data']
        exists = HMM_exists(h5_file, taste, n_states)
        if exists and not overwrite:
            return

        with tables.open_file(h5_file, 'a') as hf5:
            tmp_taste = taste.replace(' ', '_')

            # Add row to data overview
            dat = hf5.root.data_overview
            if exists:
                hf5.remove_node('/'+state_str, tmp_taste, recursive=True)
                idx = np.where((dat['taste'] == taste) &
                               (dat['n_states'] == n_states))[0]
                if len(idx) == 1:
                    dat.remove_rows(idx[0], idx[0]+1)
                elif len(idx) > 1:
                    raise ValueError('Multiple entries for %i states: %s' %
                                     (n_states, taste))

                dat.flush()
                hf5.flush()

            if state_str not in hf5.root:
                hf5.create_group('/', state_str, '%i State HMM Solutions' % n_states)

            if tmp_taste not in hf5.root[state_str]:
                hf5.create_group('/' + state_str, tmp_taste, '%s HMM Data' % taste)

            row = dat.row
            row['taste'] = taste
            row['channel'] = channel
            row['n_cells'] = n_cells
            row['n_trials'] = n_trials
            row['n_states'] = n_states
            row['dt'] = dt
            row['BIC'] = BIC
            row['converged'] = converged
            row['threshold'] = thresh
            row['cost'] = cost
            row.append()
            dat.flush()

            # Add data arrays
            array_loc = '/%s/%s' % (state_str, tmp_taste)
            hf5.create_array(array_loc, 'initial_distribution', PI)
            hf5.create_array(array_loc, 'transition', A)
            hf5.create_array(array_loc, 'emission', B)
            hf5.create_array(array_loc, 'state_sequences', bestPaths)
            hf5.flush()

        # make plots
        fn = os.path.join(plot_dir, '%s_trial_decoding.png' % taste)
        plot_hmm(hmm, self.params['time_window'], taste, save_file=fn)

    def load_hmm_from_h5(self, taste):
        n_states = self.n_states
        h5_file = self._files['hmm_data']
        spike_array, dt, _ = self.get_spike_data(taste)
        hmm = load_hmm_from_h5(h5_file, taste, n_states, spike_array, dt)
        return hmm

    def get_data_overview(self):
        h5_file = self._files['hmm_data']
        return get_data_overview(h5_file)

    def HMM_exists(taste):
        return HMM_exists(self._files['hmm_data'], taste, self.n_states)


def load_hmm_from_h5(h5_file, taste, n_states, spike_array, dt):
    state_str = '%i_states' % n_states
    taste_str = taste.replace(' ', '_')
    if not HMM_exists(h5_file, taste, n_states):
        return None

    hmm = phmm.PoissonHMM(n_states, spike_array, dt)
    with tables.open_file(h5_file, 'r') as hf5:
        taste_node = hf5.root[state_str][taste_str]
        PI = taste_node['initial_distribution'][:]
        A  = taste_node['transition'][:]
        B  = taste_node['emission'][:]
        hmm.set_matrices({'PI': PI, 'A': A, 'B': B})
        hmm.fitted = True

    return hmm


def get_data_overview(h5_file):
    with tables.open_file(h5_file, 'r') as hf5:
        return hf5.root.data_overview[:]


def HMM_exists(h5_file, taste, n_states):
    state_str = '%i_states' % n_states
    taste_str = taste.replace(' ', '_')
    dat = get_data_overview(h5_file)
    idx = np.where((dat['taste'].astype('str') == taste) &
                   (dat['n_states'] == n_states))[0]
    if len(idx) == 0:
        return False

    # Check if actual data exists
    with tables.open_file(h5_file, 'r') as hf5:
        if not state_str in hf5.root:
            return False

        if not taste_str in hf5.root[state_str]:
            return False

        if not any([x in hf5.root[state_str][taste_str]
                    for x in ['initial_distribution', 'transition',
                              'emission', 'state_sequences']]):
            return False

    return True


def rebin_spike_array(spikes, dt, time, new_dt):
    n_bins = int(new_dt/dt)
    win_starts = np.arange(time[0], time[-1], new_dt)

def plot_hmm(hmm, time_window, taste, save_file=None):
    spikes = hmm.data
    dt = hmm.dt
    BIC, paths = hmm.get_BIC()
    nTrials, nCells, nTimeSteps = spikes.shape
    time = np.arange(time_window[0], time_window[1], dt*1000)
    nStates = np.max(paths)+1
    colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, nStates)]


    fig, axes = plt.subplots(nrows=nTrials, figsize=(15,15))
    fig.subplots_adjust(right=0.9)
    y_step = np.linspace(0.05, 0.95, nCells)
    handles = []
    labels = []
    for ax, seq, trial in zip(axes, paths, spikes):
        _, leg_handles, leg_labels = plot_sequence(seq, time=time, ax=ax, colors=colors)
        for h, l in zip(leg_handles, leg_labels):
            if l not in labels:
                handles.append(h)
                labels.append(l)

        for j, cell in enumerate(trial):
            idx = np.where(cell == 1)[0]
            ax.scatter(time[idx], cell[idx]*y_step[j], marker='|', color='black')

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)


    axes[-1].get_xaxis().set_visible(True)
    axes[-1].set_xlabel('Time (ms)')
    mid = int(nTrials/2)
    axes[mid].legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                    fontsize=14)
    axes[mid].set_ylabel('Trials')
    axes[0].set_title('HMM Decoded State Sequences\n%s' % taste)
    if save_file:
        fig.savefig(save_file)
        return
    else:
        fig.show()
        return fig, ax


def plot_sequence(seq, time=None, ax=None, y_min=0, y_max=1, colors=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if time is None:
        time = np.arange(0, len(seq))

    nStates = np.max(seq)+1
    if colors is None:
        colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, nStates)]

    seq_windows = get_sequence_windows(seq)
    handles = {}
    for win in seq_windows:
        t_vec = [time[win[0]], time[win[1]]]
        h = ax.fill_between(t_vec, [y_min, y_min], [y_max, y_max],
                            color=colors[int(win[2])], alpha=0.4)
        if  win[2] not in handles:
            handles[win[2]] = h

    leg_handles = [handles[k] for k in sorted(handles.keys())]
    leg_labels = ['State %i' % k for k in sorted(handles.keys())]
    return ax, leg_handles, leg_labels


def get_sequence_windows(seq):
    t = 0
    out = []
    while t < len(seq):
        s = seq[t]
        tmp = np.where(seq[t:] != s)[0]
        if len(tmp) == 0:
            tmp = len(seq) - t
        else:
            tmp = tmp[0]

        out.append((t, tmp+t-1, s))
        t += tmp

    return out


def plot_raster(spikes, time=None, ax=None, y_min=0.1, y_max=0.9):
    pass

def plot_hmm_rates(rates, ax=None, colors=None):
    pass

def plot_hmm_transition(transition, ax=None):
    pass

def plot_forward_probs(hmm, time=None, ax=None, colors=None):
    pass

def plot_backward_probs(hmm, time=None, ax=None, colors=None):
    pass

def plot_viterbi_probs(hmm, time=None, ax=None, colors=None):
    pass
