import tables
import blechpy
import os
import numpy as np
import pandas as pd
import itertools as it
from blechpy.analysis import poissonHMM as phmm
from blechpy.utils import write_tools as wt
from blechpy.plotting import hmm_plot as hmmplt
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


class MultiInfoParticle(tables.IsDescription):
    #   HMM_ID, taste, din_channel, n_cells, time_start, time_end, thresh,
    #   unit_type, n_repeats, dt, n_states, n_iters, BIC, cost, converged
    hmm_id = tables.Int16Col()
    taste = tables.StringCol(20)
    channel = tables.Int16Col()
    n_cells = tables.Int32Col()
    unit_type = tables.StringCol(15)
    n_trials = tables.Int32Col()
    dt = tables.Float64Col()
    max_iter = tables.Int32Col()
    threshold = tables.Float64Col()
    time_start = tables.Int32Col()
    time_end = tables.Int32Col()
    n_repeats = tables.Int16Col()
    n_states = tables.Int32Col()
    n_iterations = tables.Int32Col()
    BIC = tables.Float64Col()
    cost = tables.Float64Col()
    converged = tables.BoolCol()
    fitted = tables.BoolCol()


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
        plot_dir = self._plot_dir


        for taste, row in dig_ins.iterrows():
            # Check if trained HMM already exists
            # Add overwrite flag
            if HMM_exists(h5_file, taste, n_states) and not overwrite:
                print('Detected existing fitted model for %i states - %s.'
                      '\nLoading...' % (n_states, taste))
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

            def error_call(e):
                print(e)
                raise ValueError(e)

            if parallel:
                pool = mp.get_context('spawn').Pool(mp.cpu_count()-1)
                for i in range(n_repeats):
                    pool.apply_async(phmm.fit_hmm_mp,
                                     (n_states, spike_array, dt, max_iter, thresh),
                                     callback=update, error_callback=error_call)

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
            # make plots
            save_dir = os.path.join(plot_dir, taste)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            hmmplt.plot_hmm_figures(hmm, self.params['time_window'], save_dir=save_dir)


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
    if spikes.ndim == 2:
        spikes = np.expand_dims(spikes,0)

    n_trials, n_cells, n_steps = spikes.shape
    n_bins = int(new_dt/dt)
    new_time = np.arange(time[0], time[-1], new_dt)
    new_spikes = np.zeros((n_trials, n_cells, len(new_time)))
    for i, w in enumerate(new_time):
        idx = np.where((time >= w) & (time < w+new_dt))[0]
        new_spikes[:,:,i] = np.sum(spikes[:,:,idx], axis=-1)

    return new_spikes, new_time


# Cut data to 0-2000 sec, fit hmm with 2 states and 3 states and try a different bin size
# Bin sizes: 0.001 and 0.10
# Get program to make all hmms and put into an array first and then fit them all
HMM_PARAMS = {'unit_type': 'single', 'dt': 0.001, 'threshold': 1e-4, 'max_iter': 1000,
              'time_start': 0, 'time_end': 2000, 'n_repeats': 3, 'n_states': 3}


class HmmHandler(object):
    def __init__(self, dat, params=None, save_dir=None):
        '''Takes a blechpy dataset object and fits HMMs for each tastant

        Parameters
        ----------
        dat: blechpy.dataset
        params: dict or list of dicts
            each dict must have fields:
                time_window: list of int, time window to cut around stimuli in ms
                convergence_thresh: float
                max_iter: int
                n_repeats: int
                unit_type: str, {'single', 'pyramidal', 'interneuron', 'all'}
                bin_size: time bin for spike array when fitting in seconds
                n_states: predicted number of states to fit
        '''
        if isinstance(params, dict):
            params = [params]

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
        self.h5_file = os.path.join(save_dir, '%s_HMM_Analysis.hdf5' % dat.data_name)
        dim = dat.dig_in_mapping.query('exclude==False')
        tastes = dim['name'].tolist()
        if params is None:
            # Load params and fitted models
            self.load_data()
        else:
            self.init_params(params)

        self.params = params

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.plot_dir = os.path.join(save_dir, 'HMM_Plots')
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self._setup_hdf5()

    def init_params(self, params):
        dat = self._dataset
        dim = dat.dig_in_mapping.query('exclude == False')
        tastes = dim['name'].tolist()
        dim = dim.set_index('name')
        if not hasattr(dat, 'dig_in_trials'):
            dat.create_trial_list()

        trials = dat.dig_in_trials
        data_params = []
        fit_objs = []
        for i, X in enumerate(it.product(params,tastes)):
            p = X[0].copy()
            t = X[1]
            p['hmm_id'] = i
            p['taste'] = t
            p['channel'] = dim.loc[t, 'channel']
            unit_names = query_units(dat, p['unit_type'])
            p['n_cells'] = len(unit_names)
            p['n_trials'] = len(trials.query('name == @t'))

            data_params.append(p)
            # Make fit object for each repeat
            # During fitting compare HMM as ones with the same ID are returned
            for i in range(p['n_repeats']):
                hmmFit = HMMFit(dat.root_dir, p)
                fit_objs.append(hmmFit)

        self._fit_objects = fit_objs
        self._data_params = data_params
        self._fitted_models = dict.fromkeys([x['hmm_id'] for x in data_params])

    def load_params(self):
        h5_file = self.h5_file
        if not os.path.isfile(h5_file):
            raise ValueError('No params to load')

        rec_dir = self._dataset.root_dir
        params = []
        fit_objs = []
        fitted_models = {}
        with tables.open_file(h5_file, 'r') as hf5:
            table = hf5.root.data_overview
            for row in table[:]:
                p = {}
                for k in NEW_HMM_PARAMS.keys():
                    p[k] = row[k]

                params.append(p)
                for i in range(p['n_repeats']):
                    hmmFit = HMMFit(rec_dir, p)
                    fit_objs.append(hmmFit)

        for p in params:
            hmm_id = p['hmm_id']
            fitted_models[hmm_id] = read_hmm_from_hdf5(h5_file, hmm_id)

        self._data_params = params
        self._fit_objects = fit_objs
        self._fitted_models = fitted_models


    def write_overview_to_hdf5(self):
        params = self._data_params
        h5_file = self.h5_file
        if hasattr(self, '_fitted_models'):
            models = self._fitted_models
        else:
            models = dict.fromkeys([x['hmm_id']
                                    for x in data_params])
            self._fitted_models = models


        if not os.path.isfile(h5_file):
            self._setup_hdf5()

        print('Writing data overview table to hdf5...')
        with tables.open_file(h5_file, 'a') as hf5:
            table = hf5.root.data_overview
            # Clear old table
            table.remove_rows(start=0)

            # Add new rows
            for p in params:
                row = table.row
                for k, v in p.items():
                    row[k] = v

                if models[p['hmm_id']] is not None:
                   hmm = models[p['hmm_id']]
                   row['n_iterations'] =  hmm.iterations
                   row['BIC'] = hmm.BIC
                   row['cost'] = hmm.cost
                   row['converged'] = hmm.isConverged(p['threshold'])
                   row['fitted'] = hmm.fitted

                row.append()

            table.flush()
            hf5.flush()

        print('Done!')

    def _setup_hdf5(self):
        h5_file = self.h5_file

        with tables.open_file(h5_file, 'a') as hf5:
            # Taste -> PI, A, B, BIC, state_sequences, nStates, nCells, dt
            if not 'data_overview' in hf5.root:
                # Contains taste, channel, n_cells, n_trials, n_states, dt, BIC
                table = hf5.create_table('/', 'data_overview', MultiInfoParticle,
                                         'Basic info for each digital_input')
                table.flush()


            if hasattr(self, '_data_params') and self._data_params is not None:
                for p in self._data_params:
                    hmm_str = 'hmm_%i' % p['hmm_id']
                    if hmm_str not in hf5.root:
                        hf5.create_group('/', hmm_str, 'Data for HMM #%i' % p['hmm_id'])

            hf5.flush()

    def run(self, parallel=True):
        h5_file = self.h5_file
        fit_objs = self._fit_objects
        HMMs = dict.fromkeys([x['hmm_id'] for x in self._data_params])
        errors = []

        def update(ans):
            hmm_id = ans[0]
            hmm = ans
            if hmm_id in HMMs:
                new_hmm = pick_best_hmm([HMMs[hmm_id], hmm])
                HMMs[hmm_id] = new_hmm
            else:
                # Check history for lowest BIC
                HMMs[hmm_id] = hmm.set_to_lowest_BIC()

        def error_call(e):
            errors.append(e)

        if parallel:
            n_cpu = np.min((mp.cpu_count(), len(fit_objs)))
            pool = mp.get_context('spawn').Pool(mp.cpu_count())
            for f in fit_objs:
                pool.apply_async(f.run, callback=update, error_callback=error_call)

            pool.close()
            pool.join()
        else:
            for f in fit_objs:
                try:
                    ans = f.run()
                    update(ans)
                except Exception as e:
                    raise Exception(e)
                    error_call(e)

        self._fitted_models = HMMs
        self.write_overview_to_hdf5()
        self.save_fitted_models()
        if len(errors) > 0:
            print('Encountered errors: ')
            for e in errors:
                print(e)

    def save_fitted_models(self):
        models = self._fitted_models
        for k, v in models:
            write_hmm_to_hdf5(self.h5_file, k, v)
            plot_dir = os.path.join(self.plot_dir, 'HMM_%i' % k)
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            ids = [x['hmm_id'] for x in self._data_params]
            idx = np.where(ids == k)[0]
            params = self._data_params[idx]
            time_window = [params['time_start'], params['time_end']]
            hmmplt.plot_hmm_figures(v, time_window, save_dir=plot_dir)








def read_hmm_from_hdf5(h5_file, hmm_id, rec_dir):
    with tables.open_file(h5_file, 'r') as hf5:
        h_str = 'hmm_%i' % hmm_id
        if h_str not in hf5.root:
            return None

        table = hf5.root.data_overview
        row = table.where('hmm_id == id', condvars={'id':hmm_id})
        if len(row) == 0:
            raise ValueError('Parameters not found for hmm %i' % hmm_id)
        elif len(row) > 1:
            raise ValueError('Multiple parameters found for hmm %i' % hmm_id)

        units = query_units(rec_dir, row['unit_type'])
        spikes, dt, time = get_spike_data(rec_dir, units, row['channel'],
                                          dt=row['dt'],
                                          time_start=row['time_start'],
                                          time_end=row['time_end'])
        tmp = hf5.root[h_str]
        mats = {'initial_distribution': tmp['initial_distribution'][:],
                'transition': tmp['transition'][:],
                'emission': tmp['emission'][:],
                'fitted': row['fitted']}
        hmm = phmm.PoissonHMM(row['n_states'], spikes, dt, set_data=mats)

    return hmm

def write_hmm_to_hdf5(h5_file, hmm_id, hmm):
    h_str = 'hmm_%i' % hmm_id
    with tables.open_file(h5_file, 'a') as hf5:
        if hmm_id in hf5.root:
            hf5.remove_node('/', h_str, recursive=True)

        hf5.create_group('/', h_str, 'Data for HMM #%i' % i)
        hf5.create_array('/'+h_str, 'initial_distribution',
                         hmm.initial_distribution)
        hf5.create_array('/'+h_str, 'transition', hmm.transition)
        hf5.create_array('/'+h_str, 'emission', hmm.emission)

        best_paths, _ = hmm.get_best_paths()
        hf5.create_array('/'+h_str, 'state_sequences', best_paths)


def query_units(dat, unit_type):
    '''Returns the units names of all units in the dataset that match unit_type

    Parameters
    ----------
    dat : blechpy.dataset or str
        Can either be a dataset object or the str path to the recording
        directory containing that data .h5 object
    unit_type : str, {'single', 'pyramidal', 'interneuron', 'all'}
        determines whether to return 'single' units, 'pyramidal' (regular
        spiking single) units, 'interneuron' (fast spiking single) units, or
        'all' units

    Returns
    -------
        list of str : unit_names
    '''
    if isinstance(dat, str):
        units = blechpy.dio.h5io.get_unit_table(dat)
    else:
        units = dat.get_unit_table()

    u_str = unit_type.lower()
    q_str = ''
    if u_str == 'single':
        q_str = 'single_unit == True'
    elif u_str == 'pyramidal':
        q_str = 'single_unit == True and regular_spiking == True'
    elif u_str == 'interneuron':
        q_str = 'single_unit == True and fast_spiking == True'
    elif u_str == 'all':
        return units['unit_name'].tolist()
    else:
        raise ValueError('Invalid unit_type %s. Must be '
                         'single, pyramidal, interneuron or all' % u_str)

    return units.query(q_str)['unit_name'].tolist()


    # Parameters
    # hmm_id
    # taste
    # channel
    # n_cells
    # unit_type
    # n_trials
    # dt
    # threshold
    # time_start
    # time_end
    # n_repeats
    # n_states
    # n_iterations
    # BIC
    # cost
    # converged
    # fitted
    #
    # Extras: unit_names, rec_dir


class HMMFit(object):
    def __init__(self, rec_dir, params):
        self._rec_dir = rec_dir
        self._params = params

    def run(self):
        params = self._params
        spikes, dt, time = self.get_spike_data()
        hmm = phmm.fit_hmm_mp(params['n_states'], spikes, dt,
                              params['max_iter'], params['threshold'])
        return params['hmm_id'], hmm

    def get_spike_data(self):
        p = self._params
        units = query_units(self._rec_dir, p['unit_type'])
        # Get stored spike array, time is in ms, dt is usually 1 ms
        spike_array, dt, time = get_spike_data(self._rec_dir, units,
                                               p['channel'], dt=p['dt'],
                                               time_start=p['time_start'],
                                               time_end=p['time_end'])
        return spike_array, dt, time


def get_spike_data(rec_dir, units, channel, dt=None, time_start=None, time_end=None):
    time, spike_array = blechpy.dio.h5io.get_spike_data(rec_dir, units, channel)
    curr_dt = np.unique(np.diff(time))[0] / 1000
    if dt is not None and curr_dt < dt:
        spike_array, time = rebin_spike_array(spike_array, dt, time, p['dt'])
    elif dt is not None and curr_dt > dt:
        raise ValueError('Cannot upsample spike array from %f ms '
                         'bins to %f ms bins' % (dt, curr_dt))
    else:
        dt = curr_dt

    if time_start and time_end:
        idx = np.where((time >= time_start) & (time < time_end))[0]
        time = time[idx]
        spike_array = spike_array[:, :, idx]

    return spike_array.astype('int32'), dt, time


def pick_best_hmm(HMMs):
    '''For each HMM it searches the history for the HMM with lowest BIC Then it
    compares HMMs. Those with same # of free parameters are compared by BIC
    Those with different # of free parameters (namely # of states) are compared
    by cost Best HMM is returned

    Parameters
    ----------
    HMMs : list of phmm.PoissonHmm objects

    Returns
    -------
    phmm.PoissonHmm
    '''
    # First optimize each HMMs and sort into groups based on # of states
    groups = {}
    for hmm in HMMs:
        hmm.set_to_lowest_BIC()
        if hmm.n_states in groups:
            groups[hmm.n_states].append(hmm)
        else:
            groups[hmm.n_states] = [hmm]

    best_per_state = {}
    for k, v in groups:
        BICs = np.array([x.get_BIC() for x in v])
        idx = np.argmin(BICs)
        best_per_state[k] = v[idx]

    hmm_list = best_per_state.values()
    costs = np.array([x.get_cost() for x in hmm_list])
    idx = np.argmin(costs)
    return hmm_list[idx]

    # Compare HMMs with same number of states by BIC

