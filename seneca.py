import tables
import numpy as np
from blechpy import load_dataset
from blechpy.dio import h5io
from blechpy.analysis import poissonHMM as phmm

def get_laser_trials(rec_dir, din):
    '''returns on_trials and off_trials'''
    h5_file = h5io.get_h5_filename(rec_dir)
    with tables.open_file(h5_file, 'r') as hf5:
        qstr = f'/spike_trains/dig_in_{din}/on_laser'
        if qstr in hf5:
            tbl = hf5.get_node(qstr)[:,0]
        else:
            print(f'No on_laser found in {rec_dir} for dig_in {din}')
            return

        on_trials = np.where(tbl == 1)[0]
        off_trials = np.where(tbl == 0)[0]

    return list(on_trials), list(off_trials)

def run_hmms(rec_dirs, constraint_func=None):
    base_params = {'unit_type': 'single', 'dt': 0.001,
                   'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
                   'time_end': 2000, 'n_states': 3, 'area': 'GC',
                   'hmm_class': 'PoissonHMM', 'threshold':1e-10,
                   'notes': 'sequential - low thresh'}

    params = [{'n_states': 2}, {'n_states': 3}, {'time_start': -200, 'n_states': 4}]

    for rec_dir in rec_dirs:
        units = phmm.query_units(rec_dir, 'single', area='GC')
        if len(units) < 2:
            continue
        handler = phmm.HmmHandler(rec_dir)
        dat = load_dataset(rec_dir)

        for i, row in dat.dig_in_mapping.iterrows():
            if row['laser']:
                continue

            name = row['name']
            ch = row['channel']
            on_trials, off_trials = get_laser_trials(rec_dir, ch)

            for new_params in params:
                p = base_params.copy()
                p.update(new_params)
                p['taste'] = name
                p['channel'] ch
                p['notes'] += ' - all_trials'
                handler.add_params(p)

                p = base_params.copy()
                p.update(new_params)
                p['taste'] = name
                p['channel'] ch
                p['trial_nums'] = on_trials
                p['notes'] += ' - on_trials'
                handler.add_params(p)

                p = base_params.copy()
                p.update(new_params)
                p['taste'] = name
                p['channel'] ch
                p['trial_nums'] = off_trials
                p['notes'] += ' - off_trials'
                handler.add_params(p)

        dataname = os.path.basename(rec_dir)
        print('Fitting %s' % os.path.basename(rec_dir))
        if type(constraint) == 'function':
            print('Fitting Constraint: %s' % constraint.__name__)
        else:
            print('Fitting Constraint: %s' % str(constraint))

        handler.run(constraint_func=constraint)
