import numpy as np
import pandas as pd
from blechpy.analysis import spike_analysis as sas
from sklearn.decomposition import PCA
import itertools as itt
from uncertainty import I

def apply_pca_analysis(df, params):
    '''df is held_units dataframe grouped by exp_name and held_over
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    Raises
    ------
    
    '''
    bin_size = params['pca']['win_size']
    bin_step = params['pca']['step_size']
    time_start = params['pca']['time_win'][0]
    time_end = params['pca']['time_win'][1]
    smoothing = params['pca']['smoothing_win']
    n_cells = len(df)

    exp_name = df.exp_name.unique()[0]
    time_group = df.time_group.unique()[0]
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
    if n_cells < 2:
        # No point if only 1 unit
        print('Not enough units for PCA analysis')
        return

    time, sa = h5io.get_spike_data(rd1, units1)
    fr_t, fr, fr_lbls = get_pca_data(rd1, units1, bin_size, step=bin_step,
                                     t_start=time_start, t_end=time_end)
    rates = fr
    labels = fr_lbls
    time = fr_t
    # Again with rec2
    fr_t, fr, fr_lbls = get_pca_data(rd2, units2, bin_size, step=bin_step,
                                     t_start=time_start, t_end=time_end)
    rates = np.vstack([rates, fr])
    labels = np.vstack([labels, fr_lbls])
    # So now rates is tastes*trial*times X units

    # Do PCA on all data, put in (trials*time)xcells 2D matrix
    pca = PCA(n_components=2)
    pc_values = pca.fit_transform(rates)
    out_df = pd.DataFrame(labels, columns=['taste', 'trial', 'time'])
    out_df['n_cells'] = n_cells
    out_df[['PC1', 'PC2']] = pd.DataFrame(pc_values)
    return out_df


def apply_pc_distances(df):
    '''df shoudl have columns taste, trial, PC1 and PC2 and be grouped by
    exp_name and time_group (pre vs post cta) and time (post-stimulus time)
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    Raises
    ------
    
    '''
    tastes = df.taste.unique()
    tst_pairs = list(itt.combinations(tastes, 2))
    # Output is df with taste1, taste2, mean_distance, sem_distance
    out = []
    for tst1, tst2 in tst_pairs:
        grp1 = df.query('taste == @tst1')
        grp2 = df.query('taste == @tst2')
        x1, y1 = grp1[['PC1', 'PC2']].mean()
        sx1, sy1 = grp1[['PC1', 'PC2']].std()
        x2, y2 = grp2[['PC1', 'PC2']].mean()
        sx2, sy2 = grp2[['PC1', 'PC2']].std()
        X1 = I(x1, sx1)
        Y1 = I(y1, sy1)
        X2 = I(x2, sx2)
        Y2 = I(y2, sy2)

        delta = ((X1-X2)**2 + (Y1-Y2)**2)**0.5
        out.append((tst1, tst2, delta.value, delta.delta))

    out_df = pd.DataFrame(out, columns=['taste_1', 'taste_2', 'pc_dist', 'pc_dist_std'])
    return out_df


def get_pca_data(rec, units, bin_size, step=None, t_start=None, t_end=None, baseline_win=None):
    '''Get spike data, turns it into binned firing rate traces and then
    organizes them into a format for PCA (trials*time X units)
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    Raises
    ------
    
    '''
    if step is None:
        step = bin_size

    st, sa = h5io.get_spike_data(rd1, units1)
    if t_start is None:
        t_start = st[0]

    if t_end is None:
        t_end = st[-1]

    spikes = []
    labels = []
    if isinstance(sa, dict):
        for k,v in sa.items():
            ch = int(k.split('_')[-1])
            tst = dim1.loc[ch, 'name']
            l = [(tst, i) for i in v.shape[0]]
            if len(v.shape) == 2:
                tmp = np.expand_dims(v, 1)
            else:
                tmp = v

            labels.extend(l)
            spikes.append(tmp)

    else:
        if len(sa.shape) == 2:
            tmp = np.exapnd_dims(sa, 1)
        else:
            tmp = sa

        spikes.append(tmp)
        tst = dim1.loc[0,'name']
        l = [(tst, i) for i in range(sa.shape[0])]
        labels.extend(l)

    b_idx = np.where(st < 0)[0]
    baseline_fr = np.sum(spikes[:,:,b_idx], axis=-1)/ (len(b_idx)/1000)
    baseline_fr = np.mean(baseline_fr, axis=0)
    t_idx = np.where((st >= t_start) & (st <= t_end))[0]
    spikes = spikes[:, : , t_idx]
    st = st[t_idx]

    fr_lbls = []
    fr_arr = []
    fr_time = None
    for trial_i, trial, lbl in enumerate(zip(spikes, labels)):
        fr_t, fr = sas.get_binned_firing_rate(st, trial, bin_size, bin_step)
        # fr is units x time
        fr = fr.T  # now its time x units
        fr = fr - baseline_fr  # subtract baseline firing rate
        l = [(*lbl, t) for t in fr_t]
        fr_arr.append(fr)
        fr_lbls.extend(l)
        if fr_time is None:
            fr_time = fr_t
        elif not np.array_equal(fr_t, fr_time):
            raise ValueError('Time Vectors dont match')

    # So now fr_lbls is [taste, trial, time]
    fr_out = np.vstack(fr_arr)
    fr_lbls = np.array(fr_lbls)
    return fr_time, fr_out, fr_lbls

