import os
import numpy as np
import pandas as pd
from blechpy.analysis import spike_analysis as sas
from blechpy import load_dataset
from blechpy.dio import h5io
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import itertools as itt
from scipy.stats import sem
from uncertainty import I
from scipy.spatial.distance import euclidean

def apply_pca_analysis(df, params):
    '''df is held_units dataframe grouped by exp_name, exp_group, time_group
    only contains units held over preCTA or postCTA, no units held from ctaTrain to ctaTest
    
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

    rd1 = df['rec1'].unique()
    rd2 = df['rec2'].unique()
    if len(rd1) > 1 or len(rd2) > 1:
        raise ValueError('Too many recording directories')

    rd1 = rd1[0]
    rd2 = rd2[0]
    units1 = list(df['unit1'].unique())
    units2 = list(df['unit2'].unique())
    dim1 = load_dataset(rd1).dig_in_mapping.set_index('channel')
    dim2 = load_dataset(rd2).dig_in_mapping.set_index('channel')
    if n_cells < 2:
        # No point if only 1 unit
        exp_name = os.path.basename(rd1).split('_')
        print('%s - %s: Not enough units for PCA analysis' % (exp_name[0], exp_name[-3]))
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
    # pca = MDS(n_components=2)
    pca = PCA(n_components=2)
    pc_values = pca.fit_transform(rates)
    mds = MDS(n_components=2)
    md_values = mds.fit_transform(rates)

    out_df = pd.DataFrame(labels, columns=['taste', 'trial', 'time'])
    out_df['n_cells'] = n_cells
    out_df[['PC1', 'PC2']] = pd.DataFrame(pc_values)
    out_df[['MDS1','MDS2']] = pd.DataFrame(md_values)

    # Compute the MDS distance metric using the full dimensional solution
    # For each point computes distance to mean Quinine / distance to mean NaCl
    mds = MDS(n_components=rates.shape[1])
    mds_values = mds.fit_transform(rates)
    n_idx = np.where(labels[:,0] == 'NaCl')[0]
    q_idx = np.where(labels[:,0] == 'Quinine')[0]
    q_mean = np.mean(mds_values[q_idx,:], axis=0)
    n_mean = np.mean(mds_values[n_idx,:], axis=0)
    dist_metric = [euclidean(x, q_mean)/euclidean(x, n_mean) for x in mds_values]
    assert len(dist_metric) == rates.shape[0], 'computed distances over wrong axis'
    out_df['dQ_v_dN_fullMDS'] = pd.DataFrame(dist_metric)

    # Do it again with raw rates
    q_mean = np.mean(rates[q_idx, :], axis=0)
    n_mean = np.mean(rates[n_idx, :], axis=0)
    raw_metric = [euclidean(x, q_mean)/euclidean(x, n_mean) for x in rates]
    assert len(raw_metric) == rates.shape[0], 'computed distances over wrong axis'
    out_df['dQ_v_dN_rawRates'] = pd.DataFrame(raw_metric)


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
        sx1, sy1 = grp1[['PC1', 'PC2']].agg(sem)
        x2, y2 = grp2[['PC1', 'PC2']].mean()
        sx2, sy2 = grp2[['PC1', 'PC2']].agg(sem)
        X1 = I(x1, sx1)
        Y1 = I(y1, sy1)
        X2 = I(x2, sx2)
        Y2 = I(y2, sy2)
        pc_delta = ((X1-X2)**2 + (Y1-Y2)**2)**0.5

        x1, y1 = grp1[['MDS1', 'MDS2']].mean()
        sx1, sy1 = grp1[['MDS1', 'MDS2']].agg(sem)
        x2, y2 = grp2[['MDS1', 'MDS2']].mean()
        sx2, sy2 = grp2[['MDS1', 'MDS2']].agg(sem)
        X1 = I(x1, sx1)
        Y1 = I(y1, sy1)
        X2 = I(x2, sx2)
        Y2 = I(y2, sy2)
        mds_delta = ((X1-X2)**2 + (Y1-Y2)**2)**0.5

        out.append((tst1, tst2, pc_delta.value, pc_delta.delta, mds_delta.value, mds_delta.delta))

    out_df = pd.DataFrame(out, columns=['taste_1', 'taste_2', 'pc_dist', 'pc_dist_sem', 'mds_dist','mds_dist_sem'])
    return out_df


def apply_pc_dist_metric(df):
    '''grouped by exp_name, exp_group, time_group, time
    '''
    tastes = df.taste.unique()
    nacl_x, nacl_y = df.query('taste == "NaCl"')[['PC1','PC2']].mean()
    nacl_dx, nacl_dy = df.query('taste == "NaCl"')[['PC1','PC2']].agg(sem)
    Q_x, Q_y = df.query('taste == "Quinine"')[['PC1','PC2']].mean()
    Q_dx, Q_dy = df.query('taste == "Quinine"')[['PC1','PC2']].agg(sem)
    sacc = df.query('taste == "Saccharin"').copy()
    Xn = I(nacl_x, nacl_dx)
    Yn = I(nacl_y, nacl_dy)
    Xq = I(Q_x, Q_dx)
    Yq = I(Q_y, Q_dy)
    def get_metric(z):
        x = z[0]
        y = z[1]
        dN = ((Xn-x)**2 + (Yn - y)**2)**0.5
        dQ = ((Xq-x)**2 + (Yq-y)**2)**0.5
        dS = dQ/dN
        if np.isnan(dS.value):
            raise ValueError()

        return pd.Series({'PC_dQ_v_dN': dS.value, 'PC_dQ_v_dN_sem': dS.delta})

    sacc[['PC_dQ_v_dN', 'PC_dQ_v_dN_sem']] = sacc[['PC1','PC2']].apply(get_metric, axis=1)

    return sacc


def apply_mds_dist_metric(df):
    '''grouped by exp_name, exp_group, time_group, time
    '''
    tastes = df.taste.unique()
    nacl_x, nacl_y = df.query('taste == "NaCl"')[['MDS1','MDS2']].mean()
    nacl_dx, nacl_dy = df.query('taste == "NaCl"')[['MDS1','MDS2']].agg(sem)
    Q_x, Q_y = df.query('taste == "Quinine"')[['MDS1','MDS2']].mean()
    Q_dx, Q_dy = df.query('taste == "Quinine"')[['MDS1','MDS2']].agg(sem)
    sacc = df.query('taste == "Saccharin"').copy()
    Xn = I(nacl_x, nacl_dx)
    Yn = I(nacl_y, nacl_dy)
    Xq = I(Q_x, Q_dx)
    Yq = I(Q_y, Q_dy)
    dQN = ((Xn - Xq)**2 + (Yn - Yq)**2)**0.5
    def get_metric(z):
        x = z[0]
        y = z[1]
        dN = ((Xn-x)**2 + (Yn - y)**2)**0.5
        dQ = ((Xq-x)**2 + (Yq-y)**2)**0.5
        dS = dQ/dN
        dQ = dQ/dQN
        dN = dN/dQN
        dS2 = (dQ - dN)
        if np.isnan(dS.value):
            raise ValueError()

        return pd.Series({'MDS_dQ': dQ.value, 'MDS_dN': dN.value, 'MDS_dQ_v_dN': dS.value,
                          'MDS_dQ_v_dN_sem': dS.delta, 'MDS_dQ_minus_dN': dS2.value,
                          'MDS_dQ_minus_dN_sem':dS2.delta})

    sacc[['MDS_dQ', 'MDS_dN', 'MDS_dQ_v_dN', 'MDS_dQ_v_dN_sem',
          'MDS_dQ_minus_dN', 'MDS_dQ_minus_dN_sem']] = sacc[['MDS1','MDS2']].apply(get_metric, axis=1)

    return sacc


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

    st, sa = h5io.get_spike_data(rec, units)
    if t_start is None:
        t_start = st[0]

    if t_end is None:
        t_end = st[-1]

    spikes = []
    labels = []
    dim = load_dataset(rec).dig_in_mapping.set_index('channel')
    if isinstance(sa, dict):
        for k,v in sa.items():
            ch = int(k.split('_')[-1])
            tst = dim.loc[ch, 'name']
            l = [(tst, i) for i in range(v.shape[0])]
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
        tst = dim.loc[0,'name']
        l = [(tst, i) for i in range(sa.shape[0])]
        labels.extend(l)

    spikes = np.vstack(spikes).astype('float64')
    b_idx = np.where(st < 0)[0]
    baseline_fr = np.sum(spikes[:,:,b_idx], axis=-1)/ (len(b_idx)/1000)
    baseline_fr = np.mean(baseline_fr, axis=0)
    t_idx = np.where((st >= t_start) & (st <= t_end))[0]
    spikes = spikes[:, : , t_idx]
    st = st[t_idx]

    fr_lbls = []
    fr_arr = []
    fr_time = None
    for trial_i, (trial, lbl) in enumerate(zip(spikes, labels)):
        fr_t, fr = sas.get_binned_firing_rate(st, trial, bin_size, step)
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


def fulldim_mds_analysis(df):
    """ Should be receiving held_units grouped by exp_name, exp_group, time_group
    """
