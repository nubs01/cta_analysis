# Single Cell Analysis

## Non-held units
### % taste responsive neurons (pre vs post CTA) (data readout per animal) [Bar plot GFP vs Cre]
### % palatability responsive neurons (pre vs post CTA) (data readout per animal) [Bar plot GFP vs Cre]

## Held-units (baseline removed: subtract mean firing rate when t<0)
### Use palatability responsive units held over 2 recordings to deduce palatability rank order of saccharin pre and post
### % neurons with significant change in response over CTA [data readout per animal] (Bar plot GFP vs Cre)
### Initial time of significant change [histogram per animal, average and deviation in data readout] (Bar plot or dot & box plot GFP vs Cre)
### % units changed at each time point of response [step plot per animal and GFP vs Cre]
### Average change in magnitude of response at each time point [plot per animal and GFP vs Cre]
### % units with change in baseline firing rate [data readout per animal] (bar plot GFP vs Cre)

# Population analysis
## Semi-held units: Units held across pre-CTA compared to units held across post-CTA
### Run HMM and get time of switch on each trial (make plots of HMM results for each trial) [HMM probabilities in data file]
### Plot MDS of each trial's ID phase and PAL phase seperately [per animal and GFP vs CTA]
### Compute relative distances between tastants (mean MDS) pre-CTA and post-CTA in ID and PAL [data readout: table per animal](bar plot GFP vs Cre]
### Plot PCA trajectory of each trial [per animal, pre vs post CTA, use paired color scheme] 

# So for each dataset, get:
# - Number of taste responsive neurons
# - Number of palatability responsive units

# For each experiment, compute:


from blechpy.dio import h5io
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, sem
from blechpy import load_experiment, load_dataset
from blechpy.analysis import stat_tests as stt


def get_baseline_firing(rec, unit, win_size=1500):
    '''Returns a vectors of baseline firing rates for the unit aggregated from
    win_size ms before every stimulus delivery
    
    Parameters
    ----------
    rec : str, path to recording directory
    unit: str or int, unit name or number
    win_size: int (optional), ms before stimulus delivery to use (default=1500)
    
    Returns
    -------
    numpy.ndarray, vector of baseline firing rates in Hz
    '''
    dat = load_dataset(rec)
    if dat is None:
        raise FileNotFoundError('Dataset not found for %s' % rec)

    if isinstance(unit, str):
        unit_num = h5io.parse_unit_number(unit)
    elif isinstance(unit, int):
        unit_num = unit
        unit = 'unit%03i' % unit_num

    dim = dat.dig_in_mapping.copy()
    dins = dim.query('exclude==False and spike_array==True')['channel'].tolist()
    baselines = []
    for channel in dins:
        times, spikes = h5io.get_spike_data(rec, unit_num, channel)
        idx = np.where((time >= -win_start) & (time < 0))[0]
        tmp = np.sum(spikes[:, idx], axis=1) / abs(win_size)
        tmp = tmp / 1000  # convert to Hz
        baselines.append(tmp)

    return np.hstack(baselines)


def compare_baseline(rd1, u1, rd2, u2, win_size=1500):
    '''Compares baseline firing rates between 2 unit using the Mann-Whitney
    U-test. Baseline firing is taken from all trials from all digital inputs
    that are not excluded and have spike arrays.
    
    
    Parameters
    ----------
    rd1: str, path to recording directory 1
    u1: str or int, unit from recording directory 1
    rd2: str, path to recording directory 2
    u2: str or int, unit from recording directory 2
    win_size: int (optional), size of window to use, in ms (default=1500)
    
    Returns
    -------
    dict, float
    stats, p-value
    '''
    baseline1 = get_baseline_firing(rd1, u1, win_size=win_size)
    baseline2 = get_baseline_firing(rd2, u2, win_size=win_size)

    base_u, base_p = mannwhitneyu(baseline1, baseline2,
                                  alternative='two-sided')
    stats = {'u-stat': base_u, 'p-val': base_p,
             'baseline1': (np.mean(baseline1), sem(baseline1)),
             'baseline2': (np.mean(baseline2), sem(baseline2))}

    return stats, base_p


def compare_taste_response(rd1, u1, din1, rd2, u2, din2,
                           time_window=[0, 2000], bin_size=250,
                           norm_func=None):
    s_time1, spikes1 = h5io.get_spike_data(rd1, u1, din1)
    s_time2, spikes2 = h5io.get_spike_data(rd2, u2, din2)

    t_idx1 = np.where((s_time1 >= time_window[0]) & (s_time1 <= time_window[1]))[0]
    t_idx2 = np.where((s_time2 >= time_window[0]) & (s_time2 <= time_window[1]))[0]
    s_time1 = s_time1[t_idx1]
    spikes1 = spikes1[:, t_idx1]
    s_time2 = s_time2[t_idx2]
    spikes2 = spikes2[:, t_idx2]

    time1, fr1 = sas.get_binned_firing_rate(time1, spikes1, bin_size, bin_size)
    time2, fr2 = sas.get_binned_firing_rate(time2, spikes2, bin_size, bin_size)
    if norm_func is not None:
        fr1 = norm_func(time, fr1)
        fr2 = norm_func(time, fr2)

    win_starts = np.arange(time_window[0], time_window[1], bin_size)
    resp_u = np.zeros(win_starts.shape)
    resp_p = np.ones(win_starts.shape)
    for i, ws in enumerate(win_starts):
        rate1 = fr1[:, i]
        rate2 = fr2[:, i]
        try:
            resp_u[i], resp_p[i] = mannwhitneyu(rate1, rate2,
                                                alternative='two-sided')
        except ValueError:
            resp_u[i] = 0
            resp_p[i] = 1


    # Bonferroni correction
    resp_p = resp_p * len(win_starts)

    return win_starts, resp_u, resp_p

def check_taste_responsiveness(rec, unit, win_size=1500, alpha=0.05):
    '''Runs through all digital inputs (non-excluded) and determines if neuron
    is taste responsive and to which tastant. Compares win_size ms before
    stimulus to win_size ms after stimulus with Mann-Whitney U-test.
    
    Parameters
    ----------
    rec: str, path to recording directory
    unit: int or str, unit name or number
    win_size: int (optional), window size in ms (default=1500)
    alpha: float (optional), significance level for tests (default=0.5)
    
    Returns
    -------
    bool: whether unit is taste responsive at all
    dict: computed statistics from each tastant

    Raises
    ------
    FileNotFoundError: if not dataset file in recording dir
    '''
    dat = load_dataset(rec)
    if dat is None:
        raise FileNotFoundError('No dataset found in %s' % rec)

    dim = dat.dig_in_mapping.copy()
    dins = dim.query('exclude==False and spike_array==True')['channel'].tolist()
    stats = {}
    taste_responsive = False
    p_values = {}

    # Bonferroni correction
    # TODO: Is this actually needed/applicable?
    alpha = alpha/len(dins)

    for i in dins:
        s, p = stt.check_taste_response(rec, unit, i, win_size=win_size)
        if p <= alpha:
            taste_responsive = True

        stats[i] = s
        p_values[i] = p

    return taste_responsive, stats


def check_palatability_responsiveness(rec, unit, win_size=250, step_size=25):
    # TODO: make this
    pass


def deduce_palatability_rank_order(rec, unit, dins, window):
    # TODO: make this
    pass









class AnimalAnalysis(object):
    def __init__(self, experiment=None):
        if experiment is None or isinstance(experiment, str):
            experiment = load_experiment(experiment)

        save_dir = self.analysis_dir = experiment.analysis_dir
        rec_key = {}
        for i, rec in enumerate(experiment.rec_labels.items()):
            rec_key[i] = rec
            
        held_unit_dir = os.path.join(save_dir, 'held_unit_analysis')
        self._files = {'single_cell': 1}
