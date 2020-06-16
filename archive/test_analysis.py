
def analyze_single_unit(row, dir_key, params=None):
    pass


def analyze_held_unit(unit_info, rec_key, norm_func=None, params=None):
    '''Check all changes in baselines firing and taste response between two or more groups

    Parameters
    ----------
    unit_info: pd.DataFrame
        chunk of DataFrame containing info for a single held unit in tidy data
        format
        columns: unit, electrode, area, recording, rec_unit, rec_group
    rec_key: dict,
        keys are number dictating order of directories.
        values are dicts with keys: name, dir, group and values as strings
    norm_func: function
        function to use to normalize firing rate traces before comparison
        take args: time (1xN np.ndarray) and fr (MxN np.ndarray) and returns a
        transformed fr of same shape
    params: dict
    '''
    groups = unit_info['rec_group'].unique()
    group_pairs = it.combinations(groups, 2)
    dir_key = {x['name']: x['dir'] for x in  rec_key.values()}
    group_key = {v['group']: v['name'] for v in rec_key.values()}

    # Un-pack params
    bin_size = params['response_comparison']['win_size']
    bin_step = params['response_comparison']['step_size']
    time_win = params['response_comparison']['time_win']
    alpha = params['response_comparison']['alpha']
    baseline_win = params['baseline_comparison']['win_size']
    baseline_alpha = params['baseline_comparison']['alpha']
    tasty_win = params['taste_responsive']['win_size']
    tasty_alpha = params['taste_responsive']['alpha']

    data = {}
    tastants = set()
    for i, row in unit_info.iterrows():
        g = row['rec_group']
        unit = row['rec_unit']
        rn = row['recording']
        rd = dir_key[rn]

        if g not in data:
            data[g] = {}
            data[g]['tasty_ps'] = []

        # aggregate baseline firing before CTA (even over days)
        baseline = get_baseline_firing(rd, unit, win_size=baseline_win)
        if 'baseline' in data[g]:
            data[g]['baseline'] = np.hstack((data[g]['baseline'], baseline))
        else:
            data[g]['baseline'] = baseline

        dat = load_dataset(rd)
        dim = dat.dig_in_mapping
        _, tasty_stats = check_taste_responsiveness(rd, unit,
                                                    win_size=tasty_win,
                                                    alpha=tasty_alpha)
        for j, t_row in dim.iterrows():
            t = t_row['name']
            channel = t_row['channel']
            tstat = tasty_stats[channel]
            tastants.add(t)
            if not t_row['spike_array'] or t_row['exclude']:
                continue

            if t not in data[g]:
                data[g][t] = {}
                data[g][t]['taste_responsive'] = (tstat['p-val'] < tasty_alpha)
                data[g][t]['taste_responsive_p'] = tstat['p-val']
                data[g][t]['taste_responsive_u'] = tstat['u-stat']
                data[g][t]['mean_taste_response'] = tstat['delta'][0]
                data[g][t]['sem_taste_response'] = tstat['delta'][1]
                data[g]['tasty_ps'].append(tstat['p-val'])
            else:
                raise KeyError('Already existing data for group %s and tastant %s' % (g,t))

            time, spikes = h5io.get_spike_data(rd, unit, channel)
            time, fr = sas.get_binned_firing_rate(time, spikes, bin_size, bin_step)
            data[g][t]['raw_response'] = fr
            data[g][t]['time'] = time
            if norm_func:
                fr = norm_func(time, fr)
                data[g][t]['norm_response'] = fr


    out = {}
    # Loop through group pairs and compare responses
    # Store delta, p-values, u-stats, 
    # Mean response for each tastant for each group
    for g1, g2 in group_pairs:
        k = '%s_vs_%s' % (g1, g2)

        # Compare baselines
        baseline1 = data[g1]['baseline']
        baseline2 = data[g2]['baseline']
        base_u, base_p = mannwhitneyu(baseline1, baseline2,
                                      alternative='two-sided')

        # Store baseline data
        if g1 not in out:
            out[g1] = {}
            out[g1]['mean_baseline'] = np.mean(baseline1)
            out[g1]['sem_baseline'] = sem(baseline1)
            tasty_ps = data[g1]['tasty_ps']
            # Check taste response to all tastants and bonferroni correct
            out[g1]['taste_responsive_all'] = any([p < tasty_alpha/len(tasty_ps) for p in tasty_ps])

        if g2 not in out:
            out[g2] = {}
            out[g2]['mean_baseline'] = np.mean(baseline2)
            out[g2]['sem_baseline'] = sem(baseline2)
            tasty_ps = data[g2]['tasty_ps']
            out[g2]['taste_responsive_all'] = any([p < tasty_alpha/len(tasty_ps) for p in tasty_ps])

        mean_baseline_change, sem_baseline_change = stt.get_mean_difference(baseline1, baseline2)

        if k not in out:
            out[k] = {}

        # Store Baseline stats
        out[k]['baseline_shift'] = False
        out[k]['baseline_p'] = base_p
        out[k]['baseline_u'] = base_u
        out[k]['mean_baseline_change'] = mean_baseline_change
        out[k]['sem_baseline_change'] = sem_baseline_change
        if base_p <= baseline_alpha:
            out[k]['baseline_shift'] = True

        for t in tastants:
            normalize = False
            raw1 = data[g1][t]['raw_response']
            t1 = data[g1][t]['time']
            raw2 = data[g2][t]['raw_response']
            t2 = data[g2][t]['time']

            idx1 = np.where((t1>=time_win[0]) & (t1<=time_win[1]))[0]
            idx2 = np.where((t2>=time_win[0]) & (t2<=time_win[1]))[0]
            if not np.array_equal(t1, t2):
                raise ValueError('Uncomparable time vectors')

            raw_u = np.zeros((len(idx1),))
            raw_p = np.ones(raw_u.shape)

            if 'norm_response' in data[g1][t]:
                norm1 = data[g1][t]['norm_response']
                norm2 = data[g2][t]['norm_response']
                norm_u = raw_u.copy()
                norm_p = raw_p.copy()
                normalize = True

            # Store mean responses and delta
            raw_change = stt.get_mean_difference(raw1, raw2)
            if t not in out[k]:
                out[k][t] = {}

            out[k][t]['raw_mean_change'] = raw_change[0]
            out[k][t]['raw_sem_change'] = raw_change[1]
            out[k][t]['time'] = t1

            if normalize:
                norm_change = stt.get_mean_difference(norm1, norm2)
                out[k][t]['norm_mean_change'] = norm_change[0]
                out[k][t]['norm_sem_change'] = norm_change[1]

            if t not in out[g1]:
                out[g1][t] = {}
                out[g1][t]['raw_response'] = np.mean(raw1, axis=0)
                out[g1][t]['raw_sem'] = sem(raw1, axis=0)
                out[g1][t]['time'] = t1
                out[g1][t]['taste_responsive'] = data[g1][t]['taste_responsive']
                out[g1][t]['taste_responsive_p'] = data[g1][t]['taste_responsive_p']
                out[g1][t]['taste_responsive_u'] = data[g1][t]['taste_responsive_u']
                out[g1][t]['mean_taste_response'] = data[g1][t]['mean_taste_response']
                out[g1][t]['sem_taste_response'] = data[g1][t]['sem_taste_response']
                if normalize:
                    out[g1][t]['norm_response'] = np.mean(norm1, axis=0)
                    out[g1][t]['norm_sem'] = sem(norm1, axis=0)

            if t not in out[g2]:
                out[g2][t] = {}
                out[g2][t]['raw_response'] = np.mean(raw2, axis=0)
                out[g2][t]['raw_sem'] = sem(raw2, axis=0)
                out[g2][t]['time'] = t2
                out[g2][t]['taste_responsive'] = data[g2][t]['taste_responsive']
                out[g2][t]['taste_responsive_p'] = data[g2][t]['taste_responsive_p']
                out[g2][t]['taste_responsive_u'] = data[g2][t]['taste_responsive_u']
                out[g2][t]['mean_taste_response'] = data[g2][t]['mean_taste_response']
                out[g2][t]['sem_taste_response'] = data[g2][t]['sem_taste_response']
                if normalize:
                    out[g2][t]['norm_response'] = np.mean(norm2, axis=0)
                    out[g2][t]['norm_sem'] = sem(norm2, axis=0)


            out[k][t]['p_time'] = t1[idx1]
            for i, idx in enumerate(zip(idx1,idx2)):
                raw_u[i], raw_p[i] = compare_responses(raw1[:, idx[0]],
                                                       raw2[:, idx[1]])
                if normalize:
                    norm_u[i], norm_p[i] = compare_responses(norm1[:, idx[0]],
                                                             norm2[:, idx[1]])

            # Bonferroni correction
            raw_p = raw_p * len(idx1)
            out[k][t]['raw_p'] = raw_p
            out[k][t]['raw_u'] = raw_u
            raw_sig = np.where(raw_p <= alpha)[0]
            if len(raw_sig) > 0:
                out[k][t]['raw_change'] = True
                raw_sig = np.sort(raw_sig)
                out[k][t]['raw_earliest_change'] = out[k][t]['p_time'][raw_sig[0]]
                out[k][t]['raw_latest_change'] = out[k][t]['p_time'][raw_sig[-1]]
            else:
                out[k][t]['raw_change'] = False
                raw_sig = np.sort(raw_sig)
                out[k][t]['raw_earliest_change'] = None
                out[k][t]['raw_latest_change'] = None

            if normalize:
                norm_p = norm_p * len(idx1)
                norm_sig = np.where(norm_p <= alpha)[0]
                if len(norm_sig) > 0:
                    out[k][t]['norm_change'] = True
                    norm_sig = np.sort(norm_sig)
                    out[k][t]['norm_earliest_change'] = out[k][t]['p_time'][norm_sig[0]]
                    out[k][t]['norm_latest_change'] = out[k][t]['p_time'][norm_sig[-1]]
                else:
                    out[k][t]['norm_change'] = False
                    out[k][t]['norm_earliest_change'] = None
                    out[k][t]['norm_latest_change'] = None

                out[k][t]['norm_p'] = norm_p
                out[k][t]['norm_u'] = norm_u

    return out
