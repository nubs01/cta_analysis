import blechpy, os

def pre_process(dat):
    if isinstance(dat, str):
        fd = dat
        dat = blechpy.load_dataset(fd)
        if dat is None:
            dat = blechpy.dataset(fd)

    status = dat.process_status
    if not status['initialize_parameters']:
        dat.initParams()

    if not status['extract_data']:
        dat.extract_data()

    if not status['create_trial_list']:
        dat.create_trial_list()

    if not status['mark_dead_channels']:
        dat.mark_dead_channels()

    if not status['common_average_reference']:
        dat.common_average_reference()

    if not status['spike_detection']:
        dat.detect_spikes()


def clustering(exp):
    if isinstance(exp, str):
        fd = exp
        exp = blechpy.load_experiment(fd)
        if exp is None:
            exp = blechpy.experiment(fd)

    dat = blechpy.load_dataset(exp.recording_dirs[0])
    clustering_params = dat.clustering_params.copy()
    clustering_params['clustering_params']['Max Number of Clusters'] = 15
    exp.cluster_spikes(custom_params=clustering_params, umap=True)
    for fd in exp.recording_dirs:
        dat = blechpy.load_dataset(fd)
        dat.cleanup_clustering()

def post_process(dat):
    if isinstance(dat, str):
        fd = dat
        dat = blechpy.load_dataset(fd)
        if dat is None:
            raise FileNotFoundError('Dataset for %s not found.' % fd)

    dat.cleanup_lowSpiking_units(min_spikes=100)
    dat.units_similarity(shell=True)
    dat.make_unit_plots()
    dat.make_unit_arrays()
    dat.make_psth_arrays()


def main(anim):
    rec_dirs, anim_dir = get_rec_dirs(anim)
    for fd in rec_dirs:
        pre_process(fd)

    clustering(anim_dir)

    ## Sort spike manually 1 electrode and 1 recording at a time

    for fd in rec_dirs:
        dat = blechpy.load_dataset(fd)
        # Sort each electrode
        # root, ssg = dat.sort_spikes(electrode) 
        dat.cleanup_lowSpiking_units(min_spikes=100)
        dat.units_similarity(shell=True)

    ## Check units similarity and delete necessary units
    # Then
    for fd in rec_dirs:
        dat = blechpy.load_dataset(fd)
        dat.make_unit_plots()
        dat.make_unit_arrays()
        dat.make_psth_arrays()

    exp.detect_held_units()
    exp.save()



