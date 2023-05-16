"""
A function on top of MEArec simulator to simulate several drift on
a Neuropixel-128 or Neuronexus-32.

"""


from configuration import *

import elephant
import neo
import quantities as pq


def generate_drift_recordings(base_folder=None, probename='Neuronexus-32',
                              drift_mode='static', cells_position='uniform', cells_rate='homogenenous',
                              erase=False, n_jobs=1):

    assert cells_position in ['uniform', 'bimodal']
    assert cells_rate  in ['homogeneous', 'modulated']
    assert drift_mode in ['static', 'rigid', 'non-rigid', 'bumps']

    cell_folder = mr.get_default_cell_models_folder()
    template_params = mr.get_default_templates_params()


    templates_ids =  None

    # let's generate 10 templates per cell models (total 130 templates)
    template_params["n"] = 100
    template_params["drifting"] = True
    template_params["drift_steps"] = 100
    # this ensures that all cells drift on the same z trajectory, with a small xy variation
    template_params["drift_xlim"] = [0, 0]
    template_params["drift_ylim"] = [0, 0]

    max_drift = 100 # um
    margin = 100
    template_params["drift_zlim"] = [max_drift, max_drift]
    positions = mu.return_mea(probename).positions
    template_params["zlim"] = [positions[:,2].min() - max_drift - margin, positions[:,2].max() + margin]
    template_params["max_drift"] = max_drift
    template_params["overhang"] = 0
    template_params["angle_tol"] = 0
    template_params["min_amp"] = 0
    template_params["check_eap_shape"] = True

    uniform_positions = cells_position == 'uniform'

    density = 2 
    if uniform_positions:
        n_exc = int((density*len(positions))*0.75)
        n_inh = int((density*len(positions))*0.25)
    else:
        n_exc = int((0.66*density*len(positions))*0.75)
        n_inh = int((0.66*density*len(positions))*0.25)

    
    template_params["probe"] = probename

    template_filename = base_folder / 'templates' / f"templates_{probename}.h5"
        
    if not template_filename.is_file():
        print(f"Generating drifting templates for {probename}")
        tempgen = mr.gen_templates(cell_models_folder=cell_folder, params=template_params, 
                                    n_jobs=n_jobs, verbose=True, recompile=True)

        mr.save_template_generator(tempgen, filename=template_filename)
    else:
        print(f"{template_filename} already generated")

    tempgen = mr.load_templates(template_filename)

    center_positions = template_params["drift_steps"] // 2
    ptps = tempgen.templates[:].ptp(axis=(1,2,3))

    possible_inds, = np.nonzero((ptps > 5) & (ptps < 300))

    center_positions_y = tempgen.locations[:, center_positions, 2]

    valid_positions_y = center_positions_y[possible_inds]

    n_bins = 100
    x, y = np.histogram(valid_positions_y, n_bins)
    idx = np.searchsorted(y[1:], valid_positions_y, 'left')

    if uniform_positions:
        np.random.seed(42)
        templates_ids = np.random.choice(possible_inds, n_exc + n_inh, p=x[idx]/x[idx].sum(), replace=False)
    else:
        np.random.seed(42)
        basal = 0.1

        probe_length = positions[:,2].max() - positions[:,2].min()

        mu_1, sigma_1 = 0.15*probe_length + positions[:,2].min(), probe_length/10
        mu_2, sigma_2 = 0.85*probe_length + positions[:,2].min(), probe_length/12

        all_positions = np.linspace(valid_positions_y.min(), 
            valid_positions_y.max(), n_bins)

        x = basal + np.exp(-((all_positions - mu_1)**2)/(2*(sigma_1**2))) + np.exp(-((all_positions - mu_2)**2)/(2*(sigma_2**2)))
        templates_ids = np.random.choice(possible_inds, n_exc + n_inh, p=x[idx]/x[idx].sum(), replace=False)

    # set duration and number of units

    duration = 10*60 # 10 min
    # 10 min
    recordings_params = mr.get_default_recordings_params()
    recordings_params["spiketrains"]["duration"] = duration

    # 100 Excitatory, 20 inhibitory (the main difference is morphology and avg firing rates)

    recordings_params["spiketrains"]["n_exc"] = n_exc
    recordings_params["spiketrains"]["n_inh"] = n_inh
    recordings_params["spiketrains"]["f_exc"] = 5
    recordings_params["spiketrains"]["f_inh"] = 5

    recordings_params["templates"]["min_amp"] = 30
    recordings_params["templates"]["min_dist"] = 20 # um 
    recordings_params["templates"]["smooth_percent"] = 0
    recordings_params["templates"]["n_jitters"] = 3
    recordings_params["recordings"]["filter"] = False
    recordings_params["recordings"]["noise_level"] = 5
    recordings_params["recordings"]["noise_mode"] = "distance-correlated"
    recordings_params['recordings']['chunk_duration'] = 1.
    recordings_params['recordings']['modulation'] = "template"
    recordings_params['recordings']['bursting'] = False
    recordings_params['recordings']['drifting'] = True


    # (optional) set seeds for reproducibility 
    # (e.g. if you want to maintain underlying activity, but change e.g. noise level)
    recordings_params['seeds']['spiketrains'] = 42
    recordings_params['seeds']['templates'] = 42
    recordings_params['seeds']['convolution'] = 42
    recordings_params['seeds']['noise'] = 42

    # no drift for now (first location is used)

    drift_fs = 5.
    drift_dict = mr.get_default_drift_dict()
    drift_dict["drift_fs"] = drift_fs
    drift_dict["t_start_drift"] = 60

    if drift_mode == 'static':
        drift_dict["slow_drift_velocity"] = 0 # um/min
        drift_dicts = [drift_dict]
    elif drift_mode in 'rigid':
        drift_dict["drift_mode_speed"] = "slow"
        drift_dict["slow_drift_waveform"] = "triangluar"
        drift_dict["slow_drift_velocity"] = 30 # um/min
        drift_dict["slow_drift_amplitude"] = 30
        drift_dict["drift_mode_probe"] = 'rigid'
        drift_dicts = [drift_dict]
    
    elif drift_mode == 'non-rigid':
        drift_dict["drift_mode_speed"] = "slow"
        drift_dict["slow_drift_waveform"] = "triangluar"
        drift_dict["slow_drift_velocity"] = 30 # um/min
        drift_dict["slow_drift_amplitude"] = 30
        drift_dict["drift_mode_probename"] = 'non-rigid'
        drift_dict["non_rigid_gradient_mode"] = 'linear'
        drift_dict["non_rigid_linear_min_factor"] = 0.4
        drift_dicts = [drift_dict]

    elif drift_mode == 'bumps':
        # inject some random dumps up/down ward every 0.5 to 1.5s
        # and have a gradient from 0.5 to 1 on y axis
        drift_times = np.arange(0, duration, 1 / drift_fs)

        min_bump_interval = 30.
        max_bump_interval = 90.
        bumps_amplitude_um = 80.

        rg = np.random.RandomState(seed=42)
        diff = rg.uniform(min_bump_interval, max_bump_interval, size=int(duration / min_bump_interval))
        bumps_times = np.cumsum(diff)
        bumps_times = bumps_times[bumps_times<duration]
        
        drift_vector_um = np.zeros(drift_times.size) - bumps_amplitude_um / 2
        for i in range(bumps_times.size - 1):
            ind0 = int(bumps_times[i] * drift_fs )
            ind1 = int(bumps_times[i+1] * drift_fs )
            if i % 2 ==0:
                drift_vector_um[ind0:ind1] = bumps_amplitude_um / 2
            else:
                drift_vector_um[ind0:ind1] = - bumps_amplitude_um /2
        
        positions_y = tempgen.locations[:, center_positions, 2][templates_ids]
        drift_factors = (positions_y - np.min(positions_y)) / (np.max(positions_y) - np.min(positions_y))
        drift_factors = (1 - drift_factors) / 2 + 0.5
        drift_vector_um[drift_times < 60.] = 0.
        dump_drift = {'drift_fs': drift_fs, 'external_drift_times': drift_times,
            'external_drift_vector_um': drift_vector_um, 'external_drift_factors': drift_factors}

        # add a small sinus
        sinus_amplitude_um = 3 # micro meter
        sinus = np.sin(drift_times * np.pi * 0.05) * sinus_amplitude_um
        sinus[drift_times < 60.] = 0.
        sinus_drift = {'drift_fs': drift_fs, 'external_drift_times': drift_times,
            'external_drift_vector_um': sinus, 'external_drift_factors': np.ones(len(templates_ids))}
        drift_dicts = [sinus_drift, dump_drift]


    

    if cells_rate == 'homogeneous':
        # TODO generate spiketrain also here
        spgen = None
    elif cells_rate == 'modulated':
        # reference_file = base_folder / probename / "recordings" / drift_slow["drift_mode_probe"] / cells_position / "homogeneous" / "recordings.h5"
        spiketrains = []
        
        # gt_unit_positions, _ = mr.extract_units_drift_vector(reference_file, time_vector=np.arange(0, duration, 10))
        # rate_vectors = gt_unit_positions - gt_unit_positions[0]
        rate_fs = 10
        rate_times = np.arange(0, duration, 1 / rate_fs)
        av_rate = 5
        min_rate = 0.5
        modulation_freq = 1 / (3*60) # 3min.
        rate_vector = (np.sin(rate_times * np.pi * 2 * modulation_freq) + 0.8) / 1.8 * av_rate
        rate_vector = np.maximum(rate_vector, min_rate)

        rate_vectors = np.ones((rate_times.size, len(templates_ids)))
        rate_vectors[:] = rate_vector[:, None]

        for count in range(len(templates_ids)):
            rate_sig = neo.AnalogSignal(rate_vectors[:, count], units=pq.Hz, sampling_rate=rate_fs*pq.Hz)
            spiketrain = elephant.spike_train_generation.NonStationaryPoissonProcess(rate_sig, refractory_period=5*pq.ms).generate_spiketrain()
            spiketrain.annotate(cell_type='E')
            spiketrains.append(spiketrain)
        print("Spikes trains generated")
        spgen = mr.SpikeTrainGenerator(spiketrains=spiketrains)
    
    filename = base_folder / 'recordings' / f'{probename}_{drift_mode}_{cells_position}_{cells_rate}.h5'
    filename.parent.mkdir(exist_ok = True, parents=True)

    if erase or not filename.exists():
        recgen = mr.gen_recordings(params=recordings_params, tempgen=tempgen, tmp_folder=tmp_folder,
                                   n_jobs=n_jobs, verbose=True, template_ids=templates_ids,
                                   spgen=spgen, drift_dicts=drift_dicts)
        mr.save_recording_generator(recgen, filename=filename)
        print('File', filename, 'was created!')
    else:
        print(f"{filename} already generated")



## exhaustive generation of all cases
for cells_position in cells_positions:
    for cells_rate in cells_rates:
        for drift_mode in ['static'] + drift_modes:
            generate_drift_recordings(base_folder=base_folder,
                                      probename=probename, 
                                      drift_mode=drift_mode,
                                      cells_position=cells_position,
                                      cells_rate=cells_rate,
                                      erase=True,
                                      n_jobs=-1)


# debug 

# generate_drift_recordings(base_folder=base_folder, probename=probename, 
#                          drift_mode='bumps', cells_position='uniform', cells_rate='homogeneous',
#                          erase=True, n_jobs=30)
