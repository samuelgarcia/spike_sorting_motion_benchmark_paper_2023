"""
A script to run many motion estimation method.
The run many combinaisons of peak localization x estimation from spikeinterface.
This done using the BenchmarkMotionEstimationMearec class from
spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation

"""
from configuration import *


from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import BenchmarkMotionEstimationMearec
from spikeinterface.sorters import run_sorter
from spikeinterface.extractors import read_mearec



bin_duration_s = 2.


# select_kwargs = {'n_peaks' : 100000}
select_kwargs = None
detect_kwargs = {
    'method' : 'locally_exclusive',
    'detect_threshold' : 10,

}

common_localize_kwargs = {
    'ms_before': .5,
    'ms_after': .5,
}

specific_localize_kwargs = {
    'center_of_mass' :{
        'feature': 'energy',
        'local_radius_um' : 75.,

    },
    'monopolar_triangulation':{
        'feature': 'peak_voltage',
        'local_radius_um' : 75.,
    },
    'grid_convolution': {
        'local_radius_um': 150.,
        'upsampling_um': 5.,
        'percentile' : 10.,
    },
}


job_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True}

common_estimate_motion_kwargs = {
    'bin_duration_s': bin_duration_s,
    'bin_um': 5.,
    'rigid' : False,
    'win_step_um': 50,
    'win_sigma_um': 200.,
    'upsample_to_histogram_bin': False
}
specific_estimate_motion_kwargs = {
    'decentralized': {
        'histogram_depth_smooth_um' : 5.,
        # 'histogram_time_smooth_s' : 6.,
        'histogram_time_smooth_s' :None,
        # 'time_horizon_s': 120.,
        'time_horizon_s': None,
        # 'convergence_method':'lsqr_robust',
        'convergence_method':'lsmr',
        'win_shape' : 'gaussian',
        # 'force_spatial_median_continuity' : False,
        'force_spatial_median_continuity' : True,
        #'torch_device': 'cpu',
    },
    'iterative_template': {
        'win_shape' : 'rect',
    },
}



def run_all_benchmark_estimation():

    for cells_position in cells_positions:
        for cells_rate in cells_rates:
            for drift_mode in drift_modes:
                print()
                print(probename, cells_position, cells_rate, drift_mode)
                mearec_filename = base_folder / 'recordings' / f'{probename}_{drift_mode}_{cells_position}_{cells_rate}.h5'
                print(mearec_filename.exists(), mearec_filename)

                parent_folder = base_folder / 'bench_estimation' / f'{probename}_{drift_mode}_{cells_position}_{cells_rate}'
                parent_folder.mkdir(exist_ok=True, parents=True)

                # run kilosort 2.5
                ks_folder = parent_folder / 'kilosort2.5_folder'
                rec, _ = read_mearec(mearec_filename)
                if not ks_folder.exists():
                    run_sorter('kilosort2_5', rec, output_folder=ks_folder, delete_output_folder=False, verbose=True,
                            do_correction=True)
                ks_motion = np.load(ks_folder / 'sorter_output' / 'motion.npy')
                benchmark_folder = parent_folder / 'kilosort2.5'
                bench = BenchmarkMotionEstimationMearec(mearec_filename,  detect_kwargs={}, select_kwargs={},
                                                        localize_kwargs={}, estimate_motion_kwargs={},
                                                        folder=benchmark_folder, job_kwargs=job_kwargs,
                                                        overwrite=True,  title='kilosort2.5')
                # ks motion is reversed
                bench.motion = -ks_motion
                print(ks_motion.shape)
                nblock = ks_motion.shape[1]
                ylocs = rec.get_channel_locations()[:, 1]
                ystep = (np.max(ylocs) - np.min(ylocs)) / nblock
                bench.spatial_bins = np.arange(nblock) * ystep + ystep / 2. + np.min(ylocs)
                with open(ks_folder / 'spikeinterface_params.json') as f:
                    ks_params = json.load(f)
                    NT = ks_params["sorter_params"]["NT"]
                ks_bins_s = NT / rec.get_sampling_frequency()
                bench.temporal_bins = np.arange(ks_motion.shape[0]) * ks_bins_s + ks_bins_s / 2.
                bench.noise_levels = None
                bench.peaks = None
                bench.selected_peaks = None
                bench.peak_locations = None
                bench.compute_gt_motion()
                # align globally gt_motion and motion to avoid offsets
                bench.motion += np.median(bench.gt_motion - bench.motion)
                bench.save_to_folder()
                

                for localize_method in localize_methods:
                    localize_kwargs = {'method': localize_method}
                    localize_kwargs.update(common_localize_kwargs)
                    localize_kwargs.update(specific_localize_kwargs[localize_method])

                    for motion_method in estimation_methods:


                        benchmark_folder = parent_folder / f'{localize_method}_{motion_method}'

                        if benchmark_folder.exists():
                            print('ALREADY DONE', parent_folder)
                            continue

                        print(benchmark_folder)

                        estimate_motion_kwargs = {'method' : motion_method} 
                        estimate_motion_kwargs.update(common_estimate_motion_kwargs)
                        estimate_motion_kwargs.update(specific_estimate_motion_kwargs[motion_method])
                        
                        ## peak detect + localize + estimate
                        bench = BenchmarkMotionEstimationMearec(mearec_filename, 
                                                                detect_kwargs=detect_kwargs,
                                                                select_kwargs=select_kwargs,
                                                                localize_kwargs=localize_kwargs,
                                                                estimate_motion_kwargs=estimate_motion_kwargs,
                                                                folder=benchmark_folder,
                                                                job_kwargs=job_kwargs,
                                                                overwrite=True, 
                                                                title=f'{localize_method}+{motion_method}')
                        bench.run()

                        # # only the estimate motion
                        # bench = BenchmarkMotionEstimationMearec.load_from_folder(benchmark_folder)
                        # bench.estimate_motion_kwargs.update(estimate_motion_kwargs)
                        # bench.run_estimate_motion()





if __name__ == '__main__':
    run_all_benchmark_estimation()


