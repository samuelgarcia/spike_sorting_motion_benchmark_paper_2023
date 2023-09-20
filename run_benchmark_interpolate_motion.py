"""
This script run some motion correction (interpolation) on drifting recording and then run kilosort2.5.
This done using the ground truth motion vector or the estimated motion vector.
This also done for comparison on a non-drifting recording with same units and spiketrains.
"""

from configuration import *

import scipy.interpolate


from spikeinterface.sortingcomponents.benchmark.benchmark_motion_interpolation import BenchmarkMotionInterpolationMearec
from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import BenchmarkMotionEstimationMearec


# job_kwargs = {'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True}
job_kwargs = {'chunk_duration' : '1s', 'n_jobs' : 10, 'progress_bar':True}

common_correct_motion_kwargs = {
        # 'spatial_interpolation_method' : method,
        'border_mode' : 'force_extrapolate',
        'sigma_um': [20., 30],
    }

do_preprocessing = True
#do_preprocessing = False

delete_output_folder = False

n_jobs_sorter = 0.8

verbose_sorter = True
sorter_cases = [
    {
        'label': 'No drift - No motion correction',
        'sorter_name': 'kilosort2_5',
        'recording': 'raw_static',
        'sorter_params':{
            #'docker_image': True,
            'do_correction': False,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,
            'skip_kilosort_preprocessing': False,
            # 'scaleproc': 200,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,            
        }
    },
    {
        'label': 'No drift - Motion correction using KS2.5',
        'sorter_name': 'kilosort2_5',
        'recording': 'raw_static',
        'sorter_params':{
            #'docker_image': True,
            'do_correction': True,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,
            'skip_kilosort_preprocessing': False,
            # 'scaleproc': 200,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,            
        }
    },

    {
        'label': 'Drifting - Motion correction using GT',
        'sorter_name': 'kilosort2_5',
        'recording': 'corrected_gt',
        'sorter_params':{
            # 'docker_image': True,
            'do_correction': False,
            'skip_kilosort_preprocessing': True,
            'scaleproc': 200,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,
        }
    },
    {
        'label': 'Drifting - Motion correction using estimated',
        'sorter_name': 'kilosort2_5',
        'recording': 'corrected_estimated',
        'sorter_params':{
            # 'docker_image': True,
            'do_correction': False,
            'skip_kilosort_preprocessing': True,
            'scaleproc': 200,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,
        }
    },
    {
        'label' : 'Drifting - Motion correction using KS2.5',
        'sorter_name': 'kilosort2_5',
        'recording': 'raw_drifting',
        'sorter_params':{
            #'docker_image': True,
            'do_correction': True,
            'skip_kilosort_preprocessing': False,
            # 'scaleproc': 200,
            'verbose': verbose_sorter,
            'n_jobs':n_jobs_sorter,
        }
    },
]

interpolation_methods = ['kriging', 'idw', 'nearest', ]
# interpolation_methods = ['nearest',]
# interpolation_methods = ['kriging', ]


bin_duration_s = 2.
rigid = False
win_step_um = 50.
win_sigma_um =  150.
margin_um = 0

waveforms_kwargs = dict(
    ms_before=1.0,
    ms_after=3.0,
    max_spikes_per_unit=700,
)


correction_cases = [
    # ('uniform', 'homogeneous', 'rigid'),
    # ('uniform', 'homogeneous', 'non-rigid'),
    ('uniform', 'homogeneous', 'bumps'),
]

def compute_gt_motion(mearec_filename, bin_duration_s, win_step_um, win_sigma_um, margin_um):
    """
    Re interpolate GT motion vector of every cells from a mearec file into global GT motion vector
    given a bin step in time and spatial bin.
    """
    recording, _ = si.read_mearec(mearec_filename)

    template_locations = np.array(mr.load_recordings(mearec_filename).template_locations)
    assert len(template_locations.shape) == 3
    mid = template_locations.shape[1] //2
    unit_mid_positions = template_locations[:, mid, 2]

    duration = recording.get_total_duration()
    # center of the window
    temporal_bins = np.arange(bin_duration_s / 2., duration, bin_duration_s)
    
    contact_pos = recording.get_probe().contact_positions[:, 1]

    min_ = np.min(contact_pos) - margin_um
    max_ = np.max(contact_pos) + margin_um
    num_non_rigid_windows = int((max_ - min_) // win_step_um)
    border = ((max_ - min_)  %  win_step_um) / 2
    spatial_bins = np.arange(num_non_rigid_windows + 1) * win_step_um + min_ + border
    
    gt_unit_positions, _ = mr.extract_units_drift_vector(mearec_filename, time_vector=temporal_bins)
    unit_motions = gt_unit_positions - unit_mid_positions
    # unit_positions = np.mean(gt_unit_positions, axis=0)


    gt_motion = np.zeros((temporal_bins.size, spatial_bins.size))
    for t in range(gt_unit_positions.shape[0]):
        f = scipy.interpolate.interp1d(unit_mid_positions, unit_motions[t, :], fill_value="extrapolate")
        gt_motion[t, :] = f(spatial_bins)

    return gt_motion, temporal_bins, spatial_bins


def run_all_benchmark_correction():

    for cells_position, cells_rate, drift_mode  in correction_cases:

        print()
        print(probename, cells_position, cells_rate, drift_mode)
        
        static_mearec_filename = base_folder / 'recordings' / f'{probename}_static_{cells_position}_{cells_rate}.h5'
        drift_mearec_filename = base_folder / 'recordings' / f'{probename}_{drift_mode}_{cells_position}_{cells_rate}.h5'

        # print(static_mearec_filename.exists(), static_mearec_filename)
        # print(drift_mearec_filename.exists(), drift_mearec_filename)

        gt_motion, temporal_bins, spatial_bins = compute_gt_motion(drift_mearec_filename, bin_duration_s, win_step_um, win_sigma_um, margin_um)

        # take estimated motion from the other benchmark
        localize_method, motion_method = 'monopolar_triangulation', 'decentralized'
        benchmark_motion_folder = base_folder / 'bench_estimation' / f'{probename}_{drift_mode}_{cells_position}_{cells_rate}' / f'{localize_method}_{motion_method}'
        bench = BenchmarkMotionEstimationMearec.load_from_folder(benchmark_motion_folder)

        assert np.array_equal(bench.spatial_bins, spatial_bins)

        # interpolate estimated motion on this temporal_bins
        estimated_motion = np.zeros_like(gt_motion)
        for i in range(spatial_bins.size):
            f = scipy.interpolate.interp1d(bench.temporal_bins, bench.motion[:, i], fill_value="extrapolate")
            estimated_motion[:, i] = f(temporal_bins)

        parent = None
        for count, method in enumerate(interpolation_methods):
            print(method)
            correct_motion_kwargs = {}
            correct_motion_kwargs.update(common_correct_motion_kwargs)
            correct_motion_kwargs['spatial_interpolation_method'] = method
            
            benchmark_folder = base_folder / 'bench_interpolation' / f'{probename}_{drift_mode}_{cells_position}_{cells_rate}' / f'{method}'
            benchmark_folder.parent.mkdir(exist_ok=True, parents=True)

            # if benchmark_folder.exists():
            #    print('ALREADY DONE', benchmark_folder)
            #    continue

            bench = BenchmarkMotionInterpolationMearec(drift_mearec_filename, static_mearec_filename,
                                                    gt_motion, estimated_motion, temporal_bins, spatial_bins,
                                                    do_preprocessing=do_preprocessing,
                                                    delete_output_folder=delete_output_folder,
                                                    correct_motion_kwargs=correct_motion_kwargs,
                                                    sorter_cases=sorter_cases,
                                                    folder=benchmark_folder,
                                                    waveforms_kwargs=waveforms_kwargs,
                                                    job_kwargs=job_kwargs,
                                                    overwrite=True, 
                                                    title=f'{method}',
                                                    parent_benchmark=parent)
            if count == 0:
                 # 'kriging' : run sorter + waveforms
                # bench.extract_waveforms()
                # bench.save_to_folder()
                bench.run_sorters()
                bench.save_to_folder()
            else:
                # 'idw', 'nearest' : run only waveforms
                # bench.extract_waveforms()
                # bench.save_to_folder()
                pass

            if parent is None:
                parent = bench




if __name__ == '__main__':
    run_all_benchmark_correction()

