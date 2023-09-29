from configuration import *

import numpy as np
import pandas as pd
import xarray as xr

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import plot_errors_several_benchmarks, plot_speed_several_benchmarks, plot_error_map_several_benchmarks
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import _simpleaxis
from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks
from spikeinterface.widgets import plot_probe_map


from plotting_tools import removeaxis, label_panel


convert_method = {
    'center_of_mass' : 'CoM',
    'monopolar_triangulation': 'Mono',
    'grid_convolution': 'Grid',
    'decentralized' : 'Dec',
    'iterative_template': 'Iter',
}

drift_convert = {'rigid': 'ZigZag', 'non-rigid': 'ZigZag (non rigid)', 'bumps': 'Bumps'}

def load_benchmarks(drift, cells_position, cells_rate):
    from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import BenchmarkMotionEstimationMearec
    benchmarks = {}
    
    recording_folder = base_folder / 'recordings'
    parent_folder = base_folder / 'bench_estimation' / f'{probename}_{drift}_{cells_position}_{cells_rate}'
    for localize_method in localize_methods:
        for motion_method in estimation_methods:
            key = localize_method, motion_method
            benchmark_folder = parent_folder / f'{localize_method}_{motion_method}'
            bench = BenchmarkMotionEstimationMearec.load_from_folder(benchmark_folder)
            benchmarks[key] = bench

            # this fix the title
            m1, m2 = bench.title.split('+')
            new_title = convert_method[m1] + ' + ' + convert_method[m2]
            bench.title = new_title

    return benchmarks


def drift_title(key):
    k0, k1, k2 = key
    k0 = drift_convert[k0]

    title = '\n'.join((k0, k1.title(), k2.title()))
    return title



def plot_figure_individual_motion_benchmark(benchmarks, label='', figsize=(15,15), error_lim=20):
    
    fig = plt.figure(figsize=figsize)

    gs1 = fig.add_gridspec(2, 5, wspace=0.7, hspace=0.1)
    gs2 = fig.add_gridspec(2, 5, wspace=0.08, hspace=0.1)

    ax0 = fig.add_subplot(gs1[0, 0])
    ax1 = fig.add_subplot(gs2[0, 1])
    ax2 = fig.add_subplot(gs2[0, 2])
    ax3 = fig.add_subplot(gs2[0, 3])
    ax4 = fig.add_subplot(gs1[0, 4])


    bench = benchmarks[('monopolar_triangulation', 'decentralized')]
    fs = bench.recording.get_sampling_frequency()
    for d in range(0, bench.gt_motion.shape[1], 2):
        # color = drift_colors[key[0]]
        color = 'C6'
        depth = bench.spatial_bins[d]
        ax0.plot(bench.temporal_bins, bench.gt_motion[:, d] + depth, color=color, lw=3, alpha=0.5)
    x = bench.selected_peaks['sample_index']  / fs
    y = bench.peak_locations['y']

    ax1.set_ylim(0, error_lim)

    ax0.scatter(x, y, color='k', s=1, marker='.', alpha=0.04)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel(label)

    method_colors = [plt.get_cmap('tab20')(i) for i in range(6)]
    plot_errors_several_benchmarks(list(benchmarks.values()), axes=[ax1, ax2, ax3], show_legend=False, colors=method_colors)
    ax3.legend(framealpha=1, bbox_to_anchor=(1, 1.45), loc='upper right', ncols=3)
    for ax in (ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([0, 5, 10])
        ax.grid(axis='y', color='0.2', ls='--', alpha=0.2)
        ax.set_ylim(0, 13)
    # ax1.set_yticks([0, 5, 10])
    ax2.set_yticklabels(['', '', ''])
    ax3.set_yticklabels(['', '', ''])
    ax1.set_title("Time error")
    ax2.set_title("Global error")
    ax3.set_title("Depth error")



    ax1.set_ylabel('Error [μm]')
    plot_speed_several_benchmarks(list(benchmarks.values()), detailed=False, ax=ax4, colors=method_colors)
    # ax4.legend(framealpha=1, bbox_to_anchor=(1, 1.45), loc='upper right')
    ax4.set_ylim(0, 170)
    ax4.set_yticks([0, 50, 100, 150])



    # gs3 = fig.add_gridspec(80, 60, wspace=1.1, hspace=0.3)
    gs3 = fig.add_gridspec(80, 60, wspace=1.1, hspace=2.3)
    
    n = len(benchmarks)
    ncols = 2
    nrows = n // ncols
    # axes = []
    for i, (method, bench) in enumerate(benchmarks.items()):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs3[48 + r*10: 48 + r*10 + 8, c*28:c*28+28])

        channel_positions = bench.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
        errors = bench.gt_motion - bench.motion
        if c > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Depth\n[μm]')
        # if r < nrows - 1:
        #     ax.set_xticks([])
        # else:
        #     ax.set_xlabel('Time [s]')
        ax.set_xticks([])

        im = ax.imshow(
            np.abs(errors).T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(bench.temporal_bins[0], bench.temporal_bins[-1], bench.spatial_bins[0], bench.spatial_bins[-1] ),
        )
        # ax.set_title(' + '.join(method).replace('_', ' '))
        ax.set_title(bench.title)
        im.set_clim(0, error_lim)

        if i == 0:
            label_panel(ax, 'D')

    for c in range(ncols):
        ax = fig.add_subplot(gs3[75:80, c*28:c*28+28])
        # color = drift_colors[key[0]]
        color = 'C6'
        ax.plot(bench.temporal_bins, bench.gt_motion[:, -1], color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(np.arange(100, 600, 100))
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, 600)
        ax.set_yticks([])
        ax.set_ylim(-45, 45)

    
    cax = fig.add_subplot(gs3[48:75, 59])
    fig.colorbar(im, cax=cax)
    cax.set_ylabel('Error [μm]')


    label_panel(ax0, 'A')
    label_panel(ax1, 'B')
    label_panel(ax4, 'C')
    


        # axes.append(ax)
    # plot_error_map_several_benchmarks(list(benchmarks.values()), axes=axes) 

    # 

    return fig





def plot_small_summary(benchmarks, axes, scaling_probe=1.5):
    
    bench = benchmarks[0]
    ax = axes[0]
        
    for d in range(bench.gt_motion.shape[1]):
        depth = bench.spatial_bins[d]
        ax.plot(bench.temporal_bins, bench.gt_motion[:, d] + depth, color='green', lw=4)

    # ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    _simpleaxis(ax)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)

    channel_positions = bench.recording.get_channel_locations()
    probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
    ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)

    ax.axhline(probe_y_min, color='k', ls='--', alpha=0.5)
    ax.axhline(probe_y_max, color='k', ls='--', alpha=0.5)
    
    plot_firing_rate(bench, axes[1])
    ax.set_xlabel('Time [s]')



def plot_firing_rate(benchmark, ax, bin_size=10, color='k', label=None):
    times, units = benchmark.gt_sorting.get_all_spike_trains()[0]
    fr = benchmark.recording.get_sampling_frequency()
    nb_units = len(benchmark.gt_sorting.unit_ids)
    time_axis = np.arange(0, benchmark.recording.get_total_duration(), bin_size)
    x, y = np.histogram(times/fr, bins=time_axis)
    rates = x/(bin_size*nb_units)
    ax.plot(y[1:], rates, c=color, label=label)
    # _simpleaxis(ax)
    #ax.set_xlabel('time (s)')
    # ax.set_ylabel('rate (Hz)')
    # ax.set_xlabel('time (s)')


drift_colors = {'rigid' : 'C4', 
          'non-rigid' : 'C5', 
          'bumps' : 'C6', 
          'uniform' : 'C7',
          'bimodal' : 'C8',
          'homogeneous' : 'C9',
          'modulated' : 'C10'
}

# all_keys = [
#             ('rigid', 'uniform', 'homogeneous'),
            #('rigid', 'uniform', 'modulated'),
            #('rigid', 'bimodal', 'homogeneous'),
            # ('rigid', 'bimodal', 'modulated'),
            # ('non-rigid', 'uniform', 'homogeneous'),
            #('non-rigid', 'uniform', 'modulated'),
            # ('non-rigid', 'bimodal', 'homogeneous'),
            #('non-rigid', 'bimodal', 'modulated'),
            # ('bumps', 'uniform', 'homogeneous'),
            #('bumps', 'uniform', 'modulated'),
            #('bumps', 'bimodal', 'homogeneous'),
            # ('bumps', 'bimodal', 'modulated')
            # ]

selected_keys = [
    ('rigid', 'uniform', 'homogeneous'),
    ('rigid', 'bimodal', 'homogeneous'),
    ('rigid', 'uniform', 'modulated'),
    ('non-rigid', 'uniform', 'homogeneous'),
    ('bumps', 'uniform', 'homogeneous'),
]



additional_keys = [
            ## ('rigid', 'uniform', 'homogeneous'),
            ## ('rigid', 'uniform', 'modulated'),
            ## ('rigid', 'bimodal', 'homogeneous'),
            ('rigid', 'bimodal', 'modulated'),
            ## ('non-rigid', 'uniform', 'homogeneous'),
            ('non-rigid', 'uniform', 'modulated'),
            ('non-rigid', 'bimodal', 'homogeneous'),
            ('non-rigid', 'bimodal', 'modulated'),
            ## ('bumps', 'uniform', 'homogeneous'),
            ('bumps', 'uniform', 'modulated'),
            ('bumps', 'bimodal', 'homogeneous'),
            ('bumps', 'bimodal', 'modulated')
            ]






def plot_summary_errors_several_benchmarks(all_benchmarks, all_keys, figsize=(15,25)):

    fig = plt.figure(figsize=figsize)
    n = len(all_keys)
    gs1 = fig.add_gridspec(n, 4, wspace=0.7, hspace=0.1)
    gs2 = fig.add_gridspec(n, 4, wspace=0.06, hspace=0.1)

    
    for i, key in enumerate(all_keys):

        ax0 = fig.add_subplot(gs1[i, 0])
        ax1 = fig.add_subplot(gs2[i, 1])
        ax2 = fig.add_subplot(gs2[i, 2])
        ax3 = fig.add_subplot(gs2[i, 3])

        benchmarks = all_benchmarks[key]

        bench = all_benchmarks[key][('monopolar_triangulation', 'decentralized')]
        fs = bench.recording.get_sampling_frequency()
        for d in range(0, bench.gt_motion.shape[1], 2):
            color = drift_colors[key[0]]
            depth = bench.spatial_bins[d]
            ax0.plot(bench.temporal_bins, bench.gt_motion[:, d] + depth, color=color, lw=3, alpha=0.5)
        x = bench.selected_peaks['sample_index']  / fs
        y = bench.peak_locations['y']

        ax0.scatter(x, y, color='k', s=1, marker='.', alpha=0.04)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.set_xticks([])
        ax0.set_yticks([])

        method_colors = [plt.get_cmap('tab20')(i) for i in range(6)]
        plot_errors_several_benchmarks(list(benchmarks.values()), axes=[ax1, ax2, ax3], show_legend=False, colors=method_colors)
        for ax in (ax1, ax2, ax3):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticks([0, 5, 10])
            ax.grid(axis='y', color='0.2', ls='--', alpha=0.2)
            if i < n - 1:
                ax.set_xticks([])
                ax.set_xlabel('')
                
        
        if i == n - 1:
            ax1.set_xlabel('Time [s]')
            ax3.set_xlabel('Depth [μm]')

        ax1.set_yticks([0, 5, 10])
        ax1.set_ylabel('Error [μm]')
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])

        ax1.set_ylim(0, 13)
        ax2.set_ylim(0, 13)
        ax3.set_ylim(0, 13)
        


        if i == 0:
            ax3.legend(framealpha=1, bbox_to_anchor=(1, 1.45), loc='upper right', ncols=3)
            
            ax1.set_title("Time error")
            ax2.set_title("Global error")
            ax3.set_title("Depth error")


        ax0.set_xlim(0, 600)
        ax1.set_xlim(0, 600)

        ax0.set_ylim(-700, 700)

        ax0.set_ylabel(drift_title(key))
        label_panel(ax0, 'ABCDEFG'[i])

    return fig


def plot_drift_scenarios(all_benchmarks, all_keys, figsize=(15, 15), ncol=3):
    fig = plt.figure(figsize=figsize)
    n = len(all_keys)
    
    nrow = int(np.ceil(n / ncol))

    subfigs = fig.subfigures(nrow, ncol, wspace=0.07, hspace=0.07, squeeze=False)

    for i, key in enumerate(all_keys):
        r = i // ncol
        c = i % ncol
        subfig = subfigs[r, c]

        gs = subfig.add_gridspec(8, 6, wspace=0.0, hspace=0.0)
        
        bench = all_benchmarks[key][('monopolar_triangulation', 'decentralized')]
        fs = bench.recording.get_sampling_frequency()

        ax0 = subfig.add_subplot(gs[1:7, 1:5])  # raster
        ax1 = subfig.add_subplot(gs[7, 1:5], )  # rates
        ax2 = subfig.add_subplot(gs[1:7, 5])   # depth histogram
        ax3 = subfig.add_subplot(gs[1:7, 0])  # probe and units
        ax4 = subfig.add_subplot(gs[0, 1:5])   # motion

        ax5 = subfig.add_subplot(gs[7, 0], )  # yticks for axis rates
        ax6 = subfig.add_subplot(gs[0, 0], )  # yticks for motion

        
        # ax0.sharex(ax1)
        # ax0.sharey(ax2)
        # ax2.sharey(ax3)
        # ax1.sharey(ax5)
        # ax4.sharey(ax6)


        # for d in range(0, bench.gt_motion.shape[1], 3):
        #     color = drift_colors[key[0]]
        #     depth = bench.spatial_bins[d]
        #     ax0.plot(bench.temporal_bins, bench.gt_motion[:, d] + depth, color=color, lw=4, alpha=0.8)
        x = bench.selected_peaks['sample_index']  / fs
        y = bench.peak_locations['y']
        ax0.scatter(x, y, color='k', s=1, marker='.', alpha=0.03)
        ax0.set_xticks([])
        
        ax0.set_yticks([])

        color = drift_colors[key[2]]
        plot_firing_rate(bench, ax1, bin_size=10, color=color, label=None)
        ax1.plot([120, 180], [6, 6], color='k')
        ax1.text(150, 6.1, '1 min', va='bottom', ha='center')
        

        color = drift_colors[key[1]]
        ax2.hist(bench.gt_unit_positions[30,:], bins=30,
                 orientation='horizontal', color=color, label='uniform positons',
                 alpha=0.5)
        ax2.set_xticks([])
        ax2.set_yticks([])

        plot_probe_map(bench.recording, ax=ax3)
        ax3.set_xlabel('')
        ax3.set_ylabel('')
        x = bench.recording.get_channel_locations()[:, 0].max() + 40
        ax3.plot([x, x], [650, 750], color='k')
        ax3.text(x + 5, 700, '100μm', va='center', ha='left')

        ax3.set_yticks([])

        mr_recording = mr.load_recordings(bench.mearec_filename)
        # for loc in mr_recording.template_locations[::2]:
        #     if len(mr_recording.template_locations.shape) == 3:
        #         ax3.plot([loc[0, 1], loc[-1, 1]], [loc[0, 2], loc[-1, 2]], alpha=0.6, lw=2)
        #     else:
        #         ax3.scatter([loc[1]], [loc[2]], alpha=0.7, s=100)
        locs = np.array(mr_recording.template_locations)
        if len(locs.shape) == 3:
            mid = locs.shape[1] // 2
            ax3.scatter(locs[:, mid, 1], locs[:, mid, 2], s=12, alpha=0.7, color='PaleVioletRed')
        else:
            ax3.scatter(locs[:, 1], locs[:, 2], alpha=0.7, s=12, color='PaleVioletRed')


        color = drift_colors[key[0]]
        ax4.plot(bench.temporal_bins, bench.gt_motion[:, 0],color=color,) # color=color, lw=4, alpha=0.8)
        # ax4.plot(bench.temporal_bins, bench.gt_motion[:, -1],color=color,) # color=color, lw=4, alpha=0.8)
        removeaxis(ax4)

        removeaxis(ax2)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)

        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['top'].set_visible(False)

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.set_xticks([])

        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)


        # ax1.set_xlim(0, 600)

        ax1.set_ylim(0, 7)
        ax5.set_ylim(0, 7)
        ax1.set_xticks([])
        ax1.set_yticks([])
        # ax5.set_yticks([0, 2, 4, 6])
        ax5.set_yticks([0, 5])
        ax5.set_xticks([])


        ax4.set_ylim(-45, 45)
        ax6.set_ylim(-45, 45)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax6.set_yticks([-40, 0, 40])
        ax6.set_xticks([])




        ax0.set_xlim(0, 600)
        ax1.set_xlim(0, 600)
        ax4.set_xlim(0, 600)

        ax0.set_ylim(-750, 760)
        ax2.set_ylim(-750, 760)
        ax3.set_ylim(-750, 760)

        if c == 0:
            ax3.set_ylabel('Depth [μm]')
        
            ax5.set_ylabel('Rates [Hz]')
            ax6.set_ylabel('Motion [μm]')
        else:
            ax3.set_yticks([])
            ax5.set_yticks([])
            ax6.set_yticks([])
            ax3.spines['left'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            ax6.spines['left'].set_visible(False)
            
        # if r == nrow - 1:
        # ax1.set_xlabel('Time [s]')
        ax1.spines['bottom'].set_visible(False)

        ax4.set_title(drift_title(key))

        label_panel(ax6, 'ABCDEFG'[i])



    return fig



def export_errors_to_xarray(all_benchmarks):
    # export all errors to xarray
    cases = list(all_benchmarks.keys())
    case0 = cases[0]
    methods = list(all_benchmarks[case0].keys())
    methods0 = methods[0]
    bench0 = all_benchmarks[case0][methods0]
    times = bench0.temporal_bins
    depth = bench0.spatial_bins
    coords = dict(cases=[' '.join(key) for key in  cases],
                  methods=[' '.join(key) for key in  methods],
                  times=times,
                  depth=depth)

    all_errors = xr.DataArray(dims=list(coords.keys()), coords=coords)
    for case, benchmarks in all_benchmarks.items():
        for method, bench in benchmarks.items():
            errors = bench.gt_motion - bench.motion
            all_errors.loc[' '.join(case), ' '.join(method), :, :] = errors

    return all_errors



def benchmarks_to_df(all_benchmarks):
    df = []
    for case, benchmarks in all_benchmarks.items():
        drift_dig, depth_dist, firing = case
        drift_dig = drift_convert[drift_dig]
        drift_dig = drift_dig.replace('(', '\n(')
        depth_dist = depth_dist.title()
        firing = firing.title()
        
        for method, bench in benchmarks.items():
            loc_method, inf_method = method
            error = (bench.gt_motion - bench.motion).flatten()
            abs_error = np.abs(error)
            log_error = np.log(1 + abs_error)
            num_errors = len(error)
            times = bench.temporal_bins
            depths = bench.spatial_bins
            times_flattened = np.repeat(times, len(depths))
            depths_flattened = np.repeat(depths[:, None], len(times), axis=1).T.flatten()
            
            df.append(pd.DataFrame({
                      'Drift signal': [drift_dig] * num_errors,
                      'Depth distribution': [depth_dist] * num_errors,
                      'Firing rate': [firing] * num_errors,
                      'Localization method': [loc_method] * num_errors,
                      'Inference method': [inf_method] * num_errors,
                      'time': times_flattened,
                      'depth': depths_flattened,
                      'error': error,
                      'abs_error': abs_error,
                      'log_error': log_error}))
        
    df = pd.concat(df, axis=0)
    return df


