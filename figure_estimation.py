from configuration import *

import numpy as np

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import plot_errors_several_benchmarks, plot_speed_several_benchmarks, plot_error_map_several_benchmarks
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import _simpleaxis
from spikeinterface.sortingcomponents.motion_correction import correct_motion_on_peaks
from spikeinterface.widgets import plot_probe_map


from plotting_tools import removeaxis, label_panel


convert_method = {
    'center_of_mass' : 'CoM',
    'monopolar_triangulation': 'MonoT',
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
    gs2 = fig.add_gridspec(2, 5, wspace=0, hspace=0.1)

    ax0 = fig.add_subplot(gs1[0, 0])
    ax1 = fig.add_subplot(gs2[0, 1])
    ax2 = fig.add_subplot(gs2[0, 2])
    ax3 = fig.add_subplot(gs2[0, 3])
    ax4 = fig.add_subplot(gs1[0, 4])


    bench = benchmarks[('monopolar_triangulation', 'decentralized')]
    fs = bench.recording.get_sampling_frequency()
    for d in range(0, bench.gt_motion.shape[1], 2):
        # color = drift_colors[key[0]]
        color = 'm'
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
    ax3.legend(framealpha=1, bbox_to_anchor=(1, 1.2), loc='upper right', ncols=3)
    for ax in (ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([0, 5, 10])
        ax.grid(axis='y', color='0.2', ls='--', alpha=0.2)
        ax.set_ylim(0, 13)
    # ax1.set_yticks([0, 5, 10])
    ax2.set_yticklabels(['', '', ''])
    ax3.set_yticklabels(['', '', ''])


    ax1.set_ylabel('Error [μm]')
    plot_speed_several_benchmarks(list(benchmarks.values()), ax=ax4, colors=method_colors)
    ax4.legend(framealpha=1, bbox_to_anchor=(1, 1.2), loc='upper right')
    ax4.set_ylim(0, 170)
    ax4.set_yticks([0, 50, 100, 150])



    gs3 = fig.add_gridspec(7, 60, wspace=1.1, hspace=0.3)
    
    n = len(benchmarks)
    ncols = 2
    nrows = n // ncols
    # axes = []
    for i, (method, bench) in enumerate(benchmarks.items()):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs3[4 + r, c*28:c*28+28])

        channel_positions = bench.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
        errors = bench.gt_motion - bench.motion
        if c > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Depth [μm]')
        if r < nrows - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time [s]')

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
            label_panel(ax, 'C')
    
    cax = fig.add_subplot(gs3[4:, 59])
    fig.colorbar(im, cax=cax)
    cax.set_ylabel('Error [μm]')



    label_panel(ax1, 'A')
    label_panel(ax4, 'B')
    


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

all_keys = [
            ('rigid', 'uniform', 'homogeneous'),
            #('rigid', 'uniform', 'modulated'),
            #('rigid', 'bimodal', 'homogeneous'),
            ('rigid', 'bimodal', 'modulated'),
            ('non-rigid', 'uniform', 'homogeneous'),
            #('non-rigid', 'uniform', 'modulated'),
            ('non-rigid', 'bimodal', 'homogeneous'),
            #('non-rigid', 'bimodal', 'modulated'),
            ('bumps', 'uniform', 'homogeneous'),
            #('bumps', 'uniform', 'modulated'),
            #('bumps', 'bimodal', 'homogeneous'),
            ('bumps', 'bimodal', 'modulated')
            ]

"""
figure supp1
all_keys = [
            # ('rigid', 'uniform', 'homogeneous'),
            ('rigid', 'uniform', 'modulated'),
            ('rigid', 'bimodal', 'homogeneous'),
            # ('rigid', 'bimodal', 'modulated'),
            # ('non-rigid', 'uniform', 'homogeneous'),
            ('non-rigid', 'uniform', 'modulated'),
            # ('non-rigid', 'bimodal', 'homogeneous'),
            ('non-rigid', 'bimodal', 'modulated'),
            # ('bumps', 'uniform', 'homogeneous'),
            ('bumps', 'uniform', 'modulated'),
            ('bumps', 'bimodal', 'homogeneous'),
            # ('bumps', 'bimodal', 'modulated')
            ]
"""






def plot_summary_errors_several_benchmarks(all_benchmarks, figsize=(15,25)):

    fig = plt.figure(figsize=figsize)
    n = len(all_keys)
    gs1 = fig.add_gridspec(n, 4, wspace=0.7, hspace=0.1)
    gs2 = fig.add_gridspec(n, 4, wspace=0, hspace=0.1)

    
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
            ax3.legend(framealpha=1, bbox_to_anchor=(1, 1.1), loc='upper right', ncols=3)

        ax0.set_xlim(0, 600)
        ax1.set_xlim(0, 600)

        ax0.set_ylim(-700, 700)

        ax0.set_ylabel(drift_title(key))
        label_panel(ax0, 'ABCDEF'[i])


        
    return fig


def plot_drift_scenarios(all_benchmarks, figsize=(15, 15)):
    fig = plt.figure(figsize=figsize)
    n = len(all_keys)
    ncol = 3
    nrow = n // ncol

    subfigs = fig.subfigures(nrow, ncol, wspace=0.07, hspace=0.07)

    for i, key in enumerate(all_keys):
        r = i // ncol
        c = i % ncol
        subfig = subfigs[r, c]

        gs = subfig.add_gridspec(5, 6, wspace=0.0, hspace=0.0)
        
        bench = all_benchmarks[key][('monopolar_triangulation', 'decentralized')]
        fs = bench.recording.get_sampling_frequency()

        ax0 = subfig.add_subplot(gs[0:4, 1:5])
        ax1 = subfig.add_subplot(gs[4, 1:5], )
        ax2 = subfig.add_subplot(gs[0:4, 5])
        ax3 = subfig.add_subplot(gs[0:4, 0])
        
        ax0.sharex(ax1)
        ax0.sharey(ax2)
        ax2.sharey(ax3)

        for d in range(0, bench.gt_motion.shape[1], 2):
            color = drift_colors[key[0]]
            depth = bench.spatial_bins[d]
            ax0.plot(bench.temporal_bins, bench.gt_motion[:, d] + depth, color=color, lw=3, alpha=0.5)
        x = bench.selected_peaks['sample_index']  / fs
        y = bench.peak_locations['y']
        ax0.scatter(x, y, color='k', s=1, marker='.', alpha=0.04)
        ax0.set_xticks([])
        
        ax0.set_yticks([])

        ax1.set_ylim(0, 8)

        color = drift_colors[key[2]]
        plot_firing_rate(bench, ax1, bin_size=10, color=color, label=None)

        color = drift_colors[key[1]]
        ax2.hist(bench.gt_unit_positions[30,:], bins=30,
                 orientation='horizontal', color=color, label='uniform positons',
                 alpha=0.5)
        ax2.set_xticks([])
        ax2.set_yticks([])

        plot_probe_map(bench.recording, ax=ax3)
        ax3.set_xlabel('')
        ax3.set_ylabel('')

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


        removeaxis(ax2)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.set_xticks([])


        # ax1.set_xlim(0, 600)
        ax1.set_ylim(0, 7)

        ax0.set_xlim(0, 600)
        ax0.set_ylim(-750, 700)
        ax2.set_ylim(-750, 750)
        ax3.set_ylim(-750, 750)

        ax3.set_ylabel('Depth [μm]')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Rates [Hz]')
        ax0.set_title(drift_title(key))

        label_panel(ax0, 'ABCDEF'[i])



    return fig


