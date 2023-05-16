from configuration import *

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import plot_errors_several_benchmarks, plot_speed_several_benchmarks
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import _simpleaxis

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
    return benchmarks


def plot_figure_individual_motion_benchmark(benchmarks):
    
    fig = plt.figure(figsize=(15,15))
    gs = fig.add_gridspec(4, 6)

    ax_1 = fig.add_subplot(gs[0:3, 0])
    ax_2 = fig.add_subplot(gs[0:3, 1:3])
    ax_3 = fig.add_subplot(gs[0:3, 3])

    benchmarks[0].plot_true_drift(axes=[ax_1, ax_2, ax_3])

    ax_1 = fig.add_subplot(gs[0, 4:6])
    ax_2 = fig.add_subplot(gs[1, 4:6])
    benchmarks[0].plot_motion_corrected_peaks(show_probe=False, axes=[ax_1, ax_2])
    ax_1.set_ylabel('depth (um)')
    ax_2.set_ylabel('depth (um)')

    ax_1 = fig.add_subplot(gs[3, 0:2])
    ax_2 = fig.add_subplot(gs[3, 2:4])
    ax_3 = fig.add_subplot(gs[3, 4:6])


    plot_errors_several_benchmarks(benchmarks, axes=[ax_1, ax_2, ax_3])

    ax = fig.add_subplot(gs[2, 4:6])
    plot_speed_several_benchmarks(benchmarks, ax=ax)

    

    return fig



def plot_small_summary(benchmarks, axes, scaling_probe=1.5):
    
    bench = benchmarks[0]
    ax = axes[0]
        
    for i in range(bench.gt_motion.shape[1]):
        depth = bench.spatial_bins[i]
        ax.plot(bench.temporal_bins, bench.gt_motion[:, i] + depth, color='green', lw=4)

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
    ax.set_xlabel('time (s)')

def plot_firing_rate(benchmark, ax, bin_size=10, color='k', label=None):
    import numpy as np
    times, units = benchmark.gt_sorting.get_all_spike_trains()[0]
    fr = benchmark.recording.get_sampling_frequency()
    nb_units = len(benchmark.gt_sorting.unit_ids)
    time_axis = np.arange(0, benchmark.recording.get_total_duration(), bin_size)
    x, y = np.histogram(times/fr, bins=time_axis)
    rates = x/(bin_size*nb_units)
    ax.plot(y[1:], rates, c=color, label=label)
    _simpleaxis(ax)
    #ax.set_xlabel('time (s)')
    ax.set_ylabel('rate (Hz)')
    ax.set_xlabel('time (s)')


colors = {'rigid' : 'C4', 
          'non-rigid' : 'C5', 
          'bumps' : 'C6', 
          'uniform' : 'C7',
          'bimodal' : 'C8',
          'homogeneous' : 'C9',
          'modulated' : 'C10'}

def plot_summary_errors_several_benchmarks(all_benchmarks):

    fig = plt.figure(figsize=(15,25))
    gs = fig.add_gridspec(len(all_benchmarks)*3, 4)
    count = 0
    
    all_keys = [('rigid', 'uniform', 'homogeneous'),
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
                ('bumps', 'bimodal', 'modulated')]
    
    for key in all_keys:

        benchmarks = all_benchmarks[key]
        
        # print(key)
        ax_1 = fig.add_subplot(gs[3*count:3*count+3, 0])
        #ax_2 = fig.add_subplot(gs[3*count+2, 0])
        #plot_small_summary(list(benchmarks.values()), axes=[ax_1, ax_2])
        color = colors[key[0]]
        ax_1.text(0.5, 0.8, key[0], size=20, rotation=0.,
        ha="center", va="center",
        bbox=dict(boxstyle="round",
               fc=color,
               alpha=0.5)
        )
        
        color = colors[key[1]]
        ax_1.text(0.5, 0.5, key[1], size=20, rotation=0.,
        ha="center", va="center",
        bbox=dict(boxstyle="round",
               fc=color,alpha=0.5
               )
        )
        
        color = colors[key[2]]
        ax_1.text(0.5, 0.2, key[2], size=20, rotation=0.,
        ha="center", va="center",
        bbox=dict(boxstyle="round",
               fc=color,alpha=0.5
               )
        )
        
        _simpleaxis(ax_1)
        ax_1.spines['bottom'].set_visible(False)
        ax_1.spines['left'].set_visible(False)
        ax_1.set_yticks([])
        ax_1.set_xticks([])
        
        ax_1 = fig.add_subplot(gs[3*count:3*count+3, 1])
        ax_2 = fig.add_subplot(gs[3*count:3*count+3, 2])
        ax_3 = fig.add_subplot(gs[3*count:3*count+3, 3])

        if count == 0:
            show_legend = True
        else: 
            show_legend = False
        plot_errors_several_benchmarks(list(benchmarks.values()), axes=[ax_1, ax_2, ax_3], show_legend=show_legend)
        #ax_2.set_yticks([])
        #ax_3.set_yticks([])
        
        
        if count < len(all_benchmarks)-1:
            for ax in [ax_1, ax_2, ax_3]:
                ax.set_xticks([])
                ax.set_xlabel('')
        
        for ax in [ax_1, ax_2, ax_3]:
            ax.set_ylim(0, 20)
        
        count += 1

    return fig




def plot_drift_scenarios(all_benchmarks, scaling_probe=1.25):

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(6, 4)
    #plot_small_summary(benchmarks, axes, scaling_probe=1.5)
    key_1 = ('rigid', 'uniform', 'homogeneous')
    key_2 = ('non-rigid', 'uniform', 'homogeneous')
    key_3 = ('non-rigid', 'uniform', 'modulated')
    key_4 = ('rigid', 'bimodal', 'homogeneous')
    key_5 = ('bumps', 'uniform', 'homogeneous')
    
    bench = list(all_benchmarks[key_1].values())[0]
    c = colors['rigid']

    ax = fig.add_subplot(gs[0:5, 0:3])
        
    for i in range(bench.gt_motion.shape[1]):
        depth = bench.spatial_bins[i]
        if i == 0:
            label = 'rigid'
        else:
            label = None
        ax.plot(bench.temporal_bins, bench.gt_motion[:, i] + depth, color=c, lw=2, label=label)
    
    bench = list(all_benchmarks[key_2].values())[0]
    c = colors['non-rigid']
        
    for i in range(bench.gt_motion.shape[1]):
        depth = bench.spatial_bins[i]
        if i == 0:
            label = 'non-rigid'
        else:
            label = None
        ax.plot(bench.temporal_bins, bench.gt_motion[:, i] + depth, color=c, lw=2, label=label)
    
    bench = list(all_benchmarks[key_5].values())[0]
    c = colors['bumps']
        
    for i in range(bench.gt_motion.shape[1]):
        depth = bench.spatial_bins[i]
        if i == 0:
            label = 'bumps'
        else:
            label = None
        ax.plot(bench.temporal_bins, bench.gt_motion[:, i] + depth, color=c, lw=2, label=label)
    
    ax.legend()
    # ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    _simpleaxis(ax)
    #ax.set_yticks([])
    #ax.spines['left'].set_visible(False)

    channel_positions = bench.recording.get_channel_locations()
    probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
    ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)

    ax.axhline(probe_y_min, color='k', ls='--', alpha=0.5)
    ax.axhline(probe_y_max, color='k', ls='--', alpha=0.5)
    ax.set_ylabel('depth (um)')
    
    bench = list(all_benchmarks[key_2].values())[0]
    ax =  fig.add_subplot(gs[-1, 0:3])
    c = colors['homogeneous']
    plot_firing_rate(bench, ax, color=c, label='homogenous rates')

    bench = list(all_benchmarks[key_3].values())[0]
    c = colors['modulated']
    plot_firing_rate(bench, ax, color=c, label='drift modulated rates')
    
    ax.legend()
    ax = fig.add_subplot(gs[0:5, 3])
    bench = list(all_benchmarks[key_2].values())[0]
    c = colors['uniform']
    ax.hist(bench.gt_unit_positions[30,:], 30, orientation='horizontal', color=c, label='uniform positons', alpha=0.5)
    
    bench = list(all_benchmarks[key_4].values())[0]
    c = colors['bimodal']
    ax.hist(bench.gt_unit_positions[30,:], 30, orientation='horizontal', color=c, label='bimodal positons', alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel('# neurons')
    ax.axhline(probe_y_min, color='k', ls='--', alpha=0.5)
    ax.axhline(probe_y_max, color='k', ls='--', alpha=0.5)
    ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)
    _simpleaxis(ax)
    ax.legend()

    return fig