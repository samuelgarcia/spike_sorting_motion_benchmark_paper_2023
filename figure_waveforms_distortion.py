from configuration import *

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_correction import BenchmarkMotionCorrectionMearec
from spikeinterface import compute_sparsity

import MEArec as mr
import numpy as np

from plotting_tools import removeaxis, label_panel

from figure_estimation import drift_convert

# from configuration import interpolation_methods

convert_interpolation = {
    'nearest' : 'Snap',
    'idw' : 'IDW',
    'kriging' : 'Krig',
}


method_color = {
    # 'static' : 'LimeGreen',
    # 'static' : 'Black',
    # 'drifting' : 'OrangeRed',
    # 'drifting' : 'Black',

    # 'nearest' : 'Turquoise',
    # 'idw' : 'SteelBlue',
    # 'kriging' : 'Goldenrod',

    'static' : 'C7',
    'drifting' : 'C3',
    'nearest' : 'C0',
    'idw' : 'C1',
    'kriging' : 'C2',
  

}

def plot_template_and_std(benchmarks, unit_id, axes=None, figsize=(15,10)):
    if axes is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, wspace=2.2, hspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        axs = [ax0, ax1]
    else:
        ax0, ax1 = axes 
        axs = axes
        fig = axes[0].figure

    bench = next(iter(benchmarks.values()))
    static_we = bench.waveforms['static']
    drifting_we = bench.waveforms['drifting']
    unit_ids = static_we.unit_ids
    unit_ind = list(unit_ids).index(unit_id)

    gain = bench.recordings['static']._kwargs['gain'][0, 0]
    
    sparsity = static_we.sparsity
    mask = sparsity.mask[unit_ind, :]
    
    num_channels = 5
    sparsity2 = compute_sparsity(static_we, method='best_channels',num_channels=num_channels)
    mask2 = sparsity2.mask[unit_ind, :]
    
    static_template = static_we.get_template(unit_id, mode="average", force_dense=True)[:, mask2]
    static_template_std = static_we.get_template(unit_id, mode="std", force_dense=True)[:, mask2]
    static_template_std /= np.sqrt(np.mean(static_template ** 2))

    drifting_template = drifting_we.get_template(unit_id, mode="average", force_dense=True)[:, mask2]
    drifting_template_std = drifting_we.get_template(unit_id, mode="std", force_dense=True)[:, mask2]
    drifting_template_std /= np.sqrt(np.mean(drifting_template ** 2))


    
    for ax in axs:
        for c in range(1, num_channels, 2):
            ax.axvspan(c * static_we.nsamples, (c + 1) * static_we.nsamples, color='k', alpha=0.04)

    for interpolation_method, bench in benchmarks.items():
        we = bench.waveforms['corrected_gt']
        #templates = we.get_all_templates()
        template = we.get_template(unit_id, mode="average", force_dense=True)[:, mask2]
        template_std = we.get_template(unit_id, mode="std", force_dense=True)[:, mask2]
        template_std /= np.sqrt(np.mean(template ** 2))
        
        label = convert_interpolation[interpolation_method]
        ax0.plot(template.T.flatten() / gain, label=label, color=method_color[interpolation_method])
        ax1.plot(template_std.T.flatten(), label=label, color=method_color[interpolation_method])

    ax0.plot(static_template.T.flatten() / gain, label='static', color=method_color['static'])
    ax0.plot(drifting_template.T.flatten() / gain, label='drifting', color=method_color['drifting'], ls='--')
    
    ax1.plot(static_template_std.T.flatten(), label='static', color=method_color['static'])
    ax1.plot(drifting_template_std.T.flatten(), label='drifting', color=method_color['drifting'], ls='--')

    
    # ax0.legend(loc='lower right')
    ax1.set_xticks([])
    ax1.set_xlabel(f'Times for {num_channels} channel')
    ax0.set_ylabel('Template amplitude [mad]')
    ax1.set_ylabel('Template STD\n(normalized by rms)')
    ax1.set_ylim(0, 3.2)
    ax0.set_yticks([-10, -5, 0])

    return fig


def plot_distortion_distribution(benchmarks, metric='norm_std', bins=50, axes=None, figsize=(15,10), with_legend=True):
    if axes is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, wspace=0.0, hspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0:2])
        ax1 = fig.add_subplot(gs[0, 2])
        axs = [ax0, ax1]
    else:
        fig = axes[0].figure
        ax0 = axes[0]
        ax1 = axes[1]
    
    bench = next(iter(benchmarks.values()))

    recgen = mr.load_recordings(bench.mearec_filenames["static"])
    mid = recgen.template_locations.shape[1] // 2
    unit_depth = recgen.template_locations[:, mid, 2]

    
    static_values = bench.distances['static'][metric]
    if np.all(static_values == 0):
        # for some metric like distance for static we have only 0 so the ratio fails
        static_values = 1

    drifting_values = bench.distances['drifting'][metric]

    count, bins_ = np.histogram(drifting_values / static_values, bins=bins)
    ax0.scatter(unit_depth, drifting_values / static_values, color=method_color['drifting'], label='drifting', s=10)
    ax1.plot(count, bins_[:-1], color=method_color['drifting'], label='Drifting', ls='--')

    for interpolation_method, bench in benchmarks.items():
        values = bench.distances['corrected_gt'][metric]
        
        count, bins_ = np.histogram(values / static_values, bins=bins)
                 
        label = convert_interpolation[interpolation_method]
        ax0.scatter(unit_depth, values/ static_values, color=method_color[interpolation_method], label=label, s=10)
        ax1.plot(count, bins_[:-1],  color=method_color[interpolation_method], label=label)
        
    if with_legend:
        ax1.legend()
    ax0.set_ylabel('Waveforms std  / std(static)')
    ax0.set_xlabel('Depth [μm]')
    ax1.set_yticks([])

    ax1.set_ylim(*ax0.get_ylim())
    return fig



def plot_corrected_spike_locations(benchmarks, peaks_locations, axes=None, figsize=(15,10)):
    if axes is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 5, wspace=0.1, hspace=1.0)
        axs = [fig.add_subplot(gs[0, i]) for i in range(5) ]
    else:
        axs = axes
        fig = axs[0].figure
    
    bench = next(iter(benchmarks.values()))
    static_we = bench.waveforms['static']
    sr = static_we.sampling_frequency
    # drifting_we = bench.waveforms['drifting']
    
    # static_depth = static_we.load_extension('spike_locations').get_data()['y']
    # drifting_depth = drifting_we.load_extension('spike_locations').get_data()['y']
    
    peaks, locations = peaks_locations['static']
    axs[0].scatter(peaks['sample_index'] / sr, locations['y'], s=5, marker='o', alpha=0.01, color=method_color['static'])
    
    peaks, locations = peaks_locations['drifting']
    axs[1].scatter(peaks['sample_index'] / sr, locations['y'], s=5, marker='o', alpha=0.01, color=method_color['drifting'])

    for r, (interpolation_method, bench) in enumerate(benchmarks.items()):
        # we = bench.waveforms['corrected_gt']
        # depth = we.load_extension('spike_locations').get_data()['y']
        peaks, locations = peaks_locations[interpolation_method]
        axs[r + 2].scatter(peaks['sample_index'] / sr, locations['y'], s=5, marker='o', alpha=0.01, color=method_color[interpolation_method])
        

    
    for ax in axs:
        ax.set_ylim(-600, 600)
        ax.set_xticks([0, 200, 400])
        ax.set_xlim(0, 600)
        ax.set_ylim(-600, 600)

    for ax in axs[1:]:
        ax.set_yticks([])
    
    axs[0].set_ylabel('Depth [μm]')
        
    return fig
