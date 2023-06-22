from configuration import *

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_correction import BenchmarkMotionCorrectionMearec
from spikeinterface.sortingcomponents.benchmark.benchmark_motion_correction import plot_distances_to_static

from spikeinterface.qualitymetrics import compute_quality_metrics

import MEArec as mr
import numpy as np

from plotting_tools import removeaxis, label_panel

from figure_estimation import drift_convert



convert_sorter_cases = {
    'No drift - No motion correction': 'Static - No interpolation',
    'Drifting - Motion correction using GT': 'Using GT',
    'Drifting - Motion correction using estimated': 'Using Mono+Dec',
    'Drifting - Motion correction using KS2.5': 'Using KS2.5',
}

color_sorter_cases = [plt.get_cmap('Set1')(i) for i in range(6)]


def figure_sorting_accuracies(benchmarks, accuracy_thresh=0.95,
                              _mode_best_merge=False, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)

    nrows = len(benchmarks)
    gs = fig.add_gridspec(nrows + 1, 12, wspace=2.2, hspace=0.25)


    axes2 = []
    for r, (key, bench) in enumerate(benchmarks.items()):
        
        ###

        ax0 = fig.add_subplot(gs[r, :4])
        if _mode_best_merge:
            bench.plot_best_merges_accuracy(mode='ordered_accuracy', ax=ax0, legend=False, colors=color_sorter_cases)
        else:
            bench.plot_sortings_accuracy(mode='ordered_accuracy', ax=ax0, legend=False, colors=color_sorter_cases)
        name = f'drift-{key[-1]}'
        name = drift_convert[key[-1]]
        # ax0.set_title(name)
        # if r == 0:
        #     ax0.legend(framealpha=1, bbox_to_anchor=(1, 1.5), loc='upper right')

        # if r == 1:
        #     ax0.legend(framealpha=1, bbox_to_anchor=(0, 0), loc='lower left')
        if r == 2:
            ax0.legend(framealpha=1, bbox_to_anchor=(0, -0.8), loc='upper left')
        
        if r < nrows -1 :
            ax0.set_xticklabels([])
            ax0.set_xlabel('')
        ax0.set_ylim(0, 1.03)
        ax0.set_xlim(0, 250)

        ###

        ax1 = fig.add_subplot(gs[r, 4:8])
        ax1.set_title(name)

        counts = []
        if _mode_best_merge:
            comparisons = bench.merged_comparisons
        else:
            comparisons = bench.comparisons
        n = len(comparisons)
        for j, (k , comp) in enumerate(comparisons.items()):
            count = comp.count_units_categories()
            columns = ['num_well_detected', 'num_overmerged', 'num_redundant',
                        'num_false_positive', 'num_bad']
            count_colors = [plt.get_cmap('Accent')(c) for c in range(5)]
            for c, col in enumerate(columns):
                y = [count[col]]
                x = [j*2 + 1 + c * 0.18]
                color = count_colors[c]
                if j == 0:
                    label = col
                else:
                    label = None

                ax1.bar(x, y, width=0.18, color=color, label=label)
            ax1.axhline(count['num_gt'], color='k', ls='--', alpha=.5)
            ax1.set_yticks([0, 100, 200, 300])
            ax1.set_ylabel('Unit count')

        if r < nrows -1 :
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
        else:

            ax1.set_xticks(np.arange(n)*2 + 1)
            xlabels = list(bench.comparisons.keys())
            xlabels = [ e.replace(' - ', '\n') for e in xlabels]
            ax1.set_xticklabels(xlabels, rotation=35)
        ax1.set_ylim(0, 350)

        # if r == 1:
        #     ax1.legend(framealpha=1, bbox_to_anchor=(0, 1.15), loc='upper left')
        if r == 2:
            ax1.legend(framealpha=1, bbox_to_anchor=(0, -0.8), loc='upper left')
        

        ###

        ax2 = fig.add_subplot(gs[r, 8:12])
        axes2.append(ax2)


        metrics = compute_quality_metrics(bench.waveforms["static"], metric_names=["snr"], load_if_exists=True)
        snr = metrics["snr"].values
        template_locations = np.array(mr.load_recordings(bench.mearec_filenames["drifting"]).template_locations)
        assert len(template_locations.shape) == 3
        mid = template_locations.shape[1] // 2
        unit_depth = template_locations[:, mid, 2]
        chan_locations = bench.recordings["drifting"].get_channel_locations()


        k1 = convert_sorter_cases['No drift - No motion correction']
        k2 = convert_sorter_cases['Drifting - Motion correction using GT']
        if _mode_best_merge:
            acc_static = bench.merged_accuracies[k1]
            acc_drift = bench.merged_accuracies[k2]
        else:
            acc_static = bench.accuracies[k1]
            acc_drift = bench.accuracies[k2]


        # mask = acc_static >= accuracy_thresh
        mask = acc_static >= accuracy_thresh

        color_values = acc_drift[mask] - acc_static[mask]
        points = ax2.scatter(unit_depth[mask], snr[mask], c=color_values)
        # points.set_clim(0.0, 1.0)
        points.set_clim(-1.0, 0)
        ax2.set_title(label)
        ax2.axvline(np.min(chan_locations[:, 1]), ls="--", color="k")
        ax2.axvline(np.max(chan_locations[:, 1]), ls="--", color="k")
        ax2.set_ylabel("snr")
        if r < nrows -1 :
            ax2.set_xticklabels([])
            ax2.set_xlabel('')
        else:
            ax2.set_xlabel('Depth [μm]')
        
        acc_static = np.sort(acc_static)[::-1]
        ind_last = np.nonzero(acc_static < accuracy_thresh)[0][0]
        text = f'Acc={accuracy_thresh:0.2}\nn={ind_last}'
        ax0.annotate(text, xy=(ind_last, accuracy_thresh),
                    xytext=(ind_last + 10, accuracy_thresh-0.3 ),
                    arrowprops=dict(arrowstyle="->"),
                    color='Black',
                    fontsize=10
                    )

        for ax in (ax0, ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        
    gs2 = fig.add_gridspec(24, 12, wspace=2.2, hspace=0.25)
    cax = fig.add_subplot(gs2[22, 8:12])
    cbar = fig.colorbar(points, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('Accuracy loss')

    # cbar = fig.colorbar(points, ax=axes2, location='bottom', shrink=0.9)
    # cbar.ax.set_xlabel('Accuracy loss')




    return fig


def figure_sorting_accuracies_with_best_merge(benchmarks, merging_score=0.2,  **kargs):
    for key, bench in benchmarks.items():
        bench.find_best_merges(merging_score=merging_score)

    fig = figure_sorting_accuracies(benchmarks, _mode_best_merge=True, **kargs)

    return fig



def figure_sorting_accuracies_depth_snr(bench, figsize=(10, 10)):

    fig = plt.figure(figsize=figsize)
    nrows = len(bench.sorter_cases)
    gs = fig.add_gridspec(nrows, 1, wspace=0.4, hspace=0.25)
    
    axes = [fig.add_subplot(gs[r, 0]) for r in range(nrows)]
    bench.plot_sortings_accuracy(mode='depth_snr', axes=axes, legend=False)

    return fig



def figure_waveform_distortion(bench, figsize=(10, 10)):

    fig = plt.figure(figsize=figsize)
    


    return fig

