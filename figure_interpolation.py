from configuration import *

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_correction import BenchmarkMotionCorrectionMearec
from spikeinterface.sortingcomponents.benchmark.benchmark_motion_correction import plot_distances_to_static


import numpy as np

from plotting_tools import removeaxis, label_panel


count_colors = {
    'num_well_detected': 'Chartreuse',
    'num_overmerged' : 'Chocolate',
    'num_redundant' : 'DarkOrange',
    'num_false_positive' : 'DarkRed',
    'num_bad' : 'DarkViolet',
}


def figure_sorting_accuracies(benchmarks, _mode_best_merge=False, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)

    nrows = len(benchmarks)
    gs = fig.add_gridspec(nrows, 5, wspace=0.4, hspace=0.25)



    for r, (key, bench) in enumerate(benchmarks.items()):
        ax0 = fig.add_subplot(gs[r, :3])
        if _mode_best_merge:
            bench.plot_best_merges_accuracy(mode='ordered_accuracy', ax=ax0, legend=False)
        else:
            bench.plot_sortings_accuracy(mode='ordered_accuracy', ax=ax0, legend=False)
        name = f'drift-{key[-1]}'
        ax0.set_title(name)
        # if r == 0:
        #     ax0.legend(framealpha=1, bbox_to_anchor=(1, 1.5), loc='upper right')
        if r == 1:
            ax0.legend(framealpha=1, bbox_to_anchor=(0, 0), loc='lower left')
        
        if r < nrows -1 :
            ax0.set_xticklabels([])
            ax0.set_xlabel('')
        ax0.set_ylim(0, 1.03)
        ax0.set_xlim(0, 250)

        ax1 = fig.add_subplot(gs[r, 3:])

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
            for c, col in enumerate(columns):
                y = [count[col]]
                x = [j*2 + 1 + c * 0.18]
                color = count_colors[col]
                if j == 0:
                    label = col
                else:
                    label = None

                ax1.bar(x, y, width=0.18, color=color, label=label)
            ax1.axhline(count['num_gt'], color='k', ls='--', alpha=.5)

        if r < nrows -1 :
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
        else:

            ax1.set_xticks(np.arange(n)*2 + 1)
            xlabels = list(bench.comparisons.keys())
            xlabels = [ e.replace(' - ', '\n') for e in xlabels]
            ax1.set_xticklabels(xlabels, rotation=45)
        ax1.set_ylim(0, 350)

        if r == 1:
            ax1.legend(framealpha=1, bbox_to_anchor=(0, 1.15), loc='upper left')

    return fig


def figure_sorting_accuracies_with_best_merge(benchmarks, merging_score=0.2,  figsize=(10, 10)):
    for key, bench in benchmarks.items():
        bench.find_best_merges(merging_score=merging_score)

    fig = figure_sorting_accuracies(benchmarks, _mode_best_merge=True, figsize=figsize)

    return fig



def figure_sorting_accuracies_depth_snr(bench, figsize=(10, 10)):

    fig = plt.figure(figsize=figsize)
    nrows = len(bench.sorter_cases)
    gs = fig.add_gridspec(nrows, 1, wspace=0.4, hspace=0.25)
    
    axes = [fig.add_subplot(gs[r, 0]) for r in range(nrows)]
    bench.plot_sortings_accuracy(mode='depth_snr', axes=axes, legend=False)

    return fig

