import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import transforms

def panel_specs(layout, fig=None):
    # default arguments
    if fig is None:
        fig = plt.gcf()

    # format and sanity check grid
    lines = layout.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    linewidths = set(len(line) for line in lines)
    if len(linewidths)>1:
        raise ValueError('Invalid layout (all lines must have same width)')
    width = linewidths.pop()
    height = len(lines)
    panel_letters = set(c for line in lines for c in line)-set('.')
    # find bounding boxes for each panel
    panel_grid = {}
    for letter in panel_letters:
        left = min(x for x in range(width) for y in range(height) if lines[y][x]==letter)
        right = 1+max(x for x in range(width) for y in range(height) if lines[y][x]==letter)
        top = min(y for x in range(width) for y in range(height) if lines[y][x]==letter)
        bottom = 1+max(y for x in range(width) for y in range(height) if lines[y][x]==letter)
        panel_grid[letter] = (left, right, top, bottom)
        # check that this layout is consistent, i.e. all squares are filled
        valid = all(lines[y][x]==letter for x in range(left, right) for y in range(top, bottom))
        if not valid:
            raise ValueError('Invalid layout (not all square)')
    # build axis specs
    gs = gridspec.GridSpec(ncols=width, nrows=height, figure=fig)
    specs = {}
    for letter, (left, right, top, bottom) in panel_grid.items():
        specs[letter] = gs[top:bottom, left:right]
    return specs, gs

def panels(layout, fig=None, simple_axis=True, title=None, projection=False):
    # default arguments
    if fig is None:
        fig = plt.gcf()
    if title is not None:
        fig.suptitle(title, fontsize=16)
    specs, gs = panel_specs(layout, fig=fig)
    axes = {}
    for letter, spec in specs.items():
        if projection:
            axes[letter] = fig.add_subplot(spec, projection='3d')
        else:
            axes[letter] = fig.add_subplot(spec)
        if simple_axis and not projection:
            simpleaxis(axes[letter])
    return axes, gs

def label_panel(ax, letter, *,
                offset_left=0.8, offset_up=0.2, prefix='', postfix='', **font_kwds):
    kwds = dict(fontsize=18)
    kwds.update(font_kwds)
    # this mad looking bit of code says that we should put the code offset a certain distance in
    # inches (using the fig.dpi_scale_trans transformation) from the top left of the frame
    # (which is (0, 1) in ax.transAxes transformation space)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(-offset_left, offset_up, fig.dpi_scale_trans)
    ax.text(0, 1, prefix+letter+postfix, transform=trans, **kwds)

def label_panels(axes, letters=None, **kwds):
    if letters is None:
        letters = axes.keys()
    for letter in letters:
        ax = axes[letter]
        label_panel(ax, letter, **kwds)


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def removeaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_yaxis().tick_right()
    ax.get_xaxis().tick_top()