import matplotlib.pyplot as plt

def create_plot(nrows=None, ncols=None, sharex=False, sharey=False):
    if nrows is None and ncols is None:
        fig, ax = plt.subplots(sharex=sharex, sharey=sharey)
        return fig, [ax]
    elif nrows is None:
        fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=sharex, sharey=sharey)
        return fig, axes
    elif ncols is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=sharex, sharey=sharey)
        return fig, axes
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)
        return fig, axes

def set_figure_size(fig, x, y):
    fig.set_size_inches(x, y)

def set_ax_axis_labels(ax, x_label="", y_label="", font_size=14):
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)


def set_ax_title(ax, title, font_size=14, font_weight="normal", loc="center", pad=5):
    ax.set_title(title, fontsize=font_size, fontweight=font_weight, loc=loc, pad=pad)

def set_plot_title(title, font_size=14, font_weight="normal", loc="center", pad=5):
    plt.title(title, fontsize=font_size, fontweight=font_weight, loc=loc, pad=pad)

def set_plot_xticks(ax, xticks, xlabels, rotation=0):
    ax.set_xticks(xticks, xlabels, rotation=rotation)

def set_plot_yticks(ax, yticks, ylabels, rotation=0):
    ax.set_yticks(yticks, ylabels, rotation=rotation)
    
def set_plot_ticks_size(ax, size=14):
    ax.set_xticks(fontsize=size)
    ax.set_yticks(fontsize=size)
    
def set_visible_spines(ax, top=True, right=True, bottom=True, left=True):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)

def set_grid_lines(ax, flag=False, which='both', axis="both", width=1, color="b"):
    ax.grid(b=flag, which=which, axis=axis, linewidth=width, color=color)
    
def set_ax_bg_color(ax, color):
    ax.set_facecolor(color)