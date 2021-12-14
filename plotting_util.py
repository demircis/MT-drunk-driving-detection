import matplotlib.pyplot as plt

def create_plot(nrows=None, ncols=None, sharex=False, sharey=False):
    if nrows is None or ncols is None:
        fig, ax = plt.subplots(sharex=sharex, sharey=sharey)
        return fig, [ax]
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)
        return fig, axes

def set_figure_size(fig, x, y):
    fig.set_size_inches(x, y)

def set_plot_axis_labels(ax, x_label="", y_label="", font_size=14):
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    
def set_plot_title(ax, title, font_size=14, font_weight="normal", loc="center"):
    ax.set_title(title, fontsize=font_size, fontweight=font_weight, loc=loc, pad=5)

def set_plot_xticks(ax, xticks, xlabels, rotation=0):
    ax.set_xticks(xticks, xlabels, rotation=rotation)
    
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
    
def set_plot_bg_color(ax, color):
    ax.set_facecolor(color)