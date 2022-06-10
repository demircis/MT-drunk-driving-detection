import matplotlib.pyplot as plt

def create_plot(nrows=1, ncols=1, sharex=False, sharey=False, constrained_layout=False):
    if nrows == 1 and ncols == 1:
        fig, ax = plt.subplots(sharex=sharex, sharey=sharey, constrained_layout=constrained_layout)
        return fig, [ax]
    elif nrows == 1 and ncols > 1:
        fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=sharex, sharey=sharey, constrained_layout=constrained_layout)
        return fig, axes
    elif ncols == 1 and nrows > 1:
        fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=sharex, sharey=sharey, constrained_layout=constrained_layout)
        return fig, axes
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, constrained_layout=constrained_layout)
        return fig, axes

def set_figure_size(fig, x, y):
    fig.set_size_inches(x, y)

def set_plot_xlimits(xlim_min, xlim_max):
    plt.xlim(xmin=xlim_min, xmax=xlim_max)
    
def set_plot_ylimits(ylim_min, ylim_max):
    plt.ylim(ymin=ylim_min, ymax=ylim_max)

def set_ax_xlimits(ax, xlim_min, xlim_max):
    ax.set_xlim(xmin=xlim_min, xmax=xlim_max)
    
def set_ax_ylimits(ax, ylim_min, ylim_max):
    ax.set_ylim(ymin=ylim_min, ymax=ylim_max)

def set_ax_axis_labels(ax, x_label="", y_label="", font_size=16):
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

def set_ax_title(ax, title, font_size=16, font_weight="normal", loc="center", pad=5):
    ax.set_title(title, fontsize=font_size, fontweight=font_weight, loc=loc, pad=pad)

def set_fig_title(fig, title, font_size=18, font_weight="normal", ha="center", va="top"):
    fig.suptitle(title, fontsize=font_size, fontweight=font_weight, ha=ha, va=va)

def set_fig_xlabel(fig, xlabel='', xpos=0.5, ypos=0.01, font_size=14, font_weight="normal", ha="center", va="bottom"):
    fig.supxlabel(t=xlabel, x=xpos, y=ypos, fontsize=font_size, fontweight=font_weight, ha=ha, va=va)

def set_fig_ylabel(fig, ylabel='', xpos=0.02, ypos=0.5, font_size=14, font_weight="normal", ha="left", va="center"):
    fig.supylabel(t=ylabel, x=xpos, y=ypos, fontsize=font_size, fontweight=font_weight, ha=ha, va=va)

def set_plot_title(title, font_size=16, font_weight="normal", loc="center", pad=5):
    plt.title(title, fontsize=font_size, fontweight=font_weight, loc=loc, pad=pad)

def set_ax_xticks(ax, xticks, xlabels, rotation=0):
    ax.set_xticks(xticks, xlabels, rotation=rotation, fontsize=14)

def set_ax_yticks(ax, yticks, ylabels, rotation=0):
    ax.set_yticks(yticks, ylabels, rotation=rotation, fontsize=14)
    
def set_ax_ticks_size(ax, size=14):
    ax.tick_params(labelsize=size)
    
def set_ax_visible_spines(ax, top=True, right=True, bottom=True, left=True):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)

def set_ax_grid_lines(ax, flag=False, which='both', axis='both', style='-', width=1, color='b'):
    ax.grid(b=flag, which=which, axis=axis, linestyle=style,linewidth=width, color=color)

def set_grid_lines(flag=False, which='both', axis='both', style='-', width=1, color='b'):
    plt.grid(b=flag, which=which, axis=axis, linestyle=style, linewidth=width, color=color)
    
def set_ax_bg_color(ax, color):
    ax.set_facecolor(color)

def set_plot_bg_color(color):
    plt.gca().set_facecolor(color)
