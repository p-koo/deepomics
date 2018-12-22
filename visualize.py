import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.misc import imresize
from .utils import normalize_pwm
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams.update({'figure.autolayout': True})


def plot_roc_all(final_roc):
    """Plot ROC curve for each class"""

    fig = plt.figure()
    for i in range(len(final_roc)):
        plt.plot(final_roc[i][0], final_roc[i][1])
    plt.xlabel('False positive rate', fontsize=22)
    plt.ylabel('True positive rate', fontsize=22)
    plt.plot([0, 1], [0, 1], 'k--')
    ax = plt.gca()
    ax.xaxis.label.set_fontsize(17)
    ax.yaxis.label.set_fontsize(17)
    map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
    map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
    plt.tight_layout()
    # plt.legend(loc='best', frameon=False, fontsize=14)
    return fig


def plot_pr_all(final_pr):
    """Plot PR curve for each class"""

    fig = plt.figure()
    for i in range(len(final_pr)):
        plt.plot(final_pr[i][0], final_pr[i][1])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    ax = plt.gca()
    ax.xaxis.label.set_fontsize(17)
    ax.yaxis.label.set_fontsize(17)
    map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
    map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
    plt.tight_layout()
    # plt.legend(loc='best', frameon=False, fontsize=14)
    return fig


def activation_pwm(fmap, X, threshold, window):
    # find regions above threshold
    x, y = np.where(fmap > threshold)

    # sort score
    index = np.argsort(fmap[x, y])[-1:0:-1]
    data_index = x[index].astype(int)
    pos_index = y[index].astype(int)
    activation = fmap[data_index, pos_index]

    # extract sequences with aligned activation
    seq_align = []
    window = int(window / 2)
    num_dims = X.shape[2]
    count_matrix = np.zeros((window * 2, num_dims))

    for i in range(len(pos_index)):

        start_window = pos_index[i] - window
        if start_window < 0:
            start_buffer = np.zeros((-start_window, num_dims))
            start = 0
        else:
            start = start_window

        end_window = pos_index[i] + window
        end_remainder = end_window - fmap.shape[1]
        if end_remainder > 0:
            end = fmap.shape[1]
            end_buffer = np.zeros((end_remainder, num_dims))
        else:
            end = end_window

        seq = X[data_index[i], start:end, :] * activation[i]
        counts = np.ones(seq.shape) * activation[i]

        if start_window < 0:
            seq = np.vstack([start_buffer, seq])
            counts = np.vstack([start_buffer, counts])
        if end_remainder > 0:
            seq = np.vstack([seq, end_buffer])
            counts = np.vstack([counts, end_buffer])

        seq_align.append(seq)
        count_matrix += counts
    seq_align = np.array(seq_align)

    seq_align = np.sum(seq_align, axis=0) / count_matrix
    seq_align[np.isnan(seq_align)] = 0

    return seq_align


def generate_pwm(sess, nntrainer, X, guided_saliency, window=6, layer='conv1d_0_active'):
    data = {'inputs': guided_saliency}
    fmaps = nntrainer.get_activations(sess, data, layer=layer)

    num_filters = fmaps.shape[-1]

    pwm = []
    for i in range(num_filters):
        fmap = np.squeeze(fmaps[:, :, :, i])

        # get threshold
        threshold = np.max(fmap) * 0.8

        pwm.append(activation_pwm(fmap, X, threshold, window))

    return np.array(pwm)


def filter_heatmap(W, norm=True, cmap='hot_r', cbar_norm=True):
    import matplotlib
    if norm:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    else:
        norm = None
    cmap_reversed = matplotlib.cm.get_cmap(cmap)
    im = plt.imshow(W, cmap=cmap_reversed, norm=norm)

    # plt.axis('off');
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, W.shape[1], 1.), minor=True)
    ax.set_yticks(np.arange(-.5, W.shape[0], 1.), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    plt.xticks([])
    if W.shape[0] == 4:
        plt.yticks([0, 1, 2, 3], ['A', 'C', 'G', 'U'], fontsize=16)
    else:
        plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'C', 'G', 'U', 'paired', 'unpaired'], fontsize=16)

    # cbar = plt.colorbar();
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    if cbar_norm:
        cbar.set_ticks([0.0, 0.5, 1.0])


def plot_filter_logos(W, figsize=(10, 7), height=25, nt_width=10, norm=0, alphabet='dna', norm_factor=3, num_rows=None):
    W = np.squeeze(W.transpose([3, 2, 0, 1]))
    num_filters = W.shape[0]
    if not num_rows:
        num_rows = int(np.ceil(np.sqrt(num_filters)))
        num_cols = num_rows
    else:
        num_cols = int(np.ceil(num_filters // num_rows))
    grid = mpl.gridspec.GridSpec(num_rows, num_cols)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)
    fig = plt.figure(figsize=figsize)
    if norm:
        MAX = np.max(W)
    else:
        MAX = None

    for i in range(num_filters):
        plt.subplot(grid[i])
        if norm_factor:
            W_norm = normalize_pwm(W[i], factor=norm_factor, max=MAX)
        else:
            W_norm = W[i]
        logo = seq_logo(W_norm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
        plot_seq_logo(logo, nt_width=nt_width, step_multiple=None)
        # if np.mod(i, num_rows) != 0:
        plt.yticks([])
    return fig


def plot_seq_logo(logo, nt_width=None, step_multiple=None):
    plt.imshow(logo, interpolation='none')
    if nt_width:
        num_nt = logo.shape[1] / nt_width
        if step_multiple:
            step_size = int(num_nt / (step_multiple + 1))
            plt.xticks(
                [step_size * nt_width, step_size * 2 * nt_width, step_size * 3 * nt_width, step_size * 4 * nt_width],
                [str(step_size), str(step_size * 2), str(step_size * 3), str(step_size * 4)])
        else:
            plt.xticks([])
        # plt.yticks([0, 50], ['2.0','0.0'])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
    else:
        plt.imshow(logo, interpolation='none')
        plt.axis('off')


def plot_seq_struct_saliency(X, W, nt_width=100, norm_factor=3):
    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0) != 0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq = X[:4, plot_index]
    pwm_seq_logo = seq_logo(pwm_seq, height=nt_width, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')

    # structure logo
    pwm_struct = X[4:, plot_index]
    pwm_struct = normalize_pwm(pwm_struct, factor=norm_factor)
    pwm_struct_logo = seq_logo_reverse(pwm_struct, height=nt_width * 2, nt_width=nt_width, norm=0, alphabet='pu',
                                       colormap='bw')

    # sequence saliency logo
    seq_saliency = W[:4, plot_index]
    pwm_seq_saliency = normalize_pwm(seq_saliency, factor=norm_factor)
    pwm_seq_saliency_logo = seq_logo(pwm_seq_saliency, height=nt_width * 5, nt_width=nt_width, norm=0, alphabet='rna',
                                     colormap='standard')

    # structure saliency logo
    struct_saliency = W[4:, plot_index]
    pwm_struct_saliency = normalize_pwm(struct_saliency, factor=norm_factor)
    pwm_struct_saliency_logo = seq_logo_reverse(pwm_struct_saliency, height=int(nt_width * 8), nt_width=nt_width,
                                                norm=0, alphabet='pu', colormap='bw')

    # black line
    line1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between seq and struct logo
    spacer2 = np.zeros([20, num_nt * nt_width, 3], dtype=np.uint8)
    spacer2.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt * nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1, spacer2, pwm_seq_logo, spacer2,
                          pwm_struct_logo, line1, spacer6, pwm_struct_saliency_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off')


def plot_pos_saliency(W, height=500, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):
    """
    pwm = normalize_pwm(W, factor=factor)
    pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    """
    # sequence saliency logo
    pwm = normalize_pwm(W, factor=norm_factor)
    logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # plot logo image
    plt.imshow(logo)
    plt.axis('off')


def plot_seq_pos_saliency(X, W, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):
    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0) != 0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq_logo = seq_logo(X[:, plot_index], height=nt_width, nt_width=nt_width, norm=0, alphabet=alphabet,
                            colormap=colormap)

    # sequence saliency logo
    pwm_seq_saliency = normalize_pwm(W[:, plot_index], factor=norm_factor)
    pwm_seq_saliency_logo = seq_logo(pwm_seq_saliency, height=nt_width * 5, nt_width=nt_width, norm=0,
                                     alphabet=alphabet, colormap=colormap)

    # black line
    line1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt * nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1, spacer1, pwm_seq_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off')


def plot_neg_saliency(W, height=500, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):
    """
    num_rows = 2
    grid = mpl.gridspec.GridSpec(num_rows, 1)
    grid.update(wspace=0.2, hspace=0.00, left=0.1, right=0.2, bottom=0.0, top=0.05)

    fig = plt.figure(figsize=figsize);

    plt.subplot(grid[0])
    pwm = normalize_pwm(W, factor=factor)
    pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    if title:
        plt.title(title)

    plt.subplot(grid[1]);
    pwm = normalize_pwm(-W, factor=factor)
    neg_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(neg_logo[::-1,:,:], interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    fig.set_size_inches(150, 14)
    return fig, plt
    """
    num_nt = W.shape[1]

    # sequence logo
    pos_saliency = normalize_pwm(W, factor=norm_factor)
    pos_logo = seq_logo(pos_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # sequence saliency logo
    neg_saliency = normalize_pwm(-W, factor=norm_factor)
    neg_logo = seq_logo_reverse(neg_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet,
                                colormap=colormap)

    # black line
    line1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([30, num_nt * nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pos_logo, spacer6, line1, spacer6, neg_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off')


def plot_seq_neg_saliency(X, W, height=500, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):
    """
    num_rows = 3
    grid = mpl.gridspec.GridSpec(num_rows, 1)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)

    fig = plt.figure(figsize=figsize);

    plt.subplot(grid[0])
    pwm = normalize_pwm(W, factor=factor)
    pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    if title:
        plt.title(title)

    plt.subplot(grid[1])
    logo = seq_logo(np.squeeze(X), height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(logo, interpolation='none');
    plt.axis('off');

    plt.subplot(grid[2]);
    pwm = normalize_pwm(-W, factor=factor)
    neg_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(neg_logo[::-1,:,:], interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    #plt.yticks([0, 100], ['0.0','2.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    return fig, plt
    """
    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0) != 0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq_logo = seq_logo(X[:, plot_index], height=int(height / 5), nt_width=nt_width, norm=0, alphabet=alphabet,
                            colormap=colormap)
    W = W[:, plot_index]

    pos_saliency = normalize_pwm(W, factor=norm_factor)
    pos_logo = seq_logo(pos_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # sequence saliency logo
    neg_saliency = normalize_pwm(-W, factor=norm_factor)
    neg_logo = seq_logo_reverse(neg_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet,
                                colormap=colormap)

    # black line
    line1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt * nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between seq and struct logo
    spacer2 = np.zeros([20, num_nt * nt_width, 3], dtype=np.uint8)
    spacer2.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt * nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pos_logo, spacer6, line1, spacer2, pwm_seq_logo, spacer2, line1, spacer6, neg_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off')


# ------------------------------------------------------------------------------------------------
# helper functions

def fig_options(plt, options):
    if 'figsize' in options:
        fig = plt.gcf()
        fig.set_size_inches(options['figsize'][0], options['figsize'][1], forward=True)
    if 'ylim' in options:
        plt.ylim(options['ylim'][0], options['ylim'][1])
    if 'yticks' in options:
        plt.yticks(options['yticks'])
    if 'xticks' in options:
        plt.xticks(options['xticks'])
    if 'labelsize' in options:
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=options['labelsize'])
        ax.tick_params(axis='y', labelsize=options['labelsize'])
    if 'axis' in options:
        plt.axis(options['axis'])
    if 'xlabel' in options:
        plt.xlabel(options['xlabel'], fontsize=options['fontsize'])
    if 'ylabel' in options:
        plt.ylabel(options['ylabel'], fontsize=options['fontsize'])
    if 'linewidth' in options:
        plt.rc('axes', linewidth=options['linewidth'])


def subplot_grid(nrows, ncols):
    grid = mpl.gridspec.GridSpec(nrows, ncols)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)
    return grid


def load_alphabet(char_path, alphabet, colormap='standard'):
    def load_char(char_path, char, color):
        colors = {}
        colors['green'] = [10, 151, 21]
        colors['red'] = [204, 0, 0]
        colors['orange'] = [255, 153, 51]
        colors['blue'] = [0, 0, 204]
        colors['cyan'] = [153, 204, 255]
        colors['purple'] = [178, 102, 255]
        colors['grey'] = [160, 160, 160]
        colors['black'] = [0, 0, 0]

        img = mpimg.imread(os.path.join(char_path, char + '.eps'))
        img = np.mean(img, axis=2)
        x_index, y_index = np.where(img != 255)
        y = np.ones((img.shape[0], img.shape[1], 3)) * 255
        for i in range(3):
            y[x_index, y_index, i] = colors[color][i]
        return y.astype(np.uint8)

    colors = ['green', 'blue', 'orange', 'red']
    if alphabet == 'dna':
        letters = 'ACGT'
        if colormap == 'standard':
            colors = ['green', 'blue', 'orange', 'red']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    elif alphabet == 'rna':
        letters = 'ACGU'
        if colormap == 'standard':
            colors = ['green', 'blue', 'orange', 'red']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    elif alphabet == 'structure':  # structural profile
        letters = 'PHIME'
        if colormap == 'standard':
            colors = ['blue', 'green', 'orange', 'red', 'cyan']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    elif alphabet == 'pu':  # structural profile

        letters = 'PU'
        if colormap == 'standard':
            colors = ['cyan', 'purple']
        elif colormap == 'bw':
            colors = ['black', 'grey']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    return chars


def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):
    def get_nt_height(pwm, height, norm):

        def entropy(p):
            s = 0
            for i in range(len(p)):
                if p[i] > 0:
                    s -= p[i] * np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt, num_seq))
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                total_height = (np.log2(num_nt) - entropy(pwm[:, i])) * height
            if alphabet == 'pu':
                heights[:, i] = np.floor(pwm[:, i] * np.minimum(total_height, height))
            else:
                heights[:, i] = np.floor(pwm[:, i] * np.minimum(total_height, height * 2))

        return heights.astype(int)

    # get the alphabet images of each nucleotide
    package_directory = os.path.dirname(os.path.abspath(__file__))
    char_path = os.path.join(package_directory, 'chars')
    chars = load_alphabet(char_path, alphabet, colormap)

    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width * num_seq).astype(int)

    if alphabet == 'pu':
        max_height = height
    else:
        max_height = height * 2
    # total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
    logo = np.ones((max_height, width, 3)).astype(int) * 255
    for i in range(num_seq):
        nt_height = np.sort(heights[:, i])
        index = np.argsort(heights[:, i])
        remaining_height = np.sum(heights[:, i])
        offset = max_height - remaining_height

        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))

                # determine location of image
                height_range = range(remaining_height - nt_height[j], remaining_height)
                width_range = range(i * nt_width, i * nt_width + nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range + offset, width_range[m], k] = nt_img[:, m, k]

                remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


def seq_logo_reverse(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):
    def get_nt_height(pwm, height, norm):

        def entropy(p):
            s = 0
            for i in range(len(p)):
                if p[i] > 0:
                    s -= p[i] * np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt, num_seq))
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                total_height = (np.log2(num_nt) - entropy(pwm[:, i])) * height
            if alphabet == 'pu':
                heights[:, i] = np.floor(pwm[:, i] * np.minimum(total_height, height))
            else:
                heights[:, i] = np.floor(pwm[:, i] * np.minimum(total_height, height * 2))

        return heights.astype(int)

    # get the alphabet images of each nucleotide
    package_directory = os.path.dirname(os.path.abspath(__file__))
    char_path = os.path.join(package_directory, 'chars')
    chars = load_alphabet(char_path, alphabet, colormap)

    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width * num_seq).astype(int)

    if alphabet == 'pu':
        max_height = height
    else:
        max_height = height * 2
    # total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
    logo = np.ones((max_height, width, 3)).astype(int) * 255
    for i in range(num_seq):
        nt_height = np.sort(heights[:, i])
        index = np.argsort(heights[:, i])
        remaining_height = 0

        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))

                # determine location of image
                height_range = range(remaining_height, remaining_height + nt_height[j])
                width_range = range(i * nt_width, i * nt_width + nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range, width_range[m], k] = nt_img[:, m, k]

                remaining_height += nt_height[j]
    return logo.astype(np.uint8)
