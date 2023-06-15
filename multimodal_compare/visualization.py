# visualisation related functions
import os
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

marker_styles = ['X', 'v', 'p', '.', '^', '<', '>', '8', 's', ',', '*', 'h', 'H', 'o', 'd', 'P', 'D']

def custom_cmap(n):
    """Create customised colormap for scattered latent plot of n categories.
    Returns colormap object and colormap array that contains the RGB value of the colors.
    See official matplotlib document for colormap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    """
    # first color is grey from Set1, rest other sensible categorical colourmap
    cmap_array = sns.color_palette("Set1", 9)[-1:] + sns.husl_palette(n - 1, h=.6, s=0.7)
    cmap = colors.LinearSegmentedColormap.from_list('mmdgm_cmap', cmap_array)
    return cmap, cmap_array

def is_iterable(o):
    try:
        _ = iter(o)
        return True
    except TypeError as te:
       return False

def t_sne_ploting(df, data, path, labels, mod_names, K=1):
    data_labels = []
    mod_names_list = []
    for ind, mod in enumerate(data):
        if not labels:
            palette = sns.color_palette("hls", len(data))
            data_labels.append(["Modality {} ".format(ind+1) if len(data) > 1 else "Encoded latent vector"]*len(mod))
            mod_names_list.append(mod_names["mod_{}".format(ind+1)])
        else:
            palette = (sns.color_palette("hls", len(set(labels))))
            if len(data) > 1:
                l = ["{} Modality {} ".format(str(i), ind +1) for i in list(labels)]
            else:
                l = ["{}".format(str(i)) for i in list(labels)]
            K_times = [val for val in l for _ in range(K)]
            data_labels.append(K_times)
            mod_names_list += [mod_names["mod_{}".format(ind+1)] for _ in range(len(K_times))]
    if labels:
        labels = np.concatenate(data_labels)
        df["y"] = labels
        df["Classes"] = mod_names_list if len(data) > 1 else mod_names["mod_1"] * len(labels)
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=[x[:-11] for x in df.y.to_list()], palette=palette, data=df,
                             style='Classes', markers=marker_styles[:len(data)])
    else:
        labels = np.concatenate(data_labels)
        df["y"] = labels
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=palette, data=df)
    ax.set(title="T-SNE projection")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=1)
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def t_sne(data, path, dlabels, mod_names, K=1):
    tsne = TSNE(n_components=2, verbose=0, random_state=123, init='random', learning_rate="auto")
    z = tsne.fit_transform(np.concatenate([x.detach().cpu().numpy() for x in data]))
    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    if dlabels is not None and is_iterable(dlabels[0]) and not isinstance(dlabels[0], str):
        for i, _ in enumerate(dlabels[0]):
            labels = [x[i] for x in dlabels]
            dpath = path.replace(".png", "_labels{}.png".format(i))
            t_sne_ploting(df, data, dpath, labels, mod_names, K)
    else:
        t_sne_ploting(df, data, path, dlabels, mod_names, K)

def tensor_to_df(tensor, ax_names=None):
    """Taken from https://github.com/iffsid/mmvae"""
    assert tensor.ndim == 2, "Can only currently convert 2D tensors to dataframes"
    df = pd.DataFrame(data=tensor, columns=np.arange(tensor.shape[1]))
    return df.melt(value_vars=df.columns,
                   var_name=('variable' if ax_names is None else ax_names[0]),
                   value_name=('value' if ax_names is None else ax_names[1]))


def tensors_to_df(tensors, head=None, keys=None, ax_names=None):
    """Taken from https://github.com/iffsid/mmvae"""
    dfs = [tensor_to_df(tensor, ax_names=ax_names) for tensor in tensors]
    df = pd.concat(dfs, keys=(np.arange(len(tensors)) if keys is None else keys))
    df.reset_index(level=0, inplace=True)
    if head is not None:
        df.rename(columns={'level_0': head}, inplace=True)
    return df


def plot_kls_df(df, filepath):
    """Taken from https://github.com/iffsid/mmvae"""
    _, cmap_arr = custom_cmap(df[df.columns[0]].nunique() + 1)
    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.FacetGrid(df, height=12, aspect=2)
        g = g.map(sns.boxplot, df.columns[1], df.columns[2], df.columns[0], palette=cmap_arr[1:],
                  order=None, hue_order=None)
        g = g.set(yscale='log').despine(offset=10)
        plt.legend(loc='best', fontsize='22')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()