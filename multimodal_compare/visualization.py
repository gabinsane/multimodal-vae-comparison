# visualisation related functions
import os
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def t_sne(data, runPath, epoch, K, labels):
    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(np.concatenate(data))
    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    data_labels = []
    for ind, mod in enumerate(data):
        if not labels:
            palette = sns.color_palette("hls", len(data))
            data_labels.append(["Modality {}".format(ind+1) if len(data) > 1 else "Encoded latent vector"]*len(mod))
        else:
            palette = sns.color_palette("hls", len(data) * len(set(labels)))
            if len(data) > 1:
                l = ["Class {} Mod {}".format(str(i), ind) for i in list(labels)]
            else:
                l = ["Class {}".format(str(i)) for i in list(labels)]
            data_labels.append([val for val in l for _ in range(K)])
    df["y"] = np.concatenate(data_labels)
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette = palette, data = df).set(title="T-SNE projection")
    p = os.path.join(runPath, "visuals/t-sne_epoch{}.png".format(epoch)) if not ".png" in runPath else runPath
    plt.savefig(p)
    plt.clf()


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
