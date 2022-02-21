# visualisation related functions
import warnings, os
warnings.filterwarnings("ignore")
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D


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


def own_cmap(n, copies=2):
    """Create customised colormap for scattered latent plot of n categories.
    Returns colormap object and colormap array that contains the RGB value of the colors.
    See official matplotlib document for colormap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    """
    # first color is grey from Set1, rest other sensible categorical colourmap
    #cmap_array = sns.color_palette("Spectral")[-1:] + sns.husl_palette(n - 1, h=.6, s=0.7)
    cmap_all = [[[0.7734375, 0.2109375, 0.43359375]] * copies, [[0.8828125, 0.2421875, 0.2265625]]* copies,
                [[0.6015625, 0.171875, 0.04296875]]* copies, [[0.16796875, 0.19921875, 0.48828125]]* copies,
                [[0.59375, 0.59375, 0.59375]]* copies, [[0.99609375, 0.4375, 0.26171875]]* copies,
                [[0.12890625, 0.6328125, 0.69140625]]* copies, [[0.01171875, 0.01171875, 0.01171875]]* copies,
                [[0.1796875, 0.48828125, 0.1953125]]* copies, [[0.1171875, 0.53125, 0.89453125]]* copies]
    cmap_all = [val for sublist in cmap_all for val in sublist]
    marks = ["o", "v"]*20
    if n < 20:
        del cmap_all[2:6]
    cmap_array = cmap_all[:n]
    cmap = colors.LinearSegmentedColormap.from_list('mmdgm_cmap', cmap_array)
    return cmap, cmap_array, marks

def t_sne(data, runPath, epoch, K, labels):
    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(np.concatenate(data))
    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    data_labels = []
    for ind, mod in enumerate(data):
        if not labels:
            data_labels.append(["Modality {}".format(ind+1) if len(data) > 1 else "Encoded latent vector"]*len(mod))
        else:
            if len(data) > 1:
                l = ["Class {} Mod {}".format(str(i), ind) for i in list(labels)]
            else:
                l = ["Class {}".format(str(i)) for i in list(labels)]
            data_labels.append([val for val in l for _ in range(K)])
    df["y"] = np.concatenate(data_labels)
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette = sns.color_palette("hls", len(data)*len(set(labels))), data = df).set(title="T-SNE projection")
    plt.savefig(os.path.join(runPath, "visuals/t-sne_epoch{}.png".format(epoch)))
    plt.clf()


def embed_umap(data):
    """data should be on cpu, numpy"""
    x = 55
    while x != 10:
        try:
            embedding = UMAP(metric='euclidean', n_neighbors=x,)
            emb = embedding.fit_transform(data)
        except:
            x -= 15
    print("Succeeded with {} neighbors".format(x))
    return emb

def plot_embeddings(emb, emb_l, labels, filepath):
    cmap_obj, cmap_arr = custom_cmap(n=len(labels))
    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=emb_l, cmap=cmap_obj, s=25, alpha=0.2, edgecolors='none')
    l_elems = [Line2D([0], [0], marker='o', color=cm, label=l, alpha=0.5, linestyle='None')
               for (cm, l) in zip(cmap_arr, labels)]
    plt.legend(frameon=False, loc=2, handles=l_elems)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def plot_embeddings_enc(emb, emb_l, labels, filepath):
    cmap_obj, cmap_arr, marks = own_cmap(n=len(labels))
    l_elems = [Line2D([0], [0], marker=m, color=cm, label=l, alpha=0.5, linestyle='None')
               for (cm, l, m) in zip(cmap_arr, labels, marks)]
    if emb.shape[1] == 2:
        plt.scatter(emb[:int(emb.shape[0]/2), 0], emb[:int(emb.shape[0]/2), 1], marker="o", c=emb_l[:int(emb.shape[0]/2)], cmap=cmap_obj, s=25, alpha=0.85, edgecolors='none')
        plt.scatter(emb[int(emb.shape[0]/2):, 0], emb[int(emb.shape[0]/2):, 1], marker="v", c=emb_l[int(emb.shape[0]/2):], cmap=cmap_obj, s=25, alpha=0.85, edgecolors='none')
        plt.legend(loc='bottom left', bbox_to_anchor=(1, 0.5), handles=l_elems)
        plt.show()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    else:
        ax = plt.axes(projection='3d')
        x, y, z = np.array(emb[:int(emb.shape[0]/2), 0]), np.array(emb[:int(emb.shape[0]/2), 1]), np.array(emb[:int(emb.shape[0]/2), 2])
        x2, y2, z2 = np.array(emb[int(emb.shape[0]/2):, 0]), np.array(emb[int(emb.shape[0]/2):, 1]), np.array(emb[int(emb.shape[0]/2):, 2])
        ax.scatter(x,y,z, marker="o", c=emb_l[:int(emb.shape[0]/2)], cmap=cmap_obj, alpha=0.85)
        ax.scatter(x2,y2,z2, marker="v", c=emb_l[int(emb.shape[0]/2):], cmap=cmap_obj, alpha=0.85)
        plt.legend(loc='bottom left', bbox_to_anchor=(1, 0.5), handles=l_elems)
        plt.show()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()


def tensor_to_df(tensor, ax_names=None):
    assert tensor.ndim == 2, "Can only currently convert 2D tensors to dataframes"
    df = pd.DataFrame(data=tensor, columns=np.arange(tensor.shape[1]))
    return df.melt(value_vars=df.columns,
                   var_name=('variable' if ax_names is None else ax_names[0]),
                   value_name=('value' if ax_names is None else ax_names[1]))


def tensors_to_df(tensors, head=None, keys=None, ax_names=None):
    dfs = [tensor_to_df(tensor, ax_names=ax_names) for tensor in tensors]
    df = pd.concat(dfs, keys=(np.arange(len(tensors)) if keys is None else keys))
    df.reset_index(level=0, inplace=True)
    if head is not None:
        df.rename(columns={'level_0': head}, inplace=True)
    return df


def plot_kls_df(df, filepath):
    _, cmap_arr = custom_cmap(df[df.columns[0]].nunique() + 1)
    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.FacetGrid(df, height=12, aspect=2)
        g = g.map(sns.boxplot, df.columns[1], df.columns[2], df.columns[0], palette=cmap_arr[1:],
                  order=None, hue_order=None)
        g = g.set(yscale='log').despine(offset=10)
        plt.legend(loc='best', fontsize='22')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
