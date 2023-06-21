import math
import os, csv, copy
import shutil
import pathlib
import cv2
import h5py
import glob, imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from collections import defaultdict
import pickle
from visualization import tensors_to_df
from itertools import combinations


alphabet = ' abcdefghijklmnopqrstuvwxyz'

def print_save_stats(stats_dict, path, dataset_name, level=1):
    """
    Prints the results retrieved from eval modules into txt file and into terminal.

    :param stats_dict: dictionary with statistics as keys and corresponding values
    :type stats_dict: dict
    :param path: path where to save the stats
    :type path: str
    :param dataset_name: name of the dataset for file name
    :type dataset_name: str
    :param level: in case of cdsprites+, you can provide level to print out results for table
    :type level: int
    """
    print("Final results:")
    final_line = ""
    with open(os.path.join(path,'{}_stats.txt'.format(dataset_name)), 'w') as f:
        for key, value_dict in stats_dict.items():
            if value_dict["stdev"] is not None:
                if "strict" in key.lower() or "letter" in key.lower():
                    final_line += "{:.0f} ({:.0f}) & ".format(round(value_dict["value"], 0),
                                                              round(value_dict["stdev"], 0))
                else:
                    final_line += "{:.1f}~({:.1f})/{} & ".format(round(value_dict["value"] * level / 100, 2),
                                                                 round(value_dict["stdev"] * level / 100, 1), level)
                stat = "{}: {:.2f} ({:.2f})".format(key, round(value_dict["value"],2), round(value_dict["stdev"], 2))
            else:
                stat = "{}: {:.2f}".format(key, round(value_dict["value"], 2))
            print(stat)
            f.write(stat)
            f.write('\n')
    print("\n {} statistics printed in {} \n".format(dataset_name, os.path.join(path,'{}_stats.txt'.format(dataset_name))))
    print(final_line)

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + torch.nn.functional.softplus((tensor - min).float())
    return result_tensor


def find_out_batch_size(inputs):
    batch_size = None
    for key in inputs.keys():
        if inputs[key]["data"] is not None:
            batch_size = inputs[key]["data"].shape[0]
            break
    return batch_size

def check_input_unpacked(mods):
    """Checks if the input is unpacked in case of a unimodal scenario"""
    if len(mods.keys()) == 1:
        mods = mods[list(mods.keys())[0]]
    return mods

def subsample_input_modalities(mods):
    mods_inputs = []
    for m in range(len(mods) + 1):
        mods_input = copy.deepcopy(mods)
        for d in mods.keys():
            mods_input[d]["data"] = None
            mods_input[d]["masks"] = None
        if m == len(mods.keys()):
            mods_input = mods
        else:
            mods_input["mod_{}".format(m + 1)] = mods["mod_{}".format(m + 1)]
        mods_inputs.append(mods_input)
    return mods_inputs

def data_to_device(data, device):
    for key in data.keys():
        data[key] = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v for k, v in
                     data[key].items()}
    return data

def merge_dicts(d1, d2):
    dd = defaultdict(list)
    for d in (d1, d2):  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)
    return dd

def get_root_folder():
    return os.path.dirname(__file__)

def make_kl_df(qz_xs, pz):
    pz.loc = pz.loc.detach().cpu()
    pz.scale = pz.scale.detach().cpu()
    if isinstance(qz_xs, list):
        for i, qz in enumerate(qz_xs):
            qz_xs[i].loc = qz.loc.detach().cpu()
            qz_xs[i].scale = qz.scale.detach().cpu()
        kls_df = tensors_to_df(
            [*[kl_divergence(qz_x, pz) for qz_x in qz_xs],
             *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p))
               for p, q in combinations(qz_xs, 2)]],
            head='KL',
            keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                  *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                    for i, j in combinations(range(len(qz_xs)), 2)]],
            ax_names=['Dimensions', r'KL$(q\,||\,p)$'])
    else:
        qz_xs.loc = qz_xs.loc.detach().cpu()
        qz_xs.scale = qz_xs.scale.detach().cpu()
        kls_df = tensors_to_df([kl_divergence(qz_xs, pz)], head='KL',
                               keys=[r'KL$(q(z|x)\,||\,p(z))$'], ax_names=['Dimensions', r'KL$(q\,||\,p)$'])
    return kls_df


def get_path_type(path):
    """
    See if the provided data path is supported.

    :param path: Path to the dataset
    :type path: str
    :return: recognised type of the data
    :rtype: str
    """
    assert os.path.exists(path), "Path does not exist: {}".format(path)
    if os.path.isdir(path):
        return "dir"
    if path[-4:] == ".pth":
        return "torch"
    if path[-4:] == ".pkl":
        return "pickle"
    raise Exception("Unrecognized dataset format. Supported types are: .pkl, .pth or directory with images")


def pad_seq_data(data, masks):
    for i, _ in enumerate(data):
        if masks is not None:
            if masks[i] is not None:
                data[i].append(masks[i])
        else:
            data[i] = torch.tensor([o[0].clone().detach() for o in data[i][0]])
    return data


def load_images(path):
    images = sorted(glob.glob(os.path.join(path, "*.png")))
    if len(images) == 0:
        images = sorted(glob.glob(os.path.join(path, "./*/*.png")))
    dataset = []
    for i, image_path in enumerate(images):
        image = imageio.imread(image_path)
        dataset.append(image.reshape(-1) / 256)
    return np.asarray(dataset)

def load_pickle(pth):
        with open(pth, 'rb') as handle:
            return pickle.load(handle)

def load_data(path):
    """
    Returns appropriate method for data loading based on path suffix
    :param path: Path to data
    :type path: str
    :return: method to load data
    :rtype: object
    """
    if path.startswith('.'):
        path = os.path.join(get_root_folder(), path)
    assert os.path.exists(path), "Path does not exist: {}".format(path)
    if os.path.isdir(path):
        return load_images(path)
    if pathlib.Path(path).suffix in [".pt",".pth"]:
        return torch.load(path)
    if pathlib.Path(path).suffix == ".pkl":
        return load_pickle(path)
    if pathlib.Path(path).suffix == ".h5":
        return h5py.File(path, 'r')
    if pathlib.Path(path).suffix == ".npy":
        return np.load(path)
    raise Exception("Unrecognized dataset format. Supported types are: .pkl, .pth or directory with images")

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class Constants(object):
    eta = 1e-6
    eps = 1e-9
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


class Logger(object):
    """Saves training progress into csv"""

    def __init__(self, path, mods):
        self.fields = ["Epoch", "Train Loss", "Test Loss", "Train KLD", "Test KLD"]
        self.path = path
        self.dic = {}
        for m in range(len(mods)):
            self.fields.append("Train Mod_{}".format(m))
            self.fields.append("Test Mod_{}".format(m))
        self.reset()

    def reset(self):
        with open(os.path.join(self.path, "loss.csv"), mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writeheader()

    def update_train(self, val_d):
        self.dic = val_d

    def update(self, val_d):
        with open(os.path.join(self.path, "loss.csv"), mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writerow({**self.dic, **val_d})
        self.dic = {}

def traverse_line(bounds, num_samples, latent_dim, idx):
    samples = torch.zeros(num_samples, latent_dim)
    traversals = torch.linspace(*bounds, steps=num_samples)
    for i in range(num_samples):
        samples[i, idx] = traversals[i]
    return samples

def get_traversal_matrix(num_samples, latent_dim, trav_range=(-1,1)):
    latent_samples = [traverse_line(trav_range, num_samples, latent_dim, dim).cuda()
                      for dim in range(latent_dim)]
    return latent_samples


def last_letter(word):
    return word[::-1]

def listdirs(rootdir):
    """
    Lists all the subdirectories within a directory

    :param rootdir: path to the root dir
    :type rootdir: str
    :return: list of subdirectories
    :rtype: str
    """
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs

# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)
    if hasattr(model, 'vaes'):
        for vae in model.vaes:
            fdir, fext = os.path.splitext(filepath)
            save_vars(vae.state_dict(), fdir + '_' + vae.modelName + fext)


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def transpose_dataloader(data, device):
    mods = [[] for _ in range(len(data[0]))]
    for num, o in enumerate(data):
        for ix, mod in enumerate(o):
            mods[ix].append(mod[0])
    return [torch.stack(m).to(device) for m in mods]


def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
    elif torch.is_tensor(dataB):
        return dataB.to(device)


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.loc
    except NotImplementedError:
        print("could not get mean")
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def get_torch_mean(loss):
    """
    Get the mean of the list of torch tensors

    :param loss: list of loss tensors
    :type loss: list
    :return: mean of the losses
    :rtype: torch.float32
    """
    return round(float(torch.mean(torch.tensor(loss).detach().cpu())), 3)


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq, seq):
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq];
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded


def tensor_to_text(gen_t):
    if not isinstance(gen_t, list):
        gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples

def add_recon_title(recon_list, title, colour=(0,0,0)):
    images = [np.asarray(add_text_in_image(np.ones((40, recon_list[0].shape[1], 3)) * 255, title, (2, 0), 12, colour)),
              np.vstack(img_separators(recon_list, horizontal=True))]
    images = img_separators(images, horizontal=True)
    return images

def turn_text2image(string_list, img_size=(64,192,3)):
        return [np.asarray(add_text_in_image(np.ones(img_size)*255, t, (5,8),16)) for t in string_list]

def img_separators(imgs, thickness=2, horizontal=True):
    images = []
    for im in imgs:
        if horizontal:
            images.append(np.ones((thickness, im.shape[1], 3))*125)
        else:  # vertical64
            images.append(np.ones((im.shape[0], thickness, 3))*125)
        images.append(im)
    return images

def add_text_in_image(img, text, position, size, colour=(0, 0, 0)):
    """
    Add text into image.

    :param img: image numpy array
    :type img: np.array
    :param text: string with text
    :type text: str
    :param position: position of text within image (x,y)
    :type position: list
    :param size: font size
    :type size: int
    :param colour: RGB colour
    :type colour: tuple
    :return: finished image
    :rtype: np.array
    """
    img_size = img.shape[1]
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=size)
    draw = ImageDraw.Draw(img)
    lines = text_wrap(text, font, img_size)
    for line in lines:
        draw.text(position, line, font=font, fill=colour)
        pos = list(position)
        pos[1] += size
        position = tuple(pos)
    return img


def text_wrap(text, font, max_width):
    """
    Wrap text base on specified width. This is to enable text of width more than the image width to be display
    nicely.

    :param text: string wth text
    :type text: str
    :param font: font object
    :type font: obj
    :param max_width: maximum width to fit the text into
    :type max_width: int
    :return: list of separate text lines
    :rtype: list
    """
    lines = []

    # If the text width is smaller than the image width, then no need to split
    # just add it to the line list and return
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        # split the line by spaces to get words
        words = text.split(' ')
        i = 0
        # append every word to a line while its width is shorter than the image width
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)
    return lines

def output_onehot2text(recon=None, original=None):
    recon_decoded, orig_decoded = None, None
    if recon is not None:
        recons_mat = torch.softmax(recon, dim=-1)
        one_pos = torch.argmax(recons_mat, dim=2)
        rec = torch.nn.functional.one_hot(one_pos)
        recon = rec.int()
        recon_decoded = tensor_to_text(recon)
        recon_decoded = ["".join(x) for x in recon_decoded]
    if original is not None:
        orig_decoded = tensor_to_text(torch.stack(original).squeeze().int())
        orig_decoded = ["".join(x) for x in orig_decoded]
        return recon_decoded, orig_decoded
    return recon_decoded


def combinatorial(lst):
    index, pairs = 1, []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append((element1, element2))
        index += 1
    return pairs


def get_all_pairs(lst):
    output = []
    for i in range(0, len(lst)):
        for j in range(0, len(lst)):
            if (i != j):
                output.append((lst[i], lst[j]))
    return output


def partial_sum(v, keep_dims=[]):
    """Sums variable or tensor along all dimensions except those specified in `keep_dims`"""
    if len(keep_dims) == 0:
        return v.sum()
    else:
        keep_dims = sorted(keep_dims)
        drop_dims = list(set(range(v.dim())) - set(keep_dims))
        result = v.permute(*(keep_dims + drop_dims))
        size = result.size()[:len(keep_dims)] + (-1,)
        return result.contiguous().view(size).sum(-1)


def batch_sum(v, sample_dims=None, batch_dims=None):
    if sample_dims is None:
        sample_dims = ()
    elif isinstance(sample_dims, int):
        sample_dims = (sample_dims,)
    if batch_dims is None:
        batch_dims = ()
    elif isinstance(batch_dims, int):
        batch_dims = (batch_dims,)
    assert set(sample_dims).isdisjoint(set(batch_dims))
    keep_dims = tuple(sorted(set(sample_dims).union(set(batch_dims))))
    v_sum = partial_sum(v, keep_dims=keep_dims)
    if len(keep_dims) == 2 and sample_dims[0] > batch_dims[0]:
        return v_sum.permute(1, 0)
    else:
        return v_sum

def make_joint_samples(model, index, datamod, latents, traversals, savedir, num_samples, trav_range=(-1,1), current_vae=None):
    m = model.vaes[current_vae] if current_vae is not None else model
    samples = m.generate_samples(num_samples, traversals, traversal_range=trav_range)
    recon = m.decode({"latents": samples, "masks": None})[0]
    data_class = datamod.datasets[index]
    tag = "traversals_range{}".format(str(trav_range[0]).replace("-", "").replace(".", "")) if traversals else "samples"
    p = os.path.join(savedir, "{}_{}.png".format(tag, data_class.mod_type))
    rows = latents if traversals else int(math.sqrt(num_samples))
    data_class.save_traversals(recon, p, rows)
    return data_class.get_processed_recons(recon.detach().cpu()), recon



def log_batch_marginal(dists, zs, sample_dim=None, batch_dim=None, bias=1.0):
    """Computes log batch marginal probabilities. Returns the log marginal joint
        probability, the log product of marginals for individual variables, and the
        log product over both variables and individual dimensions."""
    log_pw_joints = 0.0
    log_marginals = 0.0
    log_prod_marginals = 0.0
    for value, dist in zip(zs, dists):
        # log pairwise probabilities of size (B, B, *, **)
        log_pw = dist.log_prob(value).transpose(1, batch_dim + 1)
        if sample_dim is None:
            keep_dims = (0, 1)
        else:
            keep_dims = (0, 1, sample_dim + 2)
        # log pairwise joint probabilities (B, B) or (B, B, S)
        log_pw_joint = partial_sum(log_pw, keep_dims)  # log(prod(qzi)) = logqz
        log_pw_joints = log_pw_joints + log_pw_joint

        # log average over pairs (B) or (S, B)
        log_marginal = log_mean_exp(log_pw_joint, 1).transpose(0, batch_dim)  # 128,128,1 -128,1-> 1,128

        # log product over marginals (B) or (S, B): #128,128,1 -- 128,1 -->
        log_prod_marginal = batch_sum(log_mean_exp(log_pw, 1),
                                      sample_dim + 1, 0)

        log_marginals = log_marginals + log_marginal
        log_prod_marginals = log_prod_marginals + log_prod_marginal
    # perform bias correction for log pairwise joint
    bias_mat = torch.zeros_like(log_pw_joints)
    log_pw_joints = log_pw_joints + bias_mat
    log_pw_joints = log_mean_exp(log_pw_joints, 1).transpose(0, batch_dim)
    return log_pw_joints, log_marginals, log_prod_marginals


def log_joint(dists, zs):
    """Returns the log joint probability"""
    log_prob = 0.0
    for i, d in enumerate(dists):
        log_p = d.log_prob(zs[i])
        log_prob = log_prob + log_p
    return log_prob


def check_img_normalize(data):
    """
    Normalizes image data between 0 and 1 (if needed)

    :param data: input images for normalisation
    :type data: Union[torch.tensor, list]
    :return: normalised data
    :rtype: list, torch.tensor
    """
    if isinstance(data, list):
        if torch.max(torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)) > 1:
            data = [x / 256 for x in data]
    else:
        data = data / 256 if torch.max(data) > 1 else data
    return data
