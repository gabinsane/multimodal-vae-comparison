import math
import os, csv
import shutil
import time
import torch.distributions as dist
import glob, imageio
from gensim.models import Word2Vec
import random
import numpy as np
import torch
import torch.nn.functional as F

class W2V():
    def __init__(self, dim, path):
        self.vec_dim = int(dim[0])
        self.model = self.get_w2v(self.vec_dim, path)
        self.max = None
        self.min = None

    def normalize_w2v(self, data):
        self.max = data.max()
        self.min = data.min()
        a = self.max - self.min
        d = (data - self.min) / a
        return d

    def unnormalize_w2v(self,data):
        a = self.max - self.min
        d = (data * a) + self.min
        return d

    def get_w2v(self, data_dim, path):
        if not os.path.exists(os.path.join(os.path.dirname(path), "word2vec{}d.model".format(data_dim))):
            print("Did not find {}".format(os.path.join(os.path.dirname(path), "word2vec{}d.model".format(data_dim))))
            return None
        else:
            w = Word2Vec.load(os.path.join(os.path.dirname(path), "word2vec{}d.model".format(data_dim)))
            return w

def load_images(path, dim):
    def generate(images):
        images = sorted(images)
        dataset = np.zeros((len(images), dim[0], dim[1], dim[2]), dtype=np.float)
        for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            dataset[i, :] = image / 255
        return dataset.reshape(-1, dataset.shape[-1], dataset.shape[1], dataset.shape[2])

    if any([os.path.isdir(x) for x in glob.glob(os.path.join(path, "*"))]):
        subparts = (glob.glob(os.path.join(path, "./*")))
        datasets = []
        for s in subparts:
            images = (glob.glob(os.path.join(s, "*.png")))
            d = generate(images)
            datasets.append(d)
        return np.concatenate(datasets)
    else:
        images = (glob.glob(os.path.join(path, "*.png")))
        dataset = generate(images)
        return dataset


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def create_vocab(text, add_noise):
    # create vocabulary
    vocab = Vocabulary('training')
    sentences = []
    if text.shape[1] > 1:
        for seq in range(text.shape[0]):
            sentences.append(" ".join((text[seq][0], text[seq][1])))
    else:
        for seq in text:
            sentences.append(seq[0])
    for sent in sentences:
        vocab.add_sentence(sent)
    text2idx = np.zeros(text.shape)
    for idx, sent in enumerate(text2idx):
        for word_idx, _ in enumerate(sent):
            if add_noise:
                text2idx[idx][word_idx] = vocab.to_index(text[idx][word_idx]) + round(random.uniform(-0.49,0.49),3)
            else:
                text2idx[idx][word_idx] = vocab.to_index(text[idx][word_idx])
    return text2idx, vocab


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        if isinstance(index, int):
            return self.index2word[int(index)]
        else:
            return self.index2word[int(index.round())]

    def to_index(self, word):
        try:
            return self.word2index[word]
        except:
            print("Error: {} is not in vocabulary!".format(word))
            return 0


class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting


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


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


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


def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))


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


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def pdist(sample_1, sample_2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb, data):
    indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
    # indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
    return data[indices]


class FakeCategorical(dist.Distribution):
    support = dist.constraints.real
    has_rsample = True

    def __init__(self, locs):
        self.logits = locs
        self._batch_shape = self.logits.shape

    @property
    def mean(self):
        return self.logits

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.logits.expand([*sample_shape, *self.logits.shape]).contiguous()

    def log_prob(self, value):
        # value of shape (K, B, D)
        lpx_z = -F.cross_entropy(input=self.logits.view(-1, self.logits.size(-1)),
                                 target=value.expand(self.logits.size()[:-1]).long().view(-1),
                                 reduction='none',
                                 ignore_index=0)

        return lpx_z.view(*self.logits.shape[:-1])
        # it is inevitable to have the word embedding dimension summed up in
        # cross-entropy loss ($\sum -gt_i \log(p_i)$ with most gt_i = 0, We adopt the
        # operationally equivalence here, which is summing up the sentence dimension
        # in objective.

alphabet = ' abcdefghijklmnopqrstuvwxyz'
#alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"\\/|_@#$%^&*~`+-=<>()[]{} \n'

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


def create_text_from_label_mnist(len_seq, label, alphabet):
    text = digit_text_english[label];
    sequence = len_seq * [' '];
    start_index = random.randint(0, len_seq - 1 - len(text));
    sequence[start_index:start_index + len(text)] = text;
    sequence_one_hot = one_hot_encode(len_seq, alphabet, sequence);
    return sequence_one_hot


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
