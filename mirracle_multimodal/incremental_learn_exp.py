import argparse
import configparser
import sys
import json, random
from collections import defaultdict
import numpy as np
import os
sys.path.append(os.path.join(os.getcwd(), "src/mirracle_multimodal/mirracle_multimodal"))
import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data
import torch
from torch import optim

def parse_args():
        conf_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
        conf_parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
        args, remaining_argv = conf_parser.parse_known_args()
        if args.cfg:
            conf = configparser.SafeConfigParser()
            conf.read([args.cfg])
            defaults = dict(conf.items("general"))
            defaults["cfg"] = args.cfg
        parser = argparse.ArgumentParser(parents=[conf_parser])
        parser.set_defaults(**defaults)
        parser.add_argument('--viz_freq', type=int,
                            help='frequency of visualization savings (number of iterations)')
        parser.add_argument('--modalities_num', type=int,
                            help='number of modalities to train on')
        parser.add_argument('--obj', type=str, metavar='O',
                            help='objective to use (moe_elbo/poe_elbo_semi)')
        parser.add_argument('--loss', type=str, metavar='O',
                            help='loss to use (lprob/bce)')
        parser.add_argument('--llik_scaling', type=float,
                            help='likelihood scaling for reconstruction loss'
                                 ', set as 0 to use balance the mods and 1 to not')
        parser.add_argument('--n_latents', type=int,
                            help='latent vector dimensionality')
        parser.add_argument('--pre_trained', type=str,
                            help='path to pre-trained model (train from scratch if empty)')
        parser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disable CUDA usage')
        parser.add_argument('--batch_size', type=int,
                            help='Batch size')
        parser.add_argument('--seed', type=int, metavar='S',
                            help='seed number')
        parser.add_argument('--exp_name', type=str,
                            help='name of folder')
        args = parser.parse_args(remaining_argv)
        return args

args = parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

model = str(args.modalities_num) if args.modalities_num == 1 else "_".join((str(args.modalities_num), args.mixing))
# load model
modelC = getattr(models, 'VAE_{}'.format(model))
model = modelC(vars(args)).to(device)

if args.pre_trained:
    print('Loading model {} from {}'.format(model.modelName, args.pre_trained))
    model.load_state_dict(torch.load(args.pre_trained + '/model.rar'))
    model._pz_params = model._pz_params

# set up run path
runPath = os.path.join('results/', args.exp_name)
os.makedirs(runPath, exist_ok=True)
print('Expt:', runPath)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))
# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=1e-3, amsgrad=True)

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + ("_".join((args.obj, args.mixing)) if hasattr(model, 'vaes') else args.obj))

def train(epoch, data, agg, lossmeter):
    model.train()
    b_loss = 0
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(args.modalities_num)]
    optimizer.zero_grad()
    loss, kld, partial_l = objective(model, data, K=1, ltype=args.loss)
    loss_m.append(loss)
    kld_m.append(kld)
    for i,l in enumerate(partial_l):
        partial_losses[i].append(l)
    loss.backward()
    optimizer.step()
    b_loss += loss.item()
    if epoch is not None:
        print("Iteration {}; loss: {:6.3f}".format(epoch, loss.item()))
        progress_d = {"Epoch": epoch, "Train Loss": sum_det(loss_m)/len(data), "Train KLD": sum_det(kld_m)/len(data)}
        for i, x in enumerate(partial_losses):
            progress_d["Train Mod_{}".format(i)] = sum_det(x)/len(data)
        lossmeter.update_train(progress_d)
    agg['train_loss'].append(b_loss /len(data))

def trest(epoch, data, agg, lossmeter):
    model.eval()
    b_loss = 0
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(args.modalities_num)]
    with torch.no_grad():
        loss, kld, partial_l = objective(model,  torch.tensor(data[:1000]).float(), K=1, ltype=args.loss)
        loss_m.append(loss)
        kld_m.append(kld)
        for i, l in enumerate(partial_l):
            partial_losses[i].append(l)
        b_loss += loss.item()
        if epoch % 10 == 0:
            model.reconstruct(torch.tensor(data).float(), runPath, epoch)
        #model.generate(runPath, epoch)
        #model.analyse(torch.tensor(data).float(), runPath, epoch)
    progress_d = {"Epoch": epoch, "Test Loss": sum_det(loss_m)/len(data), "Test KLD": sum_det(kld_m)/len(data)}
    for i, x in enumerate(partial_losses):
        progress_d["Test Mod_{}".format(i)] = sum_det(x)/len(data)
    lossmeter.update(progress_d)
    agg['test_loss'].append(b_loss / len(data))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))


def detach(listtorch):
    return [np.asarray(l.detach().cpu()) for l in listtorch]

def sum_det(tn):
    if isinstance(tn[0], int):
        return 0
    else:
        return float(torch.stack(tn).sum().cpu().detach())


class DataSource():
    def __init__(self):
        self.iter = 0
        # training strategy params
        self.buffer_size = int(args.buffer)  # after how many presented data samples to retrain on them
        self.batch_size = int(args.batch_size)    # batch size when retraining on the past data
        self.repeats = int(args.data_repeats)       # how many times to present each data sample

        self.lossmeter = Logger(runPath, args)
        self.task_samples = []
        self.testdata = self.load_data(args.mod_testdata)
        self.traindata = self.load_data(args.mod_path)
        self.subpart = 0
        self.comptime = []

    def load_data(self, pth,  imsize=64):
        import os, glob, numpy as np, imageio
        def generate(images):
            dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
            for i, image_path in enumerate(images[:3010]):
                image = imageio.imread(image_path)
                dataset[i, :] = image / 255
            return dataset.reshape(-1, 3, 64, 64)
        if any([os.path.isdir(x) for x in glob.glob(os.path.join(pth, "*"))]):
            subparts = (glob.glob(os.path.join(pth, "./*")))
            datasets = []
            for s in subparts:
                images = (glob.glob(os.path.join(s, "*.png")))
                d = generate(images)
                self.task_samples.append(np.asarray(random.sample(list(d), int(len(d) * (int(args.replay_samples_percent)/100)))))
                datasets.append(d)
            return datasets
        else:
            images = (glob.glob(os.path.join(pth, "*.png")))
            dataset = generate(images)
            return dataset

    def replay_train(self, agg):
        collected_d = self.traindata[self.iter - self.buffer_size:self.iter]
        if int(self.iter / self.buffer_size) > 1 and args.direct_sample_mixing == "true":
            for x in range(int(self.iter/self.buffer_size)-1):
                previous_data = self.task_samples[x]
                len_data = int(len(self.task_samples[x]) * (int(self.iter/self.buffer_size)-1) + len(collected_d))
                num_batches = int(len_data/self.batch_size)
                b = 0
                for i in range(len(previous_data)):
                    if b == num_batches:
                        b = 0
                    while True:
                        rand_insert = random.randint(b*self.batch_size, b*self.batch_size + self.batch_size-1)
                        if rand_insert < len(collected_d):
                            break
                    collected_d = np.insert(collected_d, rand_insert, previous_data[i], axis=0)
                    b += 1
            #np.random.shuffle(collected_d)
        collected_d = torch.tensor(collected_d).float()
        for x in range(self.repeats):
            print("Data replay, iteration {}/{}".format(x, self.repeats))
            for i in range(int(len(collected_d) / self.batch_size)):
                Ep = None if i != range(int(len(collected_d) / self.batch_size))[-1] else self.iter+x
                train(Ep, collected_d[(i * self.batch_size):(i * self.batch_size + self.batch_size)], agg, self.lossmeter)
            trest(self.iter+x, self.testdata, agg, self.lossmeter)
        if args.sample_replay == "true":
            replay_data = []
            for x in range(int(self.iter/self.buffer_size)-1):
                replay_data = np.append( replay_data, self.task_samples[x], axis=0)
            np.random.shuffle(replay_data)
            replay_data = torch.tensor(replay_data).float()
            for x in range(10):
                print("Replaying old data, iteration {}/{}".format(x, 10))
                for i in range(int(len(replay_data) / self.batch_size)):
                    Ep = None if i != range(int(len(replay_data) / self.batch_size))[-1] else self.iter + x
                    train(Ep, replay_data[(i * self.batch_size):(i * self.batch_size + self.batch_size)], agg,
                          self.lossmeter)
            trest(self.iter+x, self.testdata, agg, self.lossmeter)
        save_model(model, runPath + '/model.rar')

    def iterate_data(self):
        agg = defaultdict(list)
        if not isinstance(self.traindata, list):
            data_size = len(self.traindata)
        else:
            data_size = sum([len(x) for x in self.traindata])
            self.traindata = np.concatenate(self.traindata)
            #np.random.shuffle(self.traindata)
        if self.iter % self.buffer_size == 0 and self.iter != 0:
            self.replay_train(agg)
            self.comptime.append(time.time())
            trest(self.iter, self.testdata, agg, self.lossmeter)
        if self.iter == data_size:
            #self.replay_train(agg)
            #trest(self.iter, self.testdata, agg, self.lossmeter)
            return True
        self.iter += 1
        return False

if __name__ == '__main__':
    import time
    start = time.time()
    data_manager = DataSource()
    done = False
    while not done:
        done = data_manager.iterate_data()
    end = time.time()
    data_manager.comptime.append(end)
    times = [x-start for x in data_manager.comptime]
    print("Time elapsed: {}".format(end - start))
    with open(os.path.join(runPath, 'elapsedtime.txt'), 'w') as f:
        f.write(str(times))
