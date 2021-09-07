import argparse
import configparser
import sys
import json
from collections import defaultdict
import numpy as np
import os
sys.path.append(os.path.join(os.getcwd(), "src/mirracle_multimodal/mirracle_multimodal"))
import models
import objectives
import rclpy
from rclpy.node import Node
from utils import Logger, Timer, save_model, save_vars, unpack_data
from std_msgs.msg import Float32MultiArray
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
    print("Model updated; loss: {:6.3f}".format(loss.item()))
    progress_d = {"Epoch": epoch, "Train Loss": sum_det(loss_m)/len(data), "Train KLD": sum_det(kld_m)/len(data)}
    for i, x in enumerate(partial_losses):
        progress_d["Train Mod_{}".format(i)] = sum_det(x)/len(data)
    lossmeter.update_train(progress_d)
    agg['test_loss'].append(b_loss /len(data))

def test(epoch, data, agg, lossmeter):
    model.eval()
    b_loss = 0
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(args.modalities_num)]
    with torch.no_grad():
        loss, kld, partial_l = objective(model, data, K=1, ltype=args.loss)
        loss_m.append(loss)
        kld_m.append(kld)
        for i, l in enumerate(partial_l):
            partial_losses[i].append(l)
        b_loss += loss.item()
        if i == 0 and epoch % args.viz_freq == 0:
            model.reconstruct(data, runPath, epoch)
            model.generate(runPath, epoch)
            model.analyse(data, runPath, epoch)
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


class InputSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.iter = 0
        self.lossmeter = Logger(runPath, args)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'fuse_input',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Initiated, waiting for input data")

    def listener_callback(self, msg):
        self.get_logger().info("Got data from iteration %s" % self.iter)
        agg = defaultdict(list)
        d = np.asarray(msg.data)
        d1 = d[:12288].reshape(3,64,64)
        d1 = torch.tensor(d1)
        d2 = d[12288:].reshape(12288)
        d2 = torch.tensor(d2)
        train(self.iter, [d1.unsqueeze(0), d2.unsqueeze(0)], agg, self.lossmeter)
        test(self.iter,  [d1.unsqueeze(0), d2.unsqueeze(0)], agg, self.lossmeter)
        save_model(model, runPath + '/model.rar')
        if self.iter % 10 == 0:
            save_model(model, runPath + '/model_iterations{}.rar'.format(self.iter))
        self.iter += 1

def main():
    rclpy.init()
    input_subscriber = InputSubscriber()
    rclpy.spin(input_subscriber)
    input_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()