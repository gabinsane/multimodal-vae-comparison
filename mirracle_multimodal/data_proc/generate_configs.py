import argparse
import os, yaml

'''
Code to automatically generate a set of training configs covering all defined parameter combinations. 
For each argument, you can set multiple values which will be covered.

Example usage:
python generate_configs.py --cfg ../config2mods.yml --path sound_action --exp-name lr --mixing moe poe --lr 1e-3 1e-4 1e-5
'''


def exclude_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="../configs_experiment0", help="Folder in which to save the configs")
parser.add_argument('--exp-name', type=str, default="conf",
                    help='name of the config file')
parser.add_argument('--cfg', type=str, default=None,
                    help='Which config to alter (only the specified parameters will vary)')
parser.add_argument('--epochs', type=int, default=None,  help='number of training epochs')
parser.add_argument('--lr', type=str, nargs="+", default=None,  help='learning rate')
parser.add_argument('--batch_size', type=int,  nargs="+", default=None,
                    help='Size of the training batch')
parser.add_argument('--obj', type=str, metavar='O',  nargs="+", default=None,
                    help='objective to use (moe_elbo/poe_elbo_semi)')
parser.add_argument('--loss', type=str, metavar='O',  nargs="+", default=None,
                    help='loss to use (lprob/bce)')
parser.add_argument('--n-latents', type=int,  nargs="+", default=None,
                    help='latent vector dimensionality')
parser.add_argument('--pre-trained', type=str,  nargs="+", default=None,
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--seed', type=int, metavar='S',  nargs="+", default=None,
                    help='seed number')
parser.add_argument('--mixing', type=str, metavar='S',  nargs="+", default=None,
                    help='seed number')

args = parser.parse_args()

os.makedirs(args.path, exist_ok=True)
cfg_pth = os.path.join(args.path, args.exp_name + ".yml")
with open(args.cfg) as file: cfg_def = yaml.safe_load(file)

all_configs = [cfg_def]
for a in exclude_keys(vars(args), {"path", "exp_name", "cfg"}):
    if vars(args)[a] is not None:
        val_range = vars(args)[a]
        new_configs = []
        for c in all_configs:
            for v in val_range:
                new_c = c.copy()
                new_c[a] = v
                new_configs.append(new_c)
        all_configs = new_configs
for index, i in enumerate(all_configs):
    all_configs[index]["exp_name"] = "_".join([os.path.basename(args.path), args.exp_name, str(index)])
    print(i)
paths = [cfg_pth] * len(all_configs)
for i, c in enumerate(paths):
    paths[i] = c.replace(".yml", "_{}.yml".format(i))

for p, c in zip(paths, all_configs):
    with open(p, 'w') as outfile:
        print("Saving {}".format(p))
        yaml.dump(c, outfile, default_flow_style=False)