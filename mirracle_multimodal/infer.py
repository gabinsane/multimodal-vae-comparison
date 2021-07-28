import torch
from mirracle_multimodal import models
from mirracle_multimodal.utils import unpack_data

path = "/home/gabi/Desktop/64d/1"
args = torch.load(path + '/args.rar')
args.mod1 = "/home/gabi/mirracle_multimodal/data/image"
args.mod2 = "/home/gabi/mirracle_multimodal/data/64d.pkl"
device = torch.device("cuda" if args.cuda else "cpu")
modelC = getattr(models, 'VAE_{}'.format(args.model))
if args.model == "uni":
    model = modelC(args, index=0).to(device)
else:
    model = modelC(args).to(device)
print('Loading model {} from {}'.format(model.modelName, path))
model.load_state_dict(torch.load(path + '/model.rar'))
model._pz_params = model._pz_params
model.eval()
trainloader, testloader = model.getDataLoaders(16, False, device)
for i, data in enumerate(testloader):
    if "2mods" in args.model:
        d = unpack_data(data, device=device)
    else:
        d = unpack_data(data[0], device=device)
    model.reconstruct(d, path, "eval_rec{}".format(i))
    if i == 10:
        break
