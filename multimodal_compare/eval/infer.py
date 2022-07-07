import torch, numpy as np
import models, pickle
from models import objectives
from utils import unpack_data, one_hot_encode, output_onehot2text, pad_seq_data
import cv2, os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import json, yaml
import csv
import glob

def parse_args(pth):
    if os.path.isdir(pth):
        pth = os.path.join(pth, "config.yml")
        if not os.path.exists(pth) and os.path.exists(pth.replace("config.yml", "config.json")):
            os.rename(pth.replace("config.yml", "config.json"), pth)
    with open(pth, 'r') as stream:
        config = yaml.safe_load(stream)
    modalities = []
    for x in range(20):
        if "modality_{}".format(x) in list(config.keys()):
            modalities.append(config["modality_{}".format(x)])
    return config, modalities

plot_colors = ["blue", "green", "red", "cyan", "magenta", "orange", "navy", "maroon", "brown"]

def estimate_log_marginal(model, device="cuda"):
    """Compute an estimate of the log-marginal likelihood of test data."""
    train_loader, test_loader = model.load_dataset(32, device=device)
    model.eval()
    marginal_loglik = 0
    objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '')
                        + ("_".join(("elbo", model.modelName)) if hasattr(model, 'vaes') else ["elbo"]))
    with torch.no_grad():
        for ix, dataT in enumerate(test_loader):
            data, masks = dataT
            data = pad_seq_data(data, masks)
            marginal_loglik += objective(model, data, ltype="lprob")[0]

    marginal_loglik /= len(test_loader.dataset)
    return marginal_loglik

def eval_reconstruct(path):
    model, c = load_model(path, batch=1)
    device = torch.device("cuda")
    recons = []
    for i, data in enumerate(testloader):
        d = unpack_data(data[0], device=device)
        recon = model.reconstruct_data(d)
        recons.append(np.asarray(d[0].detach().cpu()))
        recons.append(np.asarray(recon[0].detach().cpu())[0])
        if i == 10:
            break
    with open(os.path.join(path, "visuals/reconstructions.pkl"), 'wb') as handle:
        pickle.dump(recons, handle)
    print("Saved reconstructions for {}".format(path))

def eval_sample(path):
    model, c = load_model(path, batch=1)
    N, K = 36, 1
    samples = model.generate_samples(N, K).cpu().squeeze()
    with open(os.path.join(path, "visuals/latent_samples.pkl"), 'wb') as handle:
        pickle.dump(np.asarray(samples.detach().cpu()), handle)
    print("Saved samples for {}".format(path))

def plot_setup(xname, yname, pth, figname):
    plt.xlabel(xname)
    plt.ylabel(yname)
    p = pth if os.path.isdir(pth) else os.path.dirname(pth)
    plt.savefig(os.path.join(p, "visuals/{}.png".format(figname)))
    plt.clf()

def plot_loss(path):
    pth = os.path.join(path, "loss.csv") if not "loss.csv" in path else path
    loss = pd.read_csv(pth, delimiter=",")
    epochs = loss["Epoch"]
    losses = loss["Test Loss"]
    kld = loss["Test KLD"]
    plt.plot(epochs, losses, color='green', linestyle='solid', label="Loss")
    plot_setup("Epochs", "Loss", path, "loss")
    plt.plot(epochs, kld,  color='blue', linestyle='solid', label="KLD")
    plot_setup("Epochs", "KLD", path, "KLD")
    print("Saved loss plot")

def compare_loss(paths, label_tag):
    for ll in ["Test Loss", "Test Mod_0", "Test Mod_1"]:
        for p in paths:
            pth = os.path.join(p, "loss.csv") if not "loss.csv" in p else p
            c =  os.path.join(p, "config.json") if not "loss.csv" in p else p.replace("loss.csv", "config.json")
            with open(c) as json_file:
                cfg = json.load(json_file)
            loss = pd.read_csv(pth, delimiter=",")
            epochs = loss["Epoch"]
            losses = loss[ll]
            plt.plot(epochs, losses, linestyle='solid', label= ", ".join(["{}: {}".format(x,str(cfg[x])) for x in label_tag]))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(paths[0])), "losstype_compared_{}.png".format(ll.lower().replace(" ", "_"))))
        plt.clf()

def get_all_csvs(pth):
    pth = pth + "/" if pth[-1] != "/" else pth
    return glob.glob(pth + "**/loss.csv", recursive=True)

def compare_models_numbers(pth):
    pth = pth + "/" if pth[-1] != "/" else pth
    f = open(os.path.join(pth, "comparison.csv"), 'w')
    writer = csv.writer(f)
    all_csvs = get_all_csvs(pth)
    header = None
    for c in all_csvs:
        plot_loss(c)
        model_csv = pd.read_csv(c, delimiter=",")
        if not header:
            writer.writerow(["Model", "Epochs"] + list(model_csv.keys())[1:])
            header = True
        row = [c.split(pth)[-1].split("/loss.csv")[0]] + [int(model_csv.values[-1][0])] + [round(x, 4) for x in list(model_csv.values[-1][1:])]
        writer.writerow(row)
    f.close()
    print("Saved comparison at {}".format(os.path.join(pth, "comparison.csv")))

def load_model(path):
    device = torch.device("cuda")
    config, mods = parse_args(path)
    model = "VAE" if len(mods) == 1 else config["mixing"].lower()
    modelC = getattr(models, model)
    params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods],
              [m["feature_dim"] for m in mods], [m["mod_type"] for m in mods]]
    if len(mods) == 1:
        params = [x[0] for x in params]
    if "model_specific" in config.keys():
        model_params = config["model_specific"]
        model = modelC(*params, model_params, config["n_latents"], config["test_split"], config["batch_size"]).to(device)
    else:
        model = modelC(*params, config["n_latents"], config["test_split"], config["batch_size"]).to(device)
    print('Loading model {} from {}'.format(model.modelName, path))
    model.load_state_dict(torch.load(os.path.join(path,'model.rar')))
    model._pz_params = model._pz_params
    model.eval()
    return model, config

def get_traversal_samples(latent_dim, n_samples_per_dim):
    all_samples = []
    for idx in range(latent_dim):
        samples = torch.zeros(n_samples_per_dim, latent_dim)
        traversals = torch.linspace(-3,3, steps=n_samples_per_dim)
        for i in range(n_samples_per_dim):
            samples[i, idx] = traversals[i]
        all_samples.append(samples)
    samples = torch.cat(all_samples)
    return samples

def text_to_image(text, model, path):
    img_outputs, txtoutputs = [], []
    for i, w in enumerate(text):
        txt_inp = one_hot_encode(len(w), w.lower())
        model.eval()
        recons = model.forward([None,[txt_inp.unsqueeze(0), None]])[1]
        if model.modelName == 'moe':
            recons = recons[0]
        recons_image = recons[0] if isinstance(recons, list) else recons
        recons_image = recons_image[0] if isinstance(recons_image, list) else recons_image
        recons_text = recons[1][0] if isinstance(recons[1], list) else recons[1]
        image = np.asarray(recons_image.loc[0].cpu().detach())
        recon_text = recons_text.loc[0]
        recon_text = output_onehot2text(recon=recon_text.unsqueeze(0))
        #print("Reconstructed text: {}".format(recon_text[0][0][:len(w)]))
        txtoutputs.append(recon_text[0][0])
        img_outputs.append(image*255)
        image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, "cross_sample_{}_{}.png".format(recon_text[0][0][:len(w)],i)), image)
    return img_outputs, txtoutputs

def listdirs(rootdir):
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs

if __name__ == "__main__":
    p = "/home/gabi/mirracle_multimodal/multimodal_compare/results/level3/subsampled/0110/htvae_0110_bs_3"
    model, c = load_model(p)
    t = ["large green pieslice", "small red circle","large yellow spiral", "small beige line", "large teal square", "small orange semicircle", "large brown pieslice", "small grey circle","small pink spiral", "large blue line", "small navy square", "small red semicircle", "large beige pieslice", "small green circle","large yellow spiral", "large navy line", "small pink square", "small grey semicircle" ]
    text_to_image(t, model, p)

