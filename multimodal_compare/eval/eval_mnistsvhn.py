"""Calculate cross and joint coherence of trained model on MNIST-SVHN dataset.
Train and evaluate a linear model for latent space digit classification.

Code adapted from https://github.com/iffsid/mmvae"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user
from mnistsvhn_helper import Latent_Classifier, SVHN_Classifier, MNIST_Classifier
from utils import data_to_device, check_input_unpacked
torch.backends.cudnn.benchmark = True

def classify_latents(model, epochs, option, train_loader, test_loader):
    optionmap = {model.config.mods[0]["mod_type"]:"mod_1", model.config.mods[1]["mod_type"]:"mod_2"}
    classifier = Latent_Classifier(model.config.n_latents, 10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_iters = len(train_loader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, data in enumerate(train_loader):
            # get the inputs
            data_i = check_input_unpacked(data_to_device(data, "cuda"))
            output = model.model.forward(data_i)
            with torch.no_grad():
                zs = output.mods[optionmap[option]].latent_samples["latents"]
            optimizer.zero_grad()
            outputs = classifier(zs)
            labels = model.datamodule.labels_train[i*model.config.batch_size:i*model.config.batch_size + model.config.batch_size]
            loss = criterion(outputs.squeeze(), torch.tensor(labels).cuda())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            #print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
    print('Finished Training, calculating test loss...')

    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_i = check_input_unpacked(data_to_device(data, "cuda"))
            output = model.model.forward(data_i)
            zs = output.mods[optionmap[option]].latent_samples["latents"]
            outputs = classifier(zs)[0]
            _, predicted = torch.max(outputs.data, -1)
            labels = torch.tensor(model.datamodule.labels_val[
                     i * model.config.batch_size:i * model.config.batch_size + model.config.batch_size])
            total += labels.size(0)
            correct += (predicted == torch.tensor(labels).cuda()).sum().item()
    print('The {} classifier correctly classified {} out of {} examples. Accuracy: '
          '{:.2f}%'.format(option, correct, total, correct / total * 100))


def _maybe_train_or_load_digit_classifier_img(model, path, epochs, train_loader, test_loader):
    optionmap = {model.config.mods[0]["mod_type"]:"mod_1", model.config.mods[1]["mod_type"]:"mod_2"}
    options = [o for o in ['mnist', 'svhn'] if not os.path.exists(path.format(o))]
    for option in options:
        print("Cannot find trained {} digit classifier in {}, training...".
              format(option, path.format(option)))
        classifier = globals()['{}_Classifier'.format(option.upper())]().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            total_iters = len(train_loader)
            print('\n====> Epoch: {:03d} '.format(epoch))
            for i, data in enumerate(train_loader):
                # get the inputs
                x = check_input_unpacked(data_to_device(data, "cuda"))[optionmap[option]]["data"]
                labels = torch.tensor(model.datamodule.labels_train[
                                      i * model.config.batch_size:i * model.config.batch_size + model.config.batch_size])
                optimizer.zero_grad()
                outputs = classifier(x.float()).squeeze()
                loss = criterion(outputs, labels.cuda())
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i + 1) % 50 == 0:
                    print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                    running_loss = 0.0
        print('Finished Training, calculating test loss...')

        classifier.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = check_input_unpacked(data_to_device(data, "cuda"))[optionmap[option]]["data"]
                labels = torch.tensor(model.datamodule.labels_val[
                                      i * model.config.batch_size:i * model.config.batch_size + model.config.batch_size])
                outputs = classifier(x.float()).squeeze()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cuda() == labels.cuda()).sum().item()
        print('The classifier correctly classified {} out of {} examples. Accuracy: '
              '{:.2f}%'.format(correct, total, correct / total * 100))

        torch.save(classifier.state_dict(), path.format(option))

    mnist_net, svhn_net = MNIST_Classifier().cuda(), SVHN_Classifier().cuda()
    mnist_net.load_state_dict(torch.load(path.format('mnist')))
    svhn_net.load_state_dict(torch.load(path.format('svhn')))
    return mnist_net, svhn_net

def cross_coherence(model, epochs, train_loader, test_loader):
    mnist_net, svhn_net = _maybe_train_or_load_digit_classifier_img(model, "../data/{}_model.pt", epochs=epochs,
                                                                    train_loader=train_loader, test_loader=test_loader)
    mnist_net.eval()
    svhn_net.eval()
    optionmap = {model.config.mods[0]["mod_type"]: "mod_1", model.config.mods[1]["mod_type"]: "mod_2"}
    total = 0
    corr_m = 0
    corr_s = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_i = check_input_unpacked(data_to_device(data, "cuda"))
            output = model.model.forward(data_i)
            mnist_mnist = mnist_net(output.mods[optionmap["mnist"]].decoder_dist.loc)
            svhn_svhn = svhn_net(output.mods[optionmap["svhn"]].decoder_dist.loc.reshape(-1,3,32,32))
            targets = torch.tensor(model.datamodule.labels_val[
                                  i * model.config.batch_size:i * model.config.batch_size + model.config.batch_size]).cuda()
            _, pred_m = torch.max(mnist_mnist, 1)
            _, pred_s = torch.max(svhn_svhn, 1)
            total += targets.size(0)
            corr_m += (pred_m == targets).sum().item()
            corr_s += (pred_s == targets).sum().item()

    print('Cross coherence: \n SVHN -> MNIST {:.2f}% \n MNIST -> SVHN {:.2f}%'.format(
        corr_m / total * 100, corr_s / total * 100))


def joint_coherence(model):
    mnist_net, svhn_net = MNIST_Classifier().cuda(), SVHN_Classifier().cuda()
    mnist_net.load_state_dict(torch.load('../data/mnist_model.pt'))
    svhn_net.load_state_dict(torch.load('../data/svhn_model.pt'))
    optionmap = {model.config.mods[0]["mod_type"]: "mod_1", model.config.mods[1]["mod_type"]: "mod_2"}
    mnist_net.eval()
    svhn_net.eval()

    total = 0
    corr = 0
    with torch.no_grad():
        pzs = model.model.pz(*model.model.pz_params).rsample(torch.Size([1000]))
        mnist = model.model.vaes[optionmap["mnist"]].decode({"latents":pzs.cuda()})[0].squeeze(1)
        svhn = model.model.vaes[optionmap["svhn"]].decode({"latents":pzs.cuda()})[0].reshape(-1,3,32,32)

        mnist_mnist = mnist_net(mnist)
        svhn_svhn = svhn_net(svhn)

        _, pred_m = torch.max(mnist_mnist, 1)
        _, pred_s = torch.max(svhn_svhn, 1)
        total += pred_m.size(0)
        corr += (pred_m == pred_s).sum().item()

    print('Joint coherence: {:.2f}%'.format(corr / total * 100))


if __name__ == "__main__":
    from eval.infer import MultimodalVAEInfer
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mpath", type=str, help="path to the .ckpt model file. Relative or absolute")
    parser.add_argument("-l", "--level", type=int, default=0, help="for multieval option, if statistics for individual models are not yet made"),
    args = parser.parse_args()
    exp = MultimodalVAEInfer(args.mpath)
    model = exp.get_wrapped_model()
    model.eval()
    train_loader = model.datamodule.train_dataloader()
    test_loader = model.datamodule.val_dataloader()
    print('-' * 25 + 'latent classification accuracy' + '-' * 25)
    print("Calculating latent classification accuracy for single MNIST VAE...")
    classify_latents(model, epochs=30, option='mnist', train_loader=train_loader, test_loader=test_loader)
    # #
    print("\n Calculating latent classification accuracy for single SVHN VAE...")
    classify_latents(model, epochs=30, option='svhn', train_loader=train_loader, test_loader=test_loader)
    #
    print('\n' + '-' * 45 + 'cross coherence' + '-' * 45)
    cross_coherence(model, epochs=30, train_loader=train_loader, test_loader=test_loader)
    #
    print('\n' + '-' * 45 + 'joint coherence' + '-' * 45)
    joint_coherence(model)