import torch.nn as nn
import math
from torch.nn import BatchNorm3d, Sequential, ReLU, ModuleList
import torch.nn.functional as F
from models.nn_modules import SamePadConv3d, AttentionResidualBlock
import numpy as np
from torch.nn import Conv2d, BatchNorm3d, Sequential, ReLU, ModuleList, Linear, SiLU
import torch, argparse
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader

class VideoGPT(nn.Module):
    def __init__(self, mod:str):
        super(VideoGPT, self).__init__()
        downsample = (2, 4, 4)
        if mod == "action":
            self.output_dim = 9
            self.outdim_true = [9]
        elif mod == "attributes":
            self.output_dim = 24
            self.outdim_true = [4, 6]
        n_times_downsample = np.array([int(math.log2(x)) for x in downsample])
        n_res_layers = 4
        self.convs = ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else self.output_dim
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, self.output_dim, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, self.output_dim, kernel_size=3)
        self.res_stack = Sequential(
            *[AttentionResidualBlock(self.output_dim)
              for _ in range(n_res_layers)],
            BatchNorm3d(self.output_dim),
            ReLU())
        self.fc = torch.nn.DataParallel(torch.nn.Linear(self.output_dim * 16 * 16 * 4, self.output_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        h = x.permute(0, 4, 1, 2, 3)
        for conv in self.convs:
            h = F.relu(conv(h.float()))
        h = self.conv_last(h)
        h = self.res_stack(h)
        h = h.reshape(x.shape[0], -1)
        h = self.fc(h.reshape(x.shape[0], -1))
        d = h.reshape(-1, *self.outdim_true)
        return d

class CNN(nn.Module):
    def __init__(self, mod:str):
        """
        CNN encoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim:
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        super(CNN, self).__init__()
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.silu = SiLU()
        self.relu = ReLU()
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = 3
        map = {"shape":3, "size":2, "color":5, "position":4, "background":2}
        self.output_dim = map[mod]
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = torch.nn.DataParallel(Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs))
        self.conv2 = torch.nn.DataParallel(Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.conv3 = torch.nn.DataParallel(Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))

        # If input image is 64x64 do fourth convolution
        self.conv_64 = torch.nn.DataParallel(Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.pooling = torch.nn.AvgPool2d(kernel_size)
        # Fully connected layers
        self.lin1 = torch.nn.DataParallel(Linear(np.product(self.reshape), hidden_dim))
        self.lin2 = torch.nn.DataParallel(Linear(hidden_dim, hidden_dim))
        self.fc = torch.nn.DataParallel(torch.nn.Linear(hidden_dim, self.output_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        if isinstance(x, dict):
            x = x["data"]
        batch_size = x.size(0) if len(x.shape) == 4 else x.size(1)
        # Convolutional layers with ReLu activations
        x = self.relu(self.conv1(x.float()))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = self.relu(self.lin1(x))
        #x = (self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        h = self.fc(x.reshape(x.shape[0], -1))
        d = h.reshape(-1, self.output_dim)
        return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modality", type=str, help="which modality/feature to train")
    parser.add_argument("-d", "--dataset", type=str, default="cdsprites", help="name of the dataset")
    parser.add_argument("-l", "--level", type=int, default=1, help="for cdsprites plus - which dataset to train on")
    args = parser.parse_args()
    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    if args.dataset.lower() == "sprites":
        from models.datasets import SPRITES
        epochs = 5
        batch_size = 64

        dataset = SPRITES("./data/sprites", "./data/sprites/test", args.modality)
        print("Loading data....")
        d = dataset.get_data()
        if args.modality == "action":
            labels = dataset.get_actions()
        else:
            labels = dataset.get_attributes()
        d_val = dataset.get_test_data()
        if args.modality == "action":
            labels_val = dataset.get_actions()
        else:
            labels_val = dataset.get_attributes()

        shuffle = np.random.permutation(len(d))
        shuffle_val = np.random.permutation(len(d_val))
        d = d[shuffle]
        d_val = d_val[shuffle_val]
        labels_train = torch.tensor(np.asarray(labels)[shuffle]).to("cuda")
        labels_val = torch.tensor(np.asarray(labels_val)[shuffle_val]).to("cuda")
        dataset_train = TensorDataset(d)
        dataset_val = TensorDataset(d_val)
        trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        testloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)


        print("Initializing network...")
        net = VideoGPT(args.modality).to("cuda")
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        trainloss_e = []
        valaccur_e = []
        for e in range(epochs):
            trainloss = []
            valaccur = []
            net.train()
            for i, data in enumerate(trainloader):
                optimizer.zero_grad()
                output = net(data.to("cuda"))
                bs = data.shape[0]
                target = labels_train[i*bs:(i*bs)+bs]
                l = loss(output, target.cuda())/bs
                print("Epoch {}, Iteration {}, Train Loss: {:.3f}".format(e, i, l))
                trainloss.append(l)
                l.backward()
                optimizer.step()
            average_loss = sum(trainloss) / len(trainloss)
            print("End of epoch {}. Final train loss: {}".format(e, average_loss))
            trainloss_e.append(average_loss)
            trainloss = []
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    acc = []
                    output = net(data.to("cuda"))
                    bs = data.shape[0]
                    target = labels_val[i*bs:(i*bs)+bs]
                    label_guessed = torch.argmax(output, dim=-1)
                    label = torch.argmax(target, dim=-1)
                    if args.modality == "action":
                        for ind, v in enumerate(label):
                            acc.append(int(v==label_guessed[ind]))
                    else:
                        for ind, v in enumerate(label):
                            for i2, p in enumerate(v):
                                acc.append(int(p==label_guessed[ind][i2]))
                    print("Test Iteration {}, Accuracy: {:.3f}".format(i, sum(acc)/len(acc)))
                    if len(target) > 10:
                        valaccur.append(sum(acc)/len(acc))
            average_loss = sum(valaccur) / len(valaccur)
            print("Final test accuracy: {:.3f}".format(average_loss))
            if len(valaccur_e) > 0 and average_loss > max(valaccur_e):
                torch.save(net.state_dict(), "sprites_images_to_{}.pth".format(args.modality))
            elif len(valaccur_e) == 0:
                torch.save(net.state_dict(), "sprites_images_to_{}.pth".format(args.modality))
            valaccur_e.append(average_loss)
        print("Accuracy history per epoch: {}".format(valaccur_e))
    elif args.dataset.lower() == "cdsprites":
        mappings = {"shape":["square", "ellipse", "heart"], "size":["big", "small"],
                    "color":["blue", "green", "red", "yellow", "pink"],
                    "position":["at top left", "at top right", "at bottom left", "at bottom right"],
                    "background":["on light", "on dark"]}
        from models.datasets import CDSPRITESPLUS
        batch_size = 64
        epochs = 20
        dataset = CDSPRITESPLUS("./data/CdSpritesplus/level{}/traindata.h5".format(args.level), None, "image")
        print("Loading data....")
        d = dataset.get_data()
        labels = dataset.labels()
        if args.level == 1:
            labels = [mappings["shape"].index(x) for x in labels]
        elif args.level == 2:
            if args.modality == "shape":
                labels = [mappings["shape"].index(x[1]) for x in labels]
            else:
                labels = [mappings["size"].index(x[0]) for x in labels]
        elif args.level == 3:
            if args.modality == "shape":
                labels = [mappings["shape"].index(x[2]) for x in labels]
            elif args.modality == "size":
                labels = [mappings["size"].index(x[0]) for x in labels]
            elif args.modality == "color":
                labels = [mappings["color"].index(x[1]) for x in labels]
        elif args.level == 4:
            if args.modality == "shape":
                labels = [mappings["shape"].index(x[2]) for x in labels]
            elif args.modality == "size":
                labels = [mappings["size"].index(x[0]) for x in labels]
            elif args.modality == "color":
                labels = [mappings["color"].index(x[1]) for x in labels]
            elif args.modality == "position":
                labels = [mappings["position"].index(x[3]) for x in labels]
        elif args.level == 5:
            if args.modality == "shape":
                epoch = 20
                labels = [mappings["shape"].index(x[2]) for x in labels]
            elif args.modality == "size":
                labels = [mappings["size"].index(x[0]) for x in labels]
            elif args.modality == "color":
                labels = [mappings["color"].index(x[1]) for x in labels]
            elif args.modality == "position":
                labels = [mappings["position"].index(x[3]) for x in labels]
            elif args.modality == "background":
                labels = [mappings["background"].index(x[4]) for x in labels]
        shuffle = np.random.permutation(len(d))
        d = d[shuffle]
        labels_all = torch.tensor(np.asarray(labels)[shuffle]).to("cuda")
        dataset_train = TensorDataset(d[:25000])
        dataset_val = TensorDataset(d[25000:])
        labels_train = labels_all[:25000]
        labels_val = labels_all[25000:]
        trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        testloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        print("Initializing network...")
        net = CNN(args.modality).to("cuda")
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        trainloss_e = []
        valaccur_e = []
        for e in range(epochs):
            trainloss = []
            valaccur = []
            net.train()
            for i, data in enumerate(trainloader):
                optimizer.zero_grad()
                output = net(data.to("cuda"))
                bs = data.shape[0]
                target = labels_train[i*bs:(i*bs)+bs]
                l = loss(output, target.cuda())/bs
                print("Epoch {}, Iteration {}, Train Loss: {:.3f}".format(e, i, l))
                trainloss.append(l)
                l.backward()
                optimizer.step()
            average_loss = sum(trainloss) / len(trainloss)
            print("End of epoch {}. Final train loss: {}".format(e, average_loss))
            trainloss_e.append(average_loss)
            trainloss = []
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    acc = []
                    output = net(data.to("cuda"))
                    bs = data.shape[0]
                    target = labels_val[i*bs:(i*bs)+bs]
                    label_guessed = torch.argmax(output, dim=-1)
                    for ind, v in enumerate(target):
                           acc.append(int(v==label_guessed[ind]))
                    print("Test Iteration {}, Accuracy: {:.3f}".format(i, sum(acc)/len(acc)))
                    if len(target) > 10:
                        valaccur.append(sum(acc)/len(acc))
            average_loss = sum(valaccur) / len(valaccur)
            print("Final test accuracy: {:.3f}".format(average_loss))
            if len(valaccur_e) > 0 and average_loss > max(valaccur_e):
                torch.save(net.state_dict(), "cdspritesplus_classifier_level{}_{}.pth".format(args.level, args.modality))
            elif len(valaccur_e) == 0:
                torch.save(net.state_dict(), "cdspritesplus_classifier_level{}_{}.pth".format(args.level, args.modality))
            valaccur_e.append(average_loss)
        print("Accuracy history per epoch: {}".format(valaccur_e))


