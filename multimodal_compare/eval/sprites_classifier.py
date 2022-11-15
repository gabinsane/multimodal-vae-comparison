import torch.nn as nn
import math
from torch.nn import BatchNorm3d, Sequential, ReLU, ModuleList
import torch.nn.functional as F
from models.nn_modules import SamePadConv3d, AttentionResidualBlock
import numpy as np
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modality", type=str, help="whether to train action or attribute classifier")
    args = parser.parse_args()
    from models.datasets import SPRITES
    epochs = 30
    batch_size = 64

    dataset = SPRITES("./data/sprites", "./data/sprites/test", "frames")
    print("Loading data....")
    d = dataset.get_data()
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    if args.modality == "action":
        labels = dataset.get_actions()
    else:
        labels = dataset.get_attributes()
    shuffle = np.random.permutation(len(d))
    d = d[shuffle]
    dataset_train = d[:int(len(d) * (1 - 0.1))]
    dataset_val = d[int(len(d) * (1 - 0.1)):]
    labels_train = torch.tensor(np.asarray(labels)[shuffle])[:int(len(d) * (1 - 0.1))].to("cuda")
    labels_val = torch.tensor(np.asarray(labels)[shuffle])[int(len(d) * (1 - 0.1)):].to("cuda")
    dataset_train = TensorDataset(dataset_train)
    dataset_val = TensorDataset(dataset_val)
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
            l = loss(output, target.cuda()).mean()
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
        valaccur_e.append(average_loss)
    print("Accuracy history per epoch: {}".format(valaccur_e))

