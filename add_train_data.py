import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import gzip, pickle
import pymatgen as mg
import pymatgen.core.periodic_table as peri
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.utils import convert_tensor
from ignite.metrics import Loss, MeanAbsoluteError
import ignite.metrics
import ignite
import ignite.contrib.handlers
from torch.utils.tensorboard import SummaryWriter
# import tensorboardX
from torchsummary import summary
import timeit
import torch
from perovskite_classes import get_mean_stndev, getRandomSets, PerovskiteDataset, create_supervised_trainer, \
    create_supervised_evaluator
from perovskite_classes import flatten, wrapped, create_dataset
from perovskite_classes import prepare_batch_conv as prepare_batch  # _conv
from perovskite_classes import generateData
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torchvision import transforms
import matplotlib.pyplot as plt


def add_train_data(trainsetaddition, NN_number):
    model_checkpoint = 'NN_'  # name
    D_in = int(6)  # !!!!!!!!! Has to be set when changing to different featurecount
    D_out = 1
    H1 = 20
    H2 = 64
    H3 = 7
    # H6 = 10


    # Sequential net, structure and functions
    # Working NOT conv NN
    # model = nn.Sequential(
    #     nn.Linear(D_in, H1),
    #     nn.LeakyReLU(),
    #     nn.Linear(H1, H2),
    #     nn.LeakyReLU(),
    #     nn.Linear(H2, H3),
    #     nn.Tanh(),
    #     nn.Linear(H3, D_out),
    # )  # lr:0.1

    model = nn.Sequential(
        # wrapped(),

        # nn.Conv1d(1, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # nn.ELU(),
        #
        # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # nn.Conv1d(25, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # nn.ELU(),
        #
        # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # nn.Conv1d(25, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # nn.ELU(),

        # nn.Dropout(0.1),
        # nn.AvgPool1d(2),

        # torch.nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(1, 20, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # in_channels, out_channels, kernel_size
        nn.ELU(),

        torch.nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(20, 20, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # in_channels, out_channels, kernel_size
        nn.ELU(),

        nn.Dropout(0.1),
        # nn.MaxPool1d(3),

        torch.nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(20, 20, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # in_channels, out_channels, kernel_size
        nn.ELU(),

        torch.nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(20, 25, 5, stride=1, padding=2, dilation=2, groups=1, bias=True),
        nn.ELU(),

        torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(25, 25, 5, stride=1, padding=3, dilation=1, groups=1, bias=True),
        nn.Dropout(0.1),
        # testarea
        nn.ELU(),

        nn.Dropout(0.01),

        torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(25, 25, 3, stride=1, padding=2, dilation=2, groups=1, bias=True),
        nn.ELU(),
        # print("1"),
        nn.Dropout(0.05),
        nn.Conv1d(25, 25, 5, stride=1, padding=3, dilation=1, groups=1, bias=True),
        nn.ELU(),

        torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(25, 25, 3, stride=1, padding=2, dilation=2, groups=1, bias=True),
        # print("2"),
        # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # nn.Conv1d(25, 25, 3, stride=1, padding=2, dilation=4, groups=1, bias=True),
        # print("3"),
        # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # nn.Conv1d(25, 25, 3, stride=2, padding=2, dilation=2, groups=1, bias=True),
        nn.ELU(),

        # nn.AvgPool1d(3),
        nn.ELU(),
        flatten(),
        nn.Dropout(0.01),
        nn.Linear(150, H2),
        nn.ELU(),
        nn.Linear(H2, H3),
        nn.Tanh(),
        nn.Linear(H3, D_out)
    )

    train_data = np.load(open('traindata.npy', 'rb'))
    val_data = np.load(open('valdata.npy', 'rb'))

    mean, stnddev = get_mean_stndev(train_data)

    # normalization
    val_data_x = (val_data[:, 1::] - mean[1::]) / stnddev[1::]
    print("val data shape and ex:", val_data_x.shape, val_data_x[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    predictions = np.zeros((NN_number, len(val_data)))
    for NN_index in range(NN_number):
        if (os.path.isfile(model_checkpoint + str(NN_index) + '.pt')):
            print("NN: ", NN_index, "loaded")
            checkpoint = torch.load(model_checkpoint + str(NN_index) + '.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # model.load_state_dict(torch.load(model_checkpoint + str(NN_index) + '.pt', map_location="gpu"))
            # max_epoch = 1000  # imprtance?
            predictions[NN_index] = model(torch.tensor(val_data_x)).detach().cpu().numpy().reshape(len(val_data_x),)  # Variable(torch.from_numpy(x)) ??????
        else:
            print('model not available')
        # predictions[i] = np.append(predictions[i], model(val_data_x), axis=0)

    std_p = np.std(predictions, axis=0)
    print(std_p.shape)

    index = np.flip(np.argsort(std_p))[0:trainsetaddition]

    new_train_data = []

    for i in range(len(index)):
        new_train_data.append(val_data[index[i]])
    new_train_data = np.asarray(new_train_data)

    n, bins, _ = plt.hist(new_train_data[:, 0], 50)
    plt.show()
    # print(train_data.shape)
    # print(new_train_data.shape)
    # n, bins, _ = plt.hist(train_data[:, 0], 50)
    # plt.show()

    # print(datatest[4995:5005])
    train_data = np.vstack((train_data, new_train_data))

    # print(train_data[])
    # print(train_data.shape)
    # n, bins, _ = plt.hist(train_data[:, 0], 50)
    # plt.show()

    val_data = np.delete(val_data, index, axis=0)

    np.save(open('traindata.npy', 'wb'), train_data, allow_pickle=True)
    np.save(open('valdata.npy', 'wb'), val_data, allow_pickle=True)
# add_train_data(1000)
