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
from perovskite_classes import flatten, wrapped, create_dataset, generateElementdict
from perovskite_classes import prepare_batch_conv as prepare_batch  # _conv
from perovskite_classes import generateData
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torchvision import transforms
import matplotlib.pyplot as plt
import pymatgen as mg
import pymatgen.core.periodic_table as peri


def add_train_data(trainsetaddition, NN_number, log, al_level):
    # Save n Load
    model_checkpoint = 'NN_'  # name
    logcount = 0
    al_level = 0
    while (os.access(log + "/run_" + str(logcount), os.F_OK) == True):  # +str(NN_index)
        logcount += 1
    while (os.access(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level), os.F_OK) == True):
        al_level += 1


    # load data
    train_data = np.load(open('traindata.npy', 'rb'))
    val_data = np.load(open('valdata.npy', 'rb'))
    print("val data shape and ex:", val_data.shape, val_data[0])


    mean, stnddev = get_mean_stndev(train_data)

    # normalization
    val_data_x = (val_data[:, 1::] - mean[1::]) / stnddev[1::]
    print("val data_x shape and ex:", val_data_x.shape, val_data_x[0])

    # for CNN
    # size = val_data_x.shape
    # val_data_x = val_data_x.reshape((size[0], 1, size[1]))
    # print("val data shape and ex:", val_data_x.shape, val_data_x[0])


    # netvariabeles
    D_in = int(val_data_x.shape[1])
    D_out = 1
    H1 = 20
    H2 = 64
    H3 = 7
    # H6 = 10


    # Sequential net, structure and functions
    # Working NOT conv NN
    model = nn.Sequential(
        nn.Linear(D_in, H1),
        nn.LeakyReLU(),
        nn.Linear(H1, H2),
        nn.LeakyReLU(),
        nn.Linear(H2, H3),
        nn.Tanh(),
        nn.Linear(H3, D_out),
    )  # lr:0.1

    # model = nn.Sequential(
    #     # wrapped(),
    #
    #     # nn.Conv1d(1, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
    #     # nn.ELU(),
    #     #
    #     # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     # nn.Conv1d(25, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
    #     # nn.ELU(),
    #     #
    #     # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     # nn.Conv1d(25, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
    #     # nn.ELU(),
    #
    #     # nn.Dropout(0.1),
    #     # nn.AvgPool1d(2),
    #
    #     # torch.nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(1, 20, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
    #     # in_channels, out_channels, kernel_size
    #     nn.ELU(),
    #
    #     torch.nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(20, 20, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
    #     # in_channels, out_channels, kernel_size
    #     nn.ELU(),
    #
    #     nn.Dropout(0.1),
    #     # nn.MaxPool1d(3),
    #
    #     torch.nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(20, 20, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
    #     # in_channels, out_channels, kernel_size
    #     nn.ELU(),
    #
    #     torch.nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(20, 25, 5, stride=1, padding=2, dilation=2, groups=1, bias=True),
    #     nn.ELU(),
    #
    #     torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(25, 25, 5, stride=1, padding=3, dilation=1, groups=1, bias=True),
    #     nn.Dropout(0.1),
    #     # testarea
    #     nn.ELU(),
    #
    #     nn.Dropout(0.01),
    #
    #     torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(25, 25, 3, stride=1, padding=2, dilation=2, groups=1, bias=True),
    #     nn.ELU(),
    #     # print("1"),
    #     nn.Dropout(0.05),
    #     nn.Conv1d(25, 25, 5, stride=1, padding=3, dilation=1, groups=1, bias=True),
    #     nn.ELU(),
    #
    #     torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Conv1d(25, 25, 3, stride=1, padding=2, dilation=2, groups=1, bias=True),
    #     # print("2"),
    #     # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     # nn.Conv1d(25, 25, 3, stride=1, padding=2, dilation=4, groups=1, bias=True),
    #     # print("3"),
    #     # torch.nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     # nn.Conv1d(25, 25, 3, stride=2, padding=2, dilation=2, groups=1, bias=True),
    #     nn.ELU(),
    #
    #     # nn.AvgPool1d(3),
    #     nn.ELU(),
    #     flatten(),
    #     nn.Dropout(0.01),
    #     nn.Linear(150, H2),
    #     nn.ELU(),
    #     nn.Linear(H2, H3),
    #     nn.Tanh(),
    #     nn.Linear(H3, D_out)
    # )


    predictions = np.zeros((NN_number, len(val_data)))
    for NN_index in range(NN_number):
        if (os.path.isfile(model_checkpoint + str(NN_index) + '.pt')):
            print("NN: ", NN_index, "loaded")
            checkpoint = torch.load(model_checkpoint + str(NN_index) + '.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # model.load_state_dict(torch.load(model_checkpoint + str(NN_index) + '.pt', map_location="gpu"))
            # max_epoch = 1000  # imprtance?

            predictions[NN_index] = model(torch.tensor(val_data_x)).detach().cpu().numpy().reshape(len(val_data_x),)
        else:
            print('model not available')
        # predictions[i] = np.append(predictions[i], model(val_data_x), axis=0)


    mean_p = np.mean(predictions, axis=0)* stnddev[0]
    # print("mean_p val_data stnddev[0] shape:", mean_p.shape, val_data.shape, stnddev[0].shape)
    errorofprediction = mean_p - val_data[:, 0]
    std_p = np.std(predictions, axis=0)
    print(std_p.shape)

    index = np.flip(np.argsort(std_p))[0:trainsetaddition]

    new_train_data = []

    for i in range(len(index)):
        new_train_data.append(val_data[index[i]])
    new_train_data = np.asarray(new_train_data)

    n, bins, _ = plt.hist(new_train_data[:, 0], 50)
    plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level-1) + "/energydistr.png")
    plt.show()


    train_data = np.vstack((train_data, new_train_data))

    # print(train_data[])
    # print(train_data.shape)
    # n, bins, _ = plt.hist(train_data[:, 0], 50)
    # plt.show()


    # loop for finding mean MAE one elements
    elemincompound = 3
    elements = generateElementdict()
    elementlabel = []
    number = 0
    # print(elements[1], len(elements))
    featperelem = 11
    elemMAE = np.zeros((len(elements) + 1, 2))
    # np.delete(elemMAE, 0, axis=0)
    # np.delete(elemMAE, len(elemMAE), 0)
    for i in range(1, len(elements)):
        number, group, row = elements[i]
        elementlabel.append(elements[i][0])
        # print("number, group, row", number, group, row)
        for j in range(len(val_data_x)):
            for k in range(elemincompound):  # count of elements in compound
                if i == int(val_data[j, k]):  # 0, 1, 2 are atomic numbers
                    elemMAE[i, 0] += mean_p[j]
                    elemMAE[i, 1] += 1
        if elemMAE[i, 1] != 0:
            elemMAE[i, 0] = elemMAE[i, 0] / elemMAE[i, 1]
            elemMAE[i, 1] = i
            print(elements[i][0], elemMAE[i, 0], elemMAE[i, 1])
        else:
            print(elements[i][0], "division by zero")
            elemMAE[i, 0] = 0
    # n, bins, _ = plt.hist(elemMAE[:, 0], 100)
    # plt.show()

    y_pos = np.arange(len(elementlabel)+2)
    # print("Len of elementlist", len(elementlabel))
    # print("shape of elemMAE", elemMAE[:, 0].shape, elemMAE[:, 0])
    plt.bar(elemMAE[:, 1], elemMAE[:, 0], align='center', alpha=0.5)
    plt.xticks(y_pos, elementlabel)
    plt.ylabel('Elements MAE meV')
    plt.title('Elements')
    plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level-1) + "/elemMAE.png")
    plt.show()


    # loop for finding witch materials are chosen
    # materials = []
    elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
    elementstoprint = 10
    featperelem = 2
    for i in range(1, len(elements)):
        number, group, row = elements[i]
        for j in range(trainsetaddition):
            for k in range(elemincompound):  # count of elements in compound
                if new_train_data[j, k] == i:
                    # materials[j].append(elements[i][0])
                    elemcountlist[i, k] += 1
                    elemcountlist[i, 3] += elemcountlist[i, k]
    # indexelement = np.flip(np.argsort(elemcountlist[:, 3]))[0:elementstoprint]
    # print("Elements choosen by AL")
    # for i in range(elementstoprint):  # print elements
    #     print(i, elements[indexelement(i)][0])
    # n, bins, _ = plt.hist(elemcountlist[:, 3], 100)
    # plt.show()
    y_pos = np.arange(len(elementlabel)+2)
    plt.bar(y_pos, elemcountlist[:, 3], align='center', alpha=0.5)
    plt.xticks(y_pos, elementlabel)
    plt.ylabel('Count of Compounds')
    plt.title('Elements chosen by AL')
    plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level-1) + "/elem_in_addtrain.png")
    plt.show()


    val_data = np.delete(val_data, index, axis=0)

    np.save(open(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level-1) + "/new_data.npy", 'wb'), new_train_data, allow_pickle=True)
    np.save(open('traindata.npy', 'wb'), train_data, allow_pickle=True)
    np.save(open('valdata.npy', 'wb'), val_data, allow_pickle=True)
# add_train_data(500, 3, 'active', 0)
