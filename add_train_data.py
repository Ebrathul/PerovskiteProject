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


def add_train_data(trainsetaddition, NN_number, log, al_level, element_cap, fill_random):
    # global variables
    elemincompound = 3
    elements = generateElementdict()
    elementlabel = []
    for i in range(1, len(elements)):
        elementlabel.append(elements[i][0])
    elementstoprint = 20
    # elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
    # element_cap = 100  # trainsetaddition  # max count of elements in new data

    # Save n Load
    model_checkpoint = 'NN_'  # name
    logcount = 0
    while (os.access(log + "/run_" + str(logcount), os.F_OK) == True):  # +str(NN_index)
        logcount += 1

    # load data
    train_data = np.load(open('traindata.npy', 'rb'))
    val_data = np.load(open('valdata.npy', 'rb'))
    # print("val data shape and ex:", val_data.shape, val_data[0])

    mean, stnddev = get_mean_stndev(train_data)

    # normalization
    val_data_x = (val_data[:, 1::] - mean[1::]) / stnddev[1::]
    # print("val data_x shape and ex:", val_data_x.shape, val_data_x[0])

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

    # prediction of ensemble
    predictions = np.zeros((NN_number, len(val_data)))
    for NN_index in range(NN_number):
        if os.path.isfile(model_checkpoint + str(NN_index) + '.pt'):
            print("NN: ", NN_index, "loaded")
            checkpoint = torch.load(model_checkpoint + str(NN_index) + '.pt')
            model.load_state_dict(checkpoint['model_state_dict'])

            predictions[NN_index] = model(torch.tensor(val_data_x)).detach().cpu().numpy().reshape(len(val_data_x),)
        else:
            print('model not available')

    mae = np.zeros(len(val_data_x))
    for i in range(NN_number):
        mae += np.abs((predictions[i, :] * stnddev[0] + mean[0]) - val_data[:, 0])
        mae = mae / NN_number
    std_p = np.std(predictions, axis=0)

    index = np.asarray(np.flip(np.argsort(std_p))[0:len(val_data_x)])


    def get_new_data_bounderies(val_data, index, element_cap, random):
        # loop for generating new data AND
        # loop for finding witch materials are chosen
        new_train_data = []
        new_index = []
        index_counter = 0
        materials_skipped = 0
        random_on_next_run = False
        print("len of elements", len(elements), "max per element", element_cap)
        while len(new_train_data) < trainsetaddition:
            if random_on_next_run:
                current_index = np.random.randint(index_counter, len(index))
                print("Random addition on cap", current_index)
                random_on_next_run = False
            else:
                current_index = index[index_counter]
            current_data = new_train_data.copy()
            current_data.append(val_data[current_index])
            current_data_array = np.asarray(current_data)
            # print("val_data[index[index_counter]]", len(val_data[index[index_counter]]), index[index_counter], index_counter)

            # print("start count loop", len(new_train_data), len(current_data), index_counter)
            # loop for finding witch materials are chosen
            elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
            for i in range(1, len(elements)):
                # print("Element", i, elemcountlist.shape, len(new_train_data), index_counter)
                for j in range(len(current_data)):  # count of compounds
                    for k in range(elemincompound):  # count of elements in compound
                        if current_data_array[j, k] == i:
                            elemcountlist[i, k] += 1
                            elemcountlist[i, 3] += 1
                if elemcountlist[i, 3] >= element_cap:
                    print("material not appended, break", i, elemcountlist[i, 3], index_counter, )
                    print("new traindata", len(new_train_data), len(new_index), index_counter)
                    index_counter += 1
                    materials_skipped += 1
                    if fill_random:
                        random_on_next_run = True
                        index_counter -= 1
                    break
                # last step
                if elemcountlist[len(elements)-1, 3] < element_cap and i == len(elements)-1:
                    # print("material appended")
                    new_train_data.append(val_data[current_index])
                    new_index.append(current_index)
                    index_counter += 1
        new_train_data = np.asarray(new_train_data)
        new_index = np.asarray(new_index)
        print("new_train_data", new_train_data.shape, new_index.shape, index_counter)
        print("materials skipped", materials_skipped)
        return new_train_data, np.asarray(elemcountlist[:, 3]), new_index
    new_train_data, elementcount, new_index = get_new_data_bounderies(val_data, index, element_cap, fill_random)

    # for i in range(changecount):
    #     new_train_data = np.vstack((new_train_data, val_data[index[(trainsetaddition+changecount)]]))

    # for i in range(trainsetaddition):  # here a reduced method to find elements can be added
    #     new_train_data.append(val_data[index[i]])
    # new_train_data = np.asarray(new_train_data)


    if al_level != 0:
        if os.path.isfile(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level-1) + '/all_new_data.npy'):
            all_new_data = np.load(open(log + "/run_" + str(logcount - 1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level-1) + "/" + "all_new_data.npy", 'rb'))
            all_new_data = np.vstack((new_train_data, all_new_data))
    else:
        print("AL_level", al_level)
        first_train_data = train_data.copy()
        all_new_data = new_train_data
        # plot first random set
        elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
        for i in range(1, len(elements)):
            number, group, row = elements[i]
            for j in range(len(first_train_data)):
                for k in range(elemincompound):  # count of elements in compound
                    if first_train_data[j, k] == i:
                        elemcountlist[i, k] += 1
                        elemcountlist[i, 3] += 1
        y_pos = np.arange(len(elementlabel) + 2)
        print(y_pos.shape, len(elementlabel))
        plt.bar(y_pos, elemcountlist[:, 3], align='center', alpha=0.5)
        plt.xticks(y_pos, elementlabel)
        plt.ylabel('Count of Compounds')
        plt.title('First Random Traindata')
        plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/first_random_traindata.png")
        plt.show()


    # loop for finding mean MAE one elements
    def get_mae_per_elem(mae):
        elemincompound = 3
        elements = generateElementdict()
        elementlabel = []
        for i in range(1, len(elements)):
            elementlabel.append(elements[i][0])
        elemMAE = np.zeros((len(elements) + 1, 2))
        for i in range(1, len(elements)):
            number, group, row = elements[i]
            # print("number, group, row", number, group, row)
            for j in range(len(val_data_x)):
                for k in range(elemincompound):  # count of elements in compound
                    if i == int(val_data[j, k]):  # 0, 1, 2 are atomic numbers
                        elemMAE[i, 0] += mae[i]
                        elemMAE[i, 1] += 1
            if elemMAE[i, 1] != 0:
                elemMAE[i, 0] = elemMAE[i, 0] / elemMAE[i, 1]
                elemMAE[i, 1] = i
                print(elements[i][0], elemMAE[i, 0], elemMAE[i, 1])
            else:
                print(elements[i][0], "division by zero")
                elemMAE[i, 0] = 0
        return elemMAE
    elemMAE = get_mae_per_elem(mae)


    # new_elementcount = np.asarray(elemcountlist[:, 3])
    print("count for each element", elementcount)  # counts how often each element was chosen
    elements_not_used = []
    print("Elements not used")
    for i in range(elementcount):
        if elementcount == 0 and i<0:
            elements_not_used.append(i)
            print(elements[i][0], i)
    elements_not_used = np.asarray(elements_not_used)

    indexelement = np.flip(np.argsort(elementcount, axis=0))
    print("index_element", indexelement.shape)
    print("Elements choosen by AL")
    for i in range(elementstoprint):  # print elements
        print(indexelement[i], elements[indexelement[i]][0], elementcount[indexelement[i]])


    train_data = np.vstack((train_data, new_train_data))  # traindata is old data + new data
    val_data = np.delete(val_data, new_index, axis=0)  # delete materials chosen by al
    # val_data = np.delete(val_data, index, axis=0)  # index needs to be changed


    # plot area
    # Energyhistogram
    n, bins, _ = plt.hist(new_train_data[:, 0], 100)
    plt.savefig(log + "/run_" + str(logcount - 1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/energydistr.png")
    plt.show()
    # Elements MAE
    y_pos = np.arange(len(elementlabel)+2)
    plt.bar(elemMAE[:, 1], elemMAE[:, 0], align='center', alpha=0.5)
    plt.xticks(y_pos, elementlabel)
    plt.ylabel('Elements MAE meV')
    plt.title('Elements')
    plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/elemMAE.png")
    plt.show()
    # Elementcount AL
    y_pos = np.arange(len(elementlabel)+2)
    # print(y_pos.shape, elementcount.shape, elementlabel)
    plt.bar(y_pos, elementcount, align='center', alpha=0.5)
    plt.xticks(y_pos, elementlabel)
    plt.ylabel('Count of Compounds')
    plt.title('Elements chosen by AL')
    plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/elem_in_addtrain.png")
    plt.show()
    # All new data
    if al_level > 0:
        elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
        for i in range(1, len(elements)):
            number, group, row = elements[i]
            for j in range(len(all_new_data)):
                for k in range(elemincompound):  # count of elements in compound
                    if all_new_data[j, k] == i:
                        elemcountlist[i, k] += 1
                        elemcountlist[i, 3] += 1
        y_pos = np.arange(len(elementlabel) + 2)
        print(y_pos.shape, len(elementlabel))
        plt.bar(y_pos, elemcountlist[:, 3], align='center', alpha=0.5)
        plt.xticks(y_pos, elementlabel)
        plt.ylabel('Count of Compounds')
        plt.title('All Elements chosen by AL')
        plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/all_elem_in_addtrain.png")
        plt.show()


    np.save(open(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/new_data.npy", 'wb'), new_train_data, allow_pickle=True)
    np.save(open(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/traindata.npy", 'wb'), train_data, allow_pickle=True)
    np.save(open(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/valdata.npy", 'wb'), val_data, allow_pickle=True)
    np.save(open(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/all_new_data.npy", 'wb'), all_new_data, allow_pickle=True)

    np.save(open('all_new_data.npy', 'wb'), all_new_data, allow_pickle=True)
    np.save(open('traindata.npy', 'wb'), train_data, allow_pickle=True)
    np.save(open('valdata.npy', 'wb'), val_data, allow_pickle=True)
# add_train_data(500, 3, 'active', 0)
