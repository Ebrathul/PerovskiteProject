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

from perovskite_classes import get_mean_stndev, getRandomSets, PerovskiteDataset, create_supervised_trainer, \
    create_supervised_evaluator
from perovskite_classes import flatten, wrapped, create_dataset, generateElementdict
from perovskite_classes import prepare_batch_conv as prepare_batch  # _conv
from perovskite_classes import generateData
from perovskite_classes import get_NN, get_CNN
from perovskite_classes import predict_MAE, get_new_data_bounderies, save_newdata_firstdata, get_mae_per_elem, find_elements_not_used
import matplotlib.pyplot as plt
import csv



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
    print("val data shape and ex:", val_data.shape, val_data[0])

    mean, stnddev = get_mean_stndev(train_data)

    # normalization
    val_data_x = (val_data[:, 1::] - mean[1::]) / stnddev[1::]
    print("val data_x shape and ex:", val_data_x.shape, val_data_x[0])

    # for CNN
    size = val_data_x.shape
    val_data_x = val_data_x.reshape((size[0], 1, size[1]))
    print("val data shape and ex:", val_data_x.shape, val_data_x[0])


    # netvariabeles
    feattotal = int(val_data_x.shape[1])

    # model = get_NN(feattotal)
    model = get_CNN(feattotal)

    # mae and stnd of NNs is calculated
    mae, mae_predicted, index = predict_MAE(NN_number, val_data, val_data_x, stnddev, mean, model, model_checkpoint)

    # find new data and witch materials are chosen
    new_train_data, elementcount, new_index = get_new_data_bounderies(val_data, elements, trainsetaddition, elemincompound, index, element_cap, fill_random)

    # save
    all_new_data = save_newdata_firstdata(train_data, new_train_data, al_level, logcount, log, elements, elemincompound, elementlabel, model_checkpoint)

    # mae per elem
    elemMAE = get_mae_per_elem(mae, val_data, val_data_x)
    elemMAE_predicted = get_mae_per_elem(mae_predicted, val_data, val_data_x)


    elements_not_used = find_elements_not_used(elementcount, elements, elementstoprint)


    # main program
    train_data = np.vstack((train_data, new_train_data))  # traindata is old data + new data
    val_data = np.delete(val_data, new_index, axis=0)  # delete materials chosen by al
    # val_data = np.delete(val_data, index, axis=0)  # index needs to be changed


    # plot area
    # Energyhistogram
    n, bins, _ = plt.hist(new_train_data[:, 0], 100)
    plt.title('Energyhistogram of AL chosen data')
    plt.savefig(log + "/run_" + str(logcount - 1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/energydistr.png")
    plt.show()
    # Elements MAE
    # y_pos = np.arange(len(elementlabel)+2)
    y_pos = np.arange(83)
    plt.bar(elemMAE[1:, 1], elemMAE[1:, 0], align='center', alpha=0.5)
    plt.xticks(y_pos, elementlabel[:83])
    plt.ylabel('Elements MAE meV')
    plt.title('Elements MAE')
    plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/elemMAE.png")
    plt.show()

    # write mae of elements to csv file for PSE generation
    with open(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + '/PSE.csv', 'w', newline='') as csvfile:
        element_writer = csv.writer(csvfile, delimiter=',')  # , quotechar='|', qouting=csv.QOUTE_MINIMAL)
        for i in range(83):
            element_writer.writerow([elementlabel[i], elemMAE[i, 0]])


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
            # number, group, row = elements[i]
            element_i = elements[i]
            number, group, row = element_i[1:4]
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
