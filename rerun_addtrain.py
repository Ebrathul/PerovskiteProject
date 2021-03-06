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
from perovskite_classes import get_NN, get_CNN
from perovskite_classes import predict_MAE, get_new_data_bounderies, save_newdata_firstdata, get_mae_per_e, find_elements_not_used
import matplotlib.pyplot as plt
import csv

# torch.manual_seed(0)
# np.random.seed(0)

def rerun_add_train_data(trainsetaddition, NN_number, log, al_level, name, element_cap):  # name as location of files
    print("Rerunning calculation:", name, al_level)
    # Save n Load
    model_checkpoint = 'NN_'  # name
    # global variables
    elemincompound = 3
    elements = generateElementdict()
    elementlabel = []
    for i in range(1, len(elements)):
        elementlabel.append(elements[i][0])
    elementstoprint = 20
    highest_element = 83
    fill_random = False
    # elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
    # element_cap = 90  # trainsetaddition  # max count of elements in new data

    logcount = 1
    # al_level = 0
    # while (os.access(log + "/run_" + str(logcount), os.F_OK) == True):  # +str(NN_index)
    #     logcount += 1
    # while (os.access(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level), os.F_OK) == True):
    #     al_level += 1


    # load data
    data = np.load(open("data11.npy", "rb"))
    train_data = np.load(open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/" + "traindata.npy", 'rb'))
    new_train_data = np.load(open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/new_data.npy", 'rb'), allow_pickle=True)

    real_train_data = np.vstack((train_data, new_train_data))
    real_train_data, index = np.unique(real_train_data, axis=0, return_counts=True)
    real_train_data = np.delete(real_train_data, np.nonzero(index > 1)[0], axis=0)

    data = np.vstack((data, real_train_data))
    data, index = np.unique(data, axis=0, return_counts=True)
    val_data = np.delete(data, np.nonzero(index > 1)[0], axis=0)

    # val_data = val_data[:3, :]
    # print("All val data", val_data[:, :4], val_data.shape, np.mean(val_data[:, 0]))
    # val_data = np.vstack((data, new_train_data))

    # print("nonzero", np.nonzero(index > 1), np.nonzero(index > 1)[0].shape)
    # print("Shape of all data", data.shape, train_data.shape, val_data.shape, index.shape, new_train_data.shape, real_train_data.shape)
    # val_data = np.load(open(log + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/" + "valdata.npy", 'rb'))
    # print("val data shape and ex:", val_data.shape, val_data[0])

    mean, stnddev = get_mean_stndev(real_train_data)
    # print("mean and stnd", mean, stnddev)

    # normalization
    val_data_x = (val_data[:, 1::] - mean[1::]) / stnddev[1::]
    # print("val data_x shape and ex:", val_data_x.shape, val_data_x[0])

    # for CNN
    size = val_data_x.shape
    val_data_x = val_data_x.reshape((size[0], 1, size[1]))
    # print("val data_x shape and ex:", val_data_x.shape, val_data_x[0])

    # netvariabeles
    feattotal = int(val_data_x.shape[1])

    # model = get_NN(feattotal)
    model = get_CNN(feattotal)

    mae, energy, index = predict_MAE(NN_number, val_data, val_data_x, stnddev, mean, model, al_level, (log + "/" + name + "/" + model_checkpoint))

    # find new data and which materials are chosen
    # train_data, elementcount, new_index = get_new_data_bounderies(val_data, elements, trainsetaddition, elemincompound, index, element_cap, fill_random)

    # save

    if os.path.isfile(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/all_new_data.npy" == True):
        print("All new data loaded")
        all_new_data = np.load(open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/all_new_data.npy", 'rb'), allow_pickle=True)
    else:
        print("All new data generated")
        all_new_data = save_newdata_firstdata(train_data, new_train_data, al_level, logcount, log, elements, elemincompound, elementlabel, name + "/" + model_checkpoint)
    print("All new data:", al_level, all_new_data.shape)
    # mae per elem
    # elemMAE = get_mae_per_e(mae, val_data)

    # elements_not_used = find_elements_not_used(elementcount, elements, elementstoprint)

    # load train and val data with chosen
    train_data = np.load(open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/traindata.npy", 'rb'), allow_pickle=True)
    # val_data = np.load(open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/valdata.npy", 'rb'), allow_pickle=True)

    # plot area
    #
    # # Energyhistogram
    # n, bins, _ = plt.hist(new_train_data[:, 0], 100)
    # plt.title('Energyhistogram of AL chosen data')
    # plt.savefig(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/energydistr.png")
    # plt.show()


    # Fixing random state for reproducibility


    def scatter_hist(x_1, y_1, x_2, y_2, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)


        volume_1 = 7
        volume_2 = 20
        close = 1

        # the scatter plot:
        ax.scatter(x_1, y_1, s=volume_1, alpha=1)  # c is color
        ax.scatter(x_2, y_2, c="red", s=volume_2, alpha=1)

        # now determine nice limits by hand:
        binwidth = 20  # 0.25
        try:
            xymax = np.max(np.max(np.max(np.abs(x_1), axis=0), np.max(np.abs(y_1), axis=0)), np.max(np.max(np.abs(x_2), axis=0), np.max(np.abs(y_2)), axis=0))
        except:
            xymax = max(np.max(np.abs(x_1)), np.max(np.abs(y_1)))


        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(0, lim + binwidth, binwidth)
        ax_histx.hist(x_1, bins=bins)
        ax_histx.set_ylabel('Count')
        ax_histy.hist(y_1, bins=bins, orientation='horizontal')
        ax_histy.set_xlabel('Count')
        ax.set_xlabel('Distance to convex Hull [meV]', fontsize=15)
        ax.set_ylabel('MAE of compound [meV]', fontsize=15)
        # plt.title('Distance to convex Hull')
        # plt.ylabel('Elements MAE meV')
        # plt.xlabel('Distance to convex Hull')


    # generate datasets for colours
    print("index", index.shape, index)
    chosen_compounds = val_data[index]
    energy_chosen = np.take_along_axis(energy, index[:trainsetaddition], axis=0)  # energy[index]
    mae_chosen = np.take_along_axis(mae, index[:trainsetaddition], axis=0)  # mae[index]
    print("mae and energy of chosen", energy.shape, mae.shape, energy_chosen, mae_chosen)
    print("mae and energy of chosen", energy_chosen.shape, mae_chosen.shape, energy_chosen, mae_chosen)


    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.030

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(energy, mae, energy_chosen, mae_chosen, ax, ax_histx, ax_histy)

    plt.savefig(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/energy_mae_chosen_red.png")
    plt.show()




    # # Elements MAE
    # # y_pos = np.arange(len(elementlabel)+2)
    # y_pos = np.arange(highest_element)
    # plt.bar(np.arange(1, 84), elemMAE[1:highest_element + 1, 0], align='center', alpha=0.5)
    # plt.xticks(y_pos, elementlabel[:83])
    # plt.ylabel('Elements MAE meV')
    # plt.title('Elements MAE')
    # plt.savefig(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/elemMAE.png")
    # plt.show()
    #
    #
    # # write mae of elements to csv file for PSE generation
    # with open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + '/ElementMAE.csv', 'w', newline='') as csvfile:
    #     element_writer = csv.writer(csvfile, delimiter=',')  # , quotechar='|', qouting=csv.QOUTE_MINIMAL)
    #     # print("CSV:")
    #     # print(elementlabel, len(elementlabel))
    #     # print(elemMAE[:, 0], elemMAE[1:, 0], elemMAE.shape)
    #     for i in range(highest_element):
    #         element_writer.writerow([elementlabel[i], elemMAE[i+1, 0]])
    #
    #
    # # Elementcount AL
    # y_pos = np.arange(highest_element)
    # # print(y_pos.shape, elementcount.shape, elementlabel)
    # plt.bar(y_pos, elementcount[1:highest_element + 1], align='center', alpha=0.5)
    # plt.xticks(y_pos, elementlabel)
    # plt.ylabel('Count of Compounds')
    # plt.title('Elements chosen by AL')
    # plt.savefig(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/elem_in_addtrain.png")
    # plt.show()
    #
    #
    # # write count of elements to csv file for PSE generation
    # with open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + '/Elementcount.csv', 'w', newline='') as csvfile:
    #     element_writer = csv.writer(csvfile, delimiter=',')  # , quotechar='|', qouting=csv.QOUTE_MINIMAL)
    #     # print("CSV:")
    #     # print(elementlabel, len(elementlabel))
    #     # print(elementcount[:], elementcount[0], elementcount.shape)
    #     for i in range(highest_element):
    #         element_writer.writerow([elementlabel[i], elementcount[i+1]])
    #
    #
    # All new data
    # if al_level > 0:
    #     elemcountlist = np.zeros((len(elements) + 1, elemincompound + 1))
    #     for i in range(1, len(elements)):
    #         # number, group, row = elements[i]
    #         element_i = elements[i]
    #         number, group, row = element_i[1:4]
    #         for j in range(len(all_new_data)):
    #             for k in range(elemincompound):  # count of elements in compound
    #                 if all_new_data[j, k] == i:
    #                     elemcountlist[i, k] += 1
    #                     elemcountlist[i, 3] += 1
    #
    #     # y_pos = np.arange(highest_element)  # np.arange(len(elementlabel) + 2)
    #     # print(y_pos.shape, len(elementlabel))
    #     # plt.bar(y_pos, elemcountlist[1:highest_element + 1, 3], align='center', alpha=0.5)
    #     # plt.xticks(y_pos, elementlabel[:highest_element])
    #     # plt.ylabel('Count of Compounds')
    #     # plt.title('All Elements chosen by AL')
    #     # plt.savefig(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/all_elem_in_addtrain.png")
    #     # plt.show()
    #
    #     # write count all of elements to csv file for PSE generation
    #     with open(log + "/" + name + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + '/Elementcount_All_AL.csv', 'w', newline='') as csvfile:
    #         element_writer = csv.writer(csvfile, delimiter=',')  # , quotechar='|', qouting=csv.QOUTE_MINIMAL)
    #         # print("CSV:")
    #         # print(elementlabel, len(elementlabel))
    #         # print(elementcount[:], elementcount[0], elementcount.shape)
    #         for i in range(highest_element):
    #             element_writer.writerow([elementlabel[i], elemcountlist[i + 1, 3]])


name = 'CNN_AL30a500_start5000_a70_MAE117.5_TEST'
log = 'active'
al_steps = 30
NN_number = 2
trainsetsize = 5000
trainsetaddition = 500
element_cap = 70


# re-run last step
# rerun_add_train_data(trainsetaddition, NN_number, log, 0, name, element_cap)  # name as location of files
# rerun_add_train_data(trainsetaddition, NN_number, log, al_steps-1, name, element_cap)  # name as location of files

# re-run all steps
for al_level in range(al_steps):
    rerun_add_train_data(trainsetaddition, NN_number, log, al_level, name, element_cap)  # name as location of files

print("END")
