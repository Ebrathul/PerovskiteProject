import os, sys
import numpy as np
from torch.utils import data
import torch.nn as nn
import torch
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
import gzip, pickle
import pymatgen as mg
import pymatgen.core.periodic_table as peri
import matplotlib.pyplot as plt


def prepare_batch(batch, device=None, non_blocking=False):
    x = convert_tensor(batch["features"], device=device, non_blocking=non_blocking)
    y = convert_tensor(batch["ehull"], device=device, non_blocking=non_blocking)
    return x, y


def prepare_batch_conv(batch, device=None, non_blocking=False):
    x = convert_tensor(batch["features"], device=device, non_blocking=non_blocking)
    size = x.shape
    x = x.reshape((size[0], 1, size[1]))
    y = convert_tensor(batch["ehull"], device=device, non_blocking=non_blocking)
    return x, y


class flatten(nn.Module):
    def forward(self, x):
        # print("flatten shape", x.shape)
        return x.view(x.size()[0], -1)


class wrapped(nn.Module):  # probably not working
    def forward(self, x):  # , n):
        n = 1
        length = x.shape[2]
        return torch.cat([x[:, :, length-n:length],  x, x[:, :, 0:n]], 2)


class PerovskiteDataset(data.Dataset):
    def __init__(self, array,  transform=None):
        self.data = torch.Tensor(array)
        self.transform = transform  # to implement normaltization

    def __len__(self):  # returns total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx, 1:]
        ehull = self.data[idx, 0]
        return {"ehull": ehull, "features": features}


def create_supervised_trainer(model, optimizer, loss_fn, std=1,
                              device=None, non_blocking=False,
                              prepare_batch=prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x).view(-1)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        loss = loss*std  # normalization
        return output_transform(x, y, y_pred, loss)
    return Engine(_update)


def create_supervised_evaluator(model, metrics={}, std=1,
                                device=None, non_blocking=False,
                                prepare_batch=prepare_batch,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x).view(-1)*std  # normalization
            return output_transform(x, y*std, y_pred)
    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def get_mean_stndev(data):
    mean = np.mean(data, axis=0)  # axis 0 is top to bottom, 1 would be sideways
    stnddev = np.std(data, axis=0)  # , ddof=1)
    # print("Mean: ", mean, "\nStnd dev: ", stnddev)
    return mean, stnddev


def getRandomSets(X, trainsetsize):  # shuffles the data and returns a training and a test set # needs 8 or more samples
       np.random.shuffle(X)
       Xtrain = X[0:trainsetsize]
       Xtest = X[trainsetsize:len(X), :]
       # print("Trainset: ", Xtrain[0][:])
       # print("Validationset:", Xtest[0][:])
       return Xtrain, Xtest


def create_dataset(data, trainsetsize):
    train_data, val_data = getRandomSets(data, trainsetsize)  # change to 5000
    print("created random datasets")
    np.save(open('traindata.npy', 'wb'), train_data, allow_pickle=True)
    np.save(open('valdata.npy', 'wb'), val_data, allow_pickle=True)


def generateData(file):
    elementdict = generateElementdict()
    if os.path.isfile(file):
        newdata, featperelem, datavariables, feattotal = readdata(file)
        return newdata, elementdict, featperelem, datavariables, feattotal
    else:
        # Variable for Elements
        featperelem = (len(elementdict[1]) - 1)  # Elementname still in list
        datavariables = (1 + 3 * featperelem)  # Ehull + feattotal
        feattotal = datavariables - 1
        newdata = rawdataprocessing(elementdict, featperelem, datavariables, feattotal, file)
    return newdata, elementdict, featperelem, datavariables, feattotal


def generateElementdict():
    # Dictionary with Elements and their properties
    # Delete preprocessed data when changing featperEle
    elementdict = {}
    for i in range(1, 100):  # 100 Elements in Dict
        commonoxidationstate = peri.Element.from_Z(i).common_oxidation_states
        orbitals = peri.Element.from_Z(i).full_electronic_structure
        sandp_count = 0
        dandf_count = 0
        ionizationenergy = 0
        valence = 0

        if len(commonoxidationstate) == 0:
            commonoxidationstate = 0
        else:
            commonoxidationstate = peri.Element.from_Z(i).common_oxidation_states[0]

        for j in range(len(orbitals)):
            for k in range(len(orbitals[j])):
                if orbitals[j][k] == "s" or orbitals[j][k] == "p":
                    sandp_count += orbitals[j][2]  # count in third position
                if orbitals[j][k] == "d" or orbitals[j][k] == "f":
                    dandf_count += orbitals[j][2]

        if i == 1:
            ionizationenergy = 13.6
        else:
            ionizationenergy = ((i - 1) ^ 2) * 13.6

        """
        if i == 4 or i == 12 or i == 20:
            valence = 2
            print("alkaine earth set to 2 valence e-")
        else: 
            valence = peri.Element.from_Z(i).valence
            print("Element: ", i, "Valence: ", valence, type(valence))
        """  # transition metals not working

        elementdict[i] = [peri.Element.from_Z(i),  # name
                            peri.Element.from_Z(i).number,  # atomic number
                            mg.Element(peri.Element.from_Z(i)).group,  # group
                            peri.Element(peri.Element.from_Z(i)).row  # row
                        #   peri.Element.from_Z(i).X,  # Pauling electronegativity (none equals zero)
                        #   peri.Element.from_Z(i).number,  # atomic number
                        #   commonoxidationstate,  # common oxidation state if non set to zero
                        #   peri.Element.from_Z(i).average_ionic_radius,  # average ionic radius
                        #   mg.Element(peri.Element.from_Z(i)).atomic_mass,  # avarage mass
                        #   sandp_count,  # count of e- in s and p orbitals
                        #   dandf_count,  # couunt of e- in d and f orbitals
                        #   ionizationenergy,  # ionizationenergy in eV
                          ]
        # peri.Element.from_Z(i).valence]  # number of valence electrons
    # print("Element and feat.:", elementdict)
    return elementdict


def readdata(file):
    print('Loaded preprocessed data from file')  # Ehull, group1, raw1, elecneg1, group2, raw2, elecneg2, group3, raw3, elecneg3
    newdata = np.load(open(file, 'rb'))
    # print("real values: ", newdata[0, :])
    # Variable for Elements
    featperelem = ((len(newdata[0, :])-1)/3)  # Elementname still in list
    datavariables = (1 + 3 * featperelem)  # Ehull + feattotal
    feattotal = datavariables - 1
    print("featperelem, datavariables, feattotal:", featperelem, datavariables, feattotal)
    return newdata, featperelem, datavariables, feattotal


def rawdataprocessing(elementdict, featperelem, datavariables, feattotal, file):
    # Read and prepare Data
    dataraw = gzip.open('data.pickle.gz')
    data = pickle.load(dataraw, encoding='latin1')
    dataraw.close()
    print("Datensatzlaenge: ", len(data), "Anzahl der Features: ", datavariables)
    newdata = np.zeros((len(data), datavariables))  # Features(Energy + 3*Elementfeatures)
    # Generate prepared Data
    count = 0
    for i in range(len(data)):
        newdata[i, 0] = data[i, 0]
        print("Ehull", newdata[i, 0])
        for j in range(1, 3 + 1):  # elementcount
            for k in range(1, featperelem + 1):
                count += 1
                print("Count", count, "| i j k ", i, j, k)
                print("data: ", data[i, j])
                print("featcount:", k + ((j - 1) * featperelem))
                elemstr = data[i, j].decode('UTF-8')  # convert b'Elem' --> Elem
                print(elementdict[mg.Element(elemstr).number])
                print("k, feat.", k, elementdict[mg.Element(elemstr).number][k])
                print("-----------")
                newdata[i, (k + ((j - 1) * featperelem))] = elementdict[mg.Element(elemstr).number][k]
        print(newdata[i, :])
    np.save(open(file, 'wb'), newdata)
    # dataraw = gzip.open('dataperpared.pickle.gz', 'wb')
    # pickle.dump(x)
    return newdata


def predict_MAE(NN_number, val_data, val_data_x, stnddev, mean, model, model_checkpoint):
    # prediction of ensemble
    print("Calculate MAE")
    predictions = np.zeros((NN_number, len(val_data)))
    for NN_index in range(NN_number):
        if os.path.isfile(model_checkpoint + str(NN_index) + '.pt'):
            print("NN: ", NN_index, "loaded")
            checkpoint = torch.load(model_checkpoint + str(NN_index) + '.pt')
            model.load_state_dict(checkpoint['model_state_dict'])

            splitsize = 10000
            endintervall = 0
            while endintervall < len(val_data_x):
                beginintervall = endintervall
                endintervall += splitsize
                if endintervall >= len(val_data_x):
                    endintervall = len(val_data_x)
                val_data_x_slice = val_data_x[beginintervall:endintervall]
                # print("begin and end", beginintervall, endintervall)
                # print("val data slice shape and ex:", val_data_x_slice.shape, len(val_data_x_slice), val_data_x[0])
                predictions[NN_index, beginintervall:endintervall] = model(torch.tensor(val_data_x_slice)).detach().cpu().numpy().reshape(len(val_data_x_slice), )
            # predictions[NN_index] = model(torch.tensor(val_data_x)).detach().cpu().numpy().reshape(len(val_data_x),)  # for all at once
        else:
            print('model not available')

        mae = np.zeros(len(val_data_x))
        mae_predicted = np.zeros(len(val_data_x))
        for i in range(NN_number):
            # print("Val_data", val_data[i, 0], val_data[:, 0], val_data.shape)
            mae_predicted += np.abs((predictions[i, :] * stnddev[0] + mean[0]))
            mae += np.abs((predictions[i, :] * stnddev[0] + mean[0]) - val_data[:, 0])  # z normalization --> prediction - exact
            mae_predicted = mae_predicted / NN_number
            mae = mae / NN_number

        std_p = np.std(predictions, axis=0)

        index = np.asarray(np.flip(np.argsort(std_p))[0:len(val_data_x)])
        return mae, mae_predicted, index


def get_new_data_bounderies(val_data, elements, trainsetaddition, elemincompound, index, element_cap, fill_random):
    # loop for generating new data AND
    # loop for finding witch materials are chosen
    new_train_data = []
    new_index = []
    index_counter = 0
    materials_skipped = 0
    random_on_next_run = fill_random
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
                        if i == 2:
                            print("should this be: ", current_data_array[j, k])
                            print("Helium, i, j, k, :", i, j, k)
                            print("compound ABC3", current_data_array[j, 0], current_data_array[j, 1], current_data_array[j, 2], current_data_array[j, 3],
                                  val_data.shape)
                            print("elemMAE value & count:", elemcountlist[i, k], elemcountlist[i, 3])
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


def save_newdata_firstdata(train_data, new_train_data, al_level, logcount, log, elements, elemincompound, elementlabel, model_checkpoint):
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
            element_i = elements[i]
            number, group, row = element_i[1:4]
            for j in range(len(first_train_data)):
                for k in range(elemincompound):  # count of elements in compound
                    if first_train_data[j, k] == i:
                        elemcountlist[i, k] += 1
                        elemcountlist[i, 3] += 1
        # y_pos = np.arange(len(elementlabel) + 2)
        y_pos = np.arange(83)
        print(y_pos.shape, len(elementlabel))
        plt.bar(y_pos, elemcountlist[1:84, 3], align='center', alpha=0.5)
        plt.xticks(y_pos, elementlabel[:83])
        plt.ylabel('Count of Compounds')
        plt.title('First Random Traindata')
        plt.savefig(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(0) + "/al_" + str(al_level) + "/first_random_traindata.png")
        plt.show()
    return all_new_data


def get_mae_per_elem(mae, val_data, val_data_x):
    # loop for finding mean MAE one elements
    elemincompound = 3
    elements = generateElementdict()
    elementlabel = []
    for i in range(1, len(elements)):
        elementlabel.append(elements[i][0])
    elemMAE = np.zeros((len(elements) + 1, 2))
    for i in range(1, len(elements)):
        element_i = elements[i]
        number, group, row = element_i[1:4]
        # number, group, row = elements[i]
        # print("number, group, row", number, group, row)
        for j in range(len(val_data_x)):  # check every compound in validation set
            for k in range(elemincompound):  # count of elements in compound
                if i == int(val_data[j, k+1]):  # 0, 1, 2 are atomic numbers not!!!!!! 1, 2, 3 are!!!!  # why are here elements that are not in the data!!!!!!!!!!!!!!!!!!!
                    elemMAE[i, 0] += mae[i]
                    elemMAE[i, 1] += 1
                    # seems to be working now
                    if i == 2:
                        print("Helium, i, j, k, :", i, j, k)
                        print("compound ABC3", val_data[j, 0], val_data[j, 1], val_data[j, 2], val_data[j, 3], val_data.shape)
                        print("elemMAE value & count:", elemMAE[i, 0], elemMAE[i, 1])
        if elemMAE[i, 1] != 0:
            elemMAE[i, 0] = elemMAE[i, 0] / elemMAE[i, 1]
            elemMAE[i, 1] = i
            # print(elements[i][0], elemMAE[i, 0], elemMAE[i, 1])  # H 144.04924302106176 1.0
        else:
            # print(elements[i][0], "division by zero")
            elemMAE[i, 0] = 0
    return elemMAE


def find_elements_not_used(elementcount, elements, elementstoprint):
    # new_elementcount = np.asarray(elemcountlist[:, 3])
    print("count for each element", elementcount)  # counts how often each element was chosen
    elements_not_used = []
    print("Elements not used")
    for i in range(len(elementcount)):
        if elementcount[i] == 0 and i > 0:
            elements_not_used.append(i)
            print(elements[i][0], i)
    elements_not_used = np.asarray(elements_not_used)

    indexelement = np.flip(np.argsort(elementcount, axis=0))
    print("index_element", indexelement.shape)
    elementcount = np.zeros((len(indexelement)))
    print("Elements choosen by AL")
    for i in range(elementstoprint):  # print elements
        print(indexelement[i], elements[indexelement[i]][0], elementcount[indexelement[i]])
        return elements_not_used






def get_NN(feattotal):
    # Netsizevariables
    D_in = int(feattotal)
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
    return model


def get_CNN(feattotal):
    # Netsizevariables
    # D_in = int(feattotal)
    D_out = 1
    H1 = 420  # 20
    H2 = 64
    H3 = 7
    # H6 = 10

    # CNN Variable
    channel_size = 12

    model = nn.Sequential(
        wrapped(),

        # nn.Conv1d(1, 25, 2, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # nn.ELU(),

        # nn.Dropout(0.1),
        # nn.AvgPool1d(2),

        # torch.nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(1, channel_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # in_channels, out_channels, kernel_size
        nn.ELU(),

        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # in_channels, out_channels, kernel_size
        nn.ELU(),


        nn.Dropout(0.1),
        # nn.MaxPool1d(3),

        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        # in_channels, out_channels, kernel_size
        nn.ELU(),

        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 5, stride=1, padding=2, dilation=2, groups=1, bias=True),
        nn.ELU(),

        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 5, stride=1, padding=3, dilation=1, groups=1, bias=True),
        nn.ELU(),

        nn.Dropout(0.1),


        nn.Dropout(0.01),

        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 3, stride=1, padding=2, dilation=2, groups=1, bias=True),
        nn.ELU(),
        # print("1"),
        nn.Dropout(0.05),

        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 5, stride=1, padding=3, dilation=1, groups=1, bias=True),
        nn.ELU(),


        torch.nn.BatchNorm1d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv1d(channel_size, channel_size, 3, stride=1, padding=2, dilation=2, groups=1, bias=True),
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
        nn.Linear(H1, H2),
        nn.ELU(),
        nn.Linear(H2, H3),
        nn.Tanh(),
        nn.Linear(H3, D_out)
    )
    return model
