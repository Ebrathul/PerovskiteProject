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
                            peri.Element.from_Z(i).number  # atomic number
                          # mg.Element(peri.Element.from_Z(i)).group,  # group
                          # peri.Element(peri.Element.from_Z(i)).row  # row
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
