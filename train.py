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
from perovskite_classes import get_NN, get_CNN
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torchvision import transforms

def train(NN_index, trainsetsize, log, max_epoch):
    if_gpu = True
    if if_gpu:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available()
                                      else torch.DoubleTensor)
        device = "cuda:0"
        # print("Graphics Power!")
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        device = None

    # file = 'grouprow.npy'
    # file = 'full_len.npy'
    file = 'data11.npy'  # feat1A, feat1B, feat1C, feat2A, feat2B
    if os.path.isfile('traindata.npy'):
        newdata, elementdict, featperelem, datavariables, feattotal = generateData(file)
        print("loaded given datasets")
        train_data = np.load(open('traindata.npy', 'rb'))
        val_data = np.load(open('valdata.npy', 'rb'))
    else:
        # disable features in classes to gen new data
        newdata, elementdict, featperelem, datavariables, feattotal = generateData(file)  # insert filename
        print("Shape of read data: ", newdata.shape)
        print("generating random files")
        create_dataset(newdata, trainsetsize)
        train_data = np.load(open('traindata.npy', 'rb'))
        val_data = np.load(open('valdata.npy', 'rb'))


    # newdata = znormalize(newdata)

    # train_data, val_data = getRandomSets(newdata)  # now in create_dataset class
    mean, stnddev = get_mean_stndev(train_data)

    # normalization
    train_data = (train_data - mean) / stnddev
    val_data = (val_data - mean) / stnddev  # welches????????????????????????
    # val_data = (val_data[:, 1::] - mean[1::]) / stnddev[1::]
    # print("val data shape and ex:", val_data.shape, val_data[0])

    train_set, val_set = PerovskiteDataset(train_data), PerovskiteDataset(val_data)

    # Variable batch and set loader
    train_batchsize = 1000
    val_batchsize = 10000  # len(val_data)  # 231472  # all or small like 2000 ?
    train_loader, val_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, drop_last=False), \
                               DataLoader(val_set, batch_size=val_batchsize, drop_last=True)  # shuffle=True

    # model = get_NN(feattotal)
    model = get_CNN(feattotal)

    # Shape for saving netstucture
    modelform = str(model)
    # print("Type:", type(modelform))
    # summary(netz, (1, train_batchsize, int(feattotal)))  # channel, H ,W


    lossMAE = nn.L1Loss()  # MAE  # to ignite
    lossMSE = nn.MSELoss()
    # torch.optim.SGD(params, lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # list of trainers?
    trainer = create_supervised_trainer(model, optimizer, lossMAE, std=stnddev[0], prepare_batch=prepare_batch)  # model[:]

    evaluator = create_supervised_evaluator(model, std=stnddev[0], prepare_batch=prepare_batch,
                                            metrics={'MAE': Loss(lossMAE),
                                                     'MSE': Loss(lossMSE),
                                                     # 'accuracy': Accuracy(),  ???
                                                     # 'NLL': Loss(lossNLL)
                                                     })  # output_transform=output_retransform_znormalize) expects (x, pred, y)

    # Progressbar
    pbar = ignite.contrib.handlers.ProgressBar(persist=False)
    pbar.attach(trainer, output_transform=lambda x: {'MAE': x})

    # Save n Load
    model_checkpoint = 'NN_'  # NN_index = sys.argv[1]
    # log = 'active'
    logcount = 0
    al_level = 0
    while (os.access(log + "/run_" + str(logcount), os.F_OK) == True):  # +str(NN_index)
        logcount += 1
    while (os.access(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(NN_index) + "/al_" + str(al_level), os.F_OK) == True):
        al_level += 1
    os.mkdir(log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(NN_index) + "/al_" + str(al_level))
    writer = SummaryWriter(log_dir=log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(NN_index) + "/al_" + str(al_level))  # +"NN_1" ? declaration for multiple NN
    print("Run: ", (logcount - 1), "NN: ", NN_index, "AL: ", al_level, "len of trainset: ", len(train_data))  # , comment=modelform)
    # print("Modelform:", modelform)

    if (os.path.isfile(model_checkpoint + str(NN_index) + '.pt')):
        print("NN: ", NN_index, "loaded")
        checkpoint = torch.load(model_checkpoint + str(NN_index) + '.pt')
        # try to load only optimizer
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("model not loaded!")


    start = timeit.default_timer()
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        iteration = trainer.state.iteration
        writer.add_scalar('loss_vs_iteration', trainer.state.output, iteration)
        # writer.close()  # generating mass of files


    @trainer.on(ignite.engine.Events.EPOCH_STARTED)
    def log_time(trainer):
        elapsed = round(timeit.default_timer() - start, 2)
        writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)
        epoch = trainer.state.epoch
        if trainer.state.epoch == 100:
            writer.add_text(str(logcount), "Netzstruktur: " + modelform)
            # writer.close()    # generating mass of files


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        if (trainer.state.epoch % evaluate_every == 0):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            print(trainer.state.epoch)
            print("\nTraining:", metrics)
            writer.add_scalar('MAEvsEpoch_training', metrics["MAE"], trainer.state.epoch)
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            print("Validation: ", metrics)
            writer.add_scalar('MAEvsEpoch_validation', metrics["MAE"], trainer.state.epoch)
        if trainer.state.epoch % evaluate_every == max_epoch:
            writer.close()

    evaluate_every = 100

    trainer.run(train_loader, max_epochs=max_epoch)

    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
               model_checkpoint + str(NN_index) + '.pt')
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
               log + "/run_" + str(logcount-1) + "/" + model_checkpoint + str(NN_index) + "/al_" + str(al_level) + "/" + model_checkpoint + str(NN_index) + '.pt')
