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
from perovskite_classes import prepare_batch as prepare_batch  # _conv
from perovskite_classes import generateData
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
    val_batchsize = len(val_data)  # 231472  # all or small like 2000 ?
    train_loader, val_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, drop_last=False), \
                               DataLoader(val_set, batch_size=val_batchsize, drop_last=True)  # shuffle=True

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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
