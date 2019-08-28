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
import tensorboardX
import timeit
import torch
from perovskite_classes import get_mean_stndev, getRandomSets,  PerovskiteDataset, create_supervised_trainer, create_supervised_evaluator, flatten
from perovskite_classes import prepare_batch_conv as prepare_batch
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torchvision import transforms
if_gpu = True
if if_gpu:
       torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available()
                                                     else torch.DoubleTensor)
       device="cuda:0"
else:
       torch.set_default_tensor_type(torch.DoubleTensor)
       device=None



newdata = pickle.load(open('DATA11.pickle', 'rb'), encoding='latin1')  # Jonathan

print(newdata.shape)
# newdata = znormalize(newdata)

train_data, val_data = getRandomSets(newdata)
mean, stnddev = get_mean_stndev(train_data)

train_data = (train_data-mean)/stnddev
val_data = (val_data-mean)/stnddev

train_set, val_set = PerovskiteDataset(train_data), PerovskiteDataset(val_data)

# Variable batch and set loader
train_batchsize = 5000
val_batchsize = 10000
train_loader, val_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, drop_last=False),DataLoader(val_set, batch_size=val_batchsize, drop_last=True)


# Netsizevariables
D_in = 33  # feattotal
D_out = 1
H1 = 20
H2 = 64
H3 = 7
# H6 = 10


# Sequential net, structure and functions
# netz = nn.Sequential(
#     nn.Linear(D_in, H1),
#     nn.LeakyReLU(),
#     nn.Linear(H1, H2),
#     nn.LeakyReLU(),
#     nn.Linear(H2, H3),
#     nn.Tanh(),
#     nn.Linear(H3,D_out),
# )
netz = nn.Sequential(
    nn.Conv1d(1, 9, 3, stride=1, padding=2, dilation=1, groups=1, bias=True),  # in_channels, out_channels, kernel_size
    nn.ELU(),
    nn.MaxPool1d(3),
    torch.nn.BatchNorm1d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.Conv1d(9, 16, 4, stride=1, padding=1, dilation=1, groups=1, bias=True),
    nn.MaxPool1d(4),
    torch.nn.BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.Conv1d(16, 25, 5, stride=1, padding=1, dilation=1, groups=1, bias=True),
    nn.MaxPool1d(3),
    nn.ELU(),
    flatten(),
    # nn.Dropout(0.01),
    nn.Linear(108, 33),
    nn.ELU(),
    nn.Linear(33, D_out),
)

class flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


lossMAE = nn.L1Loss()  # MAE  # to ignite
lossMSE = nn.MSELoss()
optimizer = torch.optim.Adam(netz.parameters(), lr=0.01)

trainer = create_supervised_trainer(netz, optimizer, lossMAE, std=stnddev[0],prepare_batch=prepare_batch)
evaluator = create_supervised_evaluator(netz, std=stnddev[0], prepare_batch=prepare_batch,
                                        metrics={'MAE': Loss(lossMAE),
                                                  'MSE': Loss(lossMSE),
                                                 # 'accuracy': Accuracy(),  ???
                                                 # 'NLL': Loss(lossNLL)
                                                 })  # output_transform=output_retransform_znormalize) expects (x, pred, y)

pbar = ignite.contrib.handlers.ProgressBar(persist=False)

pbar.attach(trainer, output_transform=lambda x: {'MAE': x})


# TensorboardX generate new file
log = 'runs'
_=0
while (os.access(log+"/run_"+str(_), os.F_OK)==True):
       _+=1
os.mkdir(log+"/run_"+str(_))
writer = tensorboardX.SummaryWriter(log_dir=log+"/run_"+str(_))

# # tensorboardlogger
# ignite.contrib.handlers.tensorboard_logger.TensorboardLogger(log_dir=log+"/run_"+str(_))
evaluate_every = 10

start= timeit.default_timer()
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    iteration = trainer.state.iteration
    writer.add_scalar('loss_vs_iteration', trainer.state.output, iteration)


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_time(trainer):
    elapsed = round(timeit.default_timer() - start, 2)
    writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    if(trainer.state.epoch%evaluate_every==0):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(trainer.state.epoch)
        print("\nTraining:", metrics)
        writer.add_scalar('MAEvsEpoch_training', metrics["MAE"], trainer.state.epoch)
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation: ", metrics)
        writer.add_scalar('MAEvsEpoch_validation', metrics["MAE"], trainer.state.epoch)

trainer.run(train_loader, max_epochs=1000)

torch.save(netz, 'testlr.pt')



