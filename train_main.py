import os
from train import train
from add_train_data import add_train_data

NN_number = 0
trainsetsize = 20000


model_checkpoint = 'NN_'  # name
log = 'runs'
logcount = 0
while (os.access(log + "/run_" + str(logcount), os.F_OK) == True):  # +str(NN_index)
    logcount += 1

os.mkdir(log + "/run_" + str(logcount))
os.mkdir(log + "/run_" + str(logcount) + "/NN_0")
for i in range(NN_number):
    os.mkdir(log + "/run_" + str(logcount) + "/" + model_checkpoint + str(i))

train(NN_number, trainsetsize, log, 1000)





