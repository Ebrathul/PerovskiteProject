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

train(NN_number, trainsetsize, log, 2000)

# move files to corresponding location
os.rename("traindata.npy", log + "/run_" + str(logcount) + "/traindata.npy")
os.rename("valdata.npy", log + "/run_" + str(logcount) + "/valdata.npy")
for i in range(NN_number):
    os.rename(model_checkpoint + str(i) + '.pt', log + "/run_" + str(logcount) + "/" + model_checkpoint + str(i) + '.pt')






