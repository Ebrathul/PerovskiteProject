import os
from train import train
from add_train_data import add_train_data

al_steps = 30
NN_number = 6
trainsetsize = 5000
trainsetaddition = 1000

model_checkpoint = 'NN_'  # name
log = 'active'
logcount = 0
while (os.access(log + "/run_" + str(logcount), os.F_OK) == True):  # +str(NN_index)
    logcount += 1

os.mkdir(log + "/run_" + str(logcount))

for i in range(NN_number):
    os.mkdir(log + "/run_" + str(logcount) + "/" + model_checkpoint + str(i))

# remove previous run
if os.path.isfile('traindata.npy'):
    print("Removing old train and validation data")
    os.remove('traindata.npy')
    if os.path.isfile('valdata.npy'):
        os.remove('valdata.npy')
    for i in range(NN_number):
        if os.path.isfile((model_checkpoint + str(i) + '.pt')):
            os.remove(model_checkpoint + str(i) + '.pt')


for i in range(al_steps):
    for j in range(NN_number):
        train(j, trainsetsize, log, 500)
    add_train_data(trainsetaddition)
# train(NN_number, trainsetsize, log, 1000)



