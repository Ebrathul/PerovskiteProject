import os
from train import train
from add_train_data import add_train_data
import shutil

al_steps = 30
NN_number = 10
trainsetsize = 5000
trainsetaddition = 500

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

# Active learning train loop
for i in range(al_steps):
    for j in range(NN_number):
        train(j, trainsetsize, log, 500 - (10*al_steps))  # max_epoch reduced in each al step sqrt to reduce?
    add_train_data(trainsetaddition, NN_number)
train(NN_number, trainsetsize + al_steps*trainsetaddition, log, 1000)


# move files to corresponding location
os.rename("traindata.npy", log + "/run_" + str(logcount) + "/traindata.npy")
# shutil.move("traindata.npy", log + "/run_" + str(logcount) + "/traindata.npy")
# os.replace("traindata.npy", log + "/run_" + str(logcount) + "/traindata.npy")

os.rename("valdata.npy", log + "/run_" + str(logcount) + "/valdata.npy")
# shutil.move("valdata.npy", log + "/run_" + str(logcount) + "/valdata.npy")
# os.replace("valdata.npy", log + "/run_" + str(logcount) + "/valdata.npy")
for i in range(NN_number):
    os.rename(model_checkpoint + str(i) + '.pt', log + "/run_" + str(logcount) + "/" + model_checkpoint + str(i) + '.pt')
    # shutil.move(model_checkpoint + str(i) + '.pt', log + "/run_" + str(logcount) + "/" + model_checkpoint + str(i) + '.pt')
    # os.replace(model_checkpoint + str(i) + '.pt', log + "/run_" + str(logcount) + "/" + model_checkpoint + str(i) + '.pt')
