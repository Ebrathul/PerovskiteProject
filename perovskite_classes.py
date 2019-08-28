import numpy as np
from torch.utils import data
import torch.nn as nn
import torch
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor


def get_mean_stndev(data):
    mean = np.mean(data, axis=0)  # axis 0 is top to bottom, 1 would be sideways
    stnddev = np.std(data, axis=0)  # , ddof=1)
    print("Mean: ", mean, "\nStnd dev: ", stnddev)
    return mean, stnddev


def getRandomSets(X):  # shuffles the data and returns a training and a test set # needs 8 or more samples
       np.random.shuffle(X)
       Xtrain = X[0:20000]
       Xtest = X[20000:len(X), :]
       # print("Trainset: ", Xtrain[0][:])
       # print("Validationset:", Xtest[0][:])
       return Xtrain, Xtest


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
        return x.view(x.size()[0], -1)

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
        loss = loss*std
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)

def create_supervised_evaluator(model, metrics={},std=1,
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
            y_pred = model(x).view(-1)*std
            return output_transform(x, y*std, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine