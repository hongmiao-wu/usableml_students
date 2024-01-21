from queue import Queue
import sys
import time
sys.path.append("ml_utils")

import numpy as np
import torch
from torch import manual_seed, Tensor
from torch.cuda import empty_cache
from torch.nn import Module, functional as F
from torch.optim import Optimizer, SGD

from data import get_data_loaders
from evaluate import accuracy
from model import ConvolutionalNeuralNetwork


def train_step(model: Module, optimizer: Optimizer, data: Tensor,
               target: Tensor, cuda: bool):
    model.train()
    if cuda:
        data, target = data.cuda(), target.cuda()
    prediction = model(data)
    loss = F.cross_entropy(prediction, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def training(model: Module, optimizer: Optimizer, cuda: bool, n_epochs: int, 
             start_epoch: int, batch_size: int, q_acc: Queue = None, q_loss: Queue = None, 
             q_epoch: Queue = None, 
             q_break_signal:Queue = None,
             stop_signal: bool = False, 
             q_stop_signal: Queue = None):
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    if cuda:
        model.cuda()
    for epoch in range(start_epoch, n_epochs):
        print(f"Epoch {epoch} starts...")
        path=f"stop{epoch}.pt"
        for batch in train_loader:
            data, target = batch
            train_step(model=model, optimizer=optimizer, cuda=cuda, data=data,
                       target=target)
        test_loss, test_acc = accuracy(model, test_loader, cuda)
        # if (q_stop_signal.empty() | (not (q_stop_signal.get()))):
        #     if q_acc is not None:
        #         q_acc.put(test_acc)
        #     if q_loss is not None:
        #         q_loss.put(test_loss)
        #     q_epoch.put(epoch) # put to listener
        #     print(f"epoch{epoch} is done!")
        #     print(f"epoch={epoch}, test accuracy={test_acc}, loss={test_loss}")
        #     save_checkpoint(model, optimizer, epoch, test_loss, test_acc, path, False)
        #     print(f"The checkpoint for epoch: {epoch} is saved!")
        # else:
        #     q_break_signal.put(True)
        #     print(f"Epoch {epoch} is aborted!")
        #     break
        # if q_stop_signal.get():
        if q_acc is not None:
            q_acc.put(test_acc)
        if q_loss is not None:
            q_loss.put(test_loss)
        q_epoch.put(epoch) # put to listener
        print(f"epoch{epoch} is done!")
        print(f"epoch={epoch}, test accuracy={test_acc}, loss={test_loss}")
        save_checkpoint(model, optimizer, epoch, test_loss, test_acc, path, False)
        print(f"The checkpoint for epoch: {epoch} is saved!")
        if q_stop_signal.empty():
            continue
        if q_stop_signal.get():
            q_break_signal.put(True)
            break
    if cuda:
        empty_cache()

def save_checkpoint(model, optimizer, epoch, loss, acc, path, print_info):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        # Add any other information you want to save
    }
    torch.save(checkpoint, path)
    if(print_info):
        print(f"The checkpoint for epoch: {epoch} is saved!")
            # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print()
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        
def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.eval()
    return checkpoint



def main(seed):
    print("init...")
    manual_seed(seed)
    np.random.seed(seed)
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=0.3, momentum=0.5)
    print("train...")
    training(
        model=model,
        optimizer=opt,
        cuda=False,     # change to True to run on nvidia gpu
        n_epochs=10,
        batch_size=256,
    )


if __name__ == "__main__":
    main(seed=0)
    # print(f"The final accuracy is: {final_test_acc}")

