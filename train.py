from __future__ import division
from __future__ import print_function

import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import GCN
from utils import load_MAO

# Model and optimizer
model = GCN(3,
            nhid=10,
            nclass=2,
            dropout=0.1)

optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs, adjs, t_classes = load_MAO()  # dataset, n_batch=1)
criterion = torch.nn.CrossEntropyLoss()
min_loss = 10000
best_acc = 0

for epoch in range(10000):  # loop over the dataset multiple times
    running_loss = 0.0
    i = 0
    acc = 0
    data = list(zip(inputs, adjs, t_classes))
    random.shuffle(data)
    for X, adj, y in data:

        # get the inputs; data is a list of [inputs, labels]
        label = torch.tensor([y]).long()
        # zero the parameter gradients

        p = torch.randperm(30)
        X = X[p, :]
        adj = adj[p, :][:, p]

        # forward + backward + optimize
        outputs = model(X, adj)
        outputs = outputs.reshape(1, -1)
        loss = criterion(outputs, label)
        pred = outputs.argmax().item()
        # print(pred)
        if (pred == y):
            acc = acc + 1

        if (i % 10 == 0):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        i = i + 1
        # print statistics
        running_loss += loss.item()
    # if epoch == 3000:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.00001

    #optimizer = adjust_learning_rate(optimizer, epoch)
    cur_loss = running_loss/len(inputs)
    if (cur_loss < min_loss):
        min_loss = cur_loss
        PATH = './weights/optimal_net.pth'
        torch.save(model.state_dict(), PATH)
    cur_acc = acc/len(t_classes)
    if (cur_acc > best_acc):
        best_acc = cur_acc
    print(f"Epoch {epoch}, loss: {cur_loss}, acc : {acc/len(t_classes)}")

print(
    f"Finished Training, best acc achieved : {best_acc}, best loss : {min_loss}")

PATH = './weights/my_net.pth'
torch.save(model.state_dict(), PATH)
