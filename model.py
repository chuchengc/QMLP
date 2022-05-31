import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise import NoiseModel


bsz = 32
epochs = 30
lr = 0.001
w_decay = 1e-4


############################################################################################################################################################################
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)

###########################################################################################################################################################################
noise_model = NoiseModel(basis_gates=['id', 'rz', 'sx', 'cx', 'x'])

# # # bit filp
# # p = 0.01
# # bitflip = noise.pauli_error([('X', p), ('I', 1-p)])
# # bitflip2 = noise.pauli_error([('X', p), ('I', 1-p)])
# # noisemy = bitflip.tensor(bitflip2)
# # print(noisemy)
# # noise_model.add_all_qubit_quantum_error(bitflip, ['id', 'rz', 'sx', 'x'])
# # noise_model.add_all_qubit_quantum_error(noisemy, ['cx'])

# phase flip
p = 0.01
phaseflip = noise.pauli_error([('Z', p), ('I', 1-p)])
phaseflip2 = noise.pauli_error([('Z', p), ('I', 1-p)])
noisemy = phaseflip.tensor(phaseflip2)
noise_model.add_all_qubit_quantum_error(phaseflip, ['id', 'rz', 'sx', 'x'])
noise_model.add_all_qubit_quantum_error(noisemy, ['cx'])

###########################################################################################################################################################################
n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)      # w/o noise
# dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)      # w noise


@qml.qnode(dev)
def qnode(inputs, w000, w001, w002, w003, w004, w005, w006, w007, w008, w009, w010, w011, w012, w013, w014, w015,
          w100, w101, w102, w103, w104, w105, w106, w107, w108, w109, w110, w111, w112, w113, w114, w115,
          x000, x001, x002, x003, x004, x005, x006, x007, x008, x009, x010, x011, x012, x013, x014, x015,
          x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115,):
    for rx in range(16):
      qml.RX(inputs[rx], wires=rx)
    qml.Rot(*w000, wires=0)
    qml.Rot(*w001, wires=1)
    qml.Rot(*w002, wires=2)
    qml.Rot(*w003, wires=3)
    qml.Rot(*w004, wires=4)
    qml.Rot(*w005, wires=5)
    qml.Rot(*w006, wires=6)
    qml.Rot(*w007, wires=7)
    qml.Rot(*w008, wires=8)
    qml.Rot(*w009, wires=9)
    qml.Rot(*w010, wires=10)
    qml.Rot(*w011, wires=11)
    qml.Rot(*w012, wires=12)
    qml.Rot(*w013, wires=13)
    qml.Rot(*w014, wires=14)
    qml.Rot(*w015, wires=15)
    
    qml.CRX(x000, wires=[0,1])
    qml.CRX(x001, wires=[1,2])
    qml.CRX(x002, wires=[2,3])
    qml.CRX(x003, wires=[3,4])
    qml.CRX(x004, wires=[4,5])
    qml.CRX(x005, wires=[5,6])
    qml.CRX(x006, wires=[6,7])
    qml.CRX(x007, wires=[7,8])
    qml.CRX(x008, wires=[8,9])
    qml.CRX(x009, wires=[9,10])
    qml.CRX(x010, wires=[10,11])
    qml.CRX(x011, wires=[11,12])
    qml.CRX(x012, wires=[12,13])
    qml.CRX(x013, wires=[13,14])
    qml.CRX(x014, wires=[14,15])
    qml.CRX(x015, wires=[15,0])

    for rx in range(16):
      qml.RX(inputs[rx], wires=rx)
    qml.Rot(*w100, wires=0)
    qml.Rot(*w101, wires=1)
    qml.Rot(*w102, wires=2)
    qml.Rot(*w103, wires=3)
    qml.Rot(*w104, wires=4)
    qml.Rot(*w105, wires=5)
    qml.Rot(*w106, wires=6)
    qml.Rot(*w107, wires=7)
    qml.Rot(*w108, wires=8)
    qml.Rot(*w109, wires=9)
    qml.Rot(*w110, wires=10)
    qml.Rot(*w111, wires=11)
    qml.Rot(*w112, wires=12)
    qml.Rot(*w113, wires=13)
    qml.Rot(*w114, wires=14)
    qml.Rot(*w115, wires=15)
    
    qml.CRX(x100, wires=[0,1])
    qml.CRX(x101, wires=[1,2])
    qml.CRX(x102, wires=[2,3])
    qml.CRX(x103, wires=[3,4])
    qml.CRX(x104, wires=[4,5])
    qml.CRX(x105, wires=[5,6])
    qml.CRX(x106, wires=[6,7])
    qml.CRX(x107, wires=[7,8])
    qml.CRX(x108, wires=[8,9])
    qml.CRX(x109, wires=[9,10])
    qml.CRX(x110, wires=[10,11])
    qml.CRX(x111, wires=[11,12])
    qml.CRX(x112, wires=[12,13])
    qml.CRX(x113, wires=[13,14])
    qml.CRX(x114, wires=[14,15])
    qml.CRX(x115, wires=[15,0])


    return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3)), qml.expval(qml.PauliZ(4)),
    qml.expval(qml.PauliZ(5)), qml.expval(qml.PauliZ(6)), qml.expval(qml.PauliZ(7)), qml.expval(qml.PauliZ(8)), qml.expval(qml.PauliZ(9)), 
    qml.expval(qml.PauliZ(10)), qml.expval(qml.PauliZ(11)), qml.expval(qml.PauliZ(12)), qml.expval(qml.PauliZ(13)), qml.expval(qml.PauliZ(14)), 
    qml.expval(qml.PauliZ(15)))

weight_shapes = {"w000": 3, "w001": 3, "w002": 3, "w003": 3, "w004": 3, "w005": 3, "w006": 3, "w007": 3, 
          "w008": 3, "w009": 3, "w010": 3, "w011": 3, "w012": 3, "w013": 3, "w014": 3, "w015": 3, 
          "x000": 1, "x001": 1, "x002": 1, "x003": 1, "x004": 1, "x005": 1, "x006": 1, "x007": 1, 
          "x008": 1, "x009": 1, "x010": 1, "x011": 1, "x012": 1, "x013": 1, "x014": 1, "x015": 1, 
          "w100": 3, "w101": 3, "w102": 3, "w103": 3, "w104": 3, "w105": 3, "w106": 3, "w107": 3, 
          "w108": 3, "w109": 3, "w110": 3, "w111": 3, "w112": 3, "w113": 3, "w114": 3, "w115": 3,
          "x100": 1, "x101": 1, "x102": 1, "x103": 1, "x104": 1, "x105": 1, "x106": 1, "x107": 1, 
          "x108": 1, "x109": 1, "x110": 1, "x111": 1, "x112": 1, "x113": 1, "x114": 1, "x115": 1}

#################################################################################################################################################################################

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.pool = torch.nn.AvgPool2d(7, 7)                      # 4x4
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc1 = torch.nn.Linear(16, 10)


    def forward(self, x):
        b = x.shape[0]
        x = self.pool(x).view(b, 16)
        x = self.qlayer(x)
        out = self.fc1(x)
        out = F.log_softmax(out, dim=1)
        return out

###########################################################################################################################################################################
def train(model, DEVICE, train_loader, optimizer, epoch):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('epoch: %d, batch_idx: %d, acc: %5f %%, loss: %.3f' % (epoch, batch_idx + 1, 100 * correct / total, running_loss / 10))
            running_loss = 0.0

##########################################################################################################################################################################
def test(model, DEVICE, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    acc = 100 * correct / total
    return acc

###########################################################################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNIST().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)


for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
