import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torchvision.transforms as transforms
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json, string
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn 
from tqdm import tqdm_notebook as tqdm
import pickle
import numpy
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import gc

class vgg16_bn_nc():
    def __init__(self):
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.classifier = None

    def forward(self, x):
        x = self.vgg16_bn.features(x)
        x = x.view(x.size(0), -1)
        return x

class custom_classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(custom_classifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        return self.seq.forward(x)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def train_model(network, vggnet, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = False):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    epochs_list = []
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
        vggnet.vgg16_bn = vggnet.vgg16_bn.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        
        if epoch and epoch%10 == 0: #modify learning rate every 8 steps by factor of 10
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        r = 0
        for (i, (inputs, labels)) in enumerate(t):
            print(i)
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            vgg_out = vggnet.forward(inputs)
            outputs = network(vgg_out)
            loss = criterion(outputs, labels)
            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
            r+=1
        train_acc.append(100*correct/counter)    
        train_loss.append(cum_loss/max(r, 1))
        # # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        r = 0
        for (i, (inputs, labels)) in enumerate(t):
            print(i)
        #     # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

        #     # Forward pass:
            vgg_out = vggnet.forward(inputs)
            outputs = network(vgg_out)
            loss = criterion(outputs, labels)

        #     # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
            r+=1
        val_acc.append(100*correct/counter)    
        val_loss.append(cum_loss/max(r, 1))
        #print("Validation accuracy at epoch", n_epochs, ": ", val_acc[-1])    
        epochs_list.append(epoch)
        fig1 = plt.figure(1)
        plt.plot(epochs_list, train_acc, 'ro')
        plt.plot(epochs_list, val_acc, 'bo')
        plt.axis([0, n_epochs + 1, 97, 100])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs. Epoch on Training (red dots) and Validation Sets (blue dots)")
        fig1.savefig("accuracy.png", bbox_inches='tight')
       
        fig2 = plt.figure(2)
        plt.plot(epochs_list, train_loss, 'ro')
        plt.plot(epochs_list, val_loss, 'bo')
        plt.axis([0, n_epochs + 1, 0, 0.05])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch on Training (red dots) and Validation (blue dots) Sets")    
        fig2.savefig("loss.png", bbox_inches='tight')
        plt.clf()    
        
        gc.collect()


learningRate = 1e-2
mo = 0.9
size = (224, 224)
imgTransform = transforms.Compose([
    transforms.Scale(size),
    transforms.ToTensor()
])

trainset = datasets.ImageFolder(root='./data/train', transform = imgTransform)
valset = datasets.ImageFolder(root='./data/test', transform = imgTransform)

trainLoader = torch.utils.data.DataLoader(trainset, batch_size = 40, shuffle = True, num_workers = 0)
valLoader = torch.utils.data.DataLoader(trainset, batch_size = 20, shuffle = True, num_workers = 0)

vgg = vgg16_bn_nc()
network = custom_classifier()

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(network.parameters(), lr = learningRate, momentum = mo)

train_model(network, vgg, criterion, optimizer, trainLoader, valLoader, n_epochs = 3, use_gpu = True)
torch.save(network.state_dict(), "model.pt")
