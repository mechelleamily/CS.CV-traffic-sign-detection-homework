from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrsche
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net, SimpleNet, SimpleNet3, SimpleNet5, MultiScaleNet, BigNet
model = SimpleNet()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.00001)
#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
#scheduler = lrsche.StepLR(optimizer, step_size=10, gamma=0.1)
#scheduler = lrsche.MultiStepLR(optimizer, milestones=[10,15,25,30], gamma=0.5)
#scheduler = lrsche.CosineAnnealingLR(optimizer, T_max=args.epochs)
scheduler = lrsche.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=0.0001)
def train(epoch):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    #scheduler.step()
    model.train()        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss.append(loss.data[0])

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(validation_loss)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    val_loss.append(validation_loss)
    val_accuracy.append(100. * correct / len(val_loader.dataset))
	
def showCurve(loss, loss_val, accuracy, accuracy_val):
    fig = plt.figure()
    #plt.subplot(1,2,1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(0,args.epochs), loss, 'o-', color='r', label="train")
    plt.plot(range(0,args.epochs), loss_val, 'o-', color='g', label="val")
    #plt.subplot(1,2,2)
    #plt.xlabel("epoch")
    #plt.ylabel("accuracy")
    #plt.plot(range(0,args.epochs), accuracy)
    #plt.plot(range(0,args.epochs), accuracy_val)
    plt.legend(loc="best")
    plt.show()

def showFilters():
    params=list(model.parameters())
    for param in params:
        print(type(param.data), param.size())
    firstfilter=params[0].data.numpy()
    row=6
    column=8
    fig=plt.figure(figsize=(10,10))
    for idx,filt in enumerate(firstfilter):
        #rgbArray = np.zeros((5,5,3), 'uint8')
        #rgbArray[..., 0] = x[0]*256
        #rgbArray[..., 1] = x[1]*256
        #rgbArray[..., 2] = x[2]*256
        #img = Image.fromarray(rgbArray)
        #img.show()
        #print(filt.shape)
        #print(filt[0, :, :])
        plt.subplot(row,column, idx + 1)
        plt.imshow(filt[0, :, :])
        plt.axis('off')    
    fig.show()

train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]
for epoch in range(1, args.epochs + 1):
	train(epoch)
	validation()
	model_file = 'model_' + str(epoch) + '.pth'
	torch.save(model.state_dict(), model_file)
	print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
showFilters()
showCurve(train_loss, val_loss, train_accuracy, val_accuracy)

