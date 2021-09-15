from dataloader import read_bci_data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from distutils.dir_util import copy_tree
import pdb
#refer to
#https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

def introduction(trainX):
    #input data
    sample = np.random.randint(0,1080)
    plt.suptitle(f'Sample no.{sample}')
    plt.subplot(2,1,1)
    plt.ylabel('Channel 1')
    plt.plot(trainX[sample][0][0])
    plt.subplot(2,1,2)
    plt.ylabel('Channel 2')
    plt.plot(trainX[sample][0][1])
    plt.show()
    #activation function
    x = torch.arange(-10,10,dtype=torch.float,requires_grad=True)
    activations = ['ReLU', 'LeakyReLU', 'ELU']
    for i in range(3):
        #compute gradient
        y = getattr(nn,f'{activations[i]}')()(x)
        if i!=0: x.grad.data.zero_() #to clear out gradient value stored in grad attribute
        y.backward(torch.ones(y.shape)) #compute the gradient of current tensor w.r.t. computational graph leaves, the value of computed gradients is added to the grad property of all leaf nodes of computational graph
        #print(x.grad) #This attribute is None by default and becomes a Tensor the first time a call to backward() computes gradients for self. The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it
        #plot
        plt.subplot(3,2,i*2+1)
        if i!=2: plt.xticks([])
        plt.yticks([])
        plt.title(f'{activations[i]}(x)')
        plt.plot(x.data,y.data)
        plt.subplot(3,2,i*2+2)
        if i!=2: plt.xticks([])
        plt.title(f'{activations[i]}(x)\'')
        plt.plot(x.data,x.grad.data)
    plt.show()
    
def modelInfo(model):
    #input model's format is model(activation).to(device)
    #print the model's architecture
    print(model) 
    #overview 1, refer to https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778
    for name, param in model.named_parameters(): 
        if param.requires_grad:
            print(name)
    #overview 2, refer to https://zhuanlan.zhihu.com/p/84794981
    for param in model.parameters():
        print(param.size())
    #overview 3
    print(model.state_dict())
    #individual-view
    g = model.parameters() #required
    print(next(g))
    print(next(g).dtype) #torch.float32
    print(next(g).shape) #shape of current next(g) is different from the shape of last next(g)

def plot_acc_comp(messages,acc_comp_path):
    plt.figure()
    #plot the model with three different activations' comparison graph    
    plt.title(f"Activation function comparison({messages[0]['model']})")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    for message in messages:
        plt.plot(message['epochs'],np.array(message['train_accuracy_curve'])*100,label=f"{message['activation']}_train")
        plt.plot(message['epochs'],np.array(message['test_accuracy_curve'])*100,label=f"{message['activation']}_test")
    plt.legend()
    plt.savefig(acc_comp_path)
    plt.show()

def plot_acc_table(accuracy_table,acc_table_path):
    #plot each (model,activation) pair's accuracy in a table
    print(accuracy_table)
    ax = plt.axes(frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    pd.plotting.table(ax, accuracy_table, loc='center')
    ax.get_figure().savefig(acc_table_path, bbox_inches="tight")
    plt.show()

class EEGNet(nn.Module):
    def __init__(self,activation):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1, 1),padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1), stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.5) #p=0.25
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0),
            nn.Dropout(p=0.5) #p=0.25
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736,out_features=2,bias=True)
        )
    def forward(self,x):
        h1 = self.firstconv(x)
        h2 = self.depthwiseConv(h1)
        h3 = self.separableConv(h2) #h3: (64, 32, 1, 23)
        h3 = h3.view(h3.shape[0],-1) #h3: (64, 736), flatten
        y = self.classify(h3)
        return y

class DeepConvNet(nn.Module):
    def __init__(self,activation):
        super(DeepConvNet, self).__init__()
        channels = (25,25,50,100,200)
        kernel_sizes = ((1,5),(2,1),(1,5),(1,5),(1,5))
        self.conv0 = nn.Conv2d(1,channels[0],kernel_size=kernel_sizes[0]) #batchsize=1, number of channels=25, kernel size = (1,5)
        for i in range(1,len(channels)):
            #seems that locals()[f'self.conv{i}'] can't run
            setattr(self,f'conv{i}',nn.Sequential(
                nn.Conv2d(channels[i-1],channels[i],kernel_size=kernel_sizes[i]),
                nn.BatchNorm2d(channels[i],eps=1e-5,momentum=0.1),
                activation(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            ))
        self.classify = nn.Linear(in_features=8600,out_features=2) #8600 is because of the comment in DeepConvNet.forward.h5
    def forward(self,x):
        h1=self.conv0(x)
        h2=self.conv1(h1)
        h3=self.conv2(h2)
        h4=self.conv3(h3)
        h5=self.conv4(h4) #h5: (64, 200, 1, 43)
        h5 = h5.view(h5.shape[0],-1) #h5: (64, 8600), flatten
        y = self.classify(h5)
        return y

def train(model, activation, device, epochsize, optimizer, loss, lr, train_dataloader, test_dataloader,mode):
    #construct model and optimizer
    model_object = model(activation).to(device)
    optimizer_object = optimizer(model_object.parameters(),lr=lr,weight_decay=1e-3)
    #to record the message(include train and test accuracy per epoch) in training phase under particular model and activation function
    epochs = []
    train_accuracies = []
    test_accuracies = []
    test_best_accuracy = 0
    for epoch in range(epochsize):
        epochs.append(epoch)
        train_accuracies.append(train_one_epoch(model_object, device, optimizer_object, loss, epoch, train_dataloader)) #model_object's weight will update automatically during training's backward process
        test_accuracies.append(test_one_epoch(model_object, model, activation, device, None, test_dataloader)) #model_object's weight won't change or update. just test current model_object's performance. this line can be commented out
        if test_accuracies[-1] > test_best_accuracy:
            best_model_weight = copy.deepcopy(model_object.state_dict()) #deep copy is necessary, or the model weight won't be saved and will be overwrited the next update process
            test_best_accuracy = test_accuracies[-1]
    message = {'model':model.__name__, 'activation':activation.__name__, 'epochs':epochs, 'train_accuracy_curve':train_accuracies, 'test_accuracy_curve':test_accuracies}
    #print("best accuracy during each epoch: ",test_best_accuracy)
    return best_model_weight, message

def train_one_epoch(model_object, device, optimizer_object, loss, epoch, dataloader):
    model_object.train() #in training mode, BatchNorm keeps running estimates of its computed mean and variance which are kept with a default momentum of 0.1,
                         #                  Dropout layers activated
    total_acc = 0
    total_L = 0
    for (minibatchX, minibatchGtY) in dataloader:
        #forward
        minibatchX = minibatchX.to(device, dtype=torch.float32) #because the dtype of model's parameter is torch.float32, here we set the variable's dtype to torch.float32
        minibatchGtY = minibatchGtY.to(device, dtype=torch.long) #because the dtype of CrossEntropyLoss's input should be long
        minibatchPredY = model_object.forward(minibatchX)
        
        #compute loss, backward and update
        minibatchL = loss()(minibatchPredY, minibatchGtY)
        optimizer_object.zero_grad()
        minibatchL.backward()
        optimizer_object.step()
        total_acc += sum(torch.eq(torch.argmax(minibatchPredY,dim=1) ,minibatchGtY)).item()
        total_L += minibatchL
    total_L /= len(dataloader.dataset)
    total_acc /= len(dataloader.dataset)
    if epoch % 10 == 0 :
        print(f'epoch:{epoch}, loss={total_L:.3f}, accuracy={total_acc:.3f}')
    return total_acc
    
def test_one_epoch(model_object, model, activation, device, model_weights_path, dataloader):
    if model_weights_path != None: #the model has been trained well
        model_object = model(activation).to(device)
        model_object.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device))) #load the model weight
    #else the model has not finished its training phase
    model_object.eval() #in evaluation mode, BatchNorm does not rely on batch statistics but uses the running_mean and running_std estimates that it computed during it's training phase
                        #                    Dropout layers de-activated
    total_acc = 0
    for (minibatchX, minibatchGtY) in dataloader:
        #forward
        minibatchX = minibatchX.to(device, dtype=torch.float32)
        minibatchGtY = minibatchGtY.to(device, dtype=torch.long)
        minibatchPredY = model_object.forward(minibatchX)
        total_acc += sum(torch.eq(torch.argmax(minibatchPredY,dim=1) ,minibatchGtY)).item()
    total_acc /= len(dataloader.dataset)
    
    if model_weights_path != None:
        #model_object.__class__.__name__
        print(f'model:{model.__name__}, activation function:{activation.__name__}, accuracy:{total_acc:.3f}')
    return total_acc

if __name__=='__main__':    
    #load data
    batchsize = 1080 #bathcsize = 64
    trainX, trainY, testX, testY = read_bci_data()
    trainDataset = TensorDataset(torch.from_numpy(trainX),torch.from_numpy(trainY))
    trainDataloader = DataLoader(trainDataset, batch_size=batchsize)
    testDataset = TensorDataset(torch.from_numpy(testX),torch.from_numpy(testY))
    testDataloader = DataLoader(testDataset, batch_size=batchsize)
    #next(iter(trainDataloader)) #return (minibatch-data x, minibatch-label y)
    #introduction(trainX)
    
    #initialize model parameter
    mode = {'demo':True, 'tune':False}
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #if GPU is avaliable, then use GPU
    lr = 1e-3 #learning rate
    epochsize = 300
    optimizer = optim.Adam #delegate
    loss = nn.CrossEntropyLoss #delegate

    activations = [nn.ReLU, nn.ELU, nn.LeakyReLU] #nn.ELU #nn.ReLU #nn.LeakyReLU #activation function #delegate
    models = [EEGNet, DeepConvNet] #EEGNet #DeepConvNet #delegate
    #modelInfo(model(activation).to(device)) #can also use during training phase
    
    #run model
    os.makedirs('./ExperimentReport', exist_ok=True)
    os.makedirs('./ModelWeights', exist_ok=True)
    accuracy_table = pd.DataFrame({'ReLU':[None,None],'ELU':[None,None],'LeakyReLU':[None,None]},index=['EEGNet','DeepConvNet'])
    for model_idx, model in enumerate(models):
        messages = [] #to record different activation function's accuracy message under particualr model
        for act_idx, activation in enumerate(activations):
            model_weights_path = f'./ModelWeights/{model.__name__}_{activation.__name__}.pt'
            #train
            if not mode['demo']:
                model_weights, message = train(model, activation, device, epochsize, optimizer, loss, lr, trainDataloader, testDataloader,mode)
                torch.save(model_weights, model_weights_path) #store the model weight
                messages.append(message)
            
            #test
            accuracy_table.loc[model.__name__, activation.__name__] = test_one_epoch(None, model, activation, device, model_weights_path, testDataloader)
        if mode['tune']: plot_acc_comp(messages, acc_comp_path = f"./ExperimentReport/{model.__name__}_comparison.png")
    plot_acc_table(accuracy_table, acc_table_path = "./ExperimentReport/Accuracy_comparison.png")
