import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from dataloader import RetinopathyDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import copy
import os
import pdb

#refer to https://stackoverflow.com/questions/60776749/plot-confusion-matrix-without-estimator
def plot_confusion_matrix(GtY, PredY, model_name, pretrained_status):
    #PredY has been choose as the highest possible value
    #GtY and PredY's type are both numpy
    matrix_array = confusion_matrix(GtY, PredY, normalize='true') #normalize must be one of {'true', 'pred', 'all', None}
    matrix_img = ConfusionMatrixDisplay(confusion_matrix=matrix_array,display_labels=None)
    try:
        matrix_img.plot(cmap=plt.cm.Blues)
    except:
        pass
    total_acc = sum(np.equal(PredY ,GtY)).item()/len(GtY)
    plt.title(f"{model_name}({pretrained_status}), accuracy:{total_acc}")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f"./ExperimentReport/{model_name}_{pretrained_status}_confusionMatrix.png")
    plt.show()
    plt.clf()
    
def plot_acc_comp(messages):
    with open(f"./ExperimentReport/{messages[0]['model']}_trainingMessages.txt", "w") as text_file: #store the message
        print(messages, file = text_file)
    plt.figure()
    #plot the model with three different activations' comparison graph    
    plt.title(f"Result Comparison - {messages[0]['model']}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    for message in messages:
        plt.plot(message['epochs'],np.array(message['train_accuracy_curve'])*100,label=f"Train({message['pretrained']})")
        plt.plot(message['epochs'],np.array(message['test_accuracy_curve'])*100,label=f"Test({message['pretrained']})")
    plt.legend()
    plt.savefig(f"./ExperimentReport/{messages[0]['model']}_comparison.png")
    plt.show()
    plt.clf()
    plt.close()

#refer to https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
class ResNet(nn.Module): 
    def __init__(self,pretrained=False,num_layers=18,epochsize=10,batchsize=4):
        super(ResNet, self).__init__()
        #model information
        self.name = f'ResNet{num_layers}' #model name, for plot
        self.pretrained_status = 'withPretraining' if pretrained else 'withoutPretraining' #pretrained flag, for plot
        self.epochsize = epochsize
        self.batchsize = batchsize
        #model architecture
        #self.model = torchvision.models.resnet18(pretrained) if num_layers==18 else torchvision.models.resnet50(pretrained) #choose between resnet18 and resnet50
        self.model = torchvision.models.__dict__[f'resnet{num_layers}'](pretrained)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=5) #fine-tune the last layer's output shpae to 5, because there are 5 classes in retinopathy problem
        
        ''' I want to start from the pretrained model to update the "whole" network instead of freezing some layers and train part of the network, so these lines are commented out
        if pretrained: #if we want to use the pretrained model, then freeze the un-fine-tune layer i.e. all the layers except for the last layer
            for param in self.model.parameters():
                param.requires_grad = False #freeze
            self.model.fc.weight.requires_grad = True #last layer's weight
            self.model.fc.bias.requires_grad = True #last layer's bias
        '''
    def forward(self,x):
        y = self.model(x)
        return y

def train(model_object, device, epochsize, optimizer, loss, lr,momentum,weight_decay, model_weights_path,messages, train_dataloader, test_dataloader):
    #construct optimizer
    ''' I want to start from the pretrained model to update the "whole" network instead of freezing some layers and just train part of the network, so these lines are commented out
    if model_object.pretrained_status == 'withPretraining': #if we use the pretrained model, then we only update the un-frozen layer i.e. the layer with requires_grad == True
        params_to_update=[]
        for name,param in model_object.named_parameters():
            if param.requires_grad:
                params_to_update.append(param) 
        optimizer_object = optimizer(params_to_update,lr=lr,momentum=momentum,weight_decay=weight_decay)
    elif model_object.pretrained_status == 'withoutPretraining': #else if we use the non-pretrained model, then update all the layers
        optimizer_object = optimizer(model_object.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    '''
    optimizer_object = optimizer(model_object.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    #to record the message(include train and test accuracy per epoch) in training phase under particular model
    epochs = []
    train_accuracies = []
    test_accuracies = []
    #to record the best model
    test_best_accuracy = 0
    message = None
    best_model_weights = None
    for epoch in range(epochsize):
        epochs.append(epoch)
        
        #train one epoch
        try:
            train_accuracies.append(train_one_epoch(model_object, device, optimizer_object, loss, epoch, train_dataloader)) #model_object's weight will update automatically during training's backward process
        except RuntimeError: #to prevent the process is suddenly killed and return the current training process 
            print("GPU out of memory")
            print(torch.cuda.memory_summary()) 
            return best_model_weights, message
        
        #test one epoch
        test_accuracies.append(test_one_epoch(model_object, device, None, test_dataloader)) #model_object's weight won't change or update. just test current model_object's performance. this line can be commented out
        if test_accuracies[-1] > test_best_accuracy: #record the best model
            best_model_weights = copy.deepcopy(model_object.state_dict()) #deep copy is necessary, or the model weight won't be saved and will be overwrited the next update process
            torch.save(best_model_weights, model_weights_path)
            test_best_accuracy = test_accuracies[-1]
        print(f"best test accuracy over {epoch} epoch: {test_best_accuracy}")
        
        #to trace the current training process to prevent the process is suddenly killed due to GPU out of memory
        message = {'model':model_object.name, 'pretrained':model_object.pretrained_status, 'epochs':epochs, 'train_accuracy_curve':train_accuracies, 'test_accuracy_curve':test_accuracies}
        plot_acc_comp(messages + [message])
        
    message = {'model':model_object.name, 'pretrained':model_object.pretrained_status, 'epochs':epochs, 'train_accuracy_curve':train_accuracies, 'test_accuracy_curve':test_accuracies}
    return message
    
def train_one_epoch(model_object, device, optimizer_object, loss, epoch, dataloader):
    model_object.train()
    total_acc = 0
    total_L = 0
    #with torch.enable_grad(): #comment out this line because not all the tensor needs to ompute gradient
    for (minibatchX, minibatchGtY) in tqdm(dataloader):
        #forward
        minibatchX = minibatchX.to(device)
        minibatchGtY = minibatchGtY.to(device, dtype=torch.long) #because the dtype of CrossEntropyLoss's input should be long
        minibatchPredY = model_object.forward(minibatchX)
        
        #compute loss, backward and update
        minibatchL = loss()(minibatchPredY, minibatchGtY)
        optimizer_object.zero_grad()
        minibatchL.backward()
        optimizer_object.step()
        total_acc += sum(torch.eq(torch.argmax(minibatchPredY,dim=1) ,minibatchGtY)).item()
        total_L += minibatchL.detach().item()
        
        #release GPU memory to prevent the process is suddenly killed
        del minibatchX, minibatchGtY, minibatchPredY, minibatchL
    if device == 'cuda' and hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache() #releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
        
    total_L /= len(dataloader.dataset)
    total_acc /= len(dataloader.dataset)
    print(f'epoch:{epoch}, loss={total_L:.4f}, accuracy={total_acc:.4f}')
    return total_acc
    
def test_one_epoch(model_object, device, model_weights_path, dataloader):
    
    if model_weights_path != None: #the model has been trained well
        model_object.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device))) #load the model weight
    #else the model has not finished its training phase
    model_object.eval()
    total_acc = 0
    wholeGtY = np.array([])
    wholePredY = np.array([]) #will store the highest possible label
    
    with torch.no_grad():
        for minibatchX, minibatchGtY in tqdm(dataloader):
            #forward
            minibatchX = minibatchX.to(device)
            minibatchGtY = minibatchGtY.to(device, dtype=torch.long)
            minibatchPredY = model_object.forward(minibatchX)
            #record to plot the confusion matrix
            wholeGtY = np.append(wholeGtY, minibatchGtY.cpu().data.numpy())
            wholePredY = np.append(wholePredY, np.argmax(minibatchPredY.cpu().data.numpy(),axis=1))
            #compute accuracy, plot and print some message
            total_acc += sum(torch.eq(torch.argmax(minibatchPredY,dim=1) ,minibatchGtY)).item()
        total_acc /= len(dataloader.dataset)
        if model_weights_path != None:
            print(f'model:{model_object.name} {model_object.pretrained_status}, accuracy:{total_acc:.5f}')
            plot_confusion_matrix(wholeGtY, wholePredY, model_object.name, model_object.pretrained_status)
    return total_acc

if __name__ == '__main__':
    #load dataset
    batchsize = 4
    trainDataset = RetinopathyDataset('./data', 'train')
    testDataset = RetinopathyDataset('./data', 'test')
    
    #initialize model parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #if GPU is avaliable, then use GPU
    learningRate = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = optim.SGD #delegate
    loss = nn.CrossEntropyLoss #delegate
    resnet18_pretrained = ResNet(pretrained=True,num_layers=18,epochsize=15,batchsize=32).to(device) #epochsize = 10->15, batchsize = 4->32; train/test one epoch on 1060 GPU requires about 10/1 mins 
    resnet18_curtrained = ResNet(pretrained=False,num_layers=18,epochsize=15,batchsize=32).to(device) #epochsize = 10->15, batchsize = 4->32; train/test one epoch on 1060 GPU requires about 10/1 mins 
    resnet50_pretrained = ResNet(pretrained=True,num_layers=50,epochsize=5,batchsize=8).to(device) #epochsize = 5->5, batchsize = 4->8; train/test one epoch on 1060 GPU requires about 30/3 mins 
    resnet50_curtrained = ResNet(pretrained=False,num_layers=50,epochsize=5,batchsize=8).to(device) #epochsize = 5->5, batchsize = 4->8; train/test one epoch on 1060 GPU requires about 30/3 mins 
    resnet18s = [resnet18_curtrained,resnet18_pretrained]
    resnet50s = [resnet50_curtrained,resnet50_pretrained]
    models = [resnet18s, resnet50s]
    mode = {'tune':True,'demo':False}
    
    #run models
    os.makedirs('./ExperimentReport', exist_ok=True)
    os.makedirs('./ModelWeights', exist_ok=True)
    accuracy_table = pd.DataFrame({'ResNet18':[None,None],'ResNet50':[None,None]},index=['withPretraining','withoutPretraining'])
    for model_types in models: #resnet18 or resnet50
        messages = [] #to record different pretrained status' accuracy message under particualr model
        for model_object in model_types: #with pretrained or withour pretrained
            #pre-setting
            trainDataloader = DataLoader(trainDataset, batch_size = model_object.batchsize, num_workers=4) #num_workers = 0
            testDataloader = DataLoader(testDataset, batch_size = model_object.batchsize, num_workers=4) #num_workers = 0
            model_weights_path = f'./ModelWeights/{model_object.name}_{model_object.pretrained_status}.pt'
            
            #train
            if mode['tune']:
                message = train(model_object, device, model_object.epochsize, optimizer, loss, learningRate,momentum,weight_decay, model_weights_path,messages, trainDataloader, testDataloader)
                messages.append(message)
            
            #test
            accuracy_table.loc[model_object.pretrained_status,model_object.name] = test_one_epoch(model_object, device, model_weights_path, testDataloader) #will also plot the confusion matrix in test function
            
            #release GPU memory to prevent the process is suddenly killed
            del model_object
            #if device == 'cuda': torch.cuda.init() #clear up all the GPU's memory to avoid error of "CUDA out of memory"
        if mode['tune']: plot_acc_comp(messages)
    print(accuracy_table)