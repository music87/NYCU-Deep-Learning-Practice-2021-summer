import matplotlib.pyplot as plt
import numpy as np
def linear_data_generator(nData):
    #generate n linear separable data between 0, 1
    X = np.random.uniform(0,1,(nData,2)) #input data
    y = (X[:,0] < X[:,1]).astype(int) #ground truth label: 0, if x1<x2; 1, if x1>x2
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    return X,y.reshape(-1,1)

def XOR_data_generator(nData):
    #generate n XOR data between 0, 1
    x1 = np.linspace(0,1,nData) #input data
    x2 = np.zeros(nData)
    mask = np.arange(nData)[np.arange(nData)%2==0]
    x2[mask] = x1[mask]
    x2[~mask] = 1-x1[~mask]
    y = np.isclose(x1,1-x2).astype(int) #ground truth label: 0, if x1==x2; 1, if x1==1-x2
    X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    return X,y.reshape(-1,1)

def sigmoid(M): #M can be scalar, vector, matrix, and etc.
    return 1.0/(1.0+np.exp(-M))

def derivative_sigmoid(M): #M can be scalar, vector, matrix, and etc.
    return sigmoid(M)*(1-sigmoid(M))

def show_result(X,gt_y,pred_y,Ls):
    plt.subplot(1,3,1)
    plt.title('Ground truth', fontsize=13)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[:,0],X[:,1],c=gt_y)
    
    plt.subplot(1,3,2)
    plt.title('Predict result', fontsize=13)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[:,0],X[:,1],c=pred_y)   
    
    #learning curve
    plt.subplot(1,3,3)
    plt.title('Learning curve', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    Ls = np.asarray(Ls)
    plt.plot(Ls[:,0],Ls[:,1])
    plt.show()

class NeuralNetwork():
    def __init__(self,nHiddenUnits,learningRate,epsilon): 
        #initialize neural network parameters which include only weights (no bias)
        (h1,h2) = nHiddenUnits #amount of hidden units for first and second layer are h1 and h2 respectively
        self.lr = learningRate
        self.eps = epsilon
        self.W1 = np.random.randn(2,h1) #(2, h1): input layer -> hidden layer 1
        self.W2 = np.random.randn(h1,h2) #(h1, h2): hidden layer 1 -> hidden layer 2
        self.W3 = np.random.randn(h2,1) #(h2, 1): hidden layer 2 -> output layer
    
    def forward(self,bX): 
        #compute predicted y
        self.inputs = bX #mini-batch input
        self.Z1 = sigmoid(self.inputs@self.W1) #(batchsize, h1): Z1=sigmoid(X@W1)
        self.Z2 = sigmoid(self.Z1@self.W2) #(batchsize, h2): Z2=sigmoid(Z1@W2)
        pred_y = sigmoid(self.Z2@self.W3) #(batchsize, 1): y=sigmoid(Z2@W3)
        return pred_y
        
    def backward(self,gt_y,pred_y): 
        #backward propogation
        #dL/d{W3} = dL/dy * dy/d{Z2W3} * d{Z2W3}/d{W3}
        batchsize = gt_y.shape[0]

        grad_L_y = (-1/batchsize) * (gt_y/pred_y - (1-gt_y)/(1-pred_y)) #(1,1)
        grad_L_Z2W3 = grad_L_y * derivative_sigmoid(self.Z2@self.W3) #(batchsize,1): (1,1)*(batchsize,1)
        grad_L_W3 = (grad_L_Z2W3.T @ self.Z2).T #(h2, 1): (((1,batchsize).T)*(batchsize,h2)).T
        
        #dL/d{W2} = dL/d{Z2W3} * d{Z2W3}/d{Z2} * d{Z2}/d{Z1W2} * d{Z1W2}/d{W2}
        grad_L_Z2 = grad_L_Z2W3 @ self.W3.T #(batchsize, h2): (batchsize,1) * ((h2,1).T)
        grad_L_Z1W2 = grad_L_Z2 * derivative_sigmoid(self.Z1@self.W2) #(batchsize, h2): element-wise multiplication
        grad_L_W2 = (grad_L_Z1W2.T @ self.Z1).T #(h1,h2): (((batchsize,h2).T) * (batchsize,h1)).T
        
        #dL/d{W1} = dL/d{Z1W2} * d{Z1W2}/d{Z1} * d{Z1}/d{XW1} * d{XW1}/d{W1}
        grad_L_Z1 = grad_L_Z1W2 @ self.W2.T #(batchsize, h1): (batchsize,h2)*((h1,h2).T)
        grad_L_XW1 = grad_L_Z1 * derivative_sigmoid(self.inputs@self.W1) #(batchsize, h1): element-wise multiplication
        grad_L_W1 = (grad_L_XW1.T @ self.inputs).T #(2, h1): (((batchsize,h1).T) * (batchsize,2)).T
        
        #update model weights
        self.W1 = self.W1 - self.lr*grad_L_W1
        self.W2 = self.W2 - self.lr*grad_L_W2
        self.W3 = self.W3 - self.lr*grad_L_W3
        
    
    def loss(self,gt_y,pred_y):
        batchsize = gt_y.shape[0]
        return (-1/batchsize) * np.sum(gt_y*np.log(gt_y+self.eps)+(1-pred_y)*np.log(1-pred_y+self.eps)) #epsilon here is to prevent undefined log(0)
    
    def train(self,X,gt_y):
        epoch=0
        Ls = [] #record the loss for each iteration to plot the learning curve
        while True: #while not converge (for each epoch)
            epoch += 1
            minibatchX = X
            batchsize = minibatchX.shape[0]
            
            #train
            pred_y = self.forward(minibatchX) #forward to compute predicted y
            self.backward(gt_y, pred_y) #backward propogation to update model weights
            L = self.loss(gt_y,pred_y) #compute loss
            
            #print some messages
            Ls.append((epoch,L)) #to plot learning curve
            pred_y = (pred_y > 0.5).astype(int)  #if probability > 0.5, then output 1; else output 0
            accuracy = np.sum(gt_y == pred_y)/batchsize #compute accuracy
            if epoch % 500 == 0: 
                print('training ... epoch:{}, loss:{:.5f}, acc:{:.2f}'.format(epoch,L,accuracy))
            if np.abs(L) < self.eps: #if converge
                show_result(X,gt_y,pred_y,Ls)
                break
    
    def test(self,X,gt_y):
        batchsize = gt_y.shape[0]
        pred_y = self.forward(X)
        pred_y = (pred_y > 0.5).astype(int) #if probability > 0.5, then output 1; else output 0
        accuracy = np.sum(gt_y == pred_y)/batchsize 
        print('testing ... accuracy:{}'.format(accuracy))

        
if __name__ == '__main__':
    #generate data
    dataType = ['linear','XOR']
    nData = 100
    
    #initialize model parameter
    nHiddenUnits=(10,10) #amount of hidden units for each layer
    learningRate = 0.3 #learning rate
    epsilon = 0.01 #to judge converge or not
    
    for datatype in dataType:
        print(f'=====data type : {datatype}=====')
        X, gt_y = locals()[f'{datatype}_data_generator'](nData) #input, gounrd true label; refer to https://stackoverflow.com/questions/28372223/python-call-function-from-string
        NN = NeuralNetwork(nHiddenUnits,learningRate,epsilon)
        NN.train(X,gt_y)
        NN.test(X,gt_y)

