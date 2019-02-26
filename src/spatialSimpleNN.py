# numpy
import numpy as np

# pandas
import pandas as pd

# torch
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# collection
from collections import OrderedDict

# file path
import os
from pathlib import Path

# loader of spatial data and extractor of features
import spatialDataLoader

# parse sprl data file 
newSprl_sentences_df = spatialDataLoader.parseSprlXML('data/newSprl2017_all.xml') 

# get features for loaded data
corpus_df = spatialDataLoader.getCorpus(newSprl_sentences_df)

# fix indexes in the spatial feature dataframe
corpus_df.reset_index(drop=True, inplace=True)

# select the feature and output columns from the dataframe
feature_df = corpus_df[['Feature_Words', 'output']]
print('feature_df head:\n', feature_df.head())
print('feature_df tail:\n', feature_df.tail())

# Pytorch Dataset for the selected feature data
class Dataset_From_DF(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, corpus_df):
    'Initialization'
    self.corpus_df = corpus_df

  def __len__(self):
    'Denotes the total number of samples'
    return self.corpus_df.index.size

  def __getitem__(self, index):
    'Generates one sample of data'

    # Get data from Dataframe
        
    X_internal1 = corpus_df['Feature_Words'][index]
    X_internal2 = X_internal1.toarray().squeeze()
    X = torch.from_numpy(X_internal2).float()
        
    y_internal = corpus_df['output'][index]
    y = torch.tensor(y_internal).float()

    return X, y

# % of training set to use as validation
valid_size = 0.2

# Learning parameters 

# N is batch size; D_in is input dimension; X = torch.randn(N, D_in)
# H is hidden dimension; D_out is output dimension. y = torch.randn(N, D_out)
N = 225
D_in = 518 # input_size
H1 = 259    # hidden_size
H2 = 130    # hidden_size
D_out = 2 # num_classes

num_epochs = 40
learning_rate = 0.001

num_train = feature_df.index.size - 6
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_data = Dataset_From_DF(feature_df)
train_loader = torch.utils.data.DataLoader(train_data,  batch_size=N, sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=N, sampler = valid_sampler)

# Fully connected neural network with three hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1).float()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2).float()
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes).float()
        #self.logsoft = nn.LogSoftmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        #out = self.logsoft(out)
        return out

modelTrajector = NeuralNet(D_in, H1, H2, D_out)
modelLandmark = NeuralNet(D_in, H1, H2, D_out)

# Train the model
total_step = 8

def perfromLearning(model, learnedConceptName):
    
    # Loss and optimizer
    criterion = nn.SmoothL1Loss() #n.NLLLoss() # nn.MSELoss(reduction='sum') #nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10) #TODO

    for epoch in range(num_epochs):
        for batch_idx, (X, y) in enumerate(train_loader):  
            def closure():
                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, y)
            
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                 
                return loss
                        
            loss = optimizer.step(closure)
    
            if (batch_idx+1) % 2 == 0:
                print (learnedConceptName + ' learning Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))
    
        #X, y = valid_dataset #TODO
        #valid_out = model(X)
        #valid_loss = loss_fn(valid_out, y)
        
        #scheduler.step(valid_loss)
        
    # Test the model
    with torch.no_grad(): # In test phase, we don't need to compute gradients (for memory efficiency)
        correct = 0
        total = 0
        for X, y in test_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
    
            #predicted_float = predicted.type_as(y)
            
            _, y_index = torch.max(y, 1)
            
            total += y_index.size(0)
            correct += (predicted == y_index).sum().item()
    
        print('Accuracy of the ' + learnedConceptName + ' network: {} %'.format(100 * correct / total))
    
    # Save the model checkpoint
    resultsPath = Path('results')
    if not resultsPath.exists():
        os.mkdir(resultsPath)
    torch.save(model.state_dict(), 'results/model' + learnedConceptName + '.ckpt')
    #model.load_state_dict('result/model' + learnedConceptName + '.ckpt')
    
perfromLearning(modelTrajector, 'Trajector')
perfromLearning(modelLandmark, 'Landmark')

