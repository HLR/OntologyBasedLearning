# file path
import os
from pathlib import Path

# collection
from collections import OrderedDict

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

# loader of spatial data and extractor of features
import spatialDataLoader

# References to ontology concept and method checking consistency  of learning results
import owlReasoning

# check if ontology path is correct
dataPath = Path(os.path.normpath("../data"))
if not os.path.isdir(dataPath.resolve()):
    print("Path to load data:", dataPath.resolve(), "does not exists")
    exit()
    
# parse sprl data file 
newSprl_sentences_df = spatialDataLoader.parseSprlXML('../data/newSprl2017_all.xml') 

# get features for loaded data
corpus_df = spatialDataLoader.getCorpus(newSprl_sentences_df)

# fix indexes in the spatial feature dataframe
corpus_df.reset_index(drop=True, inplace=True)

# select the feature and output columns from the dataframe
feature_df = corpus_df[['Feature_Words', 'output']]

#outputClasses = [(owlReasoning.mySaulSpatialOnto.lm,  "Landmark"), (owlReasoning.mySaulSpatialOnto.tr, "Trajector")]

#print('feature_df head:\n', feature_df.head())
#print('feature_df tail:\n', feature_df.tail())

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

    # Get data from DataFrame
        
    X_internal1 = corpus_df['Feature_Words'][index]
    X_internal2 = X_internal1.toarray().squeeze()
    X = torch.from_numpy(X_internal2).float()
        
    y_internal = corpus_df['output'][index]
    y = torch.tensor(y_internal).float()

    return X, y

feature_df_data_retriver = Dataset_From_DF(feature_df)

# % of training set to use as validation
valid_size = 0.2
test_size  = 0.05
# Learning parameters 

# N is batch size; D_in is input dimension; X = torch.randn(N, D_in)
# H is hidden dimension; D_out is output dimension. y = torch.randn(N, D_out)
N = 225
D_in = 518 # input_size
H1 = 259    # hidden_size
H2 = 130    # hidden_size
D_out = 2 # num_classes

num_epochs = 5
learning_rate = 0.001

num_train = feature_df.index.size - 6
split1 = int(np.floor((valid_size + test_size) * num_train))
split2 = int(np.floor(test_size * num_train))
indices = list(range(num_train))

test_idx = indices[0:split2]

indicesLandmark = indices[split2:]
indicesTrajector = indicesLandmark.copy()

# Landmark
np.random.shuffle(indicesLandmark)

Lvalid_idx, Ltrain_idx = indicesLandmark[0:split1], indicesLandmark[split1:]

Ltrain_sampler = SubsetRandomSampler(Ltrain_idx)
Lvalid_sampler = SubsetRandomSampler(Lvalid_idx)

Ltrain_loader = torch.utils.data.DataLoader(feature_df_data_retriver,  batch_size=N, sampler = Ltrain_sampler)
Lvalid_loader = torch.utils.data.DataLoader(feature_df_data_retriver, batch_size=N, sampler = Lvalid_sampler)

# Trajector
indicesTrajector = indicesLandmark.copy()
np.random.shuffle(indicesTrajector)
Tvalid_idx, Ttrain_idx = indicesTrajector[0:split1],  indicesTrajector[split1:]

Ttrain_sampler = SubsetRandomSampler(Ttrain_idx)
Tvalid_sampler = SubsetRandomSampler(Tvalid_idx)

Ttrain_loader = torch.utils.data.DataLoader(feature_df_data_retriver,  batch_size=N, sampler = Ltrain_sampler)
Tvalid_loader = torch.utils.data.DataLoader(feature_df_data_retriver, batch_size=N, sampler = Lvalid_sampler)

# Fully connected neural network with three hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1).float()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2).float()
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes).float()
        #self.batchNomral = nn.BatchNorm1d(num_classes)
        #self.logsoft = nn.LogSoftmax()
        #self.softmax2d = nn.Softmax2d()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        #out = self.batchNomral(out)
        #out = self.softmax2d(out)
        return out

modelTrajector = NeuralNet(D_in, H1, H2, D_out)
modelLandmark = NeuralNet(D_in, H1, H2, D_out)

# Train the model
total_step = 8

def perfromLearning(model, learnedConceptName, train_loader, valid_loader):
    
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
    
        # TODO: --- Integrate scheduler
        #X, y = valid_dataset #TODO
        #valid_out = model(X)
        #valid_loss = loss_fn(valid_out, y)
        
        #scheduler.step(valid_loss)
        
    # Validate the model
    with torch.no_grad(): # In validation phase, we don't need to compute gradients (for memory efficiency)
        correct = 0
        total = 0
        for X, y in valid_loader:
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
    
print('\n ---  Learning phase - Building classifiers --- \n')

perfromLearning(modelLandmark, 'Landmark', Ltrain_loader, Lvalid_loader)
perfromLearning(modelTrajector, 'Trajector', Ttrain_loader, Tvalid_loader)

# Test the model
with torch.no_grad(): # In test phase, we don't need to compute gradients (for memory efficiency)
    
    print('\n ---  Testing classifiers --- \n')

    for index in test_idx:
        
        X_internal1 = corpus_df['Feature_Words'][index]
        X_internal2 = X_internal1.toarray().squeeze()
        X = torch.from_numpy(X_internal2).float()
        
        Loutputs = modelLandmark(X)
        Toutputs = modelTrajector(X)
        
        foundClassesoOfSpacialEntity = []
        
        LValue, Lpredicted = torch.max(Loutputs.data, 0)            
        TValue, Tpredicted = torch.max(Toutputs.data, 0)          
        
        _Lpredicted = Lpredicted.item()
        _Tpredicted = Tpredicted.item()
        
        if owlReasoning.testConsistencyOfInstance(spatialDataLoader.output[_Lpredicted][0], spatialDataLoader.output[_Tpredicted][0]):
            print(newSprl_sentences_df['TEXT'][index], "  --- classified a", spatialDataLoader.output[_Lpredicted][1], '\n')
        else:
            print('Found not consistent classification will decide based on classifiers result: ', LValue.item(), TValue.item())
            if LValue.item() >= TValue.item():
                print(newSprl_sentences_df['TEXT'][index], "  --- classified a", spatialDataLoader.output[_Lpredicted][1], '\n')
            else:
                print(newSprl_sentences_df['TEXT'][index], "  --- classified a", spatialDataLoader.output[_Tpredicted][1], '\n')
        
        
        
    

