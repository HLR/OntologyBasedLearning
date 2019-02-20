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
from  collections import OrderedDict

import spatialDataLoader

# parse sprl data file 
newSprl_sentences_df = spatialDataLoader.parseSprlXML('data/newSprl2017_all.xml') 
print(newSprl_sentences_df.columns)

# get features
corpus_df = spatialDataLoader.getCorpus(newSprl_sentences_df)

corpus_df.reset_index(drop=True, inplace=True)
print(corpus_df.head())

feature_df = corpus_df[['Feature_Words', 'output']]
print(feature_df.head())

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
        i = index
        X_internal1 = corpus_df['Feature_Words'][i]
        X_internal2 = X_internal1.toarray()
        
        X = torch.from_numpy(X_internal2).float()
        y_internal = corpus_df['output'][i]
        y = torch.tensor(y_internal)

        return X, y

# % of training set to use as validation
valid_size = 0.2

# Learning parameters 

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 225
D_in = 518 # input_size
H = 259    # hidden_size
D_out = 2 # num_classes

num_epochs = 40
learning_rate = 0.001

# 3. obtain inn(feature_dfdices that will be used for validation ----
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

# Create random Tensors to hold inputs and outputs
#X = torch.randn(N, D_in)
#y = torch.randn(N, D_out)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).float()
        t1 = self.fc1.weight.dtype
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes).float()
        t2 = self.fc2.weight.dtype

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(D_in, H, D_out)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    i = 1
    for X, y in train_loader:  
        # Forward pass
        print(i)
        print(X)
        print(y)
        t = X.dtype
        
        outputs = model(X)
        print(outputs)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        i = i + 1
        if (i+1) % 2 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
