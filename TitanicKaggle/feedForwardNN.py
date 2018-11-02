import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plot
import collections
import pandas as pd

def substrings_in_string(big_string, substrings):
    for i in range(len(substrings)):
        if big_string.find(substrings[i]) != -1:
            return i + 1
    return 0


# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocessing
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)

train_df['Ticket'] = train_df['Ticket'].str.extract('(\d+)')
test_df['Ticket'] = test_df['Ticket'].str.extract('(\d+)')
train_df['Ticket'].fillna(0, inplace=True)
test_df['Ticket'].fillna(0, inplace=True)
train_df['Ticket'] = pd.to_numeric(train_df['Ticket'])
test_df['Ticket'] = pd.to_numeric(test_df['Ticket'])

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train_df['Title']=train_df['Name'].map(lambda x: substrings_in_string(x, title_list))
train_df['Deck']=train_df['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))
train_df['Family_Size']=train_df['SibSp']+train_df['Parch']
train_df['Age*Class']=train_df['Age']*train_df['Pclass']
train_df['Fare_Per_Person']=train_df['Fare']/(train_df['Family_Size']+1)

test_df['Title']=test_df['Name'].map(lambda x: substrings_in_string(x, title_list))
test_df['Deck']=test_df['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))
test_df['Family_Size']=test_df['SibSp']+test_df['Parch']
test_df['Age*Class']=test_df['Age']*test_df['Pclass']
test_df['Fare_Per_Person']=test_df['Fare']/(test_df['Family_Size']+1)

train_df.drop(columns=['PassengerId','Ticket','Name','Cabin'], inplace=True)
test_df.drop(columns=['PassengerId','Ticket','Name','Cabin'], inplace=True)

train_df['Embarked'].fillna('D', inplace=True)
test_df['Embarked'].fillna('D', inplace=True)

cleanup_nums = {"Sex":     {"male": 0, "female": 1},
                "Embarked": {"S": 1, "C": 2, "Q": 3, 'D': 0 }}

train_df.replace(cleanup_nums, inplace=True)
test_df.replace(cleanup_nums, inplace=True)

# train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)


batch_size = 100
# n_iters = 6000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = 20000
input_dim = 12
output_dim = 2

train_target = torch.tensor(train_df['Survived'].values[0:700])
train = torch.tensor(train_df.drop('Survived', axis = 1).values[0:700])
train_tensor = torch.utils.data.TensorDataset(train, train_target)

train_loader = torch.utils.data.DataLoader(dataset=train_tensor,
                                           batch_size=batch_size,
                                           shuffle=True)

val_target = torch.tensor(train_df['Survived'].values[700:-1])
val = torch.tensor(train_df.drop('Survived', axis = 1).values[700:-1])
val_tensor = torch.utils.data.TensorDataset(val, val_target)

val_loader = torch.utils.data.DataLoader(dataset=train_tensor,
                                           batch_size=batch_size,
                                           shuffle=False)

test = torch.tensor(test_df.values.astype(np.float))
test_tensor = torch.utils.data.TensorDataset(test)

test_loader = torch.utils.data.DataLoader(dataset=test_tensor,
                                          shuffle=False)

isinstance (train_loader,collections.Iterable)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_layer):
        super(LogisticRegressionModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.linears = nn.ModuleList([nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(1, len(hidden_layers))])
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        if activation_layer == 'relu':
            self.activation_layer = nn.ReLU()
        elif  activation_layer == 'sigmoid':
            self.activation_layer = nn.Sigmoid()

    def forward(self,x):
        out = self.input_layer(x)
        out = self.activation_layer(out)
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            out = self.activation_layer(out)
        out = self.output_layer(out)
        return out

hidden_layers = [35, 35, 35]
activation_layer = 'relu'

model = LogisticRegressionModel(input_dim, hidden_layers, output_dim, activation_layer)

model = model.double()

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = Variable(images.view(-1, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, input_dim))
            labels = Variable(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1
        if iter %100 == 0:
            print('Iteration: {}. Loss: {} '.format(iter, loss.data[0]))

        if iter %500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for i, (images, labels) in enumerate(val_loader):

                if torch.cuda.is_available():
                    images = Variable(images.view(-1, input_dim).cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images.view(-1, input_dim))
                    labels = Variable(labels)


                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += 1

                correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('%Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))

# Iterate through test dataset
id = 892
for images in test_loader:

    if torch.cuda.is_available():
        images = Variable(images[0].view(-1, input_dim).cuda())
    else:
        images = Variable(images[0].view(-1, input_dim))

    # Forward pass only to get logits/output
    outputs = model(images)

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)
    print(str(id) + "," + str(int(predicted.cpu())))
    id += 1