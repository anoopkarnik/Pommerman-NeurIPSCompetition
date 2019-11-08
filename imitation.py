

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from timeit import default_timer as timer

import torch 
import gym,random
from pommerman import agents
import pommerman
from IPython.display import clear_output
import os


# In[2]:


train = np.load('train_data.npz')
x_train = train['observations']
p_train = train['actions']
v_train = train['rewards']

# In[3]:


test = np.load('test_data.npz')
x_test = test['observations']
p_test = test['actions']
v_test = test['rewards']


# In[11]:


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1 = torch.nn.Conv2d(15,32,kernel_size = 3,stride = 1,padding=1)
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size = 3, stride = 1,padding=1)
        self.conv3 = torch.nn.Conv2d(32,32,kernel_size = 3, stride =1, padding=1)
        self.fc1 = torch.nn.Linear(32*11*11,128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 128+7
        self.hidden_dim = 100
        self.layer_dim = 1
        self.sequence_length = 5
        self.lstm = torch.nn.LSTM(self.input_dim,self.hidden_dim,self.layer_dim,batch_first=True)
        self.fc2 = torch.nn.Linear(100*self.sequence_length,6)
        self.fc3 = torch.nn.Linear(100*self.sequence_length,1)
        self.value_weight = 10
        self.entropy_weight = 0.001
    def forward(self,x):
        batch_size = x.shape[0]
        width = x.shape[3]
        height = x.shape[4]
        rest = x[:,8-self.sequence_length:8,15:,:,:]
        x = x[:,8-self.sequence_length:8,:15,:,:]
        h0 = torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).requires_grad_()
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        x = x.reshape(batch_size*self.sequence_length,15,width,height)
        rest = rest.reshape(batch_size*self.sequence_length,rest.size(2),rest.size(3),rest.size(4))
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))   
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(batch_size*self.sequence_length,-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.cat([x,rest[:,0,0,0].reshape(batch_size*self.sequence_length,1),rest[:,1,0,0].reshape(batch_size*self.sequence_length,1),rest[:,2,0,0].reshape(batch_size*self.sequence_length,1),rest[:,3,0,0].reshape(batch_size*self.sequence_length,1),rest[:,4,0,0].reshape(batch_size*self.sequence_length,1),rest[:,5,0,0].reshape(batch_size*self.sequence_length,1),rest[:,6,0,0].reshape(batch_size*self.sequence_length,1)],1)
        x = x.view(batch_size,self.sequence_length,-1)
        x, (hn,cn) = self.lstm(x,(h0,c0))
        x = x.contiguous().view(batch_size,-1)
        value = self.fc3(x.squeeze())
        value = torch.squeeze(value)
        logits = self.fc2(x.squeeze())
        return logits,value    


# In[12]:


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[13]:

x_train = torch.FloatTensor(x_train).to(device)
p_train = torch.LongTensor(p_train).to(device)
v_train = torch.FloatTensor(v_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
p_test = torch.LongTensor(p_test).to(device)
v_test = torch.FloatTensor(v_test).to(device)


# In[14]:


model = DQN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
action_loss = torch.nn.CrossEntropyLoss()
value_loss = torch.nn.MSELoss()
epochs = 10
batch_size = 512


# In[15]:


def get_loss(p_output,v_output,p_target,v_target):
    loss1 = action_loss(p_output,p_target)
    loss2 = value_loss(v_output,v_target)
    loss = loss1 + 10 * loss2
    return loss


# In[16]:


accuracies = []
for i in range(epochs):
    correct = 0 
    total = 512
    torch.cuda.empty_cache() 
    train_size = len(x_train)
    for j in range(int(train_size/batch_size)):
        optimizer.zero_grad()
        x_batch_train = x_train[j*512:(j+1)*512,:,:,:,:]
        p_target = p_train[j*512:(j+1)*512].view(-1)
        v_target = v_train[j*512:(j+1)*512].view(-1)
        p_output,v_output = model(x_batch_train)
        loss = get_loss(p_output,v_output,p_target,v_target) 
        output = torch.max(p_output,1)[1]
        correct += torch.sum(torch.eq(output,p_target)).cpu().numpy()
        accuracy = 100.00 * correct/total
        total += 512
        loss.backward(retain_graph=True)
        optimizer.step() 
    with open("imitation.txt", "a") as f: print('train_accuracy of',i,'=',accuracy, file=f)
    correct = 0 
    total = 100
    test_size = len(x_test)
    for j in range(int(test_size/100)):
        x_batch_test = x_test[j*100:(j+1)*100,:,:,:,:]
        p_target = p_test[j*100:(j+1)*100].view(-1)
        v_target = v_test[j*100:(j+1)*100].view(-1)
        p_output,v_output = model(x_batch_test)
        loss = get_loss(p_output,v_output,p_target,v_target) 
        output = torch.max(p_output,1)[1]
        correct += torch.sum(torch.eq(output,p_target)).cpu().numpy()
        total += 100
    accuracy = 100.00 *correct/total
    accuracies.append(accuracy)
    if i>3:
        if accuracies[i]<= accuracies[i-1] and accuracies[i-1] <= accuracies[i-2] and accuracies[i-2] <= accuracies[i-3] :
            break
    with open("imitation.txt", "a") as f: print('test_accuracy of',i,'=',accuracy, file=f)
with open("imitation.txt", "a") as f: print('final_test_accuracy =', max(accuracies),file=f)

names = 'model.pt'
torch.save(model.state_dict(), names)

