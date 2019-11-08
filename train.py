#!/usr/bin/env python
# coding: utf-8

# # Modules

# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch 
import csv
import gym
import os.path
import random
from pommerman import agents
import pommerman


# # Hyperparameters
# 

# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epsilon variables
epsilon_start = 0.9
epsilon_end = 0.99
epsilon_decay = 10
epsilon_by_frame = lambda frame_idx: (epsilon_start + (epsilon_end-epsilon_start)/epsilon_decay)

#agent variables
gamma = 0.9
lr = 0.0001
entropy_weight=0.001
value_weight=0.5

#memory
target_net_update_freq = 10
sequence_size = 5
game_count = 0
update_count = 0

#learning control variables
learn_start = 0
update_freq = 1
last_discount_factor = 0.9
middle_discount_factor = 0.5


# # Network Declaration

# In[4]:


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
        rest = x[:,:,15:,:,:]
        x = x[:,:,:15,:,:]
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


# # Functions

# In[5]:


def compute_loss(model,s,v,p,a,r,entropy_weight,value_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_output = torch.FloatTensor(np.array(p)).to(device)
    s_output = torch.FloatTensor(np.array(s)).to(device)
    v_output = torch.FloatTensor(np.array(v)).to(device)
    r_output = torch.FloatTensor(np.array(r)).to(device)
    a_output = torch.LongTensor(np.array(a)).to(device)
    advantages = r_output - v_output
    valueloss = torch.nn.MSELoss()
    p_target, v_target = model(s_output)
    log_probs = torch.nn.functional.log_softmax(p_output, dim = -1)
    action_probs = torch.nn.functional.softmax(p_output,dim=-1)
    log_prob_actions = advantages * log_probs[range(len(a_output)),a_output]
    policy_loss = -log_prob_actions.mean()
    entropy_loss = -entropy_weight * (log_probs*action_probs).mean()
    value_loss = value_weight * valueloss(v_target,r_output)  
    loss = policy_loss + entropy_loss + value_loss
    return policy_loss,entropy_loss,value_loss,loss


# In[6]:


def update(model,state,value,prob,action,reward,frame,game_count,learn_start,update_freq,entropy_weight,value_weight):
    if frame<learn_start or frame%update_freq!=0:
        return None
    state = np.array([np.array(s) for s in state])
    value = np.array([np.array(v) for v in value])
    prob = np.array([np.array(p) for p in prob])
    action = np.array([np.array(a) for a in action])
    reward = np.array([np.array(r) for r in reward ])
    policy_loss,value_loss,entropy_loss,loss = compute_loss(model,state,value,prob,action,reward,entropy_weight,value_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()        
    return [policy_loss,entropy_loss,value_loss,loss]


# In[7]:


def get_action(model,s,epsilon):
    with torch.no_grad():  
        X = torch.tensor([s],device=device,dtype=torch.float)
        probs,values = model(X)
        num_actions = probs.size(0)
        probs = probs.cpu().numpy().squeeze()
        values = values.cpu().numpy().squeeze()
        if np.random.random()<=epsilon:
            maximum= -1000
            arg_maximum =0
            for i in range(6):
                if probs[i]>maximum:
                    maximum = probs[i]
                    arg_maximum = i
            action = arg_maximum
            return action,probs,values
        else:
            random_action = np.random.randint(0,num_actions)
            return random_action,probs,values
            


# In[8]:


def save_model(model,game_count,folder_name):
    names = 'trained_model.pt'
    torch.save(model.state_dict(), names)


# # Preprocessing

# In[9]:


def translate_obs(obs,agent_no):
    obs = obs[agent_no]
    board = obs['board']
    out = [board == i for i in range(10)]
    out.append(obs['bomb_blast_strength'])
    out.append(obs['bomb_life'])
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    out.append(position)
    if obs['teammate'] is not None:
        out.append(board == obs['teammate'].value)
    else:
        out.append(np.zeros(board.shape))
    enemies = [board == e.value for e in obs['enemies']]
    out.append(np.any(enemies, axis=0))
    out.append(np.full(board.shape, obs['ammo']))
    out.append(np.full(board.shape, obs['blast_strength']))
    out.append(np.full(board.shape, obs['can_kick']))
    out.append(np.full(board.shape,obs['position'][0]))
    out.append(np.full(board.shape,obs['position'][1]))
    out.append(np.zeros((board.shape)))
    out.append(np.zeros((board.shape)))
    return np.array(out)


# In[10]:


def plot_results(rewards,policy_loss,entropy_loss,value_loss,loss):
    plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(3, 2)
    ax0 = plt.subplot(gs[0,:])
    ax0.plot(rewards)
    ax0.set_xlabel('Episode')
    plt.title('Rewards')

    ax1 = plt.subplot(gs[1, 0])
    ax1.plot(policy_loss)
    plt.title('Policy Loss')
    plt.xlabel('Update Number')

    ax2 = plt.subplot(gs[1, 1])
    ax2.plot(entropy_loss)
    plt.title('Entropy Loss')
    plt.xlabel('Update Number')

    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(value_loss)
    plt.title('Value Loss')
    plt.xlabel('Update Number')
    
    ax4 = plt.subplot(gs[2, 1])
    ax4.plot(loss)
    plt.title('Loss')
    plt.xlabel('Update Number')

    plt.tight_layout()
    plt.show()


# In[11]:


def get_middle_rewards(memory,obs,prev_obs,agent_no):
    reward = 0
    memory = np.array(memory)
    prev_kick = prev_obs[agent_no]['can_kick']
    prev_blast_strength = prev_obs[agent_no]['blast_strength']
    prev_ammo = prev_obs[agent_no]['ammo']
    next_kick = obs[agent_no]['can_kick']
    next_blast_strength = obs[agent_no]['blast_strength']
    next_ammo = obs[agent_no]['ammo']
    if next_ammo != prev_ammo:
        reward +=0.01
    prev_enemies = len(prev_obs[agent_no]['alive']) - 1
    next_enemies = len(obs[agent_no]['alive']) - 1
    if next_kick != prev_kick:
        reward +=0.02
        prev_kick = True
    reward += (next_blast_strength - prev_blast_strength)*0.01
    if reward != -1:
        reward += (prev_enemies-next_enemies)*0.5    
    prev_agent_obs = translate_obs(prev_obs,agent_no)
    agent_obs = translate_obs(obs,agent_no)
    current_x = agent_obs[18][0][0]
    current_y = agent_obs[19][0][0]
    previous_x = memory[:,18,0,0]
    previous_y = memory[:,19,0,0]
    repeat = False
    for i in range(5):
        if current_x ==previous_x[i] and current_y==previous_y[i]:
            repeat = True
    if repeat ==False:
        reward += 0.001
    #print(reward)
    return reward


# In[12]:


def rollout(env,model,frame_idx,episode_type = 'simple'):
    transitions =[]
    agent_list = [0,1,2,3]
    actions_list = []
    agent_no = np.random.choice(agent_list)
    done = False
    obs = env.reset()
    agent_obs = translate_obs(obs,agent_no)
    memory = []
    for i in range(sequence_size):
        memory.append(np.array(agent_obs))
    prev_obs = obs
    prev_enemies = 3 
    states,  probs, values,rewards,actions_list,actions = [],[],[],[],[],[]
    states.append(np.array(memory))
    previous_agent_obs = agent_obs
    while not done:
        epsilon = epsilon_by_frame(frame_idx)
        actions = env.act(obs)
        action,prob,value = get_action(model,np.array(memory),epsilon)
        probs.append(prob)
        values.append(value)  
        actions[agent_no] = action
	
        obs,reward,done,_ = env.step(actions)
        if reward[agent_no] == -1: 
            done = True 
        middle_reward = get_middle_rewards(memory,obs,prev_obs,agent_no)
        transitions.append([memory,reward[agent_no],prob,value,action,middle_reward])
        del memory[0]
        agent_obs = translate_obs(obs,agent_no)
        memory.append(np.array(agent_obs))   
        previous_agent_obs = agent_obs  
    step = 0
    discounted_reward = np.zeros(len(transitions))
    last_reward = transitions[-1][1]
    for i in reversed(range(len(transitions))):
        transitions[i][1] = last_reward*(0.9**step)
        step+=1
    return transitions


# # Environment Start

# In[13]:


agents_list = [agents.SimpleAgent(),agents.SimpleAgent(),agents.SimpleAgent(),agents.SimpleAgent()]
env = gym.make('PommeRadioCompetition-v2')
for id, agent in enumerate(agents_list):
    assert isinstance(agent, agents.BaseAgent)
    agent.init_agent(id, env.spec._kwargs['game_type'])
env.set_agents(agents_list)
env.set_init_game_state(None)
env.set_render_mode('human')
sequence_size = 5
model = DQN().to(device)
model.load_state_dict(torch.load('model.pt',map_location='cpu'))
optimizer = torch.optim.Adam(model.parameters(),lr=lr)



# In[ ]:


last_rewards,policy_losses,entropy_losses,value_losses,losses = [],[],[],[],[]
frame_idx = 0
for k in range(10000):
    avg_rewards,avg_policy_losses,avg_entropy_losses,avg_value_losses,avg_losses = [],[],[],[],[]
    for i in range(100):
        if i==0:
            transitions = rollout(env,model,frame_idx)
        else:
            transitions = np.concatenate((transitions,rollout(env,model,frame_idx,env)),0)
        avg_rewards.append(np.array(transitions).T[1][-1])
    avg_reward = sum(avg_rewards)/100
    for j in range(100):
        transition = np.array(random.sample(list(transitions),1))
        state = np.array(transition).T[0]
        reward = np.array(transition).T[1]
        prob = np.array(transition).T[2]
        value = np.array(transition).T[3]
        action = np.array(transition).T[4]
        middle_reward = np.array(transition).T[5]
        #print('reward = '+str(reward) + ', middle reward =' + str(middle_reward))
        reward = reward + middle_reward
        losss = [1,2,3,4]
        losss = update(model,state,value,prob,action,reward,frame_idx,game_count,learn_start,update_freq,entropy_weight,value_weight)
        avg_policy_losses.append(losss[0])
        avg_entropy_losses.append(losss[1])
        avg_value_losses.append(losss[2])
        avg_losses.append(losss[3])
    avg_policy_losses = sum(avg_policy_losses)/100
    avg_entropy_losses = sum(avg_entropy_losses)/100
    avg_value_losses = sum(avg_value_losses)/100
    avg_losses = sum(avg_losses)/100
    frame_idx+=1
    last_rewards.append(avg_reward)
    policy_losses.append(avg_policy_losses)
    entropy_losses.append(avg_entropy_losses)
    value_losses.append(avg_value_losses)
    losses.append(avg_losses)
    print(last_rewards)
    #plot_results(last_rewards,policy_losses,entropy_losses,value_losses,losses)
    with open("training.txt", "a") as f: print('reward of',i,'=',last_rewards, file=f)
    folder_name = 'model4'
    save_model(model,frame_idx,folder_name)





