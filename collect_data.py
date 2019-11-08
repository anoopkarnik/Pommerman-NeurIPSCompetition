#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pommerman
from pommerman import agents
import numpy as np
from copy import deepcopy



# In[2]:


def featurize(obs):
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
    out.append(np.zeros(board.shape))
    out.append(np.zeros(board.shape))
    return np.array(out)


# In[3]:


num_episodes = 150000
discount = 0.9

# Create a set of agents (exactly four)
agent_list = [
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
]
# Make the "Free-For-All" environment using the agent list
env = pommerman.make('PommeRadioCompetition-v2', agent_list)

observations = []
actions = []
rewards = []


# In[4]:


for i in range(num_episodes):
    eps_observations = [[], [], [], []]
    eps_actions = [[], [], [], []]
    eps_rewards = [[], [], [], []]

    obs = env.reset()
    done = False
    reward = [0, 0, 0, 0]
    t = 0
    while not done:
        action = env.act(obs)
        new_obs, new_reward, done, info = env.step(action)
        for j in range(4):
            if reward[j] == 0:
                eps_observations[j].append(featurize(obs[j]))
                eps_actions[j].append(action[j])
                eps_rewards[j].append(new_reward[j])
        obs = deepcopy(new_obs)
        reward = deepcopy(new_reward)
        t += 1
    #print("Episode:", i + 1, "Max length:", t, "Rewards:", reward)
    # sample one observation from each agent
    for j in range(4):
        eps_length = len(eps_observations[j])
        idx = np.random.randint(eps_length)
        observation_s = []
        action_s = []
        reward_s = []
        for k in range(idx-7,idx+1):
            steps = eps_length - k - 1
            observation_s.append(eps_observations[j][k])
        action_s.append(eps_actions[j][idx])
        reward_s.append(reward[j] * discount**steps)
        observations.append(observation_s)
        actions.append(action_s)
        rewards.append(reward_s)
        if i % 10000==0 and i!=0:
            out_file = 'train_data.npz'
            np.savez_compressed(out_file, observations=observations, actions=actions, rewards = rewards)

num_episodes = 20000
discount = 0.9

# Create a set of agents (exactly four)
agent_list = [
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
]
# Make the "Free-For-All" environment using the agent list
env = pommerman.make('PommeRadioCompetition-v2', agent_list)

observations = []
actions = []
rewards = []


for i in range(num_episodes):
    eps_observations = [[], [], [], []]
    eps_actions = [[], [], [], []]
    eps_rewards = [[], [], [], []]

    obs = env.reset()
    done = False
    reward = [0, 0, 0, 0]
    t = 0
    while not done:
        action = env.act(obs)
        new_obs, new_reward, done, info = env.step(action)
        for j in range(4):
            if reward[j] == 0:
                eps_observations[j].append(featurize(obs[j]))
                eps_actions[j].append(action[j])
                eps_rewards[j].append(new_reward[j])
        obs = deepcopy(new_obs)
        reward = deepcopy(new_reward)
        t += 1
    #print("Episode:", i + 1, "Max length:", t, "Rewards:", reward)
    # sample one observation from each agent
    for j in range(4):
        eps_length = len(eps_observations[j])
        idx = np.random.randint(eps_length)
        observation_s = []
        action_s = []
        reward_s = []
        for k in range(idx-7,idx+1):
            steps = eps_length - k - 1
            observation_s.append(eps_observations[j][k])
        action_s.append(eps_actions[j][idx])
        reward_s.append(reward[j] * discount**steps)
        observations.append(observation_s)
        actions.append(action_s)
        rewards.append(reward_s)
        if i % 10000==0 and i!=0:
            out_file = 'test_data.npz'
            np.savez_compressed(out_file, observations=observations, actions=actions, rewards = rewards)
