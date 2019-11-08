import numpy as np
import torch 
import gym
from pommerman import agents
import pommerman
from pommerman.runner import DockerAgentRunner


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


class MyAgent(DockerAgentRunner):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agents_list = [agents.SimpleAgent(),agents.SimpleAgent(),agents.SimpleAgent(),agents.SimpleAgent()]
        env = gym.make('PommeRadioCompetition-v2')
        for id, agent in enumerate(agents_list):
            assert isinstance(agent, agents.BaseAgent)
            agent.init_agent(id, env.spec._kwargs['game_type'])
        env.set_agents(agents_list)
        env.set_init_game_state(None)
        env.set_render_mode('human')
        self.sequence_size = 5
        self.model = DQN().to(self.device)
        self.model.load_state_dict(torch.load('trained_model.pt',map_location='cpu'))
        self.memory = []
    
    def translate_obs(self,obs):
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
    
    def act(self,obs,action_space=6):
        if len(self.memory) <8:
            for i in range(8):
                self.memory.append(self.translate_obs(obs))
        else:
            del memory[0]
            memory.append(self.translate_obs(obs))
        X = torch.tensor([self.memory],device=self.device,dtype=torch.float)
        probs,values = self.model(X)
        num_actions = probs.size(0)
        probs = probs.cpu().numpy().squeeze()
        values = values.cpu().numpy().squeeze()
        maximum= -1000
        arg_maximum =0
        for i in range(6):
            if probs[i]>maximum:
                maximum = probs[i]
                arg_maximum = i
        action = arg_maximum
        return action

def main():
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()

        

