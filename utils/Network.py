from math import inf
import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import random 
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class NetworkBase(nn.Module):
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        pass

    def act(self, state, *args, **kwargs):
        pass

class LinearQNetwork(nn.Module):
    '''Linear Q network
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super().__init__()
        self._num_actions = num_actions
        self._input_shape = input_shape
        self.layers = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, num_actions))
        )
        
    def forward(self, x):
        return self.layers(torch.as_tensor(x, device=torch.device(0), dtype=torch.float))
    
    def eps_greedy_act(self, state, eps, network_lock, *args, **kwargs):
        '''
        state format [X]
        '''
        eps_prob =  random.random()
        if eps_prob > eps:
            with network_lock, torch.no_grad():
                state = torch.as_tensor(state, device=torch.device(0), dtype=torch.float).unsqueeze(0)
                q_value = self.forward(state)
                return q_value.max(1)[1].data[0]
        else:
            return random.randint(0, self._num_actions-1)

class CnnQNetwork(LinearQNetwork):
    '''CNN Q network
    Nature CNN Q network
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super().__init__(input_shape, num_actions, *args, **kwargs)
        self.layers = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.feature_size(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions))
        )
        
    def forward(self, x):
        x = self.layers(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.layers(autograd.Variable(torch.zeros(1, *self._input_shape))).view(1, -1).size(1)

class CatLinearQNetwork(nn.Module):
    '''Categorical Linear Q network
    '''
    def __init__(self, input_shape, num_actions, v_min, v_max, num_atoms, *args, **kwargs):
        super(CatLinearQNetwork, self).__init__()
        self._num_actions  = num_actions
        self.input_shape = input_shape
        self.num_atoms    = num_atoms
        self.Vmin         = v_min
        self.Vmax         = v_max
        self.atoms_cpu = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        self.atoms_gpu = self.atoms_cpu.cuda()
        self.layers = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions * self.num_atoms))
        )

    def forward(self, x):
        x = self.layers(torch.as_tensor(x, device=torch.device(0), dtype=torch.float))
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self._num_actions, self.num_atoms)
        return x
    
    def forward_log(self, x):
        x = self.layers(torch.as_tensor(x, device=torch.device(0), dtype=torch.float))
        x = F.log_softmax(x.view(-1, self.num_atoms)).view(-1, self._num_actions, self.num_atoms)
        return x

    def eps_greedy_act(self, state, eps, network_lock):
        '''
        State format [X,Y]
        '''
        eps_prob =  random.random()
        if eps_prob > eps:
            with network_lock, torch.no_grad():
                state = torch.as_tensor(state, device=torch.device(0), dtype=torch.float).unsqueeze(0)
                self.action_prob = self.forward(state)
                self.action_Q = (self.action_prob * self.atoms_gpu).sum(-1)
            return torch.argmax(self.action_Q, dim=-1).item()
        else:
            return random.randint(0, self._num_actions-1)

class CatCnnQNetwork(CatLinearQNetwork):
    '''Categorical CNN Q network
    C51 CNN Q network
    '''
    def __init__(self, input_shape, num_actions, v_min, v_max, num_atoms, *args, **kwargs):
        super().__init__(input_shape, num_actions, v_min, v_max, num_atoms, *args, **kwargs)
        self.layers = nn.Sequential(
            layer_init(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.feature_size(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions * self.num_atoms))
        )

    def forward(self, x):
        x = self.layers(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(-1, self._num_actions, self.num_atoms)
        x = F.softmax(x, dim=-1)
        return x

    def forward_log(self, x):
        x = self.layers(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(-1, self._num_actions, self.num_atoms)
        x = F.log_softmax(x, dim=-1)
        return x

    def feature_size(self):
        return self.layers(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class CnnQNetwork_TicTacToe(CnnQNetwork):
    '''
    CNN Q network for tic tac toe
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super().__init__(input_shape, num_actions, *args, **kwargs)
        self.layers = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.feature_size(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions))
        )
        
    def forward(self, x):
        x = self.layers(torch.as_tensor(x, device=torch.device(0), dtype=torch.float))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def eps_greedy_act(self, state, eps, network_lock, legal_action_mask, *args, **kwargs):
        eps_prob =  random.random()
        if eps_prob > eps:
            with network_lock, torch.no_grad():
                state = torch.as_tensor(state, device=torch.device(0), dtype=torch.float).unsqueeze(0)
                q_value = self.forward(state)
                q_value[:,~legal_action_mask] = -inf
            return q_value.max(1)[1].data[0]
        else:
            return random.choice(list(range(self._num_actions))[legal_action_mask])
