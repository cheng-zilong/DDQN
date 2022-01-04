from math import inf
import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import random 
import numpy as np 

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
                return q_value.max(1)[1].item()
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

class CnnQNetwork_TicTacToe(nn.Module):
    '''
    CNN Q network for tic tac toe
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super().__init__()
        self._num_actions = num_actions
        self._input_shape = input_shape
        self.layers = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 192, kernel_size=5, padding=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(192, 1, kernel_size=1, stride=1))
        )
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.feature_size(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, num_actions)),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.layers(torch.as_tensor(x, device=torch.device(0), dtype=torch.float))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.layers(autograd.Variable(torch.zeros(1, *self._input_shape))).view(1, -1).size(1)

    def eps_greedy_act(self, state, eps, network_lock, legal_action_mask, *args, **kwargs):
        eps_prob =  random.random()
        if eps_prob > eps:
            with network_lock, torch.no_grad():
                state = torch.as_tensor(state, device=torch.device(0), dtype=torch.float).unsqueeze(0)
                q_value = self.forward(state)
                q_value[:,~legal_action_mask] = -inf
            return q_value.max(1)[1].item()
        else:
            return random.choice(np.asarray(range(self._num_actions))[legal_action_mask])



# Define block
class BasicBlock(nn.Module):
	def __init__(self, filters_num):
		super().__init__()
		
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(filters_num, filters_num, 3, padding=1, stride=1),
			nn.BatchNorm2d(filters_num),
			nn.ReLU(),
		) 
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(filters_num, filters_num, 3, padding=1, stride=1),
			nn.BatchNorm2d(filters_num),
		)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		return self.relu(self.conv_block2(self.conv_block1(x)) + x)

# Define network
class Net100(nn.Module):
	def __init__(self):
		super(Net100, self).__init__()
		channel_num = 16
		
		#TODO: 1x1 convolution -> relu (to convert feature channel number)
		self.init_block = nn.Sequential(
			nn.Conv2d(1, channel_num, 1),
			nn.ReLU(),
		) 
		#TODO: stack 100 BasicBlocks
		self.basic_blocks = nn.ModuleList([BasicBlock(channel_num) for i in range(100)])

		#TODO: 1x1 convolution -> sigmoid (to convert feature channel number)
		self.final_block = nn.Sequential(
			nn.Conv2d(channel_num, 1, 1),
			nn.Sigmoid(),
		) 

	def forward(self, x):
		
		#TODO: forward
		x = self.init_block(x)
		for i, _ in enumerate(self.basic_blocks):
			x = self.basic_blocks[i](x)
		
		out = self.final_block(x)
		return out



class AlphaZeroNetwork(nn.Module):
    '''
    TODO No layer init
    '''
    def __init__(self, input_shape, num_actions, residual_num=19, filters_num=256, *args, **kwargs):
        super().__init__()
        self._num_actions = num_actions
        self._input_shape = input_shape
        self._filters_num = filters_num
        self.body_cnn = nn.Sequential(
			nn.Conv2d(input_shape[0], self._filters_num, 3, padding=1, stride=1),
			nn.ReLU(),
            *[BasicBlock(self._filters_num) for _ in range(residual_num)],
		)  
        self.policy_head_cnn = nn.Sequential(
			nn.Conv2d(self._filters_num, 2, 1),
			nn.BatchNorm2d(2),
            nn.ReLU()
		) 
        self.policy_head_fc = nn.Sequential(
			nn.Linear(2 * input_shape[1] * input_shape[2], num_actions)
		) 
        self.value_head_cnn = nn.Sequential(
			nn.Conv2d(self._filters_num, 1, 1),
			nn.BatchNorm2d(1),
            nn.ReLU()
		) 
        self.value_head_fc = nn.Sequential(
			nn.Linear(1 * input_shape[1] * input_shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
		) 
        
    def forward(self, x):
        x = self.body_cnn(torch.as_tensor(x, device=torch.device(0), dtype=torch.float))
        p = self.policy_head_cnn(x)
        p = p.view(p.size(0), -1)
        p = self.policy_head_fc(p)
        v = self.value_head_cnn(x)
        v = v.view(v.size(0), -1)
        v = self.value_head_fc(v) 
        return p, v
