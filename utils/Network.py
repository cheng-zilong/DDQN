import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class LinearQNetwork(nn.Module):
    '''Linear Q network
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super(LinearQNetwork, self).__init__()
        self.layers = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, num_actions))
        )
        
    def forward(self, x):
        return self.layers(x.float())
    
    def act(self, state):
        with torch.no_grad():
            state   = torch.FloatTensor(state).unsqueeze(0).cuda()
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        return action.cpu().numpy()
class CnnQNetwork(nn.Module):
    '''CNN Q network
    Nature CNN Q network
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super(CnnQNetwork, self).__init__()
        self.input_shape = input_shape
        self.features = nn.Sequential(
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
        x = self.features(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        with torch.no_grad():
            state   = torch.FloatTensor(state).unsqueeze(0).cuda()
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        return action.cpu().numpy()

class CatLinearQNetwork(nn.Module):
    '''Categorical Linear Q network
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super(CatLinearQNetwork, self).__init__()
        self.num_actions  = num_actions
        self.num_atoms    = kwargs['num_atoms']
        self.Vmin         = kwargs['v_min']
        self.Vmax         = kwargs['v_max']
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
        x = self.layers(x.float())
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        return x
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).cuda()
            dist = self.forward(state).data.cpu()
            dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
            action = dist.sum(2).max(1)[1].numpy()[0]
        return action

class CatCnnQNetwork(nn.Module):
    '''Categorical CNN Q network
    C51 CNN Q network
    '''
    def __init__(self, input_shape, num_actions, *args, **kwargs):
        super(CatCnnQNetwork, self).__init__()
        self.num_actions  = num_actions
        self.num_atoms    = kwargs['num_atoms']
        self.Vmin         = kwargs['v_min']
        self.Vmax         = kwargs['v_max']
        self.input_shape = input_shape

        self.features = nn.Sequential(
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
            layer_init(nn.Linear(512, num_actions * self.num_atoms))
        )
        
        self.atoms_cpu = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        self.atoms = self.atoms_cpu.cuda()
        self.my_fig = None
        
    def forward(self, x):
        x = self.features(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(-1, self.num_actions, self.num_atoms)
        prob = F.softmax(x, dim=-1)
        return prob

    def forward_log(self, x):
        x = self.features(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(-1, self.num_actions, self.num_atoms)
        prob = F.log_softmax(x, dim=-1)
        return prob

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, device=torch.device(0)).unsqueeze(0)
            self.action_prob = self.forward(state)
            self.action_Q = (self.action_prob * self.atoms).sum(-1)
            action = torch.argmax(self.action_Q, dim=-1).item()
        return action

