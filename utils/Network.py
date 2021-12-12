import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np 

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class LinearQNetwork(nn.Module):
    '''Linear Q network
    '''
    def __init__(self, input_shape, num_actions, **args):
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

    def _render_frame(self, env, state, action, writer):
        pass
class CnnQNetwork(nn.Module):
    '''CNN Q network
    Nature CNN Q network
    '''
    def __init__(self, input_shape, num_actions, **args):
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

    def _render_frame(self, env, state, action, writer):
        pass

class CatLinearQNetwork(nn.Module):
    '''Categorical Linear Q network
    '''
    def __init__(self, input_shape, num_actions, **args):
        super(CatLinearQNetwork, self).__init__()
        self.num_actions  = num_actions
        self.num_atoms    = args['num_atoms']
        self.Vmin         = args['v_min']
        self.Vmax         = args['v_max']
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

    def _render_frame(self, env, state, action, writer):
        pass

class CatCnnQNetwork(nn.Module):
    '''Categorical CNN Q network
    C51 CNN Q network
    '''
    def __init__(self, input_shape, num_actions, **args):
        super(CatCnnQNetwork, self).__init__()
        self.num_actions  = num_actions
        self.num_atoms    = args['num_atoms']
        self.Vmin         = args['v_min']
        self.Vmax         = args['v_max']
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

    def _render_frame(self, env, state, action, writer):
        if self.my_fig is None:
            self.my_fig = plt.figure(figsize=(10, 5), dpi=160)
            plt.rcParams['font.size'] = '8'
            gs = gridspec.GridSpec(1, 2)
            self.ax_left = self.my_fig.add_subplot(gs[0])
            self.ax_right = self.my_fig.add_subplot(gs[1])
            self.my_fig.tight_layout()
            self.fig_pixel_cols, self.fig_pixel_rows = self.my_fig.canvas.get_width_height()
        action_prob = np.swapaxes(self.action_prob[0].cpu().numpy(),0, 1)
        legends = []
        for i, action_meaning in enumerate(env.unwrapped.get_action_meanings()):
            legend_text = ' (Q=%+.2e)'%(self.action_Q[0,i]) if i == action else ' (Q=%+.2e)*'%(self.action_Q[0,i])
            legends.append(action_meaning + legend_text) 
        self.ax_left.clear()
        self.ax_left.imshow(state[-1])
        self.ax_left.axis('off')
        self.ax_right.clear()
        self.ax_right.plot(self.atoms_cpu, action_prob)
        self.ax_right.legend(legends)
        self.ax_right.grid(True)
        self.my_fig.canvas.draw()
        buf = self.my_fig.canvas.tostring_rgb()
        writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(self.fig_pixel_rows, self.fig_pixel_cols, 3))