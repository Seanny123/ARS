'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from arsrl.filter import get_filter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):

    def __init__(self, input, output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input, 32)
        self.fc2 = nn.Linear(32, output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class linear(nn.Module):

    def __init__(self, input, output):
        super(linear, self).__init__()
        self.fc1 = nn.Linear(input, output)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        super(LinearPolicy, self).__init__(policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class SafeBilayerExplorerPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params, trained_weights=None):
        super(SafeBilayerExplorerPolicy, self).__init__(policy_params)
        self.net = MLP(self.ob_dim, self.ac_dim)
        self.safeQ = linear(self.ob_dim, self.ac_dim).to(device)
        self.optimizer = optim.RMSprop(self.safeQ.parameters())

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        if trained_weights:
            self.safeQ.load_state_dict(torch.load(trained_weights, map_location='cpu'))
            self.safeQ.to(device)

    def update_weights(self, new_weights):
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return

    def getQ(self, ob):
        # input_to_network = ob.astype(np.float64)
        input_to_network_ = torch.from_numpy(ob).float().to(device)
        return self.safeQ(input_to_network_).cpu().detach().double().numpy()

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        return self.net(obs).detach().double().numpy()

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        # aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux
