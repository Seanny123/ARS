'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''


import numpy as np

from arsrl.filter import get_filter


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
        Policy.__init__(self, policy_params)
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
        Policy.__init__(self, policy_params)
        # if trained_weights is not None:
        self.net = MLP(self.ob_dim, self.ac_dim)
        self.safeQ = linear(self.ob_dim, self.ac_dim).to(device)
        self.optimizer = optim.RMSprop(self.safeQ.parameters())

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        if trained_weights is not None:
            self.safeQ.load_state_dict(torch.load(trained_weights))
            self.safeQ.to(device)

    def update_weights(self, new_weights):
        print("UPDATE")
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return

    def getQ(self, ob):
        # ob = self.observation_filter(ob, update=self.update_filter)
        input_to_network = ob.astype(np.float64)
        input_to_network_ = torch.from_numpy(ob).float().to(device)
        return self.safeQ(input_to_network_).cpu().detach().double().numpy()

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        # print(np.argmax(self.net(obs).detach().double().numpy()))
        return self.net(obs).detach().double().numpy()

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        # aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux
