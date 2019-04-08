import os
import time
import numpy as np
import gym
import json
import ray
import socket

from arsrl import logz, utils, optimizers
from collections import namedtuple
from arsrl.shared_noise import SharedNoiseTable, create_shared_noise
from arsrl.policies import LinearPolicy, SafeBilayerExplorerPolicy

import torch
import torch.nn.functional as F
import random
import darwinrl.simple_envs

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


@ray.remote
class Worker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params=None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02,
                 gym_kwargs=None):

        import darwinrl.simple_envs
        self.env_name = env_name
        if gym_kwargs is not None:
            self.env = gym.make(env_name, **gym_kwargs)
        else:
            self.env = gym.make(env_name)
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'bilayer_safe_explorer':
            self.policy = SafeBilayerExplorerPolicy(policy_params)
        else:
            raise NotImplementedError

        self.delta_std = delta_std
        self.rollout_length = rollout_length

    def __str__(self):
        return "Env_NAME:{} policy_params:{}".format(self.env_name, self.policy_params)

    def __repr__(self):
        return "Env_NAME:{} policy_params:{}".format(self.env_name, self.policy_params)

    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        assert (self.policy_params['type'] == 'bilayer' or self.policy_params['type'] == 'linear' or self.policy_params[
            'type'] == 'bilayer_safe_explorer')
        return self.policy.get_weights_plus_stats()

    def rollout(self, shift=0., rollout_length=None):
        """
        Performs one rollout of maximum length rollout_length.
        At each time-step it substracts shift from the reward.
        """

        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0
        transitions = []
        record_transitions = True

        ob = self.env.reset()
        for i in range(rollout_length):

            if type(ob) is dict:
                policy_obs = np.concatenate(([ob["level"], ob["progress"]], ob["outflow"]))
            else:
                policy_obs = ob

            action = self.policy.act(policy_obs)
            next_ob, reward, done, _ = self.env.step(action)

            # Constraints for linear safety layer
            if action[0] < 0 or next_ob["level"] < 0.01:
                record_transitions = False

            if record_transitions:
                transitions.append([ob, action[0], next_ob, reward])

            steps += 1
            total_reward += (reward - shift)
            ob = next_ob
            if done:
                break

        return total_reward, steps, transitions

    def linesearch(self, delta, backtrack_ratio=0.5, num_backtracks=10):
        deltas = [delta]
        return deltas

    def do_rollouts(self, w_policy, num_rollouts=1, shift=1, evaluate=False):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        all_transitions = []
        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)

                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, transitions = self.rollout(shift=0., rollout_length=self.env.spec.timestep_limit)
                rollout_rewards.append(reward)

            else:
                idx, delta = self.deltas.get_delta(w_policy.size)

                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, transitions = self.rollout(shift=shift)
                all_transitions = all_transitions + transitions

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps, transitions = self.rollout(shift=shift)
                steps += pos_steps + neg_steps
                all_transitions = all_transitions + transitions
                rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps,
                "transitions": all_transitions}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return


class ARSLearner(object):
    """
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32,
                 num_deltas=320,
                 deltas_used=320,
                 delta_std=0.02,
                 logdir=None,
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123,
                 gym_kwargs=None):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        if gym_kwargs is not None:
            env = gym.make(env_name, **gym_kwargs)
        else:
            env = gym.make(env_name)

        self.timesteps = 0
        self.action_size = env.action_space.shape[0]

        if env.observation_space.shape is not None:
            self.ob_size = env.observation_space.shape[0]
        elif type(env.observation_space) is gym.spaces.Dict:
            self.ob_size = 0
            for val in env.observation_space.spaces.values():
                self.ob_size += val.shape[0]
        else:
            raise NotImplementedError("What space is this?")

        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        # Parameters for Q Learner
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.TARGET_UPDATE = 5

        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.')
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std,
                                      gym_kwargs=gym_kwargs) for i in range(num_workers)]

        print(self.workers[0])
        # initialize policy
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'bilayer_safe_explorer':
            self.policy = SafeBilayerExplorerPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts=None, evaluate=False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts=num_rollouts,
                                                     shift=self.shift,
                                                     evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts=1,
                                                     shift=self.shift,
                                                     evaluate=evaluate) for worker in
                           self.workers[:(num_deltas % self.num_workers)]]

        # gather results
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], []
        all_transitions = []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            all_transitions += result['transitions']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            all_transitions += result['transitions']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)

        # Push all the transitions collected in the Replay Buffer
        for tran in all_transitions:
            self.memory.push(tran[0], tran[1], tran[2], tran[3])

        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[
            max_rewards >= np.percentile(max_rewards, 100 * (1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        if np.std(rollout_rewards) != 0:
            rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size=500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat

    def train_step(self):
        """
        Perform one update step of the policy weights.
        """

        g_hat = self.aggregate_rollouts()
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        self.policy.update_weights(self.w_policy)

    def update_explorer_net(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        policy_obs = [np.concatenate((np.array([a_tran.state["level"]]),
                                      np.array([a_tran.state["progress"]]),
                                      np.array(a_tran.state["outflow"]))) for a_tran in transitions]
        policy_obs = [torch.from_numpy(an_obs) for an_obs in policy_obs]
        state_batch = torch.cat(policy_obs).view(len(transitions), -1).to(device).float()

        policy_action = torch.from_numpy(np.array([a_tran.action for a_tran in transitions]))

        # set up the costs for constraints
        # TODO: more hardcoded index?
        cost_next_state = torch.from_numpy(np.array([a_tran.next_state["level"] for a_tran in transitions]))

        cost_state = torch.from_numpy(np.array([a_tran.state["level"] for a_tran in transitions]))

        # state_batch size of (b, obs_dim)
        # action size of (b, act_dim)
        transpose_action = self.policy.safeQ(state_batch)
        transpose_action = transpose_action.unsqueeze(-1)

        policy_action = torch.reshape(policy_action, transpose_action.size()).to(device)

        mul = torch.bmm(transpose_action.float(), policy_action.float())
        cost_state = torch.reshape(cost_state, mul.size()).to(device).float()
        target = cost_state + mul

        cost_next_state = torch.reshape(cost_next_state, target.size()).to(device).float()

        loss = F.mse_loss(cost_next_state, target)

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

    def train(self, num_iter):
        max_reward_ever = -1
        start = time.time()
        for i in range(num_iter):

            t1 = time.time()
            self.train_step()
            for iter_ in range(10):
                self.update_explorer_net()
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('iter ', i, ' done')

            # record statistics every 10 iterations
            if ((i + 1) % 10) == 0:
                rewards = self.aggregate_rollouts(num_rollouts=30, evaluate=True)
                print("SHAPE", rewards.shape)
                if np.mean(rewards) > max_reward_ever:
                    max_reward_ever = np.mean(rewards)
#                w = ray.get(self.workers[0].get_weights_plus_stats.remote())

                torch.save(self.policy.safeQ.state_dict(), self.logdir + "/safeQ_torch" + str(i) + ".pt")

                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("BestRewardEver", max_reward_ever)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()

            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)


def run_ars(params):
    dir_path = params['dir_path']

    if not (os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    if params["config_file"] == "":
        gym_kwargs = None
        env = gym.make(params['env_name'])
    else:
        with open(params["config_file"], "r") as fi:
            gym_kwargs = json.load(fi)

        env = gym.make(params['env_name'], **gym_kwargs)

    if env.observation_space.shape is not None:
        ob_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.Dict):
        ob_dim = 0
        for val in env.observation_space.spaces.values():
            ob_dim += val.shape[0]
    else:
        raise NotImplementedError("What space is this?")

    ac_dim = env.action_space.shape[0]

    print(f"ob_dim: {ob_dim}")
    print(f"ac_dim: {ac_dim}")
    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params = {'type': 'bilayer_safe_explorer',
                     'ob_filter': params['filter'],
                     'ob_dim': ob_dim,
                     'ac_dim': ac_dim}

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'],
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'],
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed=params['seed'],
                     gym_kwargs=gym_kwargs)

    ARS.train(params['n_iter'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Madras-v0')
    parser.add_argument('--n_iter', '-n', type=int, default=3000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=10)
    parser.add_argument('--rollout_length', '-r', type=int, default=500)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=1)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='trained_policies/waterenv')
    parser.add_argument('--logdir', type=str, default='trained_policies/waterenv')
    parser.add_argument('--config_file', type=str, default='')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init(redis_address="11.1.2.58:6379")

    args = parser.parse_args()
    params = vars(args)
    run_ars(params)
