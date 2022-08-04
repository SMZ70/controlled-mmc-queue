"""
Implements classes to model learners for MMC queues
"""

from copy import deepcopy
from typing import List

import numpy as np
from stable_baselines3 import DQN, A2C

from stable_baselines3.common.base_class import BaseAlgorithm

from .MMCEnv import MMCEnv

from .clog import CLogger

log = CLogger("Learners", console_log_level=1, file_log_level=1)


# decorator
def check_if_trained(func):
    """Decorator to check if the learner is trained. A learner is considered trained if learner.is_trained == True

    :param func: function to be wrapped
    :raises: RuntimeError in case the method learn is not called for the learner
    :return: None
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_trained:
            raise RuntimeError("Learning has not completed. Call .run method before simulation!")
        res = func(self, *args, **kwargs)
        return res

    return wrapper


class Learner(object):
    """
    A class to implement the base Learner. Different Learners can be implemented as subclasses of :Learner:
    """
    def __init__(self, mmc_env: MMCEnv):
        self.mmc_env = mmc_env
        self.is_trained = False
        self.name = None

    def learn(self, *args, **kwargs):
        raise NotImplementedError("The method learn is not implemented!")

    def predict(self, state):
        raise NotImplementedError("The method predict is not implemented!")

    @check_if_trained
    def simulate(self, n_events, n_reps, init_state: List = None, verbose=0) -> np.ndarray:
        """Runs the simulation for `n_reps` times and for `n_events` each starting from the initial state `init_state`

        :param n_events: number of events to simulate
        :param n_reps: numer of replications to use for the simulation
        :param init_state: the initial state. If nothing passed, empty system will be assumed
        :param verbose: if 0, nothing will be logged. if 1, at the end of each replication. if 2, at the end of
        each event
        :return: a 2-d numpy array consisting of the recorded queue length at each event for each replication
        """
        if init_state is None:
            init_state = [0] * (1 + self.mmc_env.n_servers)

        q_lengths = []
        for rep in range(n_reps):
            state = deepcopy(init_state)
            q_lengths_rep = []
            for eve in range(n_events):
                if isinstance(self, BaseAlgorithm):
                    action, _ = self.predict(state, deterministic=True)
                else:
                    action = self.predict(state)
                state = self.mmc_env.action_transition(state, action)
                state, next_event = self.mmc_env.event_transition(state)
                q_lengths_rep.append(sum(state))
                if verbose == 2:
                    log.info(f"replication {rep:5d} / {n_reps} - event {eve:5d} / {n_events}", f"SIM-{self.name}")
            q_lengths.append(q_lengths_rep)
            if verbose == 1:
                log.info(f"replication {rep:5d} / {n_reps}", f"SIM-{self.name}")
        q_lengths = np.array(q_lengths)
        return q_lengths


class ValueIteration(Learner):
    """
    A Learner class to implement value iteration to compute optimal policy and its value
    """
    def __init__(self, mmc_env: MMCEnv, discount_factor: float):
        super(ValueIteration, self).__init__(mmc_env=mmc_env)
        self.discount_factor = discount_factor
        self.all_states = self.mmc_env.all_states
        self.V = None
        self.name = "ValueIteration"

    def learn(self, convergence_threshold: float = 1e-5, log_progress: bool = True):
        """implements the value iteration method.

        :param convergence_threshold: controls the stopping criterion. the algorithm stops if maximum change in the
        value of states is smaller than the :convergence_threshold:
        :param log_progress: if True, the learning progress is logged
        :type convergence_threshold: float
        """
        v = {tuple(s): 0 for s in self.all_states}
        i = 0
        while True:
            v2 = deepcopy(v)
            i += 1
            for state in self.mmc_env.all_states:
                new_states = [self.mmc_env.action_transition(state, a)
                              for a in self.mmc_env.get_possible_actions(state, return_encoded=False)]
                new_vals = [v[tuple(new_state)] for new_state in new_states]

                v2[tuple(state)] = min(new_vals)
            max_delta_v = -1
            for state in self.all_states:
                new_states, probs = self.mmc_env.get_possible_event_transitions(state)
                old_v = v[tuple(state)]
                v[tuple(state)] = sum(state) + self.discount_factor * sum(
                    v2[tuple(ns)] * p for ns, p in zip(new_states, probs))
                delta_v = abs(old_v - v[tuple(state)])
                if delta_v > max_delta_v:
                    max_delta_v = delta_v

            if log_progress:
                log.info(f"iter={i:6d}, max_delta_v={max_delta_v:13.8f}", f"LEARN-{self.name}")

            if max_delta_v < convergence_threshold:
                break

        self.is_trained = True
        self.V = v

    @check_if_trained
    def predict(self, state):
        """predicts the next action (deterministically) based on a given :state:

        :param state: the state based on which the next state is to be predicted
        :return: the action based on the state values stored
        """
        poss_actions = self.mmc_env.get_possible_actions(state, return_encoded=False)
        resulting_states = [self.mmc_env.action_transition(state, action) for action in poss_actions]
        resulting_vals = {tuple(a): self.V[tuple(state)] for a, state in zip(poss_actions, resulting_states)}
        action = min(resulting_vals, key=resulting_vals.get)

        return action


class TabularQLearner(Learner):
    """
    Implements the tabular epsilon-greedy Q-learning algorithm (off-policy)
    """
    def __init__(self, mmc_env: MMCEnv, learning_rate: float,
                 discount_factor: float, max_eps: float, min_eps: float, eps_decay: float):
        """

        :param mmc_env: the MMC queue environment of type :MMCEnv:
        :type mmc_env: MMCEnv
        :param learning_rate: the learning rate, step size or alpha
        :param discount_factor: the discount factor or gamma
        :param max_eps: the epsilon to start the algorithm. In the process of the algorithm, the epsilon decays
        exponentially to :min_eps:
        :param min_eps: the minimum value for epsilon
        :param eps_decay: the epsilon decay rate
        """
        super(TabularQLearner, self).__init__(mmc_env=mmc_env)
        self.env = mmc_env
        self.queue_cap = mmc_env.queue_cap
        self.n_servers = mmc_env.n_servers
        self.Q = np.zeros((self.queue_cap + 1, *[2] * self.n_servers, 2 ** self.env.n_servers, 1))

        # learning params
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay

        self.name = "QLearning"

    def learn(self, total_timesteps, log_progress=True, return_history=False):
        """implements the epsilon-greedy Q-learning algorithm

        :param total_timesteps: total number of episodes to use for learning
        :param log_progress: if true intermediate progress logs will be logged
        :param return_history: if true, the `training_rewards` and the `epsilons` will be returned.
        :return:
        """
        eps = self.max_eps
        self.Q = np.zeros((self.queue_cap + 1, *[2] * self.n_servers, 2 ** self.env.n_servers, 1))
        for state in self.env.all_states:
            for action in range(2 ** self.n_servers):
                if action not in self.env.get_possible_actions(state, return_encoded=True):
                    self.Q[tuple(state + [action])] = self.env.punish_reward

        episode = 0
        training_rewards = []
        epsilons = []
        while True:  # loop over episodes
            state = self.env.reset()
            total_training_rewards = 0
            old_q_vals = deepcopy(self.Q)

            while True:  # Simulation loop
                poss_actions = self.env.get_possible_actions(state, return_encoded=True)
                r = np.random.rand()
                if r <= eps:
                    action = np.random.choice(poss_actions)
                else:
                    action = np.argmax(self.Q[tuple(self.env.curr_state)])

                new_state, rew, done, info = self.env.step(action)
                if info.get("infeasible"):
                    log.warn(f"Infeasible action observed. state, poss_actions, action:\n{state, poss_actions, action}",
                             f"LEARN-{self.name}")

                next_greedy_reward = np.max(self.Q[tuple(new_state)])
                alpha = self.learning_rate

                self.Q[tuple(state + [action])] += alpha * \
                                                   (rew + self.discount_factor * next_greedy_reward -
                                                    self.Q[tuple(state + [action])])
                total_training_rewards += rew
                state = new_state
                if done:
                    break

            max_delta_q = abs(old_q_vals - self.Q).max()
            eps = self.min_eps + (self.max_eps - self.min_eps) * np.exp(-self.eps_decay * episode)
            if log_progress:
                log.info(f"episode: {episode:05d}, max_delta_q: {max_delta_q:12.5f}, "
                         f"avg_rew: {total_training_rewards / self.env.max_events:10.4f}, "
                         f"eps: {eps:0.4f}, L_q_end: {self.env.curr_state[0]}", f"LEARN-{self.name}")
            if episode > total_timesteps:
                if log_progress:
                    log.success("Learn successful", f"LEARN-{self.name}")
                break
            episode += 1

            training_rewards.append(total_training_rewards)
            epsilons.append(eps)

        self.is_trained = True

        if return_history:
            return training_rewards, epsilons

    @check_if_trained
    def predict(self, state):
        """Predicts the next action (deterministically, greedy) based on a given state and the stored Q values for that
        state

        :param state: the state ([n_in_queue, n_in_server_1, n_in_server_2, ...]) based on which the next state is to be
        predicted
        :type state: list
        :return: the predicted action based on the current state
        """
        action = np.argmax(self.Q[tuple(state)])
        return action


class SARSA(TabularQLearner):
    """
    Implements the State–action–reward–state–action (SARSA) algorithm (on-policy). Sub-class of :TabularQLearner: since
    everything is the same except for the :learn: method.
    """
    def __init__(self, *args, **kwargs):
        super(SARSA, self).__init__(*args, **kwargs)
        self.name = "SARSA"

    def learn(self, total_timesteps: int, log_progress: bool = True, return_history: bool = False):
        """Implements the SARSA algorithm. The main difference from :TabularQLearner: is that the Q-values are updated
        not based on a greedy action but based on the actual action that is going to be taken in the next timestep.

        :param total_timesteps: the total timesteps to use for learning the optimal Q values
        :param log_progress: if True, the progress of the learning process is logged
        :param return_history: if True, the history of learning, including the :training_rewards: and :epsilons: will be
        returned
        :type total_timesteps: int
        :type log_progress: bool
        :type return_history: bool
        :return: if :return_history:, :training_rewards: and :epsilons:
        """
        eps = 1
        self.Q = np.zeros((self.queue_cap + 1, *[2] * self.n_servers, 2 ** self.env.n_servers, 1))
        for state in self.env.all_states:
            for action in range(2 ** self.n_servers):
                if action not in self.env.get_possible_actions(state, return_encoded=True):
                    self.Q[tuple(state + [action])] = self.env.punish_reward

        episode = 0
        training_rewards = []
        epsilons = []
        while True:  # loop over episodes
            state1 = self.env.reset()
            action1 = 0
            total_training_rewards = 0
            old_q_vals = deepcopy(self.Q)

            while True:  # Simulation loop
                state2, rew, done, info = self.env.step(action1)

                poss_actions2 = self.env.get_possible_actions(state2, return_encoded=True)
                r = np.random.rand()
                if r <= eps:
                    action2 = np.random.choice(poss_actions2)
                else:
                    action2 = np.argmax(self.Q[tuple(state2)])

                alpha = self.learning_rate
                self.Q[tuple(state1 + [action1])] += alpha * (rew +
                                                              self.discount_factor * self.Q[tuple(state2 + [action2])] -
                                                              self.Q[tuple(state1 + [action1])])
                state1 = deepcopy(state2)
                action1 = deepcopy(action2)

                total_training_rewards += rew
                if done:
                    break

            max_delta_q = abs(old_q_vals - self.Q).max()
            eps = self.min_eps + (self.max_eps - self.min_eps) * np.exp(-self.eps_decay * episode)
            if log_progress:
                log.info(f"episode: {episode:05d}, max_delta_q: {max_delta_q:12.5f}, "
                         f"avg_rew: {total_training_rewards / self.env.max_events:10.4f}, "
                         f"eps: {eps:0.4f}, L_q_end: {self.env.curr_state[0]}", f"LEARN-{self.name}")
            # if max_delta_q < 1e-4:
            if episode > total_timesteps:
                if log_progress:
                    log.success("Learn successful", f"LEARN-{self.name}")
                break
            episode += 1

            training_rewards.append(total_training_rewards)
            epsilons.append(eps)

        self.is_trained = True

        if return_history:
            return training_rewards, epsilons


class DQNLearner(DQN, Learner):
    """
    A class to implement the DQN algorithm for MMC queues. The class uses :stable_baselines3: for DQN part
    """
    def __init__(self, *args, **kwargs):
        super(DQNLearner, self).__init__(*args, **kwargs)
        self.mmc_env = kwargs["env"]
        self.name = "DQN"
        self.is_trained = False

    def learn(self, *args, **kwargs):
        res = super(DQNLearner, self).learn(*args, **kwargs)
        self.is_trained = True
        return res


class A2CLearner(A2C, Learner):
    """
    A class to implement the A2C algorithm for MMC queues. The class uses :stable_baselines3: for DQN part
    """
    def __init__(self, *args, **kwargs):
        super(A2CLearner, self).__init__(*args, **kwargs)
        self.mmc_env = kwargs["env"]
        self.name = "A2C"
        self.is_trained = False

    def learn(self, *args, **kwargs):
        res = super(A2CLearner, self).learn(*args, **kwargs)
        self.is_trained = True
        return res
