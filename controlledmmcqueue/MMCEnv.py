from copy import deepcopy
from itertools import product, combinations
from typing import List, Union, Tuple

import gym
import numpy as np


class MMCEnv(gym.Env):
    """
    Implements controlled MMC queues as a gym environment.
    """

    def __init__(self, arr_rate: float, srv_rates: List[float], queue_cap: int,
                 max_events: int, punish_reward=-9999, scale_rewards=False):
        """

        :param arr_rate: the arrival rate to the system
        :type arr_rate: float
        :param srv_rates: the service rate of the servers. The number of servers is determined based on the elements
        in :srv_rates:
        :type srv_rates: list
        :param queue_cap: the maximum capacity of the queue
        :param max_events: number of events to use as the termination condition of each episode, used in method `step`
        :param punish_reward: the reward to use for infeasible actions. Should be a negative number
        :param scale_rewards: if True, the rewards will be scaled based on the maximum number of people possible in the
        system, i.e., `(:n_servers + :queue_cap)`
        """
        super(MMCEnv, self).__init__()
        self.arr_rate = arr_rate
        self.srv_rates = srv_rates
        self.n_servers = len(srv_rates)
        self.queue_cap = queue_cap

        self.all_states = list(
            map(list, list(product(range(queue_cap + 1), *[range(2)] * self.n_servers)))
        )

        self.action_space = gym.spaces.Discrete(2 ** self.n_servers)
        self.observation_space = gym.spaces.MultiDiscrete((self.queue_cap + 1, *[2] * self.n_servers))

        self.curr_state = [0] * (self.n_servers + 1)
        self.n_arrivals = 0
        self.n_departures = 0

        self.punish_reward = punish_reward

        self.max_events = max_events
        self.n_events = 0
        self.tot_tot_in_system = 0
        self.scale_rewards = scale_rewards

    def get_possible_actions(self, state, return_encoded):
        """

        :param: state the state based on which the possible actions are calculated
        :param: return_encoded the encoded takes the action space as a binary array of length n_servers. For instance in
        an M/M/2 system,
        - action = 0 ("0b00") refers to taking no action
        - action = 1 ("0b01") refers to navigating the first customer to server 1
        - action = 2 ("0b10") refers to navigating the first customer to server 2
        - action = 3 ("0b11") refers to navigating the first two customers to server 1 and 2 simultaneously
        If return_encoded is True, the converted decimal representation is returned, otherwise a list of
        tuples representing the servers to use. e.g., [(), (2, 3)] means that either do nothing or route to the first
        server and the second server (simultaneously).
        :return: the list of possible actions
        """
        possible_actions = []
        possible_encoded = []
        n_free_servers = self.n_servers - sum(state[1:])
        max_assignable = min(state[0], n_free_servers)
        for m in range(0, max_assignable + 1):
            free_servers = [i + 1 for i in range(self.n_servers) if state[i + 1] == 0]
            this_selection = list(combinations(free_servers, m))
            possible_actions += this_selection

            if return_encoded:
                encoded = [sum(2 ** (i - 1) for i in sel) for sel in this_selection]
                possible_encoded += encoded

        if return_encoded:
            return possible_encoded
        return possible_actions

    def decode_int_action(self, action: int):
        """decodes an action given in `int` format into `tuple` format 
        
        :param action: the action in int format
        :return: action in tuple format (including the servers to navigate the next customer(s) to)
        :raises ValueError: if the action is not in the allowed range, [0, 2^n_servers - 1]
        """
        if not (0 <= action <= 2 ** self.n_servers - 1):
            raise ValueError(f"Invalid action {action} with n_servers = {self.n_servers}")

        return tuple(np.nonzero(list(map(int, list(bin(action)[2:].zfill(self.n_servers)[-1::-1]))))[0] + 1)

    def action_transition(self, state, action: Union[Tuple, int]):
        """Calculates the next state based on a current state and an action
        
        :param state: the current state  
        :param action: the action to be taken. could be in int format or tuple format
        :return: the next state
        """
        next_state = deepcopy(state)
        return_encoded = True if not isinstance(action, tuple) else False
        possible_actions = self.get_possible_actions(state, return_encoded=return_encoded)
        if action in possible_actions:
            if not isinstance(action, tuple):
                action = self.decode_int_action(action)
            for server in action:
                next_state[server] += 1
                next_state[0] -= 1
        return next_state

    def event_transition(self, state) -> Tuple[List, str]:
        """ calculates the next state based on the transition probabilities

        :param state: the current state
        :return next_state: the resulting state based on the event transition
        """
        new_state = deepcopy(state)
        p_arr = self.arr_rate / (sum(self.srv_rates) + self.arr_rate) * (state[0] < self.queue_cap)
        p_serves = [self.srv_rates[i] / (sum(self.srv_rates) + self.arr_rate) * (state[i + 1] > 0)
                    for i in range(self.n_servers)]
        p_fake = max(0, 1 - p_arr - sum(p_serves))  # to avoid unwanted negative issues
        probs = [p_arr, *p_serves, p_fake]
        next_event: str = np.random.choice(["arr", *(f"srv_{i}" for i in range(self.n_servers)), "fake"], p=probs)
        if next_event == "arr":
            new_state[0] += 1
        if next_event.startswith("srv"):
            server = int(next_event.split("_")[1])
            new_state[server + 1] -= 1
        return new_state, next_event

    def get_possible_event_transitions(self, state):
        """Calculates all possible events and their probabilities based on a given state

        :param state: the current state
        :return poss_res_state, poss_probs: the possible resulting states and their probabilities each stored as a list
        """
        poss_res_states = []
        poss_probs = []
        if state[0] < self.queue_cap:
            poss_res_states.append([state[0] + 1] + state[1:])
            poss_probs.append(self.arr_rate / (self.arr_rate + sum(self.srv_rates)))
        for server in range(self.n_servers):
            if state[server + 1] > 0:
                temp_state = deepcopy(state)
                temp_state[server + 1] -= 1
                poss_res_states.append(temp_state)
                poss_probs.append(self.srv_rates[server] / (self.arr_rate + sum(self.srv_rates)))
        poss_res_states.append(state)
        poss_probs.append(1 - sum(poss_probs))
        return poss_res_states, poss_probs

    def reset(self):
        self.n_events = 0
        self.tot_tot_in_system = 0
        self.curr_state = [0] * (self.n_servers + 1)
        self.n_arrivals = 0
        self.n_departures = 0
        return deepcopy(self.curr_state)

    def step(self, action: Union[tuple, int]):
        """

        :param: action is a tuple of the form (server1, server2, ..., server n) representing the simultaneous assignment
        of the first n elements to the servers in the tuple. Empty tuple represents do nothing
        :return:
        """
        self.n_events += 1

        # state_before = deepcopy(self.curr_state)
        action_result = self.action_transition(self.curr_state, action)
        event_result, next_event = self.event_transition(action_result)

        if next_event.startswith("arr"):
            self.n_arrivals += 1
        if next_event.startswith("srv"):
            self.n_departures += 1

        # n_left = sum(np.maximum(np.array(state_before[1:]) - np.array(event_result[1:]), [0] * len(event_result[1:])))

        if action not in self.get_possible_actions(self.curr_state, return_encoded=True):
            infeasible = True
        else:
            infeasible = False

        self.curr_state = event_result

        obs = event_result
        # rew = n_left
        if self.scale_rewards:
            rew = -sum(self.curr_state) / (self.n_servers + self.queue_cap)
        else:
            rew = -sum(self.curr_state)

        done = self.n_events >= self.max_events

        if infeasible:
            rew = self.punish_reward
        info = dict(infeasible=infeasible)

        return obs, rew, done, info

    def render(self, mode="human"):
        print(self.curr_state)
