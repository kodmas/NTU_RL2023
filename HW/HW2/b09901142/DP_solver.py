import numpy as np
import json
from collections import defaultdict

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)

    def get_all_state_values(self) -> np.array:
        return self.values


class MonteCarloPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float):
        """Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.returns = defaultdict(list)
        self.num_episodes = 20000

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        for i in range(self.num_episodes):
            current_state = self.grid_world.reset()
            episode = []
            while self.grid_world.check():
                next_state, reward, done = self.grid_world.step()
                episode.append((current_state, reward))
                if done:
                    break
                current_state = next_state
            G = 0
            for t in reversed(range(len(episode))):
                state, reward = episode[t]
                G = self.discount_factor * G + reward
                if state not in [x[0] for x in episode[:t]]:
                    self.returns[state].append(G)
                    self.values[state] = np.mean(self.returns[state])
            # print(self.values )

class TDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.num_episodes = 20000
        self.values = np.zeros(self.state_space)

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm
        for i in range(self.num_episodes):
            current_state = self.grid_world.reset()
            while self.grid_world.check():
                # print(self.grid_world.get_step_count())
                next_state, reward, done = self.grid_world.step()
                td_error = reward + self.discount_factor * self.values[next_state] - self.values[current_state]
                if done:
                    td_error = reward - self.values[current_state]
                self.values[current_state] += self.lr * td_error
                if done:
                    break
                current_state = next_state
            # print(self.values)
                



class NstepTDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, num_step: int):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate
        self.n      = num_step
        self.num_episodes = 10000
    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with N-step TD Algorithm
        for i in range(self.num_episodes):
            current_state = self.grid_world.reset()
            T = np.inf
            t = 0
            states = [current_state]
            rewards = [0]
            while self.grid_world.check():
                if t < T:
                    next_state, reward, done = self.grid_world.step()
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += self.discount_factor ** (i - tau - 1) * rewards[i]
                    if tau + self.n < T:
                        G += self.discount_factor ** self.n * self.values[states[tau + self.n]]
                    self.values[states[tau]] += self.lr * (G - self.values[states[tau]])
                if tau == T - 1:
                    break
                t += 1
                
