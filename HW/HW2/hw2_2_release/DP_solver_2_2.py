import numpy as np
from collections import deque
from gridworld import GridWorld
import matplotlib.pyplot as plt

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
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space 
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace)-> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        G = 0
        for t in reversed(range(len(state_trace)-1)):
            state, action, reward = state_trace[t], action_trace[t], reward_trace[t]
            G = self.discount_factor * G + reward
        loss = G - self.q_values[state][action]
        self.q_values[state][action] += self.lr * loss
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        for s in range(self.state_space):
            best_action = np.argmax(self.q_values[s])
            for a in range(self.action_space):
                if a == best_action:
                    self.policy[s][a] = 1 - self.epsilon + self.epsilon / self.action_space
                else:
                    self.policy[s][a] = self.epsilon / self.action_space            

    def run(self, max_episode=1000)-> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # if np.random.rand() < self.epsilon:
            #     action = np.random.choice(self.action_space)
            # else:
            #     action = np.argmax(self.q_values[current_state])
            if iter_episode < 1000:
                action = np.random.choice(self.action_space)
            else:
                action = np.random.choice(self.action_space, p=self.policy[current_state])
            next_state, reward, done = self.grid_world.step(action)

            state_trace.append(current_state)
            action_trace.append(action)
            reward_trace.append(reward)
            current_state = next_state
            # print(current_state, action, reward, done)
            if done:
                self.policy_evaluation(state_trace, action_trace, reward_trace)
                self.policy_improvement()
                iter_episode += 1

                state_trace   = [current_state]
                action_trace  = []
                reward_trace  = []
            current_state = next_state
    
class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done)-> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        td_target = r + self.discount_factor * self.q_values[s2][a2] * (1 - is_done)
        actual = self.q_values[s][a]
        td_error = td_target - actual
        if is_done:
            self.q_values[s][a] += self.lr * (r - actual)
        else:
            self.q_values[s][a] += self.lr * td_error
        # best_action = np.argmax(self.q_values[s])
        # for action in range(self.action_space):
        #     if action == best_action:
        #         self.policy[s][action] = 1 - self.epsilon + self.epsilon / self.action_space
        #     else:
        #         self.policy[s][action] = self.epsilon / self.action_space

    def run(self, max_episode=1000)-> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                action = np.argmax(self.q_values[current_state])
            
            next_state, reward, is_done = self.grid_world.step(action)
            if np.random.rand() < self.epsilon:
                next_action = np.random.choice(self.action_space)
            else:
                next_action = np.argmax(self.q_values[next_state])
            self.policy_eval_improve(current_state, action, reward, next_state, next_action, is_done)
            current_state = next_state
            action = next_action    
            if is_done:
                iter_episode += 1
                for s in range(self.state_space):
                    best_action = np.argmax(self.q_values[s])
                    for action in range(self.action_space):
                        if action == best_action:
                            self.policy[s][action] = 1 - self.epsilon + self.epsilon / self.action_space
                        else:
                            self.policy[s][action] = self.epsilon / self.action_space
            # if iter_episode % 10000 == 0:
            #     print(iter_episode)


class Q_Learning(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        # print(self.buffer)
        batch = np.random.choice(len(self.buffer), self.sample_batch_size)
        # print(batch)
        return batch

    def policy_eval_improve(self, s, a, r, s2, is_done)-> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        td_target = r + self.discount_factor * np.max(self.q_values[s2]) * (1 - is_done)
        actual = self.q_values[s][a]
        td_error = td_target - actual
        self.q_values[s][a] += self.lr * td_error

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        transition_count = 0

        while iter_episode < max_episode:

            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                action = np.argmax(self.q_values[current_state])
            
            next_state, reward, is_done = self.grid_world.step(action)
            self.add_buffer(current_state, action, reward, next_state, is_done)
            self.policy_eval_improve(current_state, action, reward, next_state, is_done)
            transition_count += 1
            if transition_count % self.update_frequency == 0:
                batch = self.sample_batch()
                for choice in batch:
                    self.policy_eval_improve(self.buffer[choice][0], self.buffer[choice][1], self.buffer[choice][2], self.buffer[choice][3], self.buffer[choice][4])
            
            current_state = next_state

            if is_done:
                iter_episode += 1
                for s in range(self.state_space):
                    best_action = np.argmax(self.q_values[s])
                    for action in range(self.action_space):
                        if action == best_action:
                            self.policy[s][action] = 1 - self.epsilon + self.epsilon / self.action_space
                        else:
                            self.policy[s][action] = self.epsilon / self.action_space
            # if iter_episode % 1000 == 0:
            #     print(iter_episode)
            

