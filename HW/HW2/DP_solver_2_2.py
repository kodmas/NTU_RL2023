import numpy as np
from collections import deque
from gridworld import GridWorld
from collections import defaultdict
import random
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
        self.returns = defaultdict(list)


    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        g = 0
        for i in range(len(state_trace)-1):
            g = 0
            for j in range(i, len(state_trace)):
                g += self.discount_factor ** (j-i) * reward_trace[j]
            self.q_values[state_trace[i], action_trace[i]] += self.lr * (g - self.q_values[state_trace[i], action_trace[i]])


      


    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        for s_i in range(self.state_space):
            best_action = self.q_values[s_i].argmax()
            for a_i in range(self.action_space):
                if a_i == best_action:
                    self.policy[s_i, a_i] = 1 - self.epsilon + self.epsilon / self.action_space
                else:
                    self.policy[s_i, a_i] = self.epsilon / self.action_space
        #print("Policy: ", self.policy)
        
            


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        #print(self.epsilon)
        #max_episode = 1000
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = []
        action_trace  = []
        reward_trace  = []
        self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
        #plt_x = []
        #plt_y = []
        
        

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            #print("Policy: ", self.policy)
            action = np.random.choice(self.action_space, p=self.policy[current_state])
            next_state, reward, is_done = self.grid_world.step(action)
            state_trace.append(current_state)
            action_trace.append(action)
            reward_trace.append(reward)
            
            if is_done:
                print("Episode: ", iter_episode)
                #print("State trace: ", state_trace, len(state_trace))
                #print("Action trace: ", action_trace)
                #print("Reward trace: ", reward_trace)
                #print("Q values: ", self.q_values)
                self.policy_evaluation(state_trace, action_trace, reward_trace)
                self.policy_improvement()

                ###
                
                #if iter_episode % 10 == 0:
                    #plt_x.append(iter_episode)
                    #plt_y.append(self.get_max_state_values().mean())
                    
                ###
                
                
                state_trace  = []
                action_trace = []
                reward_trace = []
                iter_episode += 1

            
            current_state = next_state
            #raise NotImplementedError

        #plt.plot(plt_x, plt_y, color='b', linestyle='-', linewidth=1)
        #plt.title("MC")
        #plt.show()



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

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        if is_done:
            self.q_values[s, a] += self.lr * (r - self.q_values[s, a])
        else:
            self.q_values[s, a] += self.lr * (r + self.discount_factor * self.q_values[s2, a2] - self.q_values[s, a])

      
    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        #max_episode = 1
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        next_a = 0
        self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
        action = np.random.choice(self.action_space, p=self.policy[current_state])
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            next_state, reward, is_done = self.grid_world.step(action)
            next_a = np.random.choice(self.action_space, p=self.policy[next_state])
            self.policy_eval_improve(current_state, action, reward, next_state, next_a, is_done)
            current_state = next_state
            action = next_a
            if is_done:
                iter_episode += 1
                #print("Episode: ", iter_episode)
                for s_i in range(self.state_space):
                    best_action = self.q_values[s_i].argmax()
                    for a_i in range(self.action_space):
                        if a_i == best_action:
                            self.policy[s_i, a_i] = 1 - self.epsilon + self.epsilon / self.action_space
                        else:
                            self.policy[s_i, a_i] = self.epsilon / self.action_space
                
            #raise NotImplementedError


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
        k = random.sample(self.buffer, self.sample_batch_size)
        
        #print("Sample batch: ", k)
        return k
            
        

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        #print(s, a, r, s2, is_done)
        if is_done:
            self.q_values[s, a] += self.lr * (r - self.q_values[s, a])
        else:
            self.q_values[s, a] += self.lr * (r + self.discount_factor * self.q_values[s2].max() - self.q_values[s, a])
 

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
        next_a = 0
        self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
        

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            action = np.random.choice(self.action_space, p=self.policy[current_state])
            next_state, reward, is_done = self.grid_world.step(action)
            self.add_buffer(current_state, action, reward, next_state, is_done)
            transition_count += 1
            #print("Transition count: ", transition_count)

            k = defaultdict(list)
            if transition_count % self.update_frequency == 0 and transition_count > self.sample_batch_size:
                k = self.sample_batch()
            for i in range(len(k)):
                self.policy_eval_improve(k[i][0], k[i][1], k[i][2], k[i][3], k[i][4])

            current_state = next_state

            if is_done:
                iter_episode += 1
                #print("Episode: ", iter_episode)
                for s_i in range(self.state_space):
                    best_action = self.q_values[s_i].argmax()
                    for a_i in range(self.action_space):
                        if a_i == best_action:
                            self.policy[s_i, a_i] = 1 - self.epsilon + self.epsilon / self.action_space
                        else:
                            self.policy[s_i, a_i] = self.epsilon / self.action_space
        
            
                    
            
            





