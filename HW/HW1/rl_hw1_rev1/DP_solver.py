import numpy as np

from gridworld import GridWorld

import time
from collections import defaultdict
from queue import PriorityQueue

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
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done = self.grid_world.step(state, action)
        if done:
            q_value = reward 
        else:
            q_value = reward + self.discount_factor * self.values[next_state] 
        return q_value


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        v = 0
        # TODO: Get the value for a state by calculating the q-values
        for action in range(self.grid_world.get_action_space()):
            q_value = self.get_q_value(state,action)
            v += q_value * self.policy[state,action]
        return v

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        delta = 0
        V_new = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            next_value = self.get_state_value(state)
            prev_value = self.values[state]
            V_new[state] = next_value 
            delta = max(delta, abs(prev_value - next_value))
        self.values = V_new
        self.delta = delta


    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            self.evaluate()
            if self.delta < self.threshold:
                break
            

class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # Get the value for a state by calculating the q-values
        next_state, reward, done = self.grid_world.step(state, self.policy[state])
        if done:
            q_value = reward 
        else:
            q_value = reward + self.discount_factor * self.values[next_state] 
        return q_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        delta = 0
        V_new = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            next_value = self.get_state_value(state)
            prev_value = self.values[state]
            V_new[state] = next_value
            delta = max(delta, abs(prev_value - next_value))
        self.values = V_new
        self.delta = delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]
            action_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            self.policy[state] = best_action
        return policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        while True:
            while True:
                self.policy_evaluation()
                if self.delta < self.threshold:
                    break
            # print("finish evaluation, ",self.grid_world.get_step_count())
            policy_stable = self.policy_improvement()
            if policy_stable:
                break
        return self.values, self.policy
        

        


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        next_state, reward, done = self.grid_world.step(state, self.policy[state])
        if done:
            q_value = reward 
        else:
            q_value = reward + self.discount_factor * self.values[next_state] 
        return q_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        delta = 0
        V_new = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            action_values = [self.get_q_value(state,action) for action in range(self.grid_world.get_action_space())]
            best_action_value = np.max(action_values)
            delta = max(delta, abs(best_action_value - self.values[state]))
            V_new[state] = best_action_value
        self.values = V_new
        self.delta = delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            action_values = [self.get_q_value(state,action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(action_values)
            self.policy[state] = best_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            self.policy_evaluation()
            if self.delta < self.threshold:
                break
        self.policy_improvement()
        return self.policy
        
class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        """priority queue"""
        super().__init__(grid_world, discount_factor)
        self.Q = np.zeros((grid_world.get_state_space(), grid_world.get_action_space()))
        self.next_step_info = np.zeros((grid_world.get_state_space(), grid_world.get_action_space(), 3),dtype=np.int64)
        
    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            action_values = [self.Q[state,action] for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(action_values)
            self.policy[state] = best_action
        return self.policy
    
    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(state, action)
                self.next_step_info[state,action] = [next_state,reward,done]
                if done:
                    q_value = reward 
                else:
                    q_value = reward + self.discount_factor * self.values[next_state] 
                self.Q[state, action] = q_value

        while True:
            delta = 0
            for state in range(self.grid_world.get_state_space()):
                v = self.values[state]
                self.values[state] = max(self.Q[state, action] for action in range(self.grid_world.get_action_space()))
                delta = max(delta, abs(v - self.values[state]))
            if delta < self.threshold:  
                break
            for state in range(self.grid_world.get_state_space()):
                for action in range(self.grid_world.get_action_space()):
                    next_state, reward, done = self.next_step_info[state, action][0],self.next_step_info[state, action][1],self.next_step_info[state, action][2]
                    if done:
                        q_value = reward
                    else:
                        q_value = reward + self.discount_factor * self.values[next_state] 
                    self.Q[state, action] = q_value
                        
        self.policy = self.policy_improvement()
        return self.policy

            

                