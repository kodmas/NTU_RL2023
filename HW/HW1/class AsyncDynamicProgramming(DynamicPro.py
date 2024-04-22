class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        """priority queue"""
        super().__init__(grid_world, discount_factor)
        """declare a list named model"""
        self.model = np.zeros((grid_world.get_state_space(), grid_world.get_action_space(), 3), dtype=int)
        #print(self.model)
    def q_val(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        next_state, reward, done = self.model[state, action]
        if done:
            return reward
        else:
            q = reward + self.discount_factor * self.values[next_state]           
            return q
        raise NotImplementedError
        
    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy_stable = True
        for s in range(self.grid_world.get_state_space()):
            old_action = self.policy[s]
          
            max_q = -999999999
            best_a = 0
            for a in range(self.grid_world.get_action_space()):
                if self.q_val(s, a) > max_q:
                    max_q = self.q_val(s, a)
                    best_a = a        
            self.policy[s] = best_a
            if old_action != self.policy[s]:
                policy_stable = False
        return self.policy, policy_stable
    
    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        """perform in-place value iteration"""
        #input s,a output next_state, reward, done
        
        self.policy = np.zeros(self.grid_world.get_state_space(), dtype=int) 
        self.values = np.zeros(self.grid_world.get_state_space())   
        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(s, a)
                self.model[s, a] = [next_state, reward, done]

        while True:
            delta = 0
            for s in range(self.grid_world.get_state_space()):
                v = self.values[s]
                self.values[s] = max(self.q_val(s, a) for a in range(self.grid_world.get_action_space()))
                delta = max(delta, abs(v - self.values[s]))
            if delta < self.threshold:  
                break
        self.policy = self.policy_improvement()[0]
        print(self.values)
        print(self.policy)

class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        self.n_planning_steps = 10
        self.model = {}
        self.pq = PriorityQueue()
        self.predecessors = defaultdict(set)
        self.alpha = 0.5
        self.q_values = np.zeros((self.grid_world.get_state_space(),self.grid_world.get_action_space()))

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
    
    def _update_predecessors(self, state, action, next_state):
        # add predecessors as a set of (state, action) tuples
        self.predecessors[next_state].add((state, action))

    def get_best_value(self,state):
        action_values = [self.get_q_value(state,action) for action in range(self.grid_world.get_action_space())]
        best_action = np.argmax(action_values)
        return self.get_q_value(state,best_action)
    
    def update(self, state, action, reward, next_state):
        """ Execute the Q-learning off-policy algorithm in Section 6.5 with
        Prioritized Sweeping model update/planning in Section 8.4 """

        # update model (Sec 8.4 - line (d))
        # model assumes deterministic environment
        self.model[(state, action)] = (reward, next_state)
        # keep track of predecessors for the pq loop below
        self._update_predecessors(state, action, next_state)

        # compute q value proposed update and update priority queue (Sec 8.4 - line (e-f))
        proposed_update = reward + self.discount_factor * self.get_state_value(next_state) - self.get_q_value(state, action)
        if abs(proposed_update) > self.threshold:
            self.pq.put([(state,action), -abs(proposed_update)])
        # loop over n_planning steps while pq is not empty (Sec 8.4 - line(g)
        for i in range(self.n_planning_steps):
            if self.pq.empty():
                break

            # pop best update from queue and transition from model
            state, action = self.pq.get()[0]
            reward, next_state = self.model[(state, action)]

            # update q values for this state-action pair
            new_value = self.q_values[state,action] + self.alpha * (reward + self.discount_factor * self.get_best_value(next_state) - self.get_q_value(state,action))
            self.q_values[state,action] = new_value

            # loop for all S', A' predicted to lead to the above state
            for s, a in self.predecessors[state]:
                # get predicted reward from the predecessor leading to `state`
                r, _ = self.model[(s, a)]
                # calculate the proposed update to (s,a)
                proposed_update = r + self.discount_factor * self.get_state_value(state) - self.get_q_value(s, a)
                # add to priority queue if greater than min threshold
                if abs(proposed_update) > self.threshold:
                    self.pq.put([(state,action), -abs(proposed_update)])


    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        state = 0
        for i in range(100):
            action_values = [self.get_q_value(state,action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(action_values)
            next_state, reward, _ = self.grid_world.step(state,best_action)
            self.update(state,best_action,reward,next_state)