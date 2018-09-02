''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy
import time
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent



class QLearningAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, name="Q-learning", default_q = 0.0, alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform", anneal=False, printer=None, zero_at_terminal=False, random_opt='normal', count_visits=False):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
        '''
        name_ext = "-" + explore if explore != "uniform" else ""
        Agent.__init__(self, name=name + name_ext, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha, self.alpha_init = alpha, alpha
        if isinstance(epsilon, float):
            self.epsilon, self.epsilon_init, self.epsilon_final = epsilon, epsilon, epsilon
            self.epsilon_discount_per_action = 0.0
        elif isinstance(epsilon, tuple) and len(epsilon) == 3:
            self.epsilon = epsilon[0]
            self.epsilon_init = epsilon[0]
            self.epsilon_final = epsilon[1]
            self.epsilon_discount_per_action = epsilon[2]
        else:
            raise NotImplementedError
        self.step_number = 0
        self.anneal = anneal
        self.printer = printer
        self.default_q = default_q
        self.explore = explore

        # Q Function:
        self.random_opt = random_opt
        if isinstance(default_q, tuple):
            if random_opt == 'normal':
                self.q_func = defaultdict(lambda : defaultdict(lambda: numpy.random.normal(default_q[0], default_q[1])))
            elif random_opt == 'uniform':
                self.q_func = defaultdict(lambda : defaultdict(lambda: numpy.random.uniform(default_q[0], default_q[1])))
            else:
                raise NotImplementedError('{} random option is not implemented.'.format(random_opt))
        else:
            self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))

        if count_visits:
            self.v_counts = defaultdict(lambda : 0)
        else:
            self.v_counts = None

        # Key: state
        # Val: dict
            #   Key: action
            #   Val: q-value


    def reset_epsilon(self, epsilon):
        if isinstance(epsilon, float):
            self.epsilon, self.epsilon_init, self.epsilon_final = epsilon, epsilon, epsilon
            self.epsilon_discount_per_action = 0.0
        elif isinstance(epsilon, tuple) and len(epsilon) == 3:
            self.epsilon = epsilon[0]
            self.epsilon_init = epsilon[0]
            self.epsilon_final = epsilon[1]
            self.epsilon_discount_per_action = epsilon[2]
    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, learning=True):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
        	(str)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''

        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)
        
        if self.explore == "softmax":
            # Softmax exploration
            action = self.soft_max_policy(state)
        else:
            # Uniform exploration
            action = self.epsilon_greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if learning and self.anneal:
            self._anneal()
        if learning :
            self.epsilon -= self.epsilon_discount_per_action

        return action

    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.

        Returns:
            (str): action.
        '''
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, just return.
        if state is None:
            self.prev_state = next_state
            return

        if self.v_counts is not None:
            self.v_counts[state] = self.v_counts[state] + 1

        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state)
        prev_q_val = self.get_q_value(state, action)
        self.q_func[state][action] = (1. - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma*max_q_curr_state)

    def _anneal(self):
        # Taken from "Note on learning rate schedules for stochastic optimization, by Darken and Moody (Yale)":
        self.alpha = self.alpha_init / (1.0 +  (self.step_number / 200.0)*(self.episode_number + 1) / 2000.0 )
        self.epsilon = self.epsilon_init / (1.0 + (self.step_number / 200.0)*(self.episode_number + 1) / 2000.0 )

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def get_v_counts(self, state):
        return self.v_counts[state]

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        if state.is_terminal():
            return 0.0
        return self.q_func[state][action]

    def get_action_distr(self, state, beta=0.2):
        '''
        Args:
            state (State)
            beta (float): Softmax temperature parameter.

        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        all_q_vals = []
        for i in range(len(self.actions)):
            action = self.actions[i]
            all_q_vals.append(self.get_q_value(state, action))

        # Softmax distribution.
        total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]

        return softmax

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        if isinstance(self.default_q, tuple):
            if self.random_opt == 'normal':
                self.q_func = defaultdict(lambda : defaultdict(lambda: numpy.random.normal(self.default_q[0], self.default_q[1])))
            elif self.random_opt == 'uniform':
                self.q_func = defaultdict(lambda : defaultdict(lambda: numpy.random.uniform(self.default_q[0], self.default_q[1])))
            else:
                raise NotImplementedError('{} random option is not implemented.'.format(self.random_opt))
        else:
            self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
        if self.v_counts is not None:
            self.v_counts = defaultdict(lambda : 0)
        Agent.reset(self)

    def end_of_episode(self, testing = False):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        if self.anneal:
            self._anneal()
        if self.printer and not testing:
            self.printer(self.episode_number, self)

        Agent.end_of_episode(self)

