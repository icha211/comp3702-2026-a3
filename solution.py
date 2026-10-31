import sys
import time

from game_env import GameEnv
from game_state import GameState
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 07/10/2026
"""


class Solver:

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        #
        # TODO: Define any class instance variables you require (e.g. a dictionary mapping (state, action) to Q-value,
        #  or the weights of your linear Q-function) here.
        #
        pass

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Training rules (apply to both algorithms) ====================================================================
    #
    # Before training starts, the tester and autograder give self.game_env a training budget: a maximum number of
    # perform_action calls, and a lowest cumulative training reward. Once either runs out, perform_action raises
    # TrainingBudgetExhausted, which ends training - do not catch it. self.game_env.training_steps_used,
    # self.game_env.training_step_budget and self.game_env.training_reward_total let you plan around the budget.
    #
    # During training, perform_action must be called on either the initial state (self.game_env.get_init_state(), which
    # starts a new episode) or the state returned by your previous perform_action call.
    #
    # Your agent must learn from experience. The movement probabilities, action costs and penalties are hidden: each
    # testcase draws its own from its seed, and the environment keeps them private. Your agent may use the map (e.g.
    # distances, or what lies in an action's direction) and the rewards it receives, but must not read the
    # environment's private parameters or call its private methods, or plan with a transition model (e.g. value
    # iteration).
    #
    # Only some actions are valid on each tile (self.game_env.get_valid_actions(state)). An invalid action leaves the
    # state unchanged and still costs its action cost.

    # === Tabular Q-learning ===========================================================================================

    def ql_initialise(self):
        """
        Initialise any variables required before the start of tabular Q-learning.
        """
        #
        # TODO: Implement any initialisation for tabular Q-learning (e.g. creating your Q-table) here. This method is
        #  not timed, but it must not call perform_action (training belongs in ql_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.q_table = {}
        for mask in range(1 << self.game_env.n_crystals):
            self.q_table[self.mask_to_status(mask)] = [[0.0] * 12 for _ in range(self.n_cells)]
        self.steps = 0

    def ql_train_episode(self):
        """
        Run one episode of tabular Q-learning.
        """
        #
        # TODO: Implement a single training episode here: start from self.game_env.get_init_state() and repeatedly
        #  choose an action (e.g. epsilon-greedy), call self.game_env.perform_action and update your Q-values, until
        #  the episode ends (a terminal state, or a step limit such as self.game_env.max_episode_steps).
        #
        # The tester and autograder call this method repeatedly until the training budget is exhausted, and check how
        # well your greedy policy performs at regular checkpoints in between.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        env = self.game_env
        perform_action = env.perform_action
        actions = GameEnv.ACTIONS
        q_table = self.q_table
        cell_valid = self.cell_valid
        cell_lava = self.cell_lava
        cell_launch = self.cell_launch
        enough = self.enough_crystals
        n_cols = self.n_cols
        gamma = env.gamma
        alpha = self.ql_alpha + (self.ql_alpha_end - self.ql_alpha) * min(1.0, self.steps / self.ql_budget /
                                                                       self.ql_alpha_frac)
        rand = random.random
        choice = random.choice
        state = env.get_init_state()
        for _ in range(env.max_episode_steps):
            frac = self.steps / self.ql_budget
            epsilon = self.ql_eps_start + (self.ql_eps_end - self.ql_eps_start) * min(1.0, frac / self.ql_eps_frac)
            idx = state.row * n_cols + state.col
            cs = state.crystal_status
            q = q_table[cs][idx]
            valid = cell_valid[idx]
            if rand() < epsilon:
                a = choice(valid)
            else:
                a = valid[0]
                best = q[a]
                for i in valid:
                    if q[i] > best:
                        best = q[i]
                        a = i
            next_state, reward, _ = perform_action(state, actions[a])
            self.steps += 1
            nidx = next_state.row * n_cols + next_state.col
            ncs = next_state.crystal_status
            if cell_lava[nidx] or (cell_launch[nidx] and enough[ncs]):
                q[a] += alpha * (reward - q[a])
                break
            nq = q_table[ncs][nidx]
            nbest = max([nq[i] for i in cell_valid[nidx]])
            q[a] += alpha * (reward + gamma * nbest - q[a])
            state = next_state

    def ql_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned Q-values.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned Q-values) here.
        #  This method is used to evaluate your agent (including during training), so it should not explore or update
        #  your Q-values.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        idx = state.row * self.n_cols + state.col
        q = self.q_table[state.crystal_status][idx]
        valid = self.cell_valid[idx]
        best = valid[0]
        for i in valid:
            if q[i] > q[best]:
                best = i
        return GameEnv.ACTIONS[best]

    # === Feature-based Q-learning =====================================================================================

    def fq_initialise(self):
        """
        Initialise any variables required before the start of feature-based Q-learning.
        """
        #
        # TODO: Implement any initialisation for feature-based Q-learning (e.g. initial weights, or pre-computing
        #  information about the map that your features use) here. This method is not timed, so it is the place for
        #  precomputation, but it must not call perform_action (training belongs in fq_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_get_features(self, state: GameState, action):
        """
        Return the feature vector of the given (state, action) pair, so that Q(s, a) = w . features(s, a).
        :param state: the current state
        :param action: an element of GameEnv.ACTIONS
        :return: a list (or 1-D numpy array) of at most 256 numbers, the same length for every (state, action) pair
        """
        #
        # TODO: Implement your feature design here. Features that describe the situation (e.g. distances, risks) rather
        #  than identify the exact state are what let your agent act well in states it never visited during training.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_train_episode(self):
        """
        Run one episode of feature-based Q-learning (Q-learning with linear function approximation).
        """
        #
        # TODO: Implement a single training episode here, as for ql_train_episode, but updating the weights of your
        #  linear Q-function instead of a Q-table.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned weights.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned weights) here.
        #  This method is also evaluated from start positions your agent never trained from.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #
