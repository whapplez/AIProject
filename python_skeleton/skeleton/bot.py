'''
This file contains the base class that you should implement for your pokerbot.
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque

class Bot():
    '''
    The base class for a pokerbot.
    '''
    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        tf.random.set_seed(42)
        np.random.seed(42)

        input_shape = [35 + 6] # == env.observation_space.shape
        n_outputs = 10 # == fold, check, call, min raise, 1/4 pot, 1/2 pot, 3/4 pot, pot, 1.5 pot, all in

        model = keras.models.Sequential([
            keras.layers.Dense(32, activation="elu", input_shape=input_shape),
            keras.layers.Dense(32, activation="elu"),
            keras.layers.Dense(n_outputs)
        ])

        replay_memory = deque(maxlen=2000)

    def epsilon_greedy_policy(state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        else:
            Q_values = model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        raise NotImplementedError('handle_new_round')

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        raise NotImplementedError('handle_round_over')
    
    def getCardRank(self, s):
        d = {
            'T': 10,
            'J': 11,
            'Q': 12,
            'K': 13,
            'A': 14,
        }
        return d.get(s[0],int(s[0]))
    def getCardSuit(self, s):
        return int(s[1 == 's']), int(s[1 == 'h']), int(s[1 == 'c']), int(s[1 == 'd'])

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        card1 = my_cards[1]
        
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        input = [getCardRank(my_cards[0]), getCardRank(my_cards[1])]

        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        return CallAction()


        raise NotImplementedError('get_action')
