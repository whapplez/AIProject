'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque


class Player(Bot):
    '''
    A pokerbot.
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

        self.input_shape = [50] # == env.observation_space.shape
        self.n_outputs = 4 # == fold, check, call, min raise, 1/4 pot, 1/2 pot, 3/4 pot, pot, 1.5 pot, all in

        self.model = keras.models.Sequential([
            keras.layers.Dense(32, activation="relu", input_shape=self.input_shape),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(self.n_outputs)
        ])
        self.round_state = None
        self.terminal_state = None
        self.prev_state = None
        self.state = None
        self.iter = 0
        self.rewards = 0
        self.batch_size = 32
        self.discount_rate = 0.95
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        self.replay_memory = deque(maxlen=2000)

    def getCardRank(self, s):
        if s[0] == 'T':
            return [10]
        if s[0] == 'J':
            return [11]
        if s[0] == 'Q':
            return [12]
        if s[0] == 'K':
            return [13]
        if s[0] == 'A':
            return [14]
        return [int(s[0])]

    def getCardSuit(self, s):
        return [int(s[1] == 's'), int(s[1] == 'h'), int(s[1] == 'c'), int(s[1] == 'd')]

    def getState(self):
        ret = self.getCardRank(self.round_state.hands[self.active][0]) + self.getCardSuit(self.round_state.hands[self.active][0]) + self.getCardRank(self.round_state.hands[self.active][1]) + self.getCardSuit(self.round_state.hands[self.active][1])
        for i in range(self.round_state.street):
            ret += self.getCardRank(self.round_state.deck[i]) + self.getCardSuit(self.round_state.deck[i])
        for i in range(5 - self.round_state.street):
            ret += [0,0,0,0,0]
        if self.terminal_state == None:
            ret += [0,0,0,0,0,0,0,0,0,0]
        elif self.terminal_state.previous_state.hands[1-self.active] == []:
            ret += [0,0,0,0,0,0,0,0,0,0]
        else:
            ret += self.getCardRank(self.terminal_state.previous_state.hands[1-self.active][0]) + self.getCardSuit(self.terminal_state.previous_state.hands[1-self.active][0]) + self.getCardRank(self.terminal_state.previous_state.hands[1-self.active][1]) + self.getCardSuit(self.terminal_state.previous_state.hands[1-self.active][1])
        
        my_pip = self.round_state.pips[self.active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = self.round_state.pips[1-self.active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = self.round_state.stacks[self.active]  # the number of chips you have remaining
        opp_stack = self.round_state.stacks[1-self.active]  # the number of chips your opponent has remaining

        ret += [my_pip, opp_pip, my_stack, opp_stack, self.active]
        return ret

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        else:
            # print(np.array(state).shape)
            Q_values = self.model.predict(np.array(state)[np.newaxis])
            return np.argmax(Q_values[0])
    
    

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones       
    
    # def play_one_step(self, env, state, epsilon):
    #     action = self.epsilon_greedy_policy(state, epsilon)
    #     next_state, reward, done, info = env.step(action)
    #     self.replay_memory.append((state, action, reward, next_state, done))
    #     return next_state, reward, done, info

    

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(np.array(next_states))
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - dones) * self.discount_rate * max_next_Q_values
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(np.array(states))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # env.seed(42)
    # np.random.seed(42)
    # tf.random.set_seed(42)

    # rewards = [] 
    # best_score = 0


    # for episode in range(600):
    #     obs = env.reset()    
    #     for step in range(200):
    #         epsilon = max(1 - episode / 500, 0.01)
    #         obs, reward, done, info = play_one_step(env, obs, epsilon)
    #         if done:
    #             break
    #     rewards.append(step) # Not shown in the book
    #     if step > best_score: # Not shown
    #         best_weights = model.get_weights() # Not shown
    #         best_score = step # Not shown
    #     print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
    #     if episode > 50:
    #         training_step(batch_size)

    # model.set_weights(best_weights)

    
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
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        self.game_state = game_state
        self.round_state = round_state
        self.active = active
        self.terminal_state = None
        self.rewards = 0
        self.state = self.getState()
        self.new_round = 1
        pass

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
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        self.game_state = game_state
        self.terminal_state = terminal_state
        self.active = active
        self.rewards = 5 * terminal_state.deltas[active]
        self.prev_state = self.state
        self.state = self.getState()
        if self.new_round == 1:
            self.new_round = 0
        else:
            self.replay_memory.append((self.prev_state, self.action, self.rewards, self.state, 0))
        self.iter += 1
        if self.iter >= 50 and self.iter % 3 == 0:
            self.training_step(self.batch_size)


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
        self.prev_state = self.state
        self.game_state = game_state
        self.round_state = round_state
        self.active = active
        self.state = self.getState()
        
        action = self.epsilon_greedy_policy(self.state, max(1 - self.iter / 100, 0.01))
        self.action = action
        if self.new_round == 1:
            self.new_round = 0
        else:
            self.replay_memory.append((self.prev_state, self.action, self.rewards, self.state, 0))
        self.iter += 1
        if self.iter >= 50 and self.iter % 3 == 0:
            self.training_step(self.batch_size)

        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        # street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        # my_cards = round_state.hands[active]  # your cards
        # board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        # opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        # my_stack = round_state.stacks[active]  # the number of chips you have remaining
        # opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        # continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        # my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        # opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        # self.action_dictionary = [FoldAction(), CheckAction(), CallAction(), RaiseAction(2)]#, RaiseAction(int(.25*(2 * STARTING_STACK - self.round_state.stacks[self.active] - self.round_state.stacks[1-self.active]))), RaiseAction(int(.5*(2 * STARTING_STACK - self.round_state.stacks[self.active] - self.round_state.stacks[1-self.active]))), RaiseAction(int(.75*(2 * STARTING_STACK - self.round_state.stacks[self.active] - self.round_state.stacks[1-self.active]))), RaiseAction(int((2 * STARTING_STACK - self.round_state.stacks[self.active] - self.round_state.stacks[1-self.active]))), RaiseAction(int(1.5*(2 * STARTING_STACK - self.round_state.stacks[self.active] - self.round_state.stacks[1-self.active]))), RaiseAction(int(.5*(2 * STARTING_STACK - self.round_state.stacks[self.active] - self.round_state.stacks[1-self.active]))), RaiseAction(min(self.round_state.stacks[self.active], self.round_state.stacks[1-self.active]))]
        # ret_action = self.action_dictionary[action]
        if RaiseAction in legal_actions and action == 3:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
           return RaiseAction(max_raise)  
        if CheckAction in legal_actions and action == 2:
            return CheckAction()
        if CallAction in legal_actions and action == 1:
            return CallAction()
        if FoldAction in legal_actions and action == 0:
            return FoldAction()
        return CheckAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
