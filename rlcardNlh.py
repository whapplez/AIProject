### Settings for Evaluation ###
# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 10000

# The paths for saving the logs and learning curves
root_path = './experiments/nl_holdem_nfsp_result/'
log_path = root_path + 'log.txt'
csv_path = log_path + '/performance.csv'
figure_path = root_path + 'figures/'
save_dir = 'models/nl_holdem_nfsp/'
# checkpoint_prefix = os.path.join(save_dir, "ckpt")
### Settings for Evaluation ###

### Step 1: Make the Environment. ###
import os
import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from rlcard.utils.logger import plot
# Make environment
env = rlcard.make('no-limit-holdem')
eval_env = rlcard.make('no-limit-holdem')
# Set a global seed
set_global_seed(0)

### Step 2: Initialize the NFSP agents. ###
import tensorflow.compat.v1 as tf
from rlcard.agents.nfsp_agent import NFSPAgent
tf.disable_v2_behavior()
memory_init_size = 1000
norm_step = 100
with tf.Session() as sess:
    # Set agents
    global_step = tf.Variable(0, name='global_step', trainable=False)
    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(sess,
                          scope='nfsp' + str(i),
                          action_num=env.action_num,
                          state_shape=env.state_shape,
                          hidden_layers_sizes=[128,128],
                          min_buffer_size_to_learn=1000,
                          q_replay_memory_init_size=memory_init_size,
                          q_update_target_estimator_every=norm_step,
                          q_mlp_layers=[128,128])
        agents.append(agent)
    # with sess.as_default():  #uncomment when loading 
    #     saver = tf.train.Saver()
    #     saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    sess.run(tf.global_variables_initializer()) # comment out when loading
    env.set_agents(agents) # Setup all nfsp agents into training environments
    
    # Setup random agent for evaluation
    random_agent = RandomAgent(action_num=eval_env.action_num) 
    eval_env.set_agents([agents[0], random_agent])

    ### Step 3: Generate game data and train the agents. ###
    episode_num = 500 # set the episode number
    step_counters = [0 for _ in range(env.player_num)] # Count the number of steps

    # Init a Logger to plot the learning curve
    logger = Logger(log_path)

    for episode in range(episode_num):

        # First sample a policy for the episode
        for agent in agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for ts in trajectories[i]:
                agents[i].feed(ts)
                step_counters[i] += 1

                # Train the agent
                train_count = step_counters[i] - (memory_init_size + norm_step)
                if train_count > 0 and train_count % 64 == 0:
                    # rl_loss = agents[i].train_rl()
                    sl_loss = agents[i].train_sl()
                    print('\rINFO - Agent {}, step {}, sl-loss: {}'.format(i, step_counters[i], sl_loss), end='')
        
        ### Evaluation Step: Evaluate the agent every 100 steps and generate the plot for analysis. ###
        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            reward = 0
            for eval_episode in range(evaluate_num):
                _, payoffs = eval_env.run(is_training=False)
                reward += payoffs[0]

            logger.log('\n########## Evaluation ##########')
            logger.log('Timestep: {} Average reward is {}'.format(env.timestep, float(reward)/evaluate_num))

            # Add point to logger
            logger.log_performance(env.timestep, float(reward)/evaluate_num)

        # Make plot
        if episode % 100000 == 0 and episode > 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(save_dir, 'model'+str(episode//100000)))
        #     logger.plot('NSFP')

    # Make the final plot
    # plot(csv_path, figure_path + 'fig.png', 'NSFP')
    
    