#!/usr/bin/python3

# Imports
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO)

# Variables and constants
num_of_vars = 8
num_of_actions = 4
num_init_observations = 0
rate_of_learning = 0.001
weights_file = "LunarLander-v2-SGD-weights.h5"
model_graph_file = "LunarLander-v2-SGD-model.png"
beta_discount = 0.98
max_memory = 60000
start_exploration_prob = 0.05
num_of_epochs = 20
save_weights = True
show_graph = True
num_of_games = 500
sum_of_dof = num_of_vars + num_of_actions
training_sets = 10
training_batch_size = 32
valid_split = 0.33

# NumPy
possible_actions = np.arange(0, num_of_actions)
actions_hot_enc = np.zeros((num_of_actions, num_of_actions))
actions_hot_enc[np.arange(num_of_actions), possible_actions] = 1

# Environment creation
env = gym.make('LunarLander-v2')
env.reset()

# Initializing training array and neural net
array_x = np.random.random((5, sum_of_dof))
# Array for total scores
array_y = np.random.random((5, 1))

logging.info('Initializing keras model...')
# Creation of a Keras Sequential models
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=array_x.shape[1]))
model.add(Dense(array_y.shape[1]))
# Keras Optimizer
optmzr = optimizers.sgd(lr=rate_of_learning)
# optmzr = optimizers.adam(lr=rate_of_learning)
model.compile(loss='mse', optimizer=optmzr, metrics=['accuracy'])
logging.info('Keras model compiled.')

# Load previous weights
current_path = os.path.realpath(".")
new_path = current_path + "/" + weights_file
logging.debug('weight path: %s', new_path)

if os.path.isfile(new_path):
    logging.info('Weights h5 file found at %s', new_path)
    logging.info('Loading weights...')
    model.load_weights(weights_file)
else:
    logging.info('Weights h5 file NOT found at %s', new_path)
    logging.info('Retraining...')

# Initialize training data
array_x = np.zeros(shape=(1, sum_of_dof))
array_y = np.zeros(shape=(1, 1))

# Initialize look-back memory array
memory_x = np.zeros(shape=(1, sum_of_dof))
memory_y = np.zeros(shape=(1, 1))

logging.debug('array_x: %s', array_x)
logging.debug('array_y: %s', array_y)


# (Q State, action) -> predicted reward
# Function that returns the predicted reward received by an agent when they taken action a at state Q
def getPrediction(q_state, action):
    qs_action_hot_enc = np.concatenate((q_state, actions_hot_enc[action]), axis=0)
    pred_x = np.zeros(shape=(1, sum_of_dof))
    pred_x[0] = qs_action_hot_enc
    # Predicting reward at Q state q_state
    prediction = model.predict(pred_x.reshape(1, pred_x.shape[1]))
    sum_remembered_reward = prediction[0][0]
    return sum_remembered_reward

# Training
for cur_game in range(num_of_games):
    game_x = np.zeros(shape=(1, sum_of_dof))
    game_y = np.zeros(shape=(1, 1))
    q_state = env.reset()
    for step in range(40000):
        eps = np.random.rand(1)
        exploration_prob = start_exploration_prob - (start_exploration_prob / num_of_games) * cur_game
        logging.debug('epsilon: %s', eps)

        # Chance
        if eps < exploration_prob:
            act = env.action_space.sample()
            logging.debug('Chance action: %s', act)
        # Prediction
        else:
            # Q State approximate values for each action to maximize on
            q_val_of_actions = np.zeros(shape=(num_of_actions))
            q_val_of_actions[0] = getPrediction(q_state, 0)
            q_val_of_actions[1] = getPrediction(q_state, 1)
            q_val_of_actions[2] = getPrediction(q_state, 2)
            q_val_of_actions[3] = getPrediction(q_state, 3)
            # Argmax to maximize over Q values
            act = np.argmax(q_val_of_actions)
            logging.debug('Prediction action: %s', act)

        # Display
        env.render()
        qs_action_hot_enc = np.concatenate((q_state, actions_hot_enc[act]), axis=0)
        logging.debug('Performing optimal step...')
        # Perform optimal action
        state, reward, done, meta = env.step(act)

        # Store experience
        if step == 0:
            game_x[0] = qs_action_hot_enc
            game_y[0] = np.array([reward])
            memory_x[0] = qs_action_hot_enc
            memory_y[0] = np.array([reward])

        game_x = np.vstack((game_x, qs_action_hot_enc))
        game_y = np.vstack((game_y, np.array([reward])))

        # Lunar Land successful
        if done:
            # Calculating Q values
            for i in range(0, game_y.shape[0]):
                game_epoch = (game_y.shape[0] - 1) - i
                if i == 0:
                    game_y[game_epoch][0] = game_y[game_epoch][0]
                else:
                    game_y[game_epoch][0] = game_y[game_epoch][0] + beta_discount * game_y[game_epoch + 1][0]

                if i == (game_y.shape[0] - 1):
                    logging.info('Training game: #%s steps: %s last reward: %s end score: %s', cur_game, step, reward, game_y[game_epoch][0])

            # Memory is experience
            if memory_x.shape[0] == 1:
                memory_x = game_x
                memory_y = game_y
            # Add experience to memory
            else:
                memory_x = np.concatenate((memory_x, game_x), axis=0)
                memory_y = np.concatenate((memory_y, game_y), axis=0)

            # Clear memory if full
            if np.alen(memory_x) >= max_memory:
                logging.debug('Memory filled at %s. Clearing memory...', np.alen(memory_x))
                for exp in range(np.alen(game_x)):
                    memory_x = np.delete(memory_x, 0, axis=0)
                    memory_y = np.delete(memory_y, 0, axis=0)

        # Update states
        q_state = state

        # Retrain every 10 sets
        if done:
            if cur_game % training_sets == 0:
                logging.info('Training game: #%s Memory: %s', cur_game, memory_x.shape[0])
                history = model.fit(memory_x, memory_y, validation_split=valid_split, batch_size=training_batch_size, epochs=num_of_epochs, verbose=2)
            if (reward >= 0) and (reward < 99):
                logging.info('Game %s ended with a positive reward: %s', cur_game, reward)
            if reward > 50:
                logging.info('Game %s won!', cur_game)
            break

if save_weights:
    logging.info('Saving weights...')
    model.save_weights(weights_file)

if show_graph:
    logging.info('Saving model to %s ...', model_graph_file)
    plot_model(model=model, to_file=model_graph_file)

    # logging.info('Printing history: %s', history.history)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
