import json

import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
from collections import deque
import torch
from snake_game_design import GameDesign, DirectionMap, Coordinates
from dqn_model import LinearQNeurNet, QTrainer

max_games = 25  # Set the number of training iterations
learn_rate = 0.001  # Learning rate for the model


def plot_performance(scores, average_scores):
    """Plot the performance of the Snake AI agent over time.

    """

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.plot(scores, label='Score per Game')
    plt.plot(average_scores, label='Average Score')

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlabel('Games Played')
    plt.ylabel('Score')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1)


def get_game_state(game_instance):
    """
    Assess the current status of the game for the snake agent.

    This function evaluates the current game environment, including the snake's direction,
    potential hazards around its head, and the position of food relative to the snake's head.

    :param game_instance: The current game environment instance.
    :return: A numpy array representing the state of the game with binary indicators.

    """
    moving_left = False
    moving_right = False
    moving_up = False
    moving_down = False

    # Identify the snake's current direction
    if game_instance.direction == DirectionMap.left:
        moving_left = True
    elif game_instance.direction == DirectionMap.right:
        moving_right = True
    elif game_instance.direction == DirectionMap.up:
        moving_up = True
    else:
        moving_down = True

    # Fetch the coordinates of the snake's head
    head_position = game_instance.snake_body[0]

    # Calculate the positions adjacent to the snake's head in all four directions
    next_position_up = Coordinates(head_position.x, head_position.y - 25)
    next_position_down = Coordinates(head_position.x, head_position.y + 25)
    next_position_left = Coordinates(head_position.x - 25, head_position.y)
    next_position_right = Coordinates(head_position.x + 25, head_position.y)


    # Construct the game state
    current_state = [
        # Current movement direction flags
        moving_up,
        moving_down,
        moving_left,
        moving_right,


        # Detect potential collisions in the current movement direction
        (moving_right and game_instance.snake_crash(next_position_right)) or
        (moving_left and game_instance.snake_crash(next_position_left)) or
        (moving_up and game_instance.snake_crash(next_position_up)) or
        (moving_down and game_instance.snake_crash(next_position_down)),

        # Detect potential collisions to the right of the current direction
        (moving_up and game_instance.snake_crash(next_position_right)) or
        (moving_down and game_instance.snake_crash(next_position_left)) or
        (moving_left and game_instance.snake_crash(next_position_up)) or
        (moving_right and game_instance.snake_crash(next_position_down)),

        # Detect potential collisions to the left of the current direction
        (moving_down and game_instance.snake_crash(next_position_right)) or
        (moving_up and game_instance.snake_crash(next_position_left)) or
        (moving_right and game_instance.snake_crash(next_position_up)) or
        (moving_left and game_instance.snake_crash(next_position_down)),


        # Position of the food relative to the snake's head
        game_instance.food.x < game_instance.head.x,  # Food is left of the head
        game_instance.food.x > game_instance.head.x,  # Food is right of the head
        game_instance.food.y < game_instance.head.y,  # Food is above the head
        game_instance.food.y > game_instance.head.y  # Food is below the head
    ]

    # Return the state as a numpy array of integers
    return np.array(current_state, dtype=int)


class Snake:
    """
    Snake agent that interacts with the environment and learns using Deep Q-Learning.

    :param games_played: Number of games played by the snake.
    :param gamma: Discount rate for future game_points.
    :param epsilon: Parameter for exploration-exploitation tradeoff.
    :param storage: Memory to store experience replay.
    :param linear_neur_net: Neural network model used for predicting Q-values.
    :param trainer: QTrainer instance used to train the model.

    """

    def __init__(self, gamma=0.9, learn_rate=0.001):
        self.gamma = gamma  # discount rate
        self.epsilon = 0  # randomness for exploration-exploitation
        self.games_played= 0

        self.storage = deque(maxlen=200_000)  # experience replay memory
        self.linear_neur_net = LinearQNeurNet(11, 256,
                                    3)  # Q-Network with input 
        self.trainer = QTrainer(self.linear_neur_net, gamma=self.gamma, learn_rate=learn_rate)  # Q-learning trainer

    def store_info(self, game_state, new_state, game_points, move, game_end):
        """
        Store a new experience in storage.

        :param game_state: Current state of the game.
        :param move: Move taken by the snake.
        :param game_points: Game points received after taking the move.
        :param new_state: State of the game after taking the move.
        :param game_end: Boolean indicating whether the game is over.

        """
        self.storage.append((game_state, new_state, game_points, move, game_end))  # append experience to storage

    def decide_move(self, game_state, epsilon0=80):
        """
        Decide on a move based on the current state.

        This function uses an epsilon-greedy strategy to balance exploration
        (random moves) and exploitation (moves based on learned policy).
        Initially, the snake explores more by taking random moves, but as
        it plays more games, it gradually shifts to exploiting its learned
        knowledge by choosing moves that maximize the predicted Q-value.

        :param   game_state: The current state of the game, represented as a numpy array.
        :param   epsilon0: The epsilon-greedy?
        :return: next_move_list: A list representing the move [left, straight, right].

        """

        # Calculate exploration-exploitation tradeoff.  Epsilon decreases as the
        # number of games increases, meaning less exploration and more exploitation over time.
        self.epsilon = epsilon0 - self.games_played
        # Initialize move list where only one of the moves will be set to 1 (indicating the chosen move)
        next_move_list = [0, 0, 0]  # [left, straight, right]

        # Determine whether to explore or exploit
        if random.randint(0, 180) < self.epsilon:
            # Exploration: Randomly choose a move to explore environment
            direction_idx = random.randint(0, 2)  # randomly select an index (0: left, 1: straight, 2: right)
            next_move_list[direction_idx] = 1  # set chosen move to 1

        else:
            # Exploitation: Use learned model to predict best move based on current state
            sold_state_tensor = torch.tensor(game_state, dtype=torch.float)  # converting state to a tensor of type float
            q_value_dir = self.linear_neur_net(sold_state_tensor)  # get the Q-values for each move from the model
            direction_idx = torch.argmax(q_value_dir).item()  # torch.argmax returns the index of the highest Q-value (0, 1, or 2)
            next_move_list[direction_idx] = 1  # set the chosen move to 1

        # Return the final move list
        return next_move_list

    def short_memory_trainer(self, game_state, new_state, game_points, move, game_end):
        """
        Train the model on a single step.

        :param game_state: The current state of the game.
        :param move: The move taken by the snake.
        :param game_points: The game points received after taking the move.
        :param new_state: The state of the game after taking the move.
        :param game_end: Boolean indicating whether the game is over.
        """
        self.trainer.train_step(game_state, new_state, game_points, move, game_end)  # perform a single training step

    def long_memory_trainer(self):
        """
        Train the model on a batch of memories (experience replay).
        """

        if len(self.storage) > 2000:
            mini_sample = random.sample(self.storage, 2000)  # randomly sample a batch from storage
        else:
            mini_sample = self.storage  # if not enough storage, use all available ones

        states, new_states, game_points ,moves , game_end = zip(*mini_sample)  # unzip the batch
        self.trainer.train_step(states, new_states, game_points ,moves , game_end)  # train the model on the batch


def train_snake(n_iterations, epsilon0=80, gamma=0.9, learn_rate=0.001):
    """
    Train the snake agent by playing a set number of games and learning from them.


    """
    global mean_score
    game_scores = []  # list to store the scores of each game
    average_scores = []  # list to store the mean scores over time
    total_score = 0  # total score accumulated over all games
    record = 0  # record score
    game = GameDesign()  # create a new Snake game environment

    snake = Snake(gamma=gamma, learn_rate=learn_rate)  # create a new Snake
    # iterations = 0  # Initialize the iterations counter

    while True:
        # Get the current state of the game
        old_state = get_game_state(game)

        # Get the move based on the current state
        next_move_list = snake.decide_move(old_state, epsilon0)

        # Perform the move and get the new state and game_points
        game_points, game_end, current_score = game.play_step(next_move_list)
        new_state = get_game_state(game)

        # Store the information
        snake.store_info(old_state, new_state,  game_points, next_move_list, game_end)

        # Train the model on the short memory (single step)
        snake.short_memory_trainer(old_state, new_state, game_points, next_move_list, game_end)

        if game_end:
            # If the game is over, reset the game and train on long memory (experience replay)
            snake.games_played += 1  # increment the number of games played

            snake.long_memory_trainer()

            if current_score > record:  # update the record score if the current score is higher
                record = current_score
                snake.linear_neur_net.save()  # save the model with the highest score

            game_scores.append(current_score)

            total_score += current_score
            average_score = total_score / snake.games_played
            average_scores.append(average_score)

            # iterations += 1
            game.reset_game()

            if snake.games_played == n_iterations:
                print(f"Training completed after {snake.games_played} games.")
                plot_performance(game_scores, average_scores)

                return game_scores, average_scores


if __name__ == '__main__':
    game_scores, average_scores = train_snake(max_games, epsilon0=80)  # start the training process with the specified number of games
    #
    epsilon0_range = np.linspace(25, 50, 3)
    gamma_range = np.linspace(0.6, 1, 5)
    learn_rate_range = [0.001, 0.01, 0.1]

    epsilon_scores = {}
    gamma_scores = {}
    learn_rate_scores = {}
    for epsilon0 in epsilon0_range:
        _game_scores, _average_scores = train_snake(max_games, gamma=0.9, epsilon0=epsilon0, learn_rate=learn_rate)
        epsilon_scores[str(epsilon0)] = _game_scores, _average_scores

    with open('epsilon_scores.json', 'w') as json_file:
        json.dump(epsilon_scores,json_file)
    print(epsilon_scores)

    for gamma in gamma_range:
        _game_scores, _average_scores = train_snake(max_games, epsilon0=80, gamma=gamma, learn_rate=learn_rate)
        gamma_scores[str(gamma)] = _game_scores, _average_scores

    with open('gamma_scores.json', 'w') as json_file:
        json.dump(gamma_scores,json_file)
    print(gamma_scores)

    for learn_rate in learn_rate_range:
        _game_scores, _average_scores = train_snake(max_games, epsilon0=80,gamma=0.9,  learn_rate=learn_rate)
        learn_rate_scores[str(learn_rate)] = _game_scores, _average_scores

    with open('LR_scores.json', 'w') as json_file:
        json.dump(learn_rate_scores,json_file)
    print(learn_rate_scores)
