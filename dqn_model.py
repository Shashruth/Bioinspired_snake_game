import os
from torch import argmax, max, tensor, float, long, unsqueeze,save
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class LinearQNeurNet(nn.Module):
    """
    A simple feedforward neural network for Q-learning.
    This network takes in the current state of the game as input and outputs Q-values for each possible action.
    """

    def __init__(self, size_input, size_hidden, size_output):
        """
        Initialize the neural network.

        :param size_input: Number of input features (size of the state space).
        :param size_hidden: Number of neurons in the hidden layer.
        :param size_output: Number of output features (size of the action space).
        """
        super().__init__()
        # Define the first linear layer (input to hidden)
        self.linear1 = nn.Linear(size_input, size_hidden)
        # Define the second linear layer (hidden to output)
        self.linear2 = nn.Linear(size_hidden, size_output)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor representing the state.
        :return: Output tensor representing Q-values for each action.
        """
        # Pass the input through the first layer and apply ReLU activation
        x = F.relu(self.linear1(x))
        # Pass the result through the second layer to get the output Q-values
        x = self.linear2(x)
        return x

    def save(self, f_name='dqnmodel.pth'):
        """
        Save the model to a file.

        :param f_name: The name of the file to save the model. Defaults to 'dqnmodel.pth'.
        """
        # Define the folder where the model will be saved
        model_path = './dqnmodel'
        # Create the folder if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Join the folder path and file name
        f_name = os.path.join(model_path, f_name)
        # Save the model's state dictionary (weights)
        save(self.state_dict(), f_name)


class QTrainer:

    """
    This class handles the training of the Q-network.
    It uses the Adam optimizer and Mean Squared Error (MSE) loss function.
    """

    def __init__(self, model, learn_rate, gamma):
        """
        Initialize the trainer with the model, learning rate, and discount factor.

        :param model: The Q-network model to be trained.
        :param learn_rate: Learning rate for the optimizer.
        :param gamma: Discount factor for future game points.
        """
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.model = model
        # Initialize the Adam optimizer with the model's parameters and the specified learning rate
        self.optimizer = optim.Adam(model.parameters(), lr=self.learn_rate)
        # Use Mean Squared Error (MSE) as the loss function
        self.criterion = nn.MSELoss()

    def train_step(self, state, new_state, game_points, action, game_over):
        """
        Perform one training step, updating the model's weights based on the provided experience tuple.

        :param state: The current state of the game.
        :param action: The action taken by the agent.
        :param game_points: The game points received after taking the action.
        :param new_state: The state of the game after the action.
        :param game_over: Boolean indicating whether the game is over.
        """
        # Convert the inputs to PyTorch tensors with the appropriate data types
        state = tensor(state, dtype=float)
        new_state = tensor(new_state, dtype=float)
        action = tensor(action, dtype=long)
        game_points = tensor(game_points, dtype=float)
        # Shape of tensors: (n, x), where n is the batch size and x is the feature size

        # If the input is a single experience (not a batch), unsqueeze to add a batch dimension
        if len(state.shape) == 1:
            # Add an extra dimension to make it a batch of size 1
            state = unsqueeze(state, 0)
            new_state = unsqueeze(new_state, 0)
            action = unsqueeze(action, 0)
            game_points = unsqueeze(game_points, 0)
            game_over = (game_over,)  # Ensure 'game_over' is iterable

        # 1: Predicted Q values from the present state
        pred = self.model(state)  # Forward pass through the model to get predicted Q-values

        # Create a copy of the predicted Q-values to modify the target
        target = pred.clone()
        for index in range(len(game_over)):
            Q_update = game_points[index]  # Initialize Q_update with the game_points
            if not game_over[index]:  # If the game is not over
                # Q_update is updated using the Bellman equation
                Q_update = game_points[index] + self.gamma * max(self.model(new_state[index]))

            # Update the target Q-value for the action that was actually taken
            target[index][argmax(action[index]).item()] = Q_update

        # 2: Loss calculation
        # The loss is calculated between the predicted Q-values and the updated target Q-values
        self.optimizer.zero_grad()  # Clear the gradients from the previous step
        loss = self.criterion(target, pred)  # Calculate the loss
        loss.backward()  # Backpropagate the loss to compute gradients

        # 3: Update the model's weights
        self.optimizer.step()  # Perform a single optimization step (parameter update)