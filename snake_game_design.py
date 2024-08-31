import random
from enum import Enum
import numpy as np
import pygame
from collections import namedtuple

pygame.init()
font = pygame.font.Font('sans.ttf', 30)

# A point to represent coordinates on the screen (x, y)
Coordinates = namedtuple('Coordinates', 'x, y')

# Define RGB colors
white = (250, 250, 250)
black = (0, 0, 0)
green1 = (56, 93, 56)
green2 = (144, 238, 144)
red = (200, 0, 0)

# Define block size and speed for the snake
snake_block_size = 25
snake_speed = 40


# Possible directions the snake can move in
class DirectionMap(Enum):
    up = 1
    down = 2
    right = 3
    left = 4


class GameDesign:
    """
    This class encapsulates the design and logic of the Snake game. It includes
    methods for resetting the game, handling player input, moving the snake,
    checking for collisions, updating the display, and placing food.

    """

    def __init__(self):
        """
        :param height: Height of the game window (default is 480).
        :param width: Width of the game window (default is 640).

        """

        self.frame_iteration = None
        self.score = None
        self.head = None
        self.snake_body = None
        self.direction = None
        self.food = None

        self.height = 600
        self.width = 750
        # Initialize the game display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('RL Snake Game')  # Set the window title
        self.clock = pygame.time.Clock()  # Set up the game clock for controlling the frame rate
        self.reset_game()  # Start the game by resetting it

    def reset_game(self):
        """
        Reset the game state. This method reinitializes the snake's position,
        direction, score, and places the first food item.

        """
        # Start the snake moving left
        self.direction = DirectionMap.left

        # Initialize the snake's position
        self.head = Coordinates(self.width / 2, self.height / 2)  # Start the snake in the middle of the game screen
        self.snake_body = [self.head,
                           Coordinates(self.head.x - snake_block_size, self.head.y),
                           Coordinates(self.head.x - (2 * snake_block_size), self.head.y)]

        self.score = 0  # Initialize score
        self.food = None  # Initialize food
        self.food_placement()  # Place the first food on the screen
        self.frame_iteration = 0  # Initialize the frame counter

    def snake_crash(self, point=None):
        """
        Check for a collision with the wall or the snake's own body.

        :param point: The point to check for collision (default is the snake's head).
        :return: Boolean indicating if there is a collision.
        """
        if point is None:
            point = self.head  # Default to checking the head

        # Check if the snake hits the boundary
        if point.x < 0 or point.y < 0 or point.x > self.width - snake_block_size or point.y > self.height - snake_block_size:
            return True

        # Check if the snake hits itself
        elif point in self.snake_body[1:]:
            return True

        else:
            return False

    def play_step(self, direction_list):
        """
        Perform a single step in the game, including moving the snake, checking for collisions,
        and updating the game state.

        :param direction_list: A list representing the move [straight, right, left].
        :return: reward: The reward gained after this step.
                 game_end: Boolean indicating if the game is over.
                 score: The current score.
        """
        self.frame_iteration += 1  # Increase the frame iteration count

        # 1. Collect user input (e.g., if the player closes the game window)
        for move in pygame.event.get():
            if move.type == pygame.QUIT:
                pygame.quit()  # Quit the game
                quit()

        _new_dir = self.get_new_snake_direction(direction_list)
        self.direction = _new_dir  # Set the new direction

        # Update the position of the snake's head based on the new direction
        x = self.head.x
        y = self.head.y

        if self.direction == DirectionMap.up:
            y -= snake_block_size  # Move up
        elif self.direction == DirectionMap.down:
            y += snake_block_size  # Move down
        elif self.direction == DirectionMap.right:
            x += snake_block_size  # Move right
        elif self.direction == DirectionMap.left:
            x -= snake_block_size  # Move left
        else:
            pass

        # Update the head position
        self.head = Coordinates(x, y)

        self.snake_body.insert(0, self.head)  # Add the new head position to the snake body

        game_points = 0
        game_end = False

        if self.snake_crash() or self.frame_iteration > 100 * len(self.snake_body):
            game_end = True  # End the game if there's a collision or the snake takes too long
            game_points = -15  # Negative game points for dying
            return game_points, game_end, self.score

        # Check if the snake has eaten the food
        if self.head == self.food:
            self.score += 1  # Increase score
            game_points = 15  # Positive game points for eating food
            self.food_placement()  # Place new food
        else:
            self.snake_body.pop()  # Remove the last segment of the snake to simulate movement

        # Update the game UI and clock
        self._update_ui()  # Redraw the game window
        self.clock.tick(snake_speed)  # Control the game speed

        return game_points, game_end, self.score

    def get_new_snake_direction(self, direction_list):

        # Define the clockwise directions for easy navigation
        clock_wise = [DirectionMap.right, DirectionMap.down, DirectionMap.left, DirectionMap.up]
        idx = clock_wise.index(self.direction)  # Get the current direction index

        # Determine the new direction based on the move
        # Straight (no change in direction)
        if np.array_equal(direction_list, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Continue in the same direction

        # Right turn (clockwise rotation)
        elif np.array_equal(direction_list, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Move to the next direction in the clockwise list
            new_dir = clock_wise[next_idx]  # Update direction to the right

        # Left turn (counter-clockwise rotation)
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # Move to the previous direction in the clockwise list
            new_dir = clock_wise[next_idx]  # Update direction to the left

        return new_dir

    def food_placement(self):
        """
        Place food in a random location on the screen. Ensure that the food does not
        spawn on the snake's body.
        """
        # Randomly place food within the grid, ensuring it aligns with the grid size
        x = random.randint(0, (self.width - snake_block_size) // snake_block_size) * snake_block_size
        y = random.randint(0, (self.height - snake_block_size) // snake_block_size) * snake_block_size
        self.food = Coordinates(x, y)  # Create the food point

        # If the food is placed on the snake's body, reposition the food
        if self.food in self.snake_body:
            self.food_placement()

    def _update_ui(self):
        """
        Update the game's graphical user interface, including drawing the snake,
        food, and the current score.
        """
        self.display.fill(black)  # Fill the background with black

        # Draw the snake body
        for pt in self.snake_body:
            pygame.draw.rect(self.display, green1, pygame.Rect(pt.x, pt.y, snake_block_size, snake_block_size))
            pygame.draw.rect(self.display, green2,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # Draw inner block for 3D effect

        # Draw the food
        pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, snake_block_size, snake_block_size))

        # Display the current score
        text = font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Update the full display
