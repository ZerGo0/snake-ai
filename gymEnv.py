from typing import Deque
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.random import random
import cv2
import random

SNAKE_LEN_GOAL = 100
FONT = cv2.FONT_HERSHEY_SIMPLEX


class GymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, env_id: str):
        super().__init__()
        self.env_id = env_id

        self.action_space = spaces.Discrete(4)

        coords_min = -500
        coords_max = 500
        snake_length_min = 3
        snake_length_max = 100

        self.observation_space = spaces.Dict(
            {
                "head_position": spaces.Box(
                    low=np.array([coords_min, coords_min]),
                    high=np.array([coords_max, coords_max]),
                    dtype=np.int64,
                ),
                "apple_delta_position": spaces.Box(
                    low=np.array([coords_min, coords_min]),
                    high=np.array([coords_max, coords_max]),
                    dtype=np.int64,
                ),
                "snake_length": spaces.Box(
                    low=snake_length_min, high=snake_length_max, dtype=np.int64
                ),
                "previous_actions": spaces.Box(
                    low=-1, high=3, shape=(100,), dtype=np.int64
                ),
            }
        )

    def reset(self, seed=None, options=None):
        if self.env_id == "0":
            self.img = np.zeros((500, 500, 3), dtype="uint8")

        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]
        self.total_reward = 0.0

        self.prev_actions = Deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = self.get_observation()

        return observation, {}

    def step(self, action):
        reward = 0
        killed = False

        # Update previous actions
        self.prev_actions.append(action)

        # Display the environment
        if self.env_id == "0":
            cv2.imshow("a", self.img)
            cv2.waitKey(1)
            self.img = np.zeros((500, 500, 3), dtype="uint8")

            # Display Apple
            cv2.rectangle(
                self.img,
                (self.apple_position[0], self.apple_position[1]),
                (self.apple_position[0] + 10, self.apple_position[1] + 10),
                (0, 0, 255),
                3,
            )

            # Display Snake
            for position in self.snake_position:
                cv2.rectangle(
                    self.img,
                    (position[0], position[1]),
                    (position[0] + 10, position[1] + 10),
                    (0, 255, 0),
                    3,
                )

        button_direction = action
        # Change the head position based on the button direction
        prev_snake_head = [self.snake_head[0], self.snake_head[1]]
        if button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
            reward += 10
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

            # Reward for moving towards apple
            if abs(self.snake_head[0] - self.apple_position[0]) < abs(
                prev_snake_head[0] - self.apple_position[0]
            ) or abs(self.snake_head[1] - self.apple_position[1]) < abs(
                prev_snake_head[1] - self.apple_position[1]
            ):
                reward += 0.025

        # On collision kill the snake and print the score
        if (
            collision_with_boundaries(self.snake_head) == 1
            or collision_with_self(self.snake_position) == 1
        ):
            if self.env_id == "0":
                self.img = np.zeros((500, 500, 3), dtype="uint8")
                cv2.putText(
                    self.img,
                    "Your Score is {}".format(self.score),
                    (140, 250),
                    FONT,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("a", self.img)

            reward = -10
            killed = True

        self.total_reward += reward

        if self.env_id == "0":
            # Display reward
            cv2.putText(
                self.img,
                "Reward: {:.1f}".format(self.total_reward),
                (10, 20),
                FONT,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        observation = self.get_observation()

        return observation, reward, killed, False, {}

    def get_observation(self):
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        observation = {
            "head_position": np.array([head_x, head_y]),
            "apple_delta_position": np.array([apple_delta_x, apple_delta_y]),
            "snake_length": np.array([snake_length]),
            "previous_actions": np.array(self.prev_actions),
        }

        return observation


# region Helper functions


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if (
        snake_head[0] >= 500
        or snake_head[0] < 0
        or snake_head[1] >= 500
        or snake_head[1] < 0
    ):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


# endregion
