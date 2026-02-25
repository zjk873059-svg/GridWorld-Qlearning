# -*- coding: utf-8 -*-
"""
Grid World Q-learning with Pygame UI
Author: ZHUANG Jingkun
"""

import numpy as np
import pygame
import sys
import random

# Constants
GRID_SIZE = 6
CELL_SIZE = 80
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Map elements
EMPTY = 0
TREASURE = 1
TRAP = 2
START = 3

class GridWorld:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.start_pos = (0, 0)
        self.treasure_positions = []
        self.trap_positions = []
        self.agent_pos = None
        self.reward = 0
        self.done = False
        self.reset()

    def reset(self):
        self.grid.fill(EMPTY)
        self.start_pos = (0, 0)
        self.grid[self.start_pos] = START
        self.agent_pos = self.start_pos

        # Fixed treasure and trap positions for stable learning
        self.treasure_positions = [(1, 2), (2, 4), (4, 3)]
        self.trap_positions = [(3, 1), (5, 5)]

        for pos in self.treasure_positions:
            self.grid[pos] = TREASURE
        for pos in self.trap_positions:
            self.grid[pos] = TRAP

        self.reward = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # State = linear index of current position
        return self.agent_pos[0] * GRID_SIZE + self.agent_pos[1]

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:      # up
            x = max(0, x-1)
        elif action == 1:    # down
            x = min(GRID_SIZE-1, x+1)
        elif action == 2:    # left
            y = max(0, y-1)
        elif action == 3:    # right
            y = min(GRID_SIZE-1, y+1)

        new_pos = (x, y)
        cell_type = self.grid[new_pos]
        reward = 0
        done = False

        if new_pos == self.agent_pos:          # hit wall
            reward = -0.2
        else:
            self.agent_pos = new_pos
            if cell_type == TREASURE:
                reward = 30
                self.grid[new_pos] = EMPTY
                self.treasure_positions.remove(new_pos)
                if len(self.treasure_positions) == 0:
                    reward += 50               # extra bonus for collecting all
                    done = True
            elif cell_type == TRAP:
                reward = -30
                done = True
            else:                               # empty cell
                reward = -0.2

        self.reward = reward
        self.done = done
        return self.get_state(), reward, done


class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.05, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class GameUI:
    def __init__(self, env, agent):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Grid World Q-learning")
        self.clock = pygame.time.Clock()
        self.env = env
        self.agent = agent
        self.font = pygame.font.SysFont("Arial", 20)

    def draw_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell = self.env.grid[i][j]
                if cell == START:
                    color = GREEN
                elif cell == TREASURE:
                    color = YELLOW
                elif cell == TRAP:
                    color = RED
                else:
                    color = WHITE
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)

        # Draw agent
        x, y = self.env.agent_pos
        center = (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(self.screen, BLUE, center, CELL_SIZE//3)

    def display_info(self, episode, step, reward, epsilon):
        text = f"Episode: {episode}  Step: {step}  Reward: {reward:.1f}  Epsilon: {epsilon:.2f}"
        surf = self.font.render(text, True, BLACK)
        self.screen.blit(surf, (10, WINDOW_HEIGHT-30))

    def run_training(self, episodes=2000):
        total_rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step = 0
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1

                # Render
                self.screen.fill(WHITE)
                self.draw_grid()
                self.display_info(ep+1, step, total_reward, self.agent.epsilon)
                pygame.display.flip()
                self.clock.tick(10)

            total_rewards.append(total_reward)
            print(f"Episode {ep+1}: Total Reward = {total_reward:.1f}, Epsilon = {self.agent.epsilon:.3f}")

        pygame.quit()
        return total_rewards


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent(state_size=GRID_SIZE*GRID_SIZE, action_size=4)
    ui = GameUI(env, agent)
    rewards = ui.run_training(episodes=2000)
    np.save("q_table.npy", agent.q_table)
    print("Training finished. Q-table saved.")