# -*- coding: utf-8 -*-
"""
Demo: Load trained Q-table and watch the agent play multiple episodes.
"""

import numpy as np
import pygame
import sys
import time
from grid_world import GridWorld, QLearningAgent, GameUI, WINDOW_HEIGHT

def main():
    env = GridWorld()
    agent = QLearningAgent(state_size=6*6, action_size=4)
    agent.q_table = np.load("q_table.npy")
    agent.epsilon = 0.0          # fully exploit learned policy

    ui = GameUI(env, agent)

    num_episodes = 5
    pause_seconds = 2

    print("Starting multi-episode demo... Press close window to exit.")

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            step += 1

            # Render
            ui.screen.fill(WHITE)
            ui.draw_grid()
            info = f"Episode: {ep+1}/{num_episodes}  Step: {step}  Reward: {total_reward:.1f}"
            surf = ui.font.render(info, True, BLACK)
            ui.screen.blit(surf, (10, WINDOW_HEIGHT-30))
            pygame.display.flip()
            ui.clock.tick(5)      # slow motion

        print(f"Episode {ep+1} finished. Total reward: {total_reward:.1f}")

        # Pause between episodes
        if ep < num_episodes - 1:
            start = time.time()
            while time.time() - start < pause_seconds:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                ui.screen.fill(WHITE)
                ui.draw_grid()
                pause_text = f"Episode {ep+1} done. Next in {int(pause_seconds - (time.time()-start) + 1)}s..."
                surf = ui.font.render(pause_text, True, BLACK)
                ui.screen.blit(surf, (10, WINDOW_HEIGHT-30))
                pygame.display.flip()
                ui.clock.tick(10)

    pygame.quit()
    print("Demo finished.")


if __name__ == "__main__":
    # Import constants from grid_world
    from grid_world import WHITE, BLACK
    main()