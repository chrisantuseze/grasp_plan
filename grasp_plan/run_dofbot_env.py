import os

import gymnasium as gym
import random

from matplotlib import pyplot as plt
import mujoco
import numpy as np
from PIL import Image

import dofbot # This registers your custom environment

# env = gym.make('TableSceneEnv-v0', render_mode='human', camera_name="front_camera") # Use the stacking environment
env = gym.make('TableSceneEnv-v0', render_mode='rgb_array', camera_name="front_camera") # Use the stacking environment


# Set a seed for reproducibility
seed = random.randint(0, 1000)
observation, info = env.reset(seed=seed)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space.shape}")
print(f"Initial observation: {observation}")

onscreen_render = True # Set to True if you want to render the environment onscreen

for episode in range(2): # Run a few episodes
    print(f"\n--- Episode {episode + 1} ---")
    episode_reward = 0
    terminated = False
    truncated = False
    step_count = 0

    # Reset before each episode
    observation, info = env.reset()

    action = np.zeros(env.action_space.shape, dtype=np.float32) # Initialize action
    action[0] = 0.8 # Set a specific action for the first joint
    action[1] = -1.0 # Set a specific action for the second joint
    action[2] = -1.5 # Set a specific action for the third joint
    action[3] = -1.5 # Set a specific action for the fourth joint
    action[4] = 0.0 # Set a specific action for the fifth joint
    action[5] = 0.0 # Set a specific action for the sixth joint

    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(observation['front_camera'])
        plt.ion()

    while not terminated and not truncated and step_count < 20:
        # action = env.action_space.sample() # Sample random action
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_count + 1}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        episode_reward += reward
        step_count += 1

        # Get camera image
        # image = observation['wrist_camera']
        # if image is not None:
        #     Image.fromarray(image).save(f"step_{step_count}_wrist_camera.png")
        #     print(f"Saved custom image for step {step_count}")

        # image = observation['front_camera']
        # if image is not None:
        #     Image.fromarray(image).save(f"step_{step_count}_front_camera.png")
        #     print(f"Saved custom image for step {step_count}")

        # setup plotting
        if onscreen_render:
            plt_img.set_data(observation['front_camera'])
            plt.pause(0.002)

    # while not terminated and not truncated:
    #     # action = env.action_space.sample() # Sample random action
    #     observation, reward, _, _, info = env.step(action)
    #     print(f"Step {step_count + 1}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    #     episode_reward += reward
    #     step_count += 1

    #     env.render() # uncomment if you are capturing frames

    print(f"Episode finished after {step_count} steps. Total reward: {episode_reward}")
    plt.close()

env.close()
print("Simulation finished.")