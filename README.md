# Deep Convolutional Q-Network (DCQN) — Ms. Pac-Man

A Deep Convolutional Q-Network agent trained to play Ms. Pac-Man 
from raw pixel input, built with PyTorch and Gymnasium.

## Overview
This project implements a DCQN agent — an extension of DQN where 
convolutional layers are used to extract spatial features directly 
from game screen images, without any hand-crafted feature engineering.

## Architecture
- 4 convolutional layers with batch normalization
- 3 fully connected layers
- Dual network setup: local Q-network + target Q-network
- Experience replay buffer (10,000 transitions)
- Epsilon-greedy exploration with decay (1.0 → 0.01)
- Frame preprocessing: resized to 128×128, normalized to [0,1]

## Environment
- MsPacmanNoFrameskip-v0 (Gymnasium / Arcade Learning Environment)
- State space: raw RGB game frames (210×160×3)
- Action space: 9 discrete actions

## Training Details
- Optimizer: Adam (lr = 5e-4)
- Discount factor (γ): 0.99
- Minibatch size: 64
- Max episodes: 2000
- Convergence target: average score ≥ 500 over 100 episodes

> Note: Full training requires several hours on a GPU. 
> Due to Colab compute limits, the agent was not run 
> to full convergence. The complete training pipeline 
> is implemented and functional.

## Concepts Applied
Deep Q-Learning, CNN, experience replay, 
target network, epsilon-greedy policy, frame preprocessing

## References
- Gymnasium ALE documentation
