# DDQN-in-python-maze
Tutorial series on how to implement DQN with PyTorch from scratch.
DDQN-in-Python-Maze
This project implements a Double Deep Q-Network (DDQN) to solve randomly generated mazes using Python. The agent is trained to navigate from a starting position to a goal within the maze, learning optimal paths through reinforcement learning techniques.

Features
Random Maze Generation: Each maze is generated randomly, providing unique challenges for the agent in every training session.
Double Deep Q-Network (DDQN): Utilizes DDQN to stabilize training by decoupling action selection from evaluation, addressing the overestimation bias commonly found in standard DQNs.
Customizable Parameters: Allows adjustment of maze size, training episodes, learning rates, and other hyperparameters to experiment with different configurations.
