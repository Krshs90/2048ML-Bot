# 2048 Machine Learning Bot

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Playwright](https://img.shields.io/badge/Playwright-Enabled-brightgreen.svg)
![Numba](https://img.shields.io/badge/Numba-JIT-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An advanced, high-performance Reinforcement Learning agent designed to autonomously play and master the game 2048. Utilizing Temporal Difference Learning (TD(0)) and Symmetric N-Tuple Networks, this bot is capable of achieving superhuman scores.

## Live Demo

https://youtu.be/0cvE0muE-vo

## Overview

This project implements a headless training environment to build a massive 16.7 million state neural network, alongside three distinct browser automation scripts to deploy the trained agent in real-time. The system evaluates the board across all 8 mathematical symmetries simultaneously, allowing it to learn at an accelerated rate.

## Execution Modes

The project is split into modular scripts to accommodate different use cases:

- `train.py`: The multithreaded training environment. Run this to train the model. The weights are continuously saved to `weights.npy`. You can adjust the speed using `--speed 1/2/3`.

- `bot.py`: Target mode. Automatically clicks "Try Again" and loops infinitely until a specific target tile is reached. Defaults to 8192, but can be customized (e.g., `python bot.py 16384`).


## Setup and Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install numpy numba playwright
   playwright install chromium
   ```
3. Run `python train.py --speed 3` to begin building the neural network weights.
4. Once satisfied with the training progress, execute any of the bot scripts to watch the agent play.

## Credits

Developed by Krshs90
