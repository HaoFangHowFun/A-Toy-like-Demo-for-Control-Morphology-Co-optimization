# A Toy-like Demo for Control-Morphology Co-optimization

Co-Design Optimization for Robotic Grasping
This project implements a co-optimization framework that simultaneously evolves robot hardware parameters and control policies using Optuna and Stable Baselines3 (PPO).

🛠 Installation
To set up the environment using Conda:

Bash

# Create environment from file
conda env create -f co-design.yml

# Activate the environment
conda activate your_env_name
🚀 Usage & Expected Outcomes
1. Training (train.py)
Run the co-optimization loop to find the best hardware-control combination.

Command: python train.py

Outcome: * Generates codesign.db (Optuna database).

Saves best_model_checkpoint.zip (Trained PPO weights).

Exports demonstration videos in /best_videos.

2. Data Retrieval (query.py)
Fetch and analyze specific trial results from the database.

Command: python query.py

Outcome: * Displays a formatted table showing Trial ID, Reward, and Hardware Configuration (l1, l2, w, beta).

3. Visualization (test.py)
Validate a specific hardware design in a GUI-based simulation.

Command: python test.py

Outcome: * Launches a PyBullet GUI window.

Renders the robot performing the grasping task with the optimized hardware parameters.
