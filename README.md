# Co-Design Optimization for Robotic Grasping

A modular framework for hardware-control co-optimization. This project utilizes **Optuna** for hardware parameter evolution and **Stable Baselines3 (PPO)** for learning robust control policies in a **PyBullet** physics environment.

## 📌 Project Overview

This repository implements a "Hardware-in-the-loop" optimization cycle:
1.  **Hardware Evolution**: Optuna samples structural parameters (link lengths, widths, etc.).
2.  **Policy Learning**: Proximal Policy Optimization (PPO) trains a controller for the specific hardware.
3.  **Performance Feedback**: The resulting reward is fed back to Optuna to guide the next hardware iteration.

![Framwork Design Layout](./assets/codesign_framwork.png)

## Video Link
[Watch the video](https://youtu.be/NjFz9qAf6IE)
---

## 🛠 Installation

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

```bash
# Clone the repository
git clone https://github.com/HaoFangHowFun/A-Toy-like-Demo-for-Control-Morphology-Co-optimization.git
cd A-Toy-like-Demo-for-Control-Morphology-Co-optimization

# Create environment from the provided YAML file
conda env create -f co_design_env.yml

# Activate the environment
conda activate co_design_env

```

---

## 🚀 Usage & Expected Outcomes

### 1. Training & Optimization

Run the main co-design loop.

```bash
python train.py

```

* **Outcome**:
* Populates `codesign.db` with trial data.
* Saves the best-performing model to `best_model_checkpoint.zip`.
* Generates demo videos in the `best_videos/` directory.



### 2. Result Retrieval

Analyze specific trial results from the SQLite database.

```bash
python query.py

```

* **Outcome**: Displays a summary table of **Trial ID**, **Reward**, and **Hardware Configurations** (e.g., link lengths , joint friction , etc.).

### 3. Visual Validation

Launch a GUI simulation to observe the robot's performance with specific parameters.

```bash
python test.py

```

* **Outcome**: Opens a **PyBullet GUI** window showing the optimized robot performing grasping tasks in real-time.

---

## 🧪 Key Features

* **Dynamic Environment**: Supports real-time hardware reconfiguration within PyBullet.
* **Checkpointing**: Saves the best-performing policy and hardware metadata for reproducibility.
* **Visualization**: Automatic video logging of new performance records during training.



