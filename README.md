# Co-Design Optimization for Robotic Grasping

A modular framework for hardware-control co-optimization. This project utilizes **Optuna** for hardware parameter evolution and **Stable Baselines3 (PPO)** for learning robust control policies in a **PyBullet** physics environment.

## 📌 Project Overview

This repository implements a "Hardware-in-the-loop" optimization cycle:
1.  **Hardware Evolution**: Optuna samples structural parameters (link lengths, widths, etc.).
2.  **Policy Learning**: Proximal Policy Optimization (PPO) trains a controller for the specific hardware.
3.  **Performance Feedback**: The resulting reward is fed back to Optuna to guide the next hardware iteration.
<img src="assets/codesign_framework.png" width="500">

[Video Link](https://youtu.be/NjFz9qAf6IE)
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

## 📝 Q&A Section (Technical Discussion)

This section summarizes key technical points discussed during the project review.

### Q1: Which Reinforcement Learning model is used in this framework?
**Answer:** We utilize **PPO (Proximal Policy Optimization)** from the [Stable Baselines3](https://stable-baselines3.readthedocs.io/) library. PPO was chosen for its balance between ease of tuning, sample efficiency, and stability in continuous action spaces common in robotics.

* **Code Reference:** See the model initialization in [`train.py`](./train.py) (where `PPO("MlpPolicy", ...)` is defined).


### Q2: What parameters are configured for the Bayesian optimization?
**Answer:** The co-optimization uses **Optuna** (which implements the **TPE - Tree-structured Parzen Estimator** algorithm). Key configurations include:
* **Search Space**: Defined for link lengths ($l_1, l_2$), joint width ($w$), and end-effector tilt angle ($\beta$).
* **Local Refinement**: A `sigma` factor is used to narrow the search range around the best-known parameters to perform fine-tuning.

* **Code Reference:** See the `objective` function and `trial.suggest_float` in [`train.py`](./train.py).


### Q3: How does link width affect momentum, and how is this handled under position control?
**Answer:** In our co-optimization framework, link width ($w$) is a key morphological parameter that dictates the system's dynamic behavior:

1.  **Width-to-Mass Mapping**: In the URDF generation, the mass ($m$) and inertia tensor ($I$) of each link are calculated based on the width. Since we assume constant density, increasing the width directly increases the link's volume and thus its mass.
2.  **PD Control Interaction**: PyBullet's position control is implemented via internal PD regulators. When a wider link is used, the controller must exert higher torques to compensate for the increased inertia. This results in a much more aggressive "smacking" effect upon contact with the ball, as the kinetic energy is higher.
3.  **Hardware-Control Trade-off**: This is the core of our Co-Design: the RL agent must learn to slow down its approach (reduce $v$) when the BO chooses a wider morphology to avoid the "smacking" penalty, or the BO must find the optimal width that balances structural stability with manageable momentum.

* **Code Reference:** * Mass/Inertia Update: See [`CoDesignEnv.py`](./CoDesignEnv.py) where the link properties are defined.
    * Position Control: See `p.setJointMotorControl2` in [`CoDesignEnv.py`](./CoDesignEnv.py).
      
### Q4: How is the Reward Function defined for the RL agent?
**Answer:** The reward function is a multi-objective weighted sum designed to balance task completion, efficiency, and safety. It consists of the following components:
1.  **Task Progress (Distance)**: A dense reward based on the Euclidean distance between the ball and the target goal.
2.  **Impact Penalty (Anti-Smacking)**: To prevent aggressive striking, we penalize excessive contact forces.
3.  **Control Effort (Action Regularization)**: To ensure smooth motion and energy efficiency, we penalize large or sudden joint torques and arbitrary swinging.
4.  **Success Bonus**: A sparse positive reward granted only when the ball successfully reaches the target zone.

* **Code Reference:** Detailed logic can be found in the `compute_reward()` or `step()` function in [`CoDesignEnv.py`](./CoDesignEnv.py).


### Q5: What specific metric is passed to the Bayesian Optimizer (BO)?
**Answer:** We pass the **Mean Evaluation Reward** to the Bayesian Optimizer (Optuna). 
Since Reinforcement Learning is inherently stochastic (due to random initial states and policy noise), a single episode's reward can be noisy. To provide a stable "score" for the hardware morphology:
* After training, we run **5 evaluation episodes** using the deterministic policy.
* The **average (mean)** of these rewards is returned to the BO as the objective value.
* This ensures that the BO optimizes for **consistent hardware performance** rather than a "lucky" single trial.

* **Code Reference:** See the evaluation loop and `mean_reward` calculation in [`train.py`](./train.py).

