# This file contains the training process of the co-optimization.
# Author: Hao-Fang Cheng  
import os
import optuna
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from CoDesignEnv import CoDesignGraspEnv

def objective(trial):
    try:
        if trial.study.best_value >10000:
            sigma = 0.3
        else:
            sigma = 0.3
    except:
            sigma = 0.3
    
    try:
        # Get best parameters so far
        best_params = trial.study.best_params
        
        def get_range(name, abs_low, abs_high):
            center = best_params[name]
            span = (abs_high - abs_low) * sigma
            new_low = np.clip(center - span, abs_low, abs_high - 0.001)
            new_high = np.clip(center + span, new_low + 0.001, abs_high)
            return float(new_low), float(new_high)

        l1_range = get_range("l1", 0.2, 0.5)
        l2_range = get_range("l2", 0.2, 0.5)
        w_range = get_range("w", 0.01, 0.06)
        beta_range = get_range("beta", 0.0, 1.2)
        print(f" Trial {trial.number}: Local Refinement Mode (Sigma={sigma})")
        
    except (ValueError, KeyError):
        # Global search for initial stage
        l1_range, l2_range = (0.2, 0.6), (0.2, 0.5)
        w_range, beta_range = (0.01, 0.06), (0.0, 1.2)
        print(f" Trial {trial.number}: Global Searching...")

    # Sample parameters
    l1 = trial.suggest_float("l1", *l1_range)
    l2 = trial.suggest_float("l2", *l2_range)
    w = trial.suggest_float("w", *w_range)
    beta = trial.suggest_float("beta", *beta_range)
    
    current_hw = {"l1": l1, "l2": l2, "w": w, "beta": beta, "kp": 0.03, "kd": 0.5}

    # Env init (DIRECT mode)
    raw_env = CoDesignGraspEnv(render_mode=None, hw_params=current_hw)
    env = Monitor(raw_env)
    env = DummyVecEnv([lambda: env])

    model_path = "best_model_checkpoint.zip"
    
    if os.path.exists(model_path):
        print(f"Trial {trial.number}: Loading best weights for fine-tuning...")
        model = PPO.load(model_path, env=env, device="cpu")
        model.learning_rate = 1e-4  
    else:
        print(f"Trial {trial.number}: Training from scratch...")
        model = PPO("MlpPolicy", env, verbose=0, device="cpu", n_steps=1024, learning_rate=3e-4)

    # Training
    try:
        if trial.study.best_value >10000:
            model.learn(total_timesteps=40000)
        else:
            model.learn(total_timesteps=15000)
    except:
            model.learn(total_timesteps=15000)


    # Evaluation 
    eval_rewards = []
    for _ in range(5):
        obs, _ = raw_env.reset()
        done = False
        truncated = False
        ep_rew = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = raw_env.step(action)
            ep_rew += reward
        eval_rewards.append(ep_rew)

    mean_reward = np.mean(eval_rewards)
     
    video_dir = "best_videos"
    os.makedirs(video_dir, exist_ok=True)

    # Check for new record
    try:
        current_best = trial.study.best_value
    except ValueError:
        current_best = -np.inf

    if mean_reward > current_best:
        model.save("best_model_checkpoint")
        print(f"Trial {trial.number} New Record: {mean_reward:.2f}. Weights updated.")

        # Record demo video
        print(f"Recording demo video for Trial {trial.number}...")

        # Disconnect DIRECT mode
        raw_env.close() 
        # Release PyBullet connection
        time.sleep(0.5)
        
        record_env = CoDesignGraspEnv(render_mode="human", hw_params=current_hw)
        video_path = os.path.join(video_dir, f"best_trial_{trial.number:03d}_score_{mean_reward:.0f}.mp4")
        
        # Run one episode for recording
        obs, _ = record_env.reset()
        time.sleep(1)

        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
        
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = record_env.step(action)
            
        p.stopStateLogging(log_id)
        record_env.close()
        print(f"Video saved to: {video_path}")

    raw_env.close() 

    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="robot_codesign_study",
        direction="maximize",
        storage="sqlite:///codesign.db",
        load_if_exists=True
    )

    # Initial test with specific params
    difficult_initial_params = {"l1": 0.2, "l2": 0.2, "w": 0.02, "beta": 0.0}
    
    if len(study.trials) == 0:
        study.enqueue_trial(difficult_initial_params)
        print(f"Enqueued initial trial: {difficult_initial_params}")

    print("Starting Co-Design optimization...")
    study.optimize(objective, n_trials=100)

    # Results
    print("\n" + "="*30)
    print("Optimization Complete!")