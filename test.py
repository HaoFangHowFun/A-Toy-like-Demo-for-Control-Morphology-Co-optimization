# This script visualizes and evaluates a specific hardware configuration in a GUI environment.
# Author: Hao-Fang Cheng  
import time
from stable_baselines3 import PPO
from CoDesignEnv import CoDesignGraspEnv

def test_manual():
    # 1. Hardware parameters from a specific trial (e.g., Trial 6) or the best study result
    best_hw = {
        'l1':  0.3530200740699877, 
        'l2':  0.4266292906115354, 
        'beta': 0.2874,
        'w':  0.014858690421539509, 
        'kp': 0.03, 
        'kd': 0.5
    }

    # 2. Initialize environment with GUI enabled
    env = CoDesignGraspEnv(render_mode="human", hw_params=best_hw)
    
    # 3. Load the pre-trained model; fallback to random actions if no model exists
    try:
        model = PPO.load("best_model_checkpoint_of_all_time") 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model not found, using random actions. Error: {e}")
        model = None
    
    obs, _ = env.reset()
    print(f"Testing hardware design: {best_hw}")

    for _ in range(1000):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample() 
            
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            time.sleep(1) # Brief pause after episode ends
            obs, _ = env.reset()

if __name__ == "__main__":
    test_manual()