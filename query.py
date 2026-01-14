# This script retrieves and analyzes specific trial data from the Optuna study database.
# Author: Hao-Fang Cheng  
import optuna

# 1. Define storage and study name
storage_path = "sqlite:///codesign.db"
study_name = "robot_codesign_study"

# 2. Load the study
study = optuna.load_study(study_name=study_name, storage=storage_path)

# 3. Specify target trial IDs to inspect
target_trials = [39]

print(f"{'Trial':<10} | {'Reward':<10} | {'Parameters'}")
print("-" * 60)

for trial_no in target_trials:
    # Fetch trial data by index
    trial = study.trials[trial_no]
    
    # Check if the trial was successful
    if trial.state == optuna.trial.TrialState.COMPLETE:
        reward = trial.value
        params = trial.params
        print(f"{trial_no:<10} | {reward:<10.2f} | {params}")
    else:
        print(f"{trial_no:<10} | Incomplete (State: {trial.state})")