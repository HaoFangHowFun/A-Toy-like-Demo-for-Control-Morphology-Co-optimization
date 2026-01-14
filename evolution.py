#This file is used to plot the results of morphological evolution
#Author: Hao-Fang Cheng  
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



study = optuna.load_study(study_name="robot_codesign_study", storage="sqlite:///codesign.db")
df = study.trials_dataframe()

plot_df = df[['value', 'params_l1', 'params_l2', 'params_beta', 'params_w']].copy()
plot_df.columns = ['Reward', 'L1', 'L2', 'Beta', 'Width']
plot_df = plot_df.dropna() 

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']

plt.figure(figsize=(10, 7))
scatter = plt.scatter(plot_df['L1'], plot_df['L2'], c=plot_df['Reward'], 
                      cmap='viridis', s=100, edgecolors='w', alpha=0.8)
plt.colorbar(scatter, label='Mean Reward')
plt.xlabel('Link 1 Length (m)',fontsize=24)
plt.ylabel('Link 2 Length (m)',fontsize=24)
plt.savefig('plot_1_l1_l2_reward.png', dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
sns.regplot(data=plot_df, x='Beta', y='Reward', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('Bending Angle (Beta, rad)',fontsize=24)
plt.ylabel('Mean Reward',fontsize=24)
plt.savefig('plot_2_beta_reward.png', dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
sns.regplot(data=plot_df, x='Width', y='Reward', scatter_kws={'alpha':0.5, 'color':'green'}, line_kws={'color':'orange'})
#plt.title('Impact of Link Thickness (Width) on Reward')
plt.xlabel('Link Width (w, m)',fontsize=24)
plt.ylabel('Mean Reward',fontsize=24)
plt.savefig('plot_3_width_reward.png', dpi=300)
plt.show()