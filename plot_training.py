import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rc


import pandas as pd



rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
plt.rcParams['font.size'] = 24

baseline = pd.read_csv('navigation/paper_data/baseline/run0/results/AirSimEnv-v0_higl_2.csv')
baseline_safe = pd.read_csv('navigation/paper_data/safe_deploy_only/run0/results/AirSimEnv-v0_higl_2.csv')
ebt = pd.read_csv('navigation/paper_data/safe_layer/run0/results/AirSimEnv-v0_higl_2.csv')

reward_or_collision = 'reward'

baseline_df = np.array(baseline[reward_or_collision].ewm(span=10, adjust=False).mean())
baseline_safe_df = np.array(baseline_safe[reward_or_collision].ewm(span=10, adjust=False).mean())
ebt_df = np.array(ebt[reward_or_collision].ewm(span=10, adjust=False).mean())
for i in range(1, 4):
    tmp_baseline = pd.read_csv('navigation/paper_data/baseline/run' + str(i) +'/results/AirSimEnv-v0_higl_2.csv')
    baseline_df = np.vstack((baseline_df, np.array(tmp_baseline[reward_or_collision].ewm(span=10, adjust=False).mean())))

    tmp_baseline_safe = pd.read_csv('navigation/paper_data/safe_deploy_only/run' + str(i) + '/results/AirSimEnv-v0_higl_2.csv')
    baseline_safe_df = np.vstack((baseline_safe_df, np.array(tmp_baseline_safe[reward_or_collision].ewm(span=10, adjust=False).mean())))

    tmp_ebt = pd.read_csv('navigation/paper_data/safe_layer/run' + str(i) + '/results/AirSimEnv-v0_higl_2.csv')
    ebt_df = np.vstack(
        (ebt_df, np.array(tmp_ebt[reward_or_collision].ewm(span=10, adjust=False).mean())))


baseline_mean = np.mean(baseline_df, axis=0)
baseline_std = np.std(baseline_df, axis=0)

baseline_safe_mean = np.mean(baseline_safe_df, axis=0)
baseline_safe_std = np.std(baseline_safe_df, axis=0)

# print(ebt_df)
ebt_mean = np.mean(ebt_df, axis=0)
# print(ebt_mean)
ebt_std = np.std(ebt_df, axis=0)

x = 5e3 * np.arange(baseline_mean.shape[0])
y = np.ones(100) * 500

fig = plt.figure(figsize=(12, 8))
ax = plt.axes()
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_xlim(0, 5e5)
ax.set_ylim(-0.001, 1.01)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Success Rate")
ax.plot(x, ebt_mean, label='Penalty + Safety Layer (Full)')
ax.fill_between(x, (ebt_mean - ebt_std), (ebt_mean + ebt_std), color='blue', alpha=0.1)
ax.plot(x, baseline_safe_mean, label="Penalty Only", alpha=1.0)
ax.fill_between(x, (baseline_safe_mean - baseline_safe_std), (baseline_safe_mean + baseline_safe_std), color='orange', alpha=0.1)
ax.plot(x, baseline_mean, label="Original", alpha=1.0)
ax.fill_between(x, (baseline_mean - baseline_std), (baseline_mean + baseline_std), color='green', alpha=0.1)
ax.legend(loc='lower right')
fig.savefig('figs/training_success_safe.pdf')


# fig = plt.figure(figsize=(12, 8))
# ax = plt.axes()
# ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax.set_xlim(0, 5e5)
# ax.set_ylim(-0.001, 27000)
# ax.set_xlabel("Timesteps")
# ax.set_ylabel("Cumulative Collisions")
# ax.plot(x, ebt_mean)
# ax.fill_between(x, (ebt_mean - ebt_std), (ebt_mean + ebt_std), color='blue', alpha=0.1)
# ax.plot(x, baseline_safe_mean, alpha=1.0)
# ax.fill_between(x, (baseline_safe_mean - baseline_safe_std), (baseline_safe_mean + baseline_safe_std), color='orange', alpha=0.1)
# ax.plot(x, baseline_mean, alpha=1.0)
# ax.fill_between(x, (baseline_mean - baseline_std), (baseline_mean + baseline_std), color='green', alpha=0.1)
# # plt.plot(x, y, '--', color='red', label='Max Violations')
# # ax.legend(loc='upper right')
# fig.savefig('figs/training_collisions_safe.pdf')


plt.show()
