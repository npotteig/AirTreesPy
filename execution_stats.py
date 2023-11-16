import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd


rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
plt.rcParams['font.size'] = 12

baseline_df = pd.read_csv('navigation/paper_data/eval_results/evaluation_ansr_fps_lds_transfer.csv')


print("Collisions:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(baseline_df['collisions']), np.median(baseline_df['collisions']), np.max(baseline_df['collisions']),
                                                                                np.min(baseline_df['collisions']), np.std(baseline_df['collisions'])))
print("Goals achieved: {:.1f}%".format(100*np.mean(baseline_df['goal_success'])))
print("Steps to finish:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(baseline_df['step_count']), np.median(baseline_df['step_count']), np.max(baseline_df['step_count']),
                                                                        np.min(baseline_df['step_count']), np.std(baseline_df['step_count'])))

# print("Collisions:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(baseline_safe_df['collisions']), np.median(baseline_safe_df['collisions']), np.max(baseline_safe_df['collisions']),
#                                                                                 np.min(baseline_safe_df['collisions']), np.std(baseline_safe_df['collisions'])))
# print("Goals achieved: {:.1f}%".format(100*np.mean(baseline_safe_df['goal_success'])))
# print("Steps to finish:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(baseline_safe_df['step_count']), np.median(baseline_safe_df['step_count']), np.max(baseline_safe_df['step_count']),
#                                                                         np.min(baseline_safe_df['step_count']), np.std(baseline_safe_df['step_count'])))

# print("Collisions:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(ebt_df['collisions']), np.median(ebt_df['collisions']), np.max(ebt_df['collisions']),
#                                                                                 np.min(ebt_df['collisions']), np.std(ebt_df['collisions'])))
# print("Goals achieved: {:.1f}%".format(100*np.mean(ebt_df['goal_success'])))
# print("Steps to finish:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(ebt_df['step_count']), np.median(ebt_df['step_count']), np.max(ebt_df['step_count']),
#                                                                         np.min(ebt_df['step_count']), np.std(ebt_df['step_count'])))
