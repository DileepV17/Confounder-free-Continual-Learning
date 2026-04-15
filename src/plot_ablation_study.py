import matplotlib.pyplot as plt
import numpy as np

# 1. Define the Metrics
labels = ['ACCd\n(Accuracy Deviation)', 'BWTd\n(Catastrophic Forgetting)', 'FWTd\n(Forward Transfer)']
x = np.arange(len(labels))

# 2. Data Arrays (Means)
means_base_no_sched = [0.1286, -0.0503, 0.1711]
means_base_sched    = [0.2500,  0.0000, 0.2500] # Collapsed to 50%
means_ens_no_sched  = [0.1228, -0.0442, 0.1570] # Single seed
means_ens_sched     = [0.1225, -0.0694, 0.1980]
means_2v_no_sched   = [0.1759,  0.0228, 0.1844]
means_2v_sched      = [0.1928,  0.0288, 0.1747]
means_rmdn          = [0.3296,  0.0968, 0.2707] # Authors' Baseline

# 3. Data Arrays (Standard Deviations)
std_base_no_sched = [0.0020, 0.0038, 0.0018]
std_base_sched    = [0.0000, 0.0000, 0.0000]
std_ens_no_sched  = [0.0000, 0.0000, 0.0000] # No std dev provided for 1 seed
std_ens_sched     = [0.0819, 0.1042, 0.0199]
std_2v_no_sched   = [0.0344, 0.0782, 0.0554]
std_2v_sched      = [0.0464, 0.0274, 0.0101]
std_rmdn          = [0.0122, 0.0203, 0.0037]

# Grouping for easy plotting
all_means = [means_base_no_sched, means_base_sched, means_ens_no_sched, means_ens_sched, 
             means_2v_no_sched, means_2v_sched, means_rmdn]
all_stds = [std_base_no_sched, std_base_sched, std_ens_no_sched, std_ens_sched, 
            std_2v_no_sched, std_2v_sched, std_rmdn]
model_names = [
    'Baseline (No Sched)', 'Baseline (Sched - Collapsed)', 
    'Ensemble 1V (No Sched)', 'Ensemble 1V (Sched)', 
    'Ensemble 2V (No Sched)', 'Ensemble 2V (Sched)', 
    "Authors' R-MDN"
]
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728']

# 4. Plot Setup
fig, ax = plt.subplots(figsize=(14, 7))
width = 0.11 # Width of each bar
offsets = np.linspace(-3*width, 3*width, 7) # Calculate offsets for 7 bars

# 5. Draw Bars
for i in range(7):
    ax.bar(x + offsets[i], all_means[i], width, yerr=all_stds[i], 
           label=model_names[i], color=colors[i], capsize=4, alpha=0.9, edgecolor='black', linewidth=0.5)

# 6. Formatting
ax.set_ylabel('Deviation Score (Closer to 0 is Better)', fontsize=13, fontweight='bold')
ax.set_title('Comprehensive Ablation Study: Diversity Ensembles vs R-MDN\n(Lower Values Indicate Better Performance)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.axhline(0, color='black', linewidth=1.5, linestyle='--')

# Put legend outside the plot to avoid overlapping bars
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
plt.savefig('full_ablation_study.png', dpi=300, bbox_inches='tight')
print("Successfully generated 'full_ablation_study.png'")