import pickle
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# SETTINGS
# ----------------------------
BASELINE_PKL = "results/test3/baseline/map_007_amhe_51009_lulc_2018/0000000/debug_info.pkl"
UPPERBOUND_PKL = "results/test3/upperbound/map_007_amhe_51009_lulc_2018/0000000/debug_info.pkl"  # path to upperbound trial .pkl
DIFFUSION_PKL = "results/evaluations/all_test/diffusion/map_007_amhe_51009_lulc_2018/0000000/debug_info.pkl"
DIFF_THRESHOLD = 125
OUTPUT_PLOT = "results/test3/cumulative_cost_comparison_map7.png"

# ----------------------------
# LOAD TRIAL DATA
# ----------------------------
def load_trial(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

baseline_data = load_trial(BASELINE_PKL)
upper_data = load_trial(UPPERBOUND_PKL)
diff_data = load_trial(DIFFUSION_PKL)

baseline_cost = np.array(baseline_data['cost_history'])
upper_cost = np.array(upper_data['cost_history'])
diff_cost = np.array(diff_data['cost_history'])

# ----------------------------
# CUMULATIVE COST
# ----------------------------
cum_baseline = np.cumsum(baseline_cost)
cum_upper = np.cumsum(upper_cost)
cum_diff = np.cumsum(diff_cost)
print(cum_diff[-1],len(cum_diff))
print(cum_baseline[-1], len(cum_baseline))

# ----------------------------
# FIND DIVERGENCE STEP
# ----------------------------
#diff = np.abs(cum_baseline - cum_upper)
#divergence_steps = np.where(diff > DIFF_THRESHOLD)[0]
T = min(len(cum_diff), len(cum_baseline), len(cum_upper))
#cum_baseline = cum_baseline[:T]
#cum_upper = cum_upper[:T]
#cum_diff = cum_diff[:T]

diff = np.abs(cum_baseline[:T] - cum_upper[:T])
divergence_steps = np.where(diff > DIFF_THRESHOLD)[0]
diff_d_b = np.abs(cum_baseline[:T] - cum_diff[:T])
div_steps_d_b = np.where(diff_d_b > DIFF_THRESHOLD)[0]
#diff_d_u = np.abs(cum_upper - cum_diff)
#div_steps_d_u = np.where(diff_d_u > DIFF_THRESHOLD)[0]

if len(divergence_steps) > 0 or len(div_steps_d_b) > 0:# or len(div_steps_d_u) > 0:
    if len(divergence_steps) > 0:
        print(f"DIVERGENCE (baseline, upper) detected at step {divergence_steps[0]} (diff={diff[divergence_steps[0]]:.6f})")
    if len(div_steps_d_b) > 0:
        print(f"DIVERGENCE (baseline, diffusion) detected at step {div_steps_d_b[0]} (diff={diff[div_steps_d_b[0]]:.6f})")
 #   print(f"DIVERGENCE (upper, diffusion) detected at step {div_steps_d_u[0]} (diff={diff[div_steps_d_u[0]]:.6f})")
else:
    print("No divergence detected.")

# ----------------------------
# PLOT CUMULATIVE COST
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(cum_diff, label="Diffusion", linewidth=2)
plt.plot(cum_baseline, label="Baseline", linewidth=2)
plt.plot(cum_upper, label="Upperbound", linewidth=2)
plt.xlabel("Step #")
plt.ylabel("Cumulative Cost")
plt.title("Cumulative Cost Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()
print(f"Saved plot to {OUTPUT_PLOT}")

# state_transition_indexes: cumulative motion steps after each replan
# Example: [12, 27, 41, 60, ...]
transitions = np.array(upper_data["state_transition_indexes"])
# find the first replan index whose cumulative steps exceed diverge_step
replan_index = np.searchsorted(transitions, divergence_steps[0], side="right")

transitions_baseline = np.array(baseline_data["state_transition_indexes"])
# find the first replan index whose cumulative steps exceed diverge_step
replan_index_baseline = np.searchsorted(transitions_baseline, div_steps_d_b[0], side="right")

transitions_diff = np.array(diff_data["state_transition_indexes"])
# find the first replan index whose cumulative steps exceed diverge_step
replan_index_diff = np.searchsorted(transitions_diff, div_steps_d_b[0], side="right")

print(replan_index, replan_index_baseline, replan_index_diff)

#print(f"Divergence at motion step {diverge_step}")
#print(f"Corresponding replan index: {replan_index}")
#print(f"Local costmap file: {replan_index:07d}.png")
