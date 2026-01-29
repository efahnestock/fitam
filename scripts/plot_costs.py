import json
import glob
import os
import matplotlib.pyplot as plt

BASELINE_ROOT = "results/test3/baseline"#"results/evaluations/all_test/baseline"
DIFFUSION_ROOT = "results/evaluations/all_test/diffusion"
UPPERBOUND_ROOT= "results/evaluations/all_test/perfect_vision"

baseline_costs = []
diffusion_costs = []
upperbound_costs = []
per_map_diffs = []

# find all map_* folders that exist in both runs
baseline_maps = sorted(glob.glob(os.path.join(BASELINE_ROOT, "map_*")))

for baseline_map in baseline_maps:
    map_name = os.path.basename(baseline_map)
    diffusion_map = os.path.join(DIFFUSION_ROOT, map_name)
    upperbound_map = os.path.join(UPPERBOUND_ROOT, map_name)

    if not os.path.isdir(diffusion_map):
        continue
    #if not os.path.isdir(upperbound_map):
     #   continue

    baseline_trial = os.path.join(
        baseline_map, "0000000", "trial_info.json"
    )
    diffusion_trial = os.path.join(
        diffusion_map, "0000000", "trial_info.json"
    )
    #    upperbound_trial = os.path.join(
     #   upperbound_map, "0000005", "trial_info.json"
    #)

    if not (os.path.isfile(baseline_trial) and os.path.isfile(diffusion_trial)):
        continue

    with open(baseline_trial, "r") as f:
        baseline_data = json.load(f)

    with open(diffusion_trial, "r") as f:
        diffusion_data = json.load(f)

    # optionally skip failed trials
    if not (baseline_data["reached_goal"] and diffusion_data["reached_goal"]):
        continue

    baseline_costs.append(baseline_data["total_cost"])
    diffusion_costs.append(diffusion_data["total_cost"])

    b_cost = baseline_data["total_cost"]
    u_cost = diffusion_data["total_cost"]
    per_map_diffs.append({
        "map": map_name,
        "baseline": b_cost,
        "diffusion": u_cost,
        "diff": b_cost - u_cost
    })
print(f"Loaded {len(baseline_costs)} paired trials")

top5 = sorted(
    per_map_diffs,
    key=lambda x: abs(x["diff"]),
    reverse=True
)[:5]

for i, entry in enumerate(top5, 1):
    print(
        f"{i}. {entry['map']}: "
        f"baseline={entry['baseline']:.2f}, "
        f"diffusion={entry['diffusion']:.2f}, "
        f"diff={entry['diff']:.2f}"
    )

# plotting
plt.figure()
plt.scatter(baseline_costs, diffusion_costs)

# y = x line
min_cost = min(baseline_costs + diffusion_costs)
max_cost = max(baseline_costs + diffusion_costs)
plt.plot([min_cost, max_cost], [min_cost, max_cost])

plt.xlabel("Baseline cost")
plt.ylabel("Diffusion cost")
plt.title("Baseline vs Diffusion Cost")
out_path = "results/evaluations/all_test/baseline_vs_diffusion.png"
plt.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
plt.tight_layout()
plt.show()
#    upperbound_trial = os.path.join(
#        upperbound_map, "0000005", "trial_info.json"
#    )

#    if not (os.path.isfile(baseline_trial) and os.path.isfile(upperbound_trial)):
#        continue
#
#    with open(baseline_trial, "r") as f:
#        baseline_data = json.load(f)
#
#    with open(upperbound_trial, "r") as f:
#        upperbound_data = json.load(f)
#
#    # optionally skip failed trials
#    if not (baseline_data["reached_goal"] and upperbound_data["reached_goal"]):
#        continue
#
#    baseline_costs.append(baseline_data["total_cost"])
#    upperbound_costs.append(upperbound_data["total_cost"])
#    
#    b_cost = baseline_data["total_cost"]
#    u_cost = upperbound_data["total_cost"]
#    per_map_diffs.append({
#        "map": map_name,
#        "baseline": b_cost,
#        "upperbound": u_cost,
#        "diff": b_cost - u_cost
#    })
#print(f"Loaded {len(baseline_costs)} paired trials")
#
#top5 = sorted(
#    per_map_diffs,
#    key=lambda x: abs(x["diff"]),
#    reverse=True
#)[:5]
#
#for i, entry in enumerate(top5, 1):
#    print(
#        f"{i}. {entry['map']}: "
#        f"baseline={entry['baseline']:.2f}, "
#        f"upperbound={entry['upperbound']:.2f}, "
#        f"diff={entry['diff']:.2f}"
#    )
#
## plotting
#plt.figure()
#plt.scatter(baseline_costs, upperbound_costs)
#
## y = x line
#min_cost = min(baseline_costs + upperbound_costs)
#max_cost = max(baseline_costs + upperbound_costs)
#plt.plot([min_cost, max_cost], [min_cost, max_cost])
#
#plt.xlabel("Baseline cost")
#plt.ylabel("Upper bound cost")
#plt.title("Baseline vs Upper Bound Cost")
#out_path = "results/evaluations/all_test/baseline_vs_upperbound.png"
#plt.savefig(out_path, dpi=200)
#print(f"Saved plot to {out_path}")
#plt.tight_layout()
#plt.show()
#
