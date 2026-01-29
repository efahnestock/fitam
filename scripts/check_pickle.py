import lzma
import pickle
import numpy as np
import matplotlib.pyplot as plt
import inspect
from IPython import embed

path = "results/test3/baseline2/map_008_anne_24003_lulc_2018/0000000/debug_info.pkl"  # adjust name
#path = "results/test3/upperbound/map_008_anne_24003_lulc_2018/0000000/local_costmap/0000099.xz"

with open(path, "rb") as f:
    dts = pickle.load(f)

embed()

with lzma.open(path, "rb") as f:
    #data = pickle.load(f)
    belief = pickle.load(f)

print(type(belief))
print(belief.keys() if isinstance(belief, dict) else belief)

print("Belief type:", type(belief))
print("\nBelief attributes:")
print([a for a in dir(belief) if not a.startswith("_")])

costmap = belief.get_full_costmap()   # numpy array

print("Costmap shape:", costmap.shape)
print("Min / Max:", costmap.min(), costmap.max())

plt.figure(figsize=(6,6))
plt.imshow(costmap, origin="lower", cmap="inferno")
plt.colorbar(label="Cost")
plt.gca().invert_yaxis()
plt.title("Planner Costmap (from Belief)")

out_path = "results/test3/baseline2/grid_check.png"
plt.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
plt.tight_layout()
plt.show()

#if hasattr(belief, "occupancy_grid"):
#    grid = belief.occupancy_grid
#elif hasattr(belief, "grid"):
#    grid = belief.grid
#else:
#    raise RuntimeError("Could not find occupancy grid in belief")
#
#print("\nGrid type:", type(grid))
#print("\nGrid attributes:")
#print([a for a in dir(grid) if not a.startswith("_")])
#
#candidates = ["values", "data", "grid", "log_odds", "probabilities"]
#
#for name in candidates:
#    if hasattr(grid, name):
#        arr = getattr(grid, name)
#        print(f"Found grid array: {name}, shape = {np.array(arr).shape}")
