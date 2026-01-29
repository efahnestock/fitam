import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from fitam.mapping.land_cover_complex_map import semantic_class_to_color_map
from fitam.mapping.land_cover_complex_map import semantic_class_to_occ_grid_cost

def plot_semantic_legend(save_path=None):
    patches = []

    for cls, color in semantic_class_to_color_map.items():
        cost = semantic_class_to_occ_grid_cost.get(cls, None)

        if cost is None:
            label = f"{cls}"
        elif np.isinf(cost):
            label = f"{cls} (blocked)"
        elif cost < 0:
            label = f"{cls} (unknown)"
        else:
            label = f"{cls} ({cost:.2f} s/m)"

        patches.append(mpatches.Patch(color=color, label=label))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    ax.legend(
        handles=patches,
        loc="center",
        frameon=True,
        ncol=1
    )

    ax.set_title("Semantic Class Legend", fontsize=14)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Legend saved to {save_path}")

    plt.show()


# usage
plot_semantic_legend("semantic_legend.png")

