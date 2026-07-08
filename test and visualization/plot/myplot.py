import json
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from air_corridor.tools.util import generate_hexagon_grid

# ─── Global Style ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# ─── Data Loading ─────────────────────────────────────────────────────────────
with open("test_data_time.json", "r") as f:
    data = json.load(f)

level_set = reversed([20, 21])
model_set = reversed(["3-1", "3-2", "3-3", "3-4", "3-5", "3-6"])
model_dic = {j: i for i, j in enumerate(model_set)}
level_dic = {j: i for i, j in enumerate(level_set)}
agents_dic = {6: 0, 9: 1, 12: 2, 18: 3, 24: 4, 36: 5}

won_matrix = np.zeros([len(model_dic), len(agents_dic), len(level_dic)])
won_rate_matrix = np.zeros([len(model_dic), len(agents_dic), len(level_dic)])
for result in data:
    if result:
        model_key, num_agents, level_key, won_value, won_rate, status_count, ave_won_speed, ave_won_time = result
        won_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_value
        won_rate_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_rate
won_times = won_matrix
won_rate = won_rate_matrix

model_lst = sorted(
    [(key, value) for key, value in model_dic.items() if key not in ["1-1", "1-2"]],
    key=lambda x: x[1], reverse=True,
)

# ─── Lookup Dictionaries ──────────────────────────────────────────────────────
dic_id_name = {
    "1-1": "10-circle", "1-2": "10-grid",
    "2-1": "3e-T6",     "2-2": "3e-F6",
    "2-3": "HD-T6",     "2-4": "HD-F6",
    "3-1": "HD-F",      "3-2": "HD-T",
    "3-3": "HTransRL-F","3-4": "HTransRL-T",
    "3-5": "DS-F",      "3-6": "DS-T",
}
dic_id_marker = {
    "1-1": "o", "1-2": "v",
    "2-1": "s", "2-2": "^",
    "2-3": "p", "2-4": "*",
    "3-1": "h", "3-2": "<",
    "3-3": ">", "3-4": "D",
    "3-5": "x", "3-6": "+",
}
# Improved color palette — consistent, publication-ready
dic_id_color = {
    "1-1": "#FFBE0B", "1-2": "#9E9E9E",
    "2-1": "#8D8D00", "2-2": "#00B4D8",
    "2-3": "#C77DFF", "2-4": "#1B1B1B",
    "3-1": "#E63946",  # HD-F    — vivid red
    "3-2": "#457B9D",  # HD-T    — steel blue
    "3-3": "#2DC653",  # HTransRL-F — emerald green (highlight)
    "3-4": "#F4A261",  # HTransRL-T — warm orange (highlight)
    "3-5": "#7B2D8B",  # DS-F    — purple
    "3-6": "#E76F51",  # DS-T    — burnt sienna
}
dic_level_name = {14: "cttc", 19: "training env", 20: "cttcttc", 21: "cttcttcttc"}

model_fraction = {"3-5": "DS", "3-1": "HD", "3-4": "HTransRL"}


# ─── Helper: annotate line endpoints ─────────────────────────────────────────
def annotate_endpoint(ax, x_vals, y_vals, label, color):
    ax.annotate(
        label, xy=(x_vals[-1], y_vals[-1]),
        xytext=(4, 0), textcoords="offset points",
        color=color, fontsize=7, va="center",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Arrival Rate Comparison (2 levels, side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, tight_layout=True)
for j, level_key in enumerate([20, 21]):
    ax = axs[j]
    for model_id, i in model_lst:
        agents = list(agents_dic.values())
        level_index = level_dic[level_key]
        wr = won_rate[i, agents, level_index]
        lw = 2.2 if model_id in ("3-3", "3-4") else 1.3
        ms = 7 if model_id in ("3-3", "3-4") else 5
        ax.plot(
            list(agents_dic.keys()), wr,
            marker=dic_id_marker[model_id],
            label=dic_id_name[model_id],
            color=dic_id_color[model_id],
            linewidth=lw, markersize=ms,
        )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Number of UAVs\n({chr(ord('a') + j)}) {dic_level_name[level_key]}", labelpad=6)
    ax.set_xticks(list(agents_dic.keys()))
    if j == 0:
        ax.set_ylabel("Arrival Rate")
axs[1].legend(loc="lower left", ncol=2)
fig.suptitle("Arrival Rate vs. Number of UAVs", fontweight="bold", fontsize=11, y=1.01)
fig.savefig("test_2.jpg")
fig.savefig("test_2.pdf")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Hexagon Grid + Arrival Rate (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
circle_radius = 2
vertex_distance = 0.6
vertices = generate_hexagon_grid(circle_radius, vertex_distance)
vertex_x_vals, vertex_y_vals = zip(*vertices)

fig = plt.figure(figsize=(10, 3.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 3, 3], wspace=0.35)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Panel (a) — Hexagon grid
ax0 = axs[0]
ax0.scatter(vertex_x_vals, vertex_y_vals, color="#E63946", s=30, zorder=5, label="Vertices")
circle_patch = plt.Circle((0, 0), circle_radius, color="#457B9D", fill=False, linewidth=1.5)
ax0.add_patch(circle_patch)
ax0.set_xlim(-circle_radius - 0.1, circle_radius + 0.1)
ax0.set_ylim(-circle_radius - 0.1, circle_radius + 0.1)
ax0.set_aspect("equal", adjustable="box")
ax0.set_xlabel("\n\n(a) Hexagonal grid layout", labelpad=6)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.spines[:].set_visible(False)

# Panels (b) & (c)
for j, level_key in enumerate([20, 21], start=1):
    ax = axs[j]
    for model_id, i in model_lst:
        agents = list(agents_dic.values())
        wr = won_rate[i, agents, level_dic[level_key]]
        lw = 2.2 if model_id in ("3-3", "3-4") else 1.3
        ms = 7 if model_id in ("3-3", "3-4") else 5
        ax.plot(
            list(agents_dic.keys()), wr,
            marker=dic_id_marker[model_id],
            label=dic_id_name[model_id],
            color=dic_id_color[model_id],
            linewidth=lw, markersize=ms,
        )
    ax.set_ylim(0.15, 1.05)
    ax.set_xticks(list(agents_dic.keys()))
    ax.set_xlabel(f"Number of UAVs\n({chr(ord('b') + j - 1)}) {dic_level_name[level_key]}", labelpad=6)
    if j == 1:
        ax.set_ylabel("Arrival Rate")
    if j == 1:
        ax.legend(loc="lower left", ncol=1, fontsize=7.5)

fig.suptitle("Hexagonal Grid & Arrival Rate by Scenario", fontweight="bold", fontsize=11, y=1.01)
fig.savefig("test_3.jpg")
fig.savefig("test_3.pdf")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Failure Reason Stacked Bar (normalised %)
# ══════════════════════════════════════════════════════════════════════════════
fail_colors = ["#E63946", "#457B9D", "#F4A261", "#2DC653"]
fail_labels = ["Cross Torus", "Cross Cylinder", "Collision w/ NCFOs", "Collision Among UAVs"]
fail_reason_color = {
    "cross torus": "#E63946",
    "cross cylinder": "#457B9D",
    "collide with NFCOs": "#F4A261",
    "collide among UAVs": "#2DC653",
}


def generate_data(models, agent_key, level_key, unified=True):
    fail_data = []
    for model_key in models:
        row_data = [0] * 4
        try:
            index = next(i for i, row in enumerate(data) if row[:3] == [model_key, agent_key, level_key])
        except StopIteration:
            continue
        for key, value in data[index][5].items():
            if key in {"breached_wall", "breached_rad_t_wall", "breached_rad"}:
                row_data[0] += value
            elif key == "breached_c":
                row_data[1] += value
            elif key == "collided_UAV":
                row_data[2] += value
            elif key == "collided":
                row_data[3] += value
        total = sum(row_data) or 1
        fail_data.append(np.array(row_data) / total if unified else np.array(row_data))
    return np.transpose(np.array(fail_data))


settings = [[20, 12], [21, 12], [20, 36], [21, 36]]
model_names = list(model_fraction.values())

fig, axs = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
for ax, (lk, ak) in zip(axs, settings):
    fd = generate_data(model_fraction.keys(), ak, lk)
    left = np.zeros(3)
    for i in range(4):
        ax.barh(range(3), fd[i], left=left, color=fail_colors[i],
                edgecolor="white", linewidth=0.5, height=0.55)
        left += fd[i]
    ax.set_yticks(range(3))
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_title(f"UAVs: {ak} | Scenario: {dic_level_name[lk]}", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

patches = [mpatches.Patch(color=fail_colors[i], label=fail_labels[i]) for i in range(4)]
fig.legend(handles=patches, loc="lower center", ncol=2, frameon=True,
           bbox_to_anchor=(0.5, -0.04), fontsize=8)
axs[3].set_xlabel("Proportion of Failure Cases")
fig.suptitle("Failure Reason Distribution (Normalised)", fontweight="bold", fontsize=11, y=1.01)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Combined: Failure Raw + Speed + Travel Time
# ══════════════════════════════════════════════════════════════════════════════
with open("data_use_for_graph/test_data_time.json", "r") as f:
    data_speed = json.load(f)

level_key = 21
speed_21 = []
arrival_time_lst = []
for model_key in model_fraction.keys():
    s_row, t_row = [], []
    for agent_key in [6, 9, 12, 18, 24, 36]:
        try:
            idx = next(i for i, row in enumerate(data_speed) if row[:3] == [model_key, agent_key, level_key])
        except StopIteration:
            continue
        s_row.append(data_speed[idx][-2])
        t_row.append(data_speed[idx][-1])
    speed_21.append(s_row)
    arrival_time_lst.append(t_row)

x = np.arange(len(agents_dic))
width = 0.22
bar_colors = [dic_id_color[k] for k in model_fraction.keys()]

fig = plt.figure(figsize=(13, 5))
column = 3
left_axes = [fig.add_subplot(4, column, (i * column) + 1) for i in range(4)]
colors_bar = fail_colors

for ax, (lk, ak) in zip(left_axes, settings):
    fd = generate_data(model_fraction.keys(), ak, lk, unified=False)
    left = np.zeros(3)
    for i in range(4):
        ax.barh(range(3), fd[i], left=left, color=colors_bar[i],
                edgecolor="white", linewidth=0.5, height=0.55)
        left += fd[i]
    ax.set_yticks(range(3))
    ax.set_yticklabels(model_names, fontsize=7.5)
    ax.set_title(f"UAVs:{ak} | {dic_level_name[lk]}", fontsize=8)
for ax, lim in zip(left_axes, [1200, 1200, 6500, 6500]):
    ax.set_xlim(0, lim)
left_axes[3].set_xlabel("Count\n(a) Failure distribution (300 eps.)", labelpad=6)
left_axes[0].legend(
    handles=[mpatches.Patch(color=colors_bar[i], label=fail_labels[i]) for i in range(4)],
    fontsize=6.5, loc="lower right",
)

# Speed bars
ax_speed = fig.add_subplot(1, column, 2)
speed_data = np.transpose(np.array(speed_21))
for i, (mk, mn) in enumerate(model_fraction.items()):
    bars = ax_speed.bar(x + i * width, speed_data[:, i], width,
                        label=mn, color=bar_colors[i], edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax_speed.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                      f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)
ax_speed.set_ylabel("Speed (m/s)")
ax_speed.set_xticks(x + width * (len(model_fraction) - 1) / 2)
ax_speed.set_xticklabels(list(agents_dic.keys()))
ax_speed.set_xlabel("Number of UAVs\n(b) Avg. Speed for Arrivals")
ax_speed.set_ylim(1.18, 1.27)
ax_speed.legend(fontsize=7.5)

# Travel time bars
ax_time = fig.add_subplot(1, column, 3)
time_data = np.transpose(np.array(arrival_time_lst))
for i, (mk, mn) in enumerate(model_fraction.items()):
    bars = ax_time.bar(x + i * width, time_data[:, i], width,
                       label=mn, color=bar_colors[i], edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                     f"{h:.0f}", ha="center", va="bottom", fontsize=6.5)
ax_time.set_ylabel("Time (s)")
ax_time.set_xticks(x + width * (len(model_fraction) - 1) / 2)
ax_time.set_xticklabels(list(agents_dic.keys()))
ax_time.set_xlabel("Number of UAVs\n(c) Avg. Travel Time for Arrivals")
ax_time.set_ylim(90, 98)

plt.tight_layout()
fig.savefig("test_4.jpg")
fig.savefig("test_4.pdf")
plt.show()
