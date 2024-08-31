import matplotlib.pyplot as plt
import json

plt.ion()

# Load the JSON files
with open('epsilon_scores_20,30,40.json', 'r') as json_file:
    epsilon_scores1 = json.load(json_file)

with open('epsilon_scores_50_100.json', 'r') as json_file:
    epsilon_scores2 = json.load(json_file)

# Define line styles and a thickness for the plots
line_styles = ['-', '--', '-.', ':']
line_thickness = 2

# Use a colormap for gradient colors
cmap = plt.get_cmap('plasma')

# Plot epsilon scores
plt.figure(figsize=(8, 5))

all_scores = list(epsilon_scores1.items()) + list(epsilon_scores2.items())
num_lines = len(all_scores)

for idx, (epsilon0, scores) in enumerate(all_scores):
    plot_scores, plot_mean_scores = scores
    color = cmap(idx / num_lines)  # Assign color based on line index
    plt.plot(range(len(plot_mean_scores)), plot_mean_scores,
             label=fr'$\epsilon$={round(float(epsilon0))}',
             linestyle=line_styles[idx % len(line_styles)],
             linewidth=line_thickness, color=color)

plt.xlabel('Number of Games')
plt.ylabel('Mean Score')
plt.legend()
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim(0, 300)
plt.show()

# Plot gamma scores
plt.figure(figsize=(8, 5))

with open('gamma1.json', 'r') as json_file:
    gamma_scores1 = json.load(json_file)

num_lines = len(gamma_scores1)

for idx, (gamma, scores) in enumerate(gamma_scores1.items()):
    plot_scores, plot_mean_scores = scores
    color = cmap(idx / num_lines)
    plt.plot(range(len(plot_mean_scores)), plot_mean_scores,
             label=fr'$\gamma$={gamma}',
             linestyle=line_styles[idx % len(line_styles)],
             linewidth=line_thickness, color=color)

plt.xlabel('Number of Games')
plt.ylabel('Mean Score')
plt.legend()
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim(0, 300)
plt.show()

# Plot learning rate scores (lr_1)
plt.figure(figsize=(8, 5))

with open('lr_1.json', 'r') as json_file:
    lr_scores1 = json.load(json_file)

num_lines = len(lr_scores1)

for idx, (lr_1, scores) in enumerate(lr_scores1.items()):
    plot_scores, plot_mean_scores = scores
    color = cmap(idx / num_lines)
    plt.plot(range(len(plot_mean_scores)), plot_mean_scores,
             label=fr'LR={lr_1}',
             linestyle=line_styles[idx % len(line_styles)],
             linewidth=line_thickness, color=color)

plt.xlabel('Number of Games')
plt.ylabel('Mean Score')
plt.legend()
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim(0, 300)
plt.show()

# Plot detailed learning rate scores (lr_2_detailed)
plt.figure(figsize=(8, 5))

with open('lr_2_detailed.json', 'r') as json_file:
    lr_scores2 = json.load(json_file)

num_lines = len(lr_scores2)

for idx, (lr_2_detailed, scores) in enumerate(lr_scores2.items()):
    plot_scores, plot_mean_scores = scores
    color = cmap(idx / num_lines)
    plt.plot(range(len(plot_mean_scores)), plot_mean_scores,
             label=fr'LR={round(float(lr_2_detailed),4)}',
             linestyle=line_styles[idx % len(line_styles)],
             linewidth=line_thickness, color=color)

plt.xlabel('Number of Games')
plt.ylabel('Mean Score')
plt.legend()
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim(0, 200)

plt.show()
