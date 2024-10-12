import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('results0.csv', names=['epoch', 'step', 'loss', 'accuracy'])

# Reset index to get sequential step numbers
df = df.reset_index(drop=True)
df['global_step'] = df.index

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))

# Plot loss over global steps
sns.lineplot(x='global_step', y='loss', data=df, color='b', ax=ax[0], linewidth=2)

# Plot accuracy over global steps
sns.lineplot(x='global_step', y='accuracy', data=df, color='k', ax=ax[1], linewidth=2)

# Set labels and titles
ax[0].set_xlabel('Step', size=13)
ax[0].set_ylabel('Loss', size=13)
ax[0].set_title('Loss over Steps', size=15)

ax[1].set_xlabel('Step', size=13)
ax[1].set_ylabel('Accuracy', size=13)
ax[1].set_title('Accuracy over Steps', size=15)

# Add grid
for a in ax:
    a.grid(True, linestyle=':')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
