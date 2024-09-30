# I will first read the uploaded CSV files to examine their contents and structure.
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
results_file_path = 'results.csv'
test_results_file_path = 'test_results.csv'

# Reading the CSV files
results_df = pd.read_csv(results_file_path)
test_results_df = pd.read_csv(test_results_file_path)

results_df.columns = ['epoch', 'step', 'loss']
test_results_df.columns = ['epoch', 'loss']


# Filter the results_df to only include rows where the step is 0 (i.e., the first step of each epoch)
filtered_results_df = results_df[results_df['step'] == 0]
# Creating a combined plot with both filtered results.csv and test_results.csv on the same graph
plt.figure(figsize=(8, 6))

# Plot for filtered results.csv (loss by epoch, first step only)
plt.plot(filtered_results_df['epoch'], filtered_results_df['loss'], label='Train Loss', color='blue')

# Plot for test_results.csv (loss by epoch)
plt.plot(test_results_df['epoch'], test_results_df['loss'], label='Test Loss', color='orange')

# Adding titles and labels
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Display the combined plot
plt.show()
