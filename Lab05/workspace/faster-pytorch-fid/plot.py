import pandas as pd
import matplotlib.pyplot as plt



# Function to plot the graph based on the CSV data and save the image
def plot_fid_graph(file_name='fid_scores.csv', image_name='different_fid.png'):
    # Load the data from the CSV file
    df = pd.read_csv(file_name)

    # Plot a line for each mask_func
    for mask_func in df['mask_func'].unique():
        mask_func_data = df[df['mask_func'] == mask_func]
        plt.plot(mask_func_data['iter'], mask_func_data['fid_value'], label=mask_func)

    # Add labels and legend to the plot
    plt.xlabel('Iteration')
    plt.ylabel('FID Value')
    plt.title('FID Score vs Iteration for Different Mask Functions')
    plt.legend()

    # Save the plot as an image
    plt.savefig(image_name)
    plt.close()  # Close the plot to avoid display

# Plotting the graph
plot_fid_graph(file_name='fid_scores.csv',image_name="loss_mask_only_cross_entropy_fid_scores.png")