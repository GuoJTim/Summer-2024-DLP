import re
import matplotlib.pyplot as plt
import math

import pandas as pd

def extract_data_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Initialize lists to store the extracted data
    epochs = []
    tf_status = []
    tf_ratio = []
    beta_values = []
    learning_rate = []
    avg_loss_values = []
    avg_psnr_values = []
    val_loss_values = []
    val_psnr_values = []
    
    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        v_epoch = row['epoch']
        v_beta = row['beta']
        v_avg_loss = row['avg_loss']
        v_avg_psnr = row['avg_psnr']
        v_data_type = row['type']
        v_tf_ratio = row['teacher forcing ratio']
        v_tf_status = row['teacher forcing']
        v_lr = math.log10(row['learning_rate'])
        
        # Collect the data based on the type
        if v_data_type == 'train':
            epochs.append(v_epoch)
            beta_values.append(v_beta)
            avg_loss_values.append(v_avg_loss)
            avg_psnr_values.append(v_avg_psnr)
            tf_ratio.append(v_tf_ratio)
            tf_status.append(v_tf_status)
            learning_rate.append(v_lr)
        elif v_data_type == 'valid':
            val_loss_values.append(v_avg_loss)
            val_psnr_values.append(v_avg_psnr)
    
    return (epochs, tf_status, tf_ratio, beta_values, learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values)


# Function to read and extract data from the file
def extract_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        output = file.read()

    # Updated regex patterns to match the desired data
    train_pattern = r"\(train \[TeacherForcing: (ON|OFF), ([\deE\+\-\.]+)\], beta: ([\deE\+\-\.]+)\) Epoch ([\deE\+\-\.]+), lr:([\deE\+\-\.]+).+?avg_loss=([\deE\+\-\.]+), avg_psnr=([\deE\+\-\.]+)"
    val_pattern = r"\(val\).+?loss=([\deE\+\-\.]+), psnr=([\deE\+\-\.]+)"

    train_data = re.findall(train_pattern, output)
    val_data = re.findall(val_pattern, output)

    epochs = range(len(train_data))

    tf_status = [data[0]=='ON' for data in train_data]
    tf_ratio = [float(data[1]) for data in train_data]
    beta_values = [float(data[2]) for data in train_data]
    learning_rate = [math.log10(float(data[4])) for data in train_data]
    avg_loss_values = [float(data[5]) for data in train_data]
    avg_psnr_values = [float(data[6]) for data in train_data]

    val_loss_values = [float(data[0]) for data in val_data]
    val_psnr_values = [float(data[1]) for data in val_data]
    
    print(tf_status)
    return epochs, tf_status, tf_ratio, beta_values,learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values

# Function to plot the graphs
def plot_graphs(epochs, tf_status, tf_ratio, beta_values,learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values):
    plt.figure(figsize=(14, 10))

    # Subplot for training data
    plt.subplot(7, 1, 1)
    plt.plot(epochs, tf_status, label='TF', color='green')
    plt.plot(epochs, tf_ratio, label='TF Ratio', color='purple')
    plt.title('Training Data - Teacher Forcing Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('TF Ratio')
    plt.legend()
    plt.grid(True)

    plt.subplot(7, 1, 2)
    plt.plot(epochs, beta_values, label='Beta')
    plt.title('Training Data')
    plt.xlabel('Epoch')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(7, 1, 3)
    plt.plot(epochs, avg_loss_values, label='Avg Loss')
    plt.title('Training Data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(7, 1, 4)
    plt.plot(epochs, avg_psnr_values, label='Avg PSNR')
    plt.title('Training Data')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    #plt.ylim(20, 40)
    plt.legend()
    plt.grid(True)

    # Subplot for validation data
    plt.subplot(7, 1, 5)
    plt.plot(epochs, val_loss_values, label='Validation Loss',  color='orange')
    plt.title('Validation Data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot for validation data
    plt.subplot(7, 1, 6)
    plt.plot(epochs, val_psnr_values, label='Validation PSNR',  color='green')
    plt.title('Validation Data')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    #plt.ylim(34, 37)
    plt.legend()
    plt.grid(True)
    
    # Subplot for validation data
    plt.subplot(7, 1, 7)
    plt.plot(epochs, learning_rate, label='Learning Rate',  color='green')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



# Main function
def main():
    
    file_path = 'finetune output.txt'  # Replace with your file path
    file_path = "exp-9-finetune-4/training_results.csv"
    epochs, tf_status, tf_ratio, beta_values,learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values = extract_data_csv(file_path)
    plot_graphs(epochs, tf_status, tf_ratio, beta_values,learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values)

if __name__ == "__main__":
    main()
