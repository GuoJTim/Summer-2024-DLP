import re
import matplotlib.pyplot as plt
import math

import pandas as pd


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


def plot_graphs(epochs, tf_status, tf_ratio, beta_values, learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values,filename):
    plt.figure(figsize=(14, 12))

    # 1,1 - PSNR for training and validation
    plt.subplot(2, 2, 1)
    plt.plot(epochs, avg_psnr_values, label='Training PSNR', color='blue')
    plt.plot(epochs, val_psnr_values, label='Validation PSNR', color='orange')
    plt.title('PSNR (Training vs Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid(True)

    # 1,2 - Loss for training and validation
    plt.subplot(2, 2, 2)
    plt.plot(epochs, avg_loss_values, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss_values, label='Validation Loss', color='orange')
    plt.title('Loss (Training vs Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,25)
    plt.legend()
    plt.grid(True)

    # 2,1 - Beta, Teacher Forcing Ratio, and Teacher Forcing Status (step plots)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, beta_values, label='Beta', color='red')
    plt.plot(epochs, tf_ratio, label='TF Ratio', color='purple')
    plt.step(epochs, tf_status, label='TF Status', color='green', where='post')
    plt.title('Beta and Teacher Forcing')
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    # 2,2 - Learning rate (log10 scale)
    plt.subplot(2, 2, 4)
    plt.step(epochs, learning_rate, label='Learning Rate (log10)', color='green', where='post')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log10)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    #plt.show()
    plt.savefig(filename+".png")
    

def extract_data_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Initialize dictionaries to store the extracted data by epoch
    data_dict = {
        'epochs': [],
        'tf_status': [],
        'tf_ratio': [],
        'beta_values': [],
        'learning_rate': [],
        'avg_loss_values': [],
        'avg_psnr_values': [],
        'val_loss_values': [],
        'val_psnr_values': []
    }
    
    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        v_epoch = row['epoch']
        #if (v_epoch >= 70):
            
        #    break
        v_beta = row['beta']
        v_avg_loss = row['avg_loss']
        v_avg_psnr = row['avg_psnr']
        v_data_type = row['type']
        v_tf_ratio = row['teacher forcing ratio']
        v_tf_status = 1 if row['teacher forcing'] else 0 
        v_lr = math.log10(row['learning_rate'])

        # Only add epoch if it's not already in the list
        if v_epoch not in data_dict['epochs']:
            data_dict['epochs'].append(v_epoch)
            data_dict['tf_status'].append(v_tf_status)
            data_dict['tf_ratio'].append(v_tf_ratio)
            data_dict['beta_values'].append(v_beta)
            data_dict['learning_rate'].append(v_lr)
            data_dict['avg_loss_values'].append(v_avg_loss if v_data_type == 'train' else None)
            data_dict['avg_psnr_values'].append(v_avg_psnr if v_data_type == 'train' else None)
            data_dict['val_loss_values'].append(v_avg_loss if v_data_type == 'valid' else None)
            data_dict['val_psnr_values'].append(v_avg_psnr if v_data_type == 'valid' else None)
        else:
            idx = data_dict['epochs'].index(v_epoch)
            if v_data_type == 'train':
                data_dict['avg_loss_values'][idx] = v_avg_loss
                data_dict['avg_psnr_values'][idx] = v_avg_psnr
            elif v_data_type == 'valid':
                data_dict['val_loss_values'][idx] = v_avg_loss
                data_dict['val_psnr_values'][idx] = v_avg_psnr

    # Ensure there are no None values in the lists
    avg_loss_values = [val if val is not None else 0 for val in data_dict['avg_loss_values']]
    avg_psnr_values = [val if val is not None else 0 for val in data_dict['avg_psnr_values']]
    val_loss_values = [val if val is not None else 0 for val in data_dict['val_loss_values']]
    val_psnr_values = [val if val is not None else 0 for val in data_dict['val_psnr_values']]

    return (
        data_dict['epochs'],
        data_dict['tf_status'],
        data_dict['tf_ratio'],
        data_dict['beta_values'],
        data_dict['learning_rate'],
        avg_loss_values,
        avg_psnr_values,
        val_loss_values,
        val_psnr_values
    )

import os
# Main function
def main():
    
    file_path = 'finetune output.txt'  # Replace with your file path
    print(os.listdir("."))
    for folder in os.listdir("."):
        if (folder.count(".") >= 1):
            continue
        file_path = folder+"/training_results.csv"
        print(file_path)
        epochs, tf_status, tf_ratio, beta_values,learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values = extract_data_csv(file_path)
        plot_graphs(epochs, tf_status, tf_ratio, beta_values,learning_rate, avg_loss_values, avg_psnr_values, val_loss_values, val_psnr_values,folder)

if __name__ == "__main__":
    main()
