import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_model_performance(csv_path,tr_csv_path, output_dir='./', output_prefix='model_comparison'):
    # 读取保存的准确度结果
    df = pd.read_csv(csv_path)
    
    # 绘制 Accuracy based on Best Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['accuracy'], label='Accuracy based on Best Loss')

    # 绘制 Accuracy for Every 10 Epochs
    df_10_epochs = df[df['epoch'] % 10 == 0]  # 只保留 epoch 为 10 的倍数的行
    plt.plot(df_10_epochs['epoch'], df_10_epochs['accuracy'], label='Accuracy for Every 10 Epochs', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    accuracy_output_path = os.path.join(output_dir, f'{output_prefix}_accuracy_vs_epoch.png')
    plt.savefig(accuracy_output_path)
    plt.show()
    
    df = pd.read_csv(tr_csv_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss'], label='Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function vs. Epoch')
    plt.legend()
    plt.grid(True)
    loss_output_path = os.path.join(output_dir, f'{output_prefix}_loss_vs_epoch.png')
    plt.savefig(loss_output_path)
    plt.show()

    return accuracy_output_path, loss_output_path



def compare_multiple_models(csv_files,tr_csv_files, output_dir='./', output_prefix='overall_comparison'):
    # 初始化字典以存储所有模型的结果
    all_results = {}
    all_training = {}
    for csv_file in csv_files:
        # 读取每个模型的 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 假设文件名中包含模型名称，例如 "modelA_accuracy_results.csv"
        model_name = "class_emb_size "+csv_file.split("/")[1].split("_")[-1]
        
        all_results[model_name] = df

    for csv_file in tr_csv_files:
        # 读取每个模型的 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 假设文件名中包含模型名称，例如 "modelA_accuracy_results.csv"
        model_name = "class_emb_size "+csv_file.split("/")[1].split("_")[-1]
        
        all_training[model_name] = df
        
    
    # 绘制整体 accuracy 比较图
    plt.figure(figsize=(10, 5))
    for model_name, df in all_results.items():
        df_10_epochs = df[(df['epoch'] % 10 == 0) & (df['epoch'] <= 300)]  # 只保留 epoch 为 10 的倍数的行
        plt.plot(df['epoch'], df['accuracy'], label=f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    accuracy_output_path = os.path.join(output_dir, f'{output_prefix}_accuracy_comparison.png')
    plt.savefig(accuracy_output_path)
    plt.show()

    # 绘制整体 loss 比较图 (假设每个 CSV 文件都有 loss 数据)
    plt.figure(figsize=(10, 5))
    for model_name, df in all_training.items():
        plt.plot(df['epoch'], df['loss'], label=f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overall Loss Comparison')
    plt.legend()
    plt.grid(True)
    loss_output_path = os.path.join(output_dir, f'{output_prefix}_loss_comparison.png')
    plt.savefig(loss_output_path)
    plt.show()

    return accuracy_output_path, loss_output_path


if __name__ == "__main__":
    csv_files = ['./ddpm_ckpt_128/test.json_accuracy_results.csv',
                './ddpm_ckpt_256/test.json_accuracy_results.csv',
                './ddpm_ckpt_512/test.json_accuracy_results.csv',
                './ddpm_ckpt_1024/test.json_accuracy_results.csv',]
    tr_csv_files = ['./ddpm_ckpt_128/training_results.csv',
                './ddpm_ckpt_256/training_results.csv',
                './ddpm_ckpt_512/training_results.csv',
                './ddpm_ckpt_1024/training_results.csv',]
    compare_multiple_models(csv_files,tr_csv_files)
    # plot_model_performance(csv_files[0],tr_csv_files[0])
    
