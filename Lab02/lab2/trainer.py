# implement your training script here
# Implement the code for training the SCCNet model, including functions
# related to training, losses, optimizer, backpropagation, etc, remember to save
# the model weight.

import Dataloader
import utils
import model.SCCNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import os.path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    test_loader = None
    def __init__(self, numClasses=4, timeSample=438, Nu=20, C=22, Nc=20, Nt=1, dropout_rate=0.5,weight_decay=0.0001,learning_rate = 0.0001,optimizer_name="adamw"):
        # 可調參數 Nu,Nt,Learning_rate,optimizer,batch_size
        self.model = model.SCCNet.SCCNet(numClasses, timeSample, Nu, C, Nc, Nt, dropout_rate).to(device)
        optimizers = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,

        }
        optimizer_class = optimizers.get(optimizer_name.lower())
        
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.loss_function = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.Nu = Nu
        self.Nt = Nt
        

    def load_training_dataset(self,train_dataset,batch_size,train_method="default"):
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        self.batch_size = batch_size
        self.train_method = train_method

    def load_testing_dataset(self,test_dataset,batch_size,train_method="default"):
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)
        self.batch_size = batch_size
        self.train_method = train_method
        
    def train(self,model_name,num_epochs,avg_epoch_save=100,save_to_db=False,max_train_epoch=-1): 
        self.model_name = model_name
        self.max_acc = 0
        # create /model_name/{train_method}/Nu=*-Nt=*-LR=*-Opt=*,BZ=*,epoch=*
        start_epoch = self.load_model_weight()
        if (max_train_epoch != -1 and start_epoch >= max_train_epoch):
            return
        total_step  = len(self.train_loader)
        total_loss = []
        for epoch in range(start_epoch,start_epoch+num_epochs):
            for batch_idx, (data,target) in enumerate(self.train_loader):
                target = target.type(torch.LongTensor)
                data, target = data.to(device),target.to(device)
                self.model.train()

                output = self.model(data)
                self.loss = self.loss_function(output,target)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs+start_epoch}], Loss: {self.loss.item():.4f}')
            total_loss.append(self.loss.item())
            accuracy = self.evaluate()
            if (save_to_db):
                utils.insert_data(self.model_name, epoch+1, total_loss[-1], accuracy, self.batch_size, self.Nu, self.Nt, self.learning_rate, self.optimizer_name, self.train_method)
            
            if ((epoch+1) % avg_epoch_save == 0):
                self.save_model_weight(epoch+1,total_loss[-1],accuracy)
                if (accuracy > self.max_acc):
                    self.max_acc = accuracy    
            elif (accuracy > self.max_acc):
                self.max_acc = accuracy
                self.save_model_weight(epoch+1,total_loss[-1],accuracy)
            
            if (epoch+1 == max_train_epoch and max_train_epoch != -1):
                break
                
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            if (self.test_loader is None):
                return -1
            for data, target in self.test_loader:
                target = target.type(torch.LongTensor)
                data, target = data.to(device),target.to(device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
        return  float(100 * float(correct) / total)
        

    def save_model_weight(self, epoch,last_loss,last_acc):
        file_path = f"{self.train_method}/{self.model_name}/Nu={self.Nu}-Nt={self.Nt}-LR={self.learning_rate}-Opt={self.optimizer_name}-BZ={self.batch_size}/epoch={epoch}.pt"
        

        # Create directory if it does not exist
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the model and optimizer state
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss' : last_loss,
            'accuracy' : last_acc
        }
        torch.save(state, file_path)
        print(f"Checkpoint saved to {file_path}")
    def load_model_weight(self):
        # Define the directory path
        dir_path = f"{self.train_method}/{self.model_name}/Nu={self.Nu}-Nt={self.Nt}-LR={self.learning_rate}-Opt={self.optimizer_name}-BZ={self.batch_size}"
        
        # List all files in the directory
        if os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
            
            if files:
                # Find the file with the highest epoch number
                latest_file = max(files, key=lambda f: int(f.split('epoch=')[1].split('.pt')[0]))
                file_path = os.path.join(dir_path, latest_file)

                # Load the model state
                state = torch.load(file_path)
                self.model.load_state_dict(state['model_state'])
                self.optimizer.load_state_dict(state['optimizer_state'])
                last_epoch = state['epoch']
                last_acc = state['accuracy']
                self.max_acc = last_acc
                print(f"Checkpoint loaded from {file_path}")
                return last_epoch
            else:
                print(f"No checkpoint files found in {dir_path}")
                return 0
        else:
            print(f"No directory found at {dir_path}")
        return 0


trainer = Trainer(numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropout_rate=0.5,learning_rate=0.0001,optimizer_name="adamw")
train_dataset = Dataloader.MIBCI2aDataset("SD-train")
#test_dataset = Dataloader.MIBCI2aDataset("SD-test")
trainer.load_training_dataset(train_dataset,32)
#trainer.load_testing_dataset(test_dataset,32)
trainer.train("my_SD_train",num_epochs=10,avg_epoch_save=10000,max_train_epoch=1000)
