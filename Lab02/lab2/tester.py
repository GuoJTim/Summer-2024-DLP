# implement your testing script here
# ▪ Implement the code for testing, load the model, print out the accuracy for lab
#   demo and report.

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

class Tester:
    def __init__(self, numClasses=4, timeSample=438, Nu=20, C=22, Nc=20, Nt=1, dropout_rate=0.5,weight_decay=0.0001,learning_rate = 0.0001,optimizer_name="adamw"):

        self.model = model.SCCNet.SCCNet(numClasses, timeSample, Nu, C, Nc, Nt, dropout_rate).to(device)
        
    def load_testing_dataset(self,test_dataset):
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    shuffle=True)
          
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                target = target.type(torch.LongTensor)
                data, target = data.to(device),target.to(device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
        return  float(100 * float(correct) / total)
        

    def load_model_weight(self,file_path):

        state = torch.load(file_path)
        self.model.load_state_dict(state['model_state'])
        #self.optimizer.load_state_dict(state['optimizer_state'])
        print(f"train epoch: {state['epoch']} , last loss: {state['loss']} , last accuracy: {state['accuracy']}")


tester = Tester(Nu=22,Nt=1)
tester.load_model_weight("FT\\tmp5_LOSO-FT_train\\Nu=22-Nt=1-LR=0.001-Opt=adamw-BZ=32\\epoch=2102.pt")
test_dataset = Dataloader.MIBCI2aDataset("LOSO-test")
tester.load_testing_dataset(test_dataset)
tester.evaluate()