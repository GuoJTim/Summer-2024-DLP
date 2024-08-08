import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

def draw_line_chart(x, y, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def draw_scatter_plot(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True)
    plt.show()


np.random.seed(1)

def relu(x):
    return x * (x > 0)

def derivative_relu(x):
    return 1. * (x > 0)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1- np.tanh(x)**2


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x,1.0 - x)

def no_function(X):
    return X


class Optimizer:
    def __init__(self):
        pass

    def calc(self,gradient):
        return None



class Momentum(Optimizer):
    m = 0.9
    # v^t = [ learning_rate * gradient           t=0
    #       [ mv(t-1) + learning_rate * gradient t>=1
    #
    # theta = theta - v(t)

    last_v = None
    def __init__(self,m=0.9):
        self.m = m
    
    # calc  v^(t)
    def calc(self,gradient):
        if (self.last_v is None):
            ret = gradient
        else:
            ret = self.last_v * self.m + gradient
        self.last_v = ret
        return ret
class nn:
    # activate function
    #    py funfiton
    # input
    #    np.array
    # output
    #    np.array
    
    W = None
    b = None
    dW = None
    db = None
    activate_function = None
    derivative_function = None
    a_output = None # self a Value (after activation)
    a_input = None # input (or last layer output)
    optimizer = None
    bias_optimizer = None
    def __init__(self,input,output,activate_function=no_function,derivative_function =no_function,weight_optimizer=None,bias_optimizer=None):
        self.W = np.random.rand(input,output) # default value
        self.b = np.random.rand(1,output)
        self.activate_function = activate_function # for all value pass the activate function get the result
        self.derivative_function = derivative_function # for all value pass the activate function get the result
        self.weight_optimizer = weight_optimizer
        self.bias_optimizer = bias_optimizer
        pass
    def forward(self,feature):
        # give feature with size input then result size output feature
        # linear function
        # y = σ(WX + b)
        self.a_input = feature
        self.a_output = self.activate_function(np.dot(feature,self.W) + self.b) 
        #print(self.a_output)
        return self.a_output
    def backward(self,output_error):
        z = output_error * self.derivative_function(self.a_output)  # calc this layer new error
        #print("->",self.a_input.T.shape,z.shape)
        self.dW = np.dot(self.a_input.T , z)
        self.db = np.sum(z, axis=0, keepdims=True)
        #print(dW.shape)
        #print(self.W.shape)
        return np.dot(z, self.W.T)
    
    def update(self,learning_rate):
        if (self.weight_optimizer != None and self.bias_optimizer!= None ):
            self.W -= self.weight_optimizer.calc(learning_rate * self.dW)
            self.b -= self.bias_optimizer.calc(learning_rate * self.db)
        else:
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

class model:
    network = []
    optimizer = None
    memory_learning_epoch = [] 
    memory_learning_curve_loss = []
    memory_learning_curve_accuracy = []
    memory_epoch = 100
    def __init__(self,memory_epoch=100):
        self.memory_epoch = memory_epoch
        pass
    def clear(self):
        self.network = []
        self.memory_learning_epoch = [] 
        self.memory_learning_curve_loss = []
        self.memory_learning_curve_accuracy = []
    def set_optimizer(self,optimizer):
        self.optimizer = optimizer
    def add_layer(self,input,output,activate_function=no_function,derivative_function=no_function):
        weight_optimizer = deepcopy(self.optimizer)
        bias_optimizer = deepcopy(self.optimizer)
        self.network.append(nn(input,output,activate_function,derivative_function,weight_optimizer,bias_optimizer))
    def forward(self,last_feature):
        #last_feature = None # input data

        for layer in self.network:
            last_feature = layer.forward(last_feature)
        prediction = last_feature
        
        # last_feature is the last result
        return prediction

    def backward(self,y,pred_y):
        error = -2*(y - pred_y) # L2'
        #print(error.shape)
        for layer in reversed(self.network):
            error = layer.backward(error)
        
    
    def update(self,learning_rate):
        for layer in self.network:
            layer.update(learning_rate)

    def training(self,n,training_data,ground_truth,learning_rate=0.1,show_info=5000,batch=32):
        training_set = np.hstack((training_data,ground_truth))
        tot_dataset = len(training_data)
        acc = None
        for epoch in range(n):
            
            np.random.shuffle(training_set)
            batch_training_data = training_set[:,:-1]
            batch_prediction = training_set[:,-1:]
            for i in range(int(math.ceil(tot_dataset / batch))):    
                st = batch * i
                ed = min(batch * (i+1),tot_dataset)
                
                prediction = self.forward(batch_training_data[st:ed])
                # backward pass compute \delta weights
                self.backward(batch_prediction[st:ed],prediction)

                # update
                self.update(learning_rate)
            
            if (epoch+1) % show_info == 0:
                prediction = self.forward(training_data)
                acc = self.calc_accuracy(ground_truth,prediction)
                loss = self.calc_loss(ground_truth,prediction)
                if (epoch+1) % self.memory_epoch == 0:
                    self.memory_learning_epoch.append(epoch+1)
                    self.memory_learning_curve_loss.append(loss)
                    self.memory_learning_curve_accuracy.append(acc)
                print(f"epoch {epoch+1}: loss : {loss:.10f} accuracy={acc:03.2f}%")
            
                if (acc == 100):
                    break
            elif (epoch+1) % self.memory_epoch == 0:
                prediction = self.forward(training_data)
                acc = self.calc_accuracy(ground_truth,prediction)
                loss = self.calc_loss(ground_truth,prediction)
                self.memory_learning_epoch.append(epoch+1)
                self.memory_learning_curve_loss.append(loss)
                self.memory_learning_curve_accuracy.append(acc)

        if acc == None:
            prediction = self.forward(training_data)
            acc = self.calc_accuracy(ground_truth,prediction)
        print(f"last accuracy={acc:03.2f}%")
    def calc_loss(self, y, y_pred):
        return np.mean(np.square(y - y_pred))

    def calc_accuracy(self,y,y_pred):
        n = len(y)
        return (y == (y_pred > 0.5)).sum()*100 / n
    def testing(self,training_data,ground_truth,show_info=True):
        prediction = self.forward(training_data)
        for i in range(len(training_data)):
            if (not show_info):
                break
            print(f"Iter {i:3d} |    Ground truth: {ground_truth[i][0]:f} |   prediction: {prediction[i][0]:f} |")
        loss = self.calc_loss(ground_truth,prediction)
        acc = self.calc_accuracy(ground_truth,prediction)
        print(f"loss={loss:.4f} accuracy={acc:03.2f}%")
        return prediction
    def show_learning_curve_loss(self):
        draw_line_chart(self.memory_learning_epoch,self.memory_learning_curve_loss,'Loss curve','epoch','loss')
    
    def show_learning_curve_accuracy(self):
        draw_line_chart(self.memory_learning_epoch,self.memory_learning_curve_accuracy,'Accuracy curve','epoch','Accuracy')
def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0] - pt[1] ) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)
def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        
        if 0.1 * i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21,1)

def generate_XOR_hard():
    import numpy as np
    inputs = []
    labels = []

    for i in range(101):
        for j in range(101):
            x1 = 0.01 * i
            x2 = 0.01 * j
            inputs.append([x1, x2])
            labels.append(int((x1 > 0.5) != (x2 > 0.5)))

    return np.array(inputs), np.array(labels).reshape(-1, 1)

def generate_XOR_hard_2(noise_level=0.1):
    inputs = []
    labels = []

    for i in range(101):
        for j in range(101):
            x1 = 0.01 * i
            x2 = 0.01 * j
            # 引入隨機噪音
            x1_noisy = x1 + np.random.uniform(-noise_level, noise_level)
            x2_noisy = x2 + np.random.uniform(-noise_level, noise_level)
            inputs.append([x1_noisy, x2_noisy])
            labels.append(int((x1 > 0.5) != (x2 > 0.5)))

    return np.array(inputs), np.array(labels).reshape(-1, 1)


def show_result(x,y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize=18)
    for i in range(x.shape[0]):
        if (y[i] == 0):
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result',fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()


#x,y = generate_XOR_hard()
#x,y= generate_XOR_hard_2()



#x,y = generate_linear(n = 100)
x,y = generate_XOR_easy()
m = model()
m.set_optimizer(Momentum(0.9))
m.add_layer(2,6,relu,derivative_relu)
m.add_layer(6,6,relu,derivative_relu)
m.add_layer(6,1,sigmoid,derivative_sigmoid)
m.training(10000,x,y,0.001,1000,batch=100)

pred_y = m.testing(x,y,show_info=False)
show_result(x,y,(pred_y > 0.5))

