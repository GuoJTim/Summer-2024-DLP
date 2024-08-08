import numpy as np


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
x,y = generate_linear(n = 1000)
#x,y = generate_XOR_easy()


# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Seed for reproducibility
np.random.seed(1)
# Initialize weights
W1 = np.random.rand(2, 4)
W2 = np.random.rand(4, 4)
W3 = np.random.rand(4, 1)
# Initialize biases
b1 = np.random.rand(1, 4)
b2 = np.random.rand(1, 4)
b3 = np.random.rand(1, 1)

# Learning rate
learning_rate = 0.1

# Number of epochs
epochs = 10000

# Training the network
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(x, W1) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    
    # Compute the error
    error = y - A3
    
    # Backpropagation
    dA3 = error * sigmoid_derivative(A3)
    dW3 = np.dot(A2.T, dA3)
    db3 = np.sum(dA3, axis=0, keepdims=True)
    
    dA2 = np.dot(dA3, W3.T) * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)
    
    dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(x.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)
    
    # Update the weights and biases
    W3 += learning_rate * dW3
    b3 += learning_rate * db3
    
    W2 += learning_rate * dW2
    b2 += learning_rate * db2
    
    W1 += learning_rate * dW1
    b1 += learning_rate * db1

# After training, let's print the final weights
print("W1:", W1)
print("W2:", W2)
print("W3:", W3)

# And the final output
print("Output after training:")
print(A3)

show_result(x,y,(A3 > 0.5))