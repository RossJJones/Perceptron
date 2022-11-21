import numpy as np

epoch = 200000 #how many training iterations

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training = np.array([[0,0,1],
                     [1,1,1],
                     [1,0,1],
                     [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

weights = 2 * np.random.random((3,1)) - 1

input_layer = training
output_layer = sigmoid(np.dot(input_layer, weights))
print(output_layer)

#train
for i in range(epoch):
    #training
    input_layer = training
    output_layer = sigmoid(np.dot(input_layer, weights))
    #backpropergation
    error = training_outputs - output_layer
    adjustment = error * sigmoid_derivative(output_layer)
    weights += np.dot(input_layer.T, adjustment)
print()
print(output_layer)
print()

#test unseen data
input_layer = np.array([[0,0,0]])
output_layer = sigmoid(np.dot(input_layer, weights))

print(output_layer)
