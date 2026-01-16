import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def generate_dataset(
    setName: str, nrPoints: int, noise_factor: float = 0, normalizeData: bool = True
):
    """
    Argument:
    setName -- name of set to be generated: blobs, circles, flower, moons,
               spiral1, spiral2, spiral3
    nrPoints -- number of sample points
    noise_factor -- the higher, the more noise; set to 0 for no noise
    normalizeData -- transforms dataset such that mean is zero and standard
                     deviation equals 1

    Returns:
    X -- coordinates of the sampled points
    Y -- class of the sampled points
    """

    def generate_dataset_flower(nrPoints):
        N = int(nrPoints / 2)
        X = np.zeros((2, nrPoints))
        Y = np.zeros((1, nrPoints), dtype="uint8")

        for cat in range(2):
            idx = range(N * cat, N * (cat + 1))
            theta = np.linspace(cat * np.pi, (cat + 1) * np.pi, N)
            radius = 4 * np.sin(4 * theta)

            X[0, idx] = radius * np.sin(theta)
            X[1, idx] = radius * np.cos(theta)
            Y[0, idx] = cat

        return X, Y

    def generate_archimedesSpiral(nrPoints, rounds):
        def archimedesSpiral(a, n):
            phi = np.linspace(2 * np.pi, (rounds * 2 + 2) * np.pi, n)
            X1 = [a * phi * np.cos(phi)]
            X2 = [a * phi * np.sin(phi)]
            X = np.vstack((X1, X2))

            return X

        N = int(nrPoints / 2)

        firstSpiralX = archimedesSpiral(1.25 * np.pi, N)

        secondSpiralX = archimedesSpiral(np.pi, N)

        X = np.hstack((firstSpiralX, secondSpiralX))

        Y = np.ones((1, X.shape[1]))
        Y[:, : firstSpiralX.shape[1]] = np.zeros_like(firstSpiralX[0, :])

        Y = Y.astype(int)

        return X, Y

    def generate_blobs(nrPoints):
        X, Y = skd.make_blobs(n_samples=nrPoints, centers=2, n_features=2)
        X = X.T
        Y = Y[np.newaxis, :]

        return X, Y

    def generate_circles(nrPoints):
        X, Y = skd.make_circles(n_samples=nrPoints)
        X = X.T
        Y = Y[np.newaxis, :]

        return X, Y

    def generate_moons(nrPoints):
        X, Y = skd.make_moons(n_samples=nrPoints)
        X = X.T
        Y = Y[np.newaxis, :]

        return X, Y


    if setName == "flower":
        X, Y = generate_dataset_flower(nrPoints)
    elif setName == "spiral1":
        X, Y = generate_archimedesSpiral(nrPoints, 1)
    elif setName == "spiral2":
        X, Y = generate_archimedesSpiral(nrPoints, 1.5)
    elif setName == "spiral3":
        X, Y = generate_archimedesSpiral(nrPoints, 2)
    elif setName == "blobs":
        X, Y = generate_blobs(nrPoints)
    elif setName == "circles":
        X, Y = generate_circles(nrPoints)
    elif setName == "moons":
        X, Y = generate_moons(nrPoints)
    else:
        raise Exception("Dataset " + setName + " unknown")

    if normalizeData:
        X = (X - np.mean(X)) / np.std(X)

    X += np.random.randn(X.shape[0], X.shape[1]) * noise_factor * np.std(X)

    return X, Y


def plot_decision_boundary(model, X, Y):
    """
    Argument:
    model -- prediction function
    X -- coordinates of sampled points
    Y -- class of sampled points
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.1, X[0, :].max() + 0.1
    y_min, y_max = X[1, :].min() - 0.1, X[1, :].max() + 0.1

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[0, :], X[1, :], c=Y[0, :], cmap=plt.cm.Spectral)
    plt.show()

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C1 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def initialize_weights(nodes0, nodes1, nodes2, nodes3):
    """
    Argument:
    nodes0 -- size of the input layer
    nodes1 -- size of the first hidden layer
    nodes2 -- size of the second hidden layer
    nodes3 -- size of the output layer

    Returns:
    weights -- python dictionary containing weight 
                and bias matrices
    """        
    # initialize the weights and biases as numpy arrays
    #  with the correct shapes.
    A1 = np.random.normal(0, 1./np.sqrt(nodes0), (nodes1, nodes0))
    b1 = np.zeros((nodes1, 1))
    A2 = np.random.normal(0, 1./np.sqrt(nodes1), (nodes2, nodes1))
    b2 = np.zeros((nodes2, 1))
    A3 = np.random.normal(0, 1./np.sqrt(nodes2), (nodes3, nodes2))
    b3 = np.zeros((nodes3, 1))
        
    assert(A1.shape == (nodes1, nodes0))
    assert(b1.shape == (nodes1, 1))
    assert(A2.shape == (nodes2, nodes1))
    assert(b2.shape == (nodes2, 1))
    assert(A3.shape == (nodes3, nodes2))
    assert(b3.shape == (nodes3, 1))


    weights = {"A1": A1,
               "b1": b1,
               "A2": A2,
               "b2": b2,
               "A3": A3,
               "b3": b3}
    
    return weights

np.random.seed(42) # Make sure to have a reproducible result
weights = initialize_weights(2, 8, 5, 1)

result = np.array([np.mean(weights["A1"]), np.mean(weights["A2"]), 
                   np.mean(weights["A3"]), np.mean(weights["b1"]),
                   np.mean(weights["b2"]), np.mean(weights["b3"])])
reference = np.array([-0.0179896655582674, -0.0798199810455688,
                   -0.0286928288067441, 0., 0., 0.])
np.testing.assert_allclose(result, reference)
print("Looks like your implementation for the weights is working :)")

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C2 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def actFunctionLinear(x, derivative: bool = False):
    if derivative:
        return np.ones_like(x) # derivative of linear function is 1
    else:
        return x


def actFunctionSigmoid(x, derivative: bool = False):        
    if derivative:
        s = 1. / (1 + np.exp(-x)) 
        return s * (1 - s)
    else:
        return 1. / (1 + np.exp(-x))        #1. is for avoiding integer division = 1.0 (float)

        
        
def actFunctionRelu(x, derivative: bool = False): 
    if derivative:
        return np.array(x > 0).astype(int) 
    else:
        return np.maximum(x, 0)         # is equivalent to: if x>0: return x else: return 0 but efficient for arrays
        

def actFunctionTanh(x, derivative: bool = False): 
    if derivative:
        return 1. / np.cosh(x)**2
    else:
        return np.tanh(x)
    


# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C3 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def forward_propagation(X, weights, f1, f2, f3):
    """
    Argument:
    X -- input data
    weights -- python dictionary containing the weight and bias matrices
    f1 -- python function as activation function layer 1
    f2 -- python function as activation function layer 2
    f3 -- python function as activation function layer 3

    Returns:
    outputLayer -- The output of the last layer
    cache -- a dictionary containing (pre)activations of every layer
    """    
    A1 = weights["A1"]
    b1 = weights["b1"]
    A2 = weights["A2"]
    b2 = weights["b2"]
    A3 = weights["A3"]
    b3 = weights["b3"]
    
    x0 = X
    z1 = A1 @ x0 + b1
    x1 = f1(z1)
    z2 = A2 @ x1 + b2
    x2 = f2(z2)
    z3 = A3 @ x2 + b3
    x3 = f3(z3)
      
    outputLayer = x3

    cache = {"x0": x0,
             "z1": z1,
             "x1": x1,
             "z2": z2,
             "x2": x2,
             "z3": z3,
             "x3": x3}


    return outputLayer, cache

f1 = actFunctionLinear # function used for hidden layer 1
f2 = actFunctionLinear # function used for hidden layer 2
f3 = actFunctionSigmoid # function used for output layer
np.random.seed(42) # Make sure to have a reproducible result
X, Y = generate_dataset("blobs", 1000, 0.5, False)
np.random.seed(42) # Make sure to have a reproducible result
weights = initialize_weights(2, 8, 5, 1)

outputLayer, cache = forward_propagation(X, weights, f1, f2, f3)
result = np.array([np.mean(cache['x0']), np.mean(cache['z1']), 
                   np.mean(cache['x1']), np.mean(cache['z2']),
                   np.mean(cache['x2']), np.mean(cache['z3']),
                   np.mean(cache['x3'])])
reference = np.array([3.3138498526746014, -0.22806136844696653,
                     -0.22806136844696653, -1.5318738340852145,
                     -1.5318738340852145, 1.1329044156610317, 
                      0.7420574800205959])
np.testing.assert_allclose(result, reference)
print("Looks like the forward propagation is working :)")
print(X.shape)

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C4 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def backward_propagation(weights, cache, X, Y):
    """
    Arguments:
    weights -- python dictionary containing our weights
    cache -- a dictionary containing "pA1", "A1", "pA2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with 
             respect to different weights
    """

    A1 = weights["A1"]
    A2 = weights["A2"]
    A3 = weights["A3"]
    
    b1 = weights["b1"]
    b2 = weights["b2"]
    b3 = weights["b3"]

    x0 = cache["x0"]
    x1 = cache["x1"]
    x2 = cache["x2"]
    x3 = cache["x3"]

    z1 = cache["z1"]
    z2 = cache["z2"]
    z3 = cache["z3"]

    N = X.shape[1]

    delta3 =  -1. /N *(Y -x3)
    delta2 =  (A3.T @ delta3) * f2(z2, derivative=True)
    delta1 = (A2.T @ delta2) * f1(z1, derivative=True)
    dA3 = delta3 @ x2.T 
    dA2 = delta2 @ x1.T 
    dA1 = delta1 @ x0.T 
    db3 = delta3 @  np.ones_like(Y).T
    db2 = delta2 @  np.ones_like(Y).T
    db1 = delta1 @  np.ones_like(Y).T
    
    grads = {"dA1": dA1,
             "db1": db1,
             "dA2": dA2,
             "db2": db2,
             "dA3": dA3,
             "db3": db3}

    return grads

f1 = actFunctionLinear # function used for hidden layer 1
f2 = actFunctionLinear # function used for hidden layer 2
f3 = actFunctionSigmoid # function used for output layer
np.random.seed(42) # Make sure to have a reproducible result
X, Y = generate_dataset("blobs", 1000, 0.5, False)
np.random.seed(42) # Make sure to have a reproducible result
weights = initialize_weights(2, 8, 5, 1)

outputLayer, cache = forward_propagation(X, weights, f1, f2, f3)
grads = backward_propagation(weights, cache, X, Y)
result = np.array([np.mean(grads["dA1"]), np.mean(grads["dA2"]),
                   np.mean(grads["dA3"]), np.mean(grads["db1"]),
                   np.mean(grads["db2"]), np.mean(grads["db3"])])
reference = np.array([0.002965957830197586, 0.004537290504686634,
                      0.14652013603577085, 0.0008235332844678776,
                      -0.006945313835622825, 0.24205748002059585])

np.testing.assert_allclose(result, reference)
print("Looks like backward propagation is working :)")

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C5 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def update_weights(weights, grads, learning_rate):
    """
    Arguments:
    weights -- python dictionary containing your weights
    grads -- python dictionary containing your gradients
    learning_rate

    Returns:
    weights -- python dictionary containing your updated weights
    """
    A1 = weights["A1"]
    b1 = weights["b1"]
    A2 = weights["A2"]
    b2 = weights["b2"]
    A3 = weights["A3"]
    b3 = weights["b3"]

    dA1 = grads["dA1"]
    db1 = grads["db1"]
    dA2 = grads["dA2"]
    db2 = grads["db2"]
    dA3 = grads["dA3"]
    db3 = grads["db3"]

    # Update rule for each weight
    A1 = A1 - learning_rate * dA1
    b1 = b1 - learning_rate * db1
    A2 = A2 - learning_rate * dA2
    b2 = b2 - learning_rate * db2
    A3 = A3 - learning_rate * dA3
    b3 = b3 - learning_rate * db3

    weights["A1"] = A1
    weights["b1"] = b1
    weights["A2"] = A2
    weights["b2"] = b2
    weights["A3"] = A3
    weights["b3"] = b3

    return weights

f1 = actFunctionLinear # function used for hidden layer 1
f2 = actFunctionLinear # function used for hidden layer 2
f3 = actFunctionSigmoid # function used for output layer
np.random.seed(42) # Make sure to have a reproducible result
X, Y = generate_dataset("blobs", 1000, 0.5, True)
np.random.seed(42) # Make sure to have a reproducible result
weights = initialize_weights(2, 8, 5, 1)

outputLayer, cache = forward_propagation(X, weights, f1, f2, f3)
grads = backward_propagation(weights, cache, X, Y)
weights_upd = update_weights(weights, grads, learning_rate=0.05)

result = np.array([np.mean(weights_upd["A1"]), np.mean(weights_upd["A2"]), 
                   np.mean(weights_upd["A3"]), np.mean(weights_upd["b1"]),
                   np.mean(weights_upd["b2"]), np.mean(weights_upd["b3"])])
reference = np.array([-0.01798925865224338, -0.07985076364711673,
                      -0.034801399603810054, -2.269537873037303e-06,
                       1.9140274093792395e-05, -0.0006670751853262199])

np.testing.assert_allclose(result, reference)
print("Looks like the weights are updated correctly :)")

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C6 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def predict(weights, X, f1, f2, f3):
    """
    Using the learned weights, predicts a class for each example in X

    Arguments:
    weights -- python dictionary containing your weights
    X -- input data of size (n_x, m) 
    f1 -- python function as activation function layer 1
    f2 -- python function as activation function layer 2
    f3 -- python function as activation function layer 3

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    # 1. Computes probabilities using forward propagation,
    # 2. and classifies to 0/1 using 0.5 as the threshold.
    outputLayer, _ = forward_propagation(X, weights, f1, f2, f3)
    predictions = (outputLayer > 0.5).astype(int)

    return predictions

f1 = actFunctionLinear # function used for hidden layer 1
f2 = actFunctionLinear # function used for hidden layer 2
f3 = actFunctionSigmoid # function used for output layer
np.random.seed(42) # Make sure to have a reproducible result
X, Y = generate_dataset("blobs", 1000, 0.5, True)
np.random.seed(42) # Make sure to have a reproducible result
weights = initialize_weights(2, 8, 5, 1)

p = predict(weights, X, f1, f2, f3)
np.testing.assert_allclose(np.sum(p), 645)
print("Looks like your implementation is working, congratulation!")

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C6 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def compute_loss(nn_output, Y):
    # This function takes the nn_outputs and Ys of the training set
    # and returns the log loss
    N = Y.shape[1]
    return (-1./N) * np.sum(Y * np.log(nn_output) + (1 - Y) * np.log(1 - nn_output))

f1 = actFunctionLinear # function used for hidden layer 1
f2 = actFunctionLinear # function used for hidden layer 2
f3 = actFunctionSigmoid # function used for output layer
np.random.seed(42) # Make sure to have a reproducible result
X, Y = generate_dataset("blobs", 1000, 0.5, True)
np.random.seed(42) # Make sure to have a reproducible result
weights = initialize_weights(2, 8, 5, 1)
outputLayer, cache = forward_propagation(X, weights, f1, f2, f3)
result = compute_loss(outputLayer, Y)
np.testing.assert_allclose(result, 0.7360633961687931)
print("Looks like your implementation is working, congratulation!")

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C7 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

def nn_model(X_train, Y_train, nodes0, nodes1, nodes2, nodes3, 
             f1, f2, f3, lossfun, learning_rate, predict, 
             num_iterations, print_cost=False, X_test=None, Y_test=None):
    """
    Arguments:
    X_train -- training dataset of shape (2, number of examples)
    Y_train -- training labels of shape (1, number of examples)
    nodes_in -- size of the input layer
    nodes_h1 -- size of the first hidden layer
    nodes_h2 -- size of the second hidden layer
    nodes_out -- size of the input layer
    f1 -- activation function for the first layer
    f2 -- activation function for the second layer
    f3 -- activation function for the third layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    X_test, Y_test -- test datasets; when set, accuracy is printed

    Returns:
    weights -- weights learnt by the model. 
               They can then be used to predict.
    """

    #np.random.seed(42 if DISABLE_RANDOM_SEED else None)    
    
    # Use the previously coded function to generate the weights
    weights = initialize_weights(nodes0, nodes1, nodes2, nodes3)
    
    # Loop (training with gradient descent)
    for i in range(1, num_iterations + 1):
        # 2. forward propagation; the variable holding the network's
        #    outputs should be called outputLayer
        outputLayer, cache = forward_propagation(X_train, weights, f1, f2, f3)
        # 3. use the results to calculate the backward propagation
        grads = backward_propagation(weights, cache, X_train, Y_train)

        # 4. update the weights according to the results of the
        #    backward propagation
        weights = update_weights(weights, grads, learning_rate)

        # Print the cost and accuracy every 500 iterations
        if print_cost and i % 500 == 0:      
            accStr = ''
            
            try:
                predictions = predict(weights, X_test, f1, f2, f3)
                
                if (X_test is not None) and (Y_test is not None ):
                    accuracy = accuracy_score(predictions.flatten(),
                                              Y_test.flatten())
                    accStr = "(Acc: %f)" %(accuracy)
            except:
                pass # will be implemented later; no action                
                
            loss = lossfun(outputLayer, Y_train)
            print("Cost after iteration %i: %f %s" %(i, loss, accStr))

    return weights

f1 = actFunctionLinear # function used for hidden layer 1
f2 = actFunctionLinear # function used for hidden layer 2
f3 = actFunctionSigmoid # function used for output layer
loss_fun = compute_loss
predict_fun = predict
np.random.seed(42) # Make sure to have a reproducible result
X, Y = generate_dataset("blobs", 1000, 0.4, False)
np.random.seed(42) # Make sure to have a reproducible result
weights = nn_model(X, Y, 2, 5, 5, 1, f1, f2, f3, loss_fun, 
                   0.05, predict_fun, 1000)

result = np.array([np.mean(weights["A1"]), np.mean(weights["A2"]),
                   np.mean(weights["A3"]), np.mean(weights["b1"]),
                   np.mean(weights["b2"]), np.mean(weights["b3"])])
reference = np.array([0.28620967900953265, -0.13481751949641932,
                     -0.4163217207836666, 0.014763485056667431,
                     -0.1323179529728214, 0.33857324167613034])
np.testing.assert_allclose(result, reference)
print("Looks like your implementation is working, congratulation!")

# -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+ C9 -+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+

datasetName = 'spiral2' # see function generateDataset()
nrSamplePoints = 1000
noise_factor = 0.1
normalize_data = True

# neural network architecture options
f1 = actFunctionRelu
f2 = actFunctionRelu

# learning options
learning_rate = 0.1
nrIterations = 10000

# do not change the following settings
f3 = actFunctionSigmoid
nodes0 = 2 # nr of nodes of the input layer
nodes3 = 1 # nr of nodes of the output layer
np.random.seed(42) # Make sure to have a reproducible result
X, y = generate_dataset(datasetName, nrSamplePoints, 
                        noise_factor, normalize_data)
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, 
                        test_size=0.2, random_state=42)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

nodes1 = 15
nodes2 = 10

np.random.seed(42) # Make sure to have a reproducible result
weights = nn_model(X_train, y_train, nodes0, nodes1, nodes2, nodes3,
                   f1, f2, f3, loss_fun, learning_rate, predict_fun,
                   nrIterations, True, X_test, y_test)
predictions = predict(weights, X_test, f1, f2, f3)
accuracy = accuracy_score(predictions.flatten(), y_test.flatten())

#config InlineBackend.figure_formats = ["svg"]

plot_decision_boundary(lambda x: predict(weights, x.T, f1, f2, f3),
                       X_train, y_train)