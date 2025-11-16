import numpy as np

class ANN:
    def __init__(self, layer_sizes, activations):
        
        #layer size = list specifying no of neurons in each layer [3(input),5(hidden layer),2(output)] 
        #output layer is always linear 
        """
        layer_sizes: [input_dim, h1, h2, ..., output_dim]
        activations: list of activation names, one per HIDDEN layer
                     e.g. ["relu", "tanh"] (output layer is always linear)
        """
        #storing layer sizes and activation in the object for later use 
        self.layer_sizes = layer_sizes
        self.activations = activations

        # Weight and bias shapes
        self.weight_shapes = [] #initialises empty list to store the shapes of weight matrices for each layer 
        for i in range(len(layer_sizes) - 1): #loops thoguh each layer to determine shape of the weight matrix connecting to next layer 
            in_size = layer_sizes[i] #no of neurons in layer i
            out_size = layer_sizes[i + 1] #no of nuerons in layer i+1 
            self.weight_shapes.append((in_size, out_size))

        self.bias_shapes = [(1, size) for size in layer_sizes[1:]] #creates a list of bias vector shapes. eahc bias is a row vector (1, no of neurons in layer)

        # Weights/biases placeholders (set later by PSO)
        self.weights = None
        self.biases = None

    # -------------------------------------------------------------
    # Activation functions
    # -------------------------------------------------------------
    def activation(self, x, name):
        if name == "sigmoid": #outputs between 0 and 1 ; non linear
            return 1 / (1 + np.exp(-x))
        elif name == "tanh": #outputs (-1,1) ; non -linear 
            return np.tanh(x)
        elif name == "relu": #outputs (0,x) ; non-linear 
            return np.maximum(0, x)
        elif name == "linear": #outputs x
            return x
        else:
            raise ValueError(f"Unknown activation: {name}")

    # -------------------------------------------------------------
    # Required by PSO: return total number of parameters
    # -------------------------------------------------------------
    def num_params(self): #calaculates total no of parameters (weights + biases) in network, need to know how long the vector of particles should be 
        total = 0
        for shape in self.weight_shapes:
            total += shape[0] * shape[1]
        for shape in self.bias_shapes:
            total += shape[1]
        return total

    # -------------------------------------------------------------
    # Required by PSO: set weights from a flat vector
    # -------------------------------------------------------------
    def set_param_vector(self, vector):
        """
        Convert a 1D vector (from PSO) into all weight matrices + bias vectors.
        """
        #initialises index counter and empty list to stpre weight and biases 
        idx = 0
        self.weights = []
        self.biases = []

        for shape in self.weight_shapes: #for each weight matrix shape, slice the corresponding vales from the flat vector and reshape to correct size 
            size = shape[0] * shape[1]
            w = vector[idx:idx + size].reshape(shape)
            self.weights.append(w) #append the matrix here 
            idx += size #increment to move through the vector 

        for shape in self.bias_shapes: #same thing happening here, but for biases 
            size = shape[1]
            b = vector[idx:idx + size].reshape(shape)
            self.biases.append(b)
            idx += size

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------
    def forward(self, X):
        a = X # x = input data , a stores the current layers output 
        L = len(self.weights) # total no of layers (excluding input layer)

        for i in range(L): 
            z = np.dot(a, self.weights[i]) + self.biases[i] #computing linear combination 

            # Last layer must be linear for regression
            if i == L - 1:
                a = self.activation(z, "linear") #applying activation func
            else:
                a = self.activation(z, self.activations[i]) #linear output 

        return a #returns the final output of hte netwrok 

    def predict(self, X):
        return self.forward(X) 
