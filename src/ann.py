import numpy as np

class ANN:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: [input_dim, h1, h2, ..., output_dim]
        activations: list of activation names, one per HIDDEN layer
                     e.g. ["relu", "tanh"] (output layer is always linear)
        """
        self.layer_sizes = layer_sizes
        self.activations = activations

        # Weight and bias shapes
        self.weight_shapes = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            self.weight_shapes.append((in_size, out_size))

        self.bias_shapes = [(1, size) for size in layer_sizes[1:]]

        # Weights/biases placeholders (set later by PSO)
        self.weights = None
        self.biases = None

    # -------------------------------------------------------------
    # Activation functions
    # -------------------------------------------------------------
    def activation(self, x, name):
        if name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif name == "tanh":
            return np.tanh(x)
        elif name == "relu":
            return np.maximum(0, x)
        elif name == "linear":
            return x
        else:
            raise ValueError(f"Unknown activation: {name}")

    # -------------------------------------------------------------
    # Required by PSO: return total number of parameters
    # -------------------------------------------------------------
    def num_params(self):
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
        idx = 0
        self.weights = []
        self.biases = []

        for shape in self.weight_shapes:
            size = shape[0] * shape[1]
            w = vector[idx:idx + size].reshape(shape)
            self.weights.append(w)
            idx += size

        for shape in self.bias_shapes:
            size = shape[1]
            b = vector[idx:idx + size].reshape(shape)
            self.biases.append(b)
            idx += size

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------
    def forward(self, X):
        a = X
        L = len(self.weights)

        for i in range(L):
            z = np.dot(a, self.weights[i]) + self.biases[i]

            # Last layer must be linear for regression
            if i == L - 1:
                a = self.activation(z, "linear")
            else:
                a = self.activation(z, self.activations[i])

        return a

    def predict(self, X):
        return self.forward(X)
