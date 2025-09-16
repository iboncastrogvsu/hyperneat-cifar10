import numpy as np

def xavier_init(in_dim, out_dim):
    """Xavier initialization for weight matrices."""
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))

class Genome:
    """
    Compact encoding of a neural network (a CPPN).

    Stores weight matrices and biases for a small MLP, which
    maps substrate connection features → connection weights.
    """
    def __init__(self, input_dim=6, hidden_dims=(32, 32, 16)):
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        dims = [input_dim] + list(hidden_dims) + [1]
        self.weights, self.biases = [], []
        for i in range(len(dims)-1):
            self.weights.append(xavier_init(dims[i], dims[i+1]))
            self.biases.append(np.zeros((dims[i+1],), dtype=float))

    @classmethod
    def random(cls, input_dim=6, hidden_dims=(32,32,16)):
        """Factory method to sample a random genome."""
        np.random.seed()
        return cls(input_dim=input_dim, hidden_dims=hidden_dims)

    def copy(self):
        """Return a deep copy of the genome."""
        g = Genome(self.input_dim, self.hidden_dims)
        g.weights = [w.copy() for w in self.weights]
        g.biases = [b.copy() for b in self.biases]
        return g

    def mutate(self, weight_sigma=0.1, bias_sigma=0.02, rate=0.1):
        """
        Apply Gaussian noise mutations to weights and biases.

        Args:
            weight_sigma (float): Std for weight mutations.
            bias_sigma (float): Std for bias mutations.
            rate (float): Probability of mutation per parameter.
        """
        for i in range(len(self.weights)):
            mask = np.random.rand(*self.weights[i].shape) < rate
            self.weights[i] += mask * np.random.normal(scale=weight_sigma, size=self.weights[i].shape)
            maskb = np.random.rand(*self.biases[i].shape) < rate
            self.biases[i] += maskb * np.random.normal(scale=bias_sigma, size=self.biases[i].shape)

    def crossover(self, other):
        """
        Create a child genome by combining weights/biases
        with uniform crossover.
        """
        child = self.copy()
        for i in range(len(self.weights)):
            mask = np.random.rand(*self.weights[i].shape) < 0.5
            child.weights[i][mask] = other.weights[i][mask]
            maskb = np.random.rand(*self.biases[i].shape) < 0.5
            child.biases[i][maskb] = other.biases[i][maskb]
        return child

    def forward(self, inputs):
        """
        Forward pass through the CPPN to produce connection weights.

        Args:
            inputs (np.ndarray): Shape (n_edges, input_dim).
        Returns:
            np.ndarray: Output values (connection weights).
        """
        a = inputs
        for i in range(len(self.weights)-1):
            a = np.tanh(a.dot(self.weights[i]) + self.biases[i])
        out = a.dot(self.weights[-1]) + self.biases[-1]
        return out
