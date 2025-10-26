import numpy as np

class NeuralNetwork:
    """
    Multi-layer Neural Network for binary classification
    """
    
    def __init__(self, hidden_layers=[64, 32], activation='relu', learning_rate=0.01, 
                 epochs=1000, batch_size=32):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        
    def _activation_function(self, z, derivative=False):
        """Apply activation function"""
        if self.activation == 'relu':
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(z) ** 2
            return np.tanh(z)
        elif self.activation == 'leaky_relu':
            if derivative:
                return np.where(z > 0, 1, 0.01)
            return np.where(z > 0, z, 0.01 * z)
        else:
            return z
    
    def _initialize_weights(self, n_features):
        """Initialize weights using He initialization"""
        layer_sizes = [n_features] + self.hidden_layers + [1]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward_propagation(self, X):
        """Forward pass through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self._activation_function(z)
            activations.append(a)
        
        # Output layer (sigmoid)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
        activations.append(output)
        
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        """Backward pass to compute gradients"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient
        delta = activations[-1] - y.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_function(z_values[i-1], derivative=True)
        
        return gradients_w, gradients_b
    
    def fit(self, X, y):
        """Train the neural network"""
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward propagation
                activations, z_values = self._forward_propagation(X_batch)
                
                # Backward propagation
                gradients_w, gradients_b = self._backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update weights
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients_w[j]
                    self.biases[j] -= self.learning_rate * gradients_b[j]
    
    def predict_proba(self, X):
        """Predict probability estimates"""
        activations, _ = self._forward_propagation(X)
        return activations[-1].flatten()
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def get_params(self):
        """Return model parameters"""
        return {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate test data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    print("Testing Neural Network...")
    nn = NeuralNetwork(hidden_layers=[64, 32], activation='relu', 
                      learning_rate=0.01, epochs=100, batch_size=32)
    nn.fit(X, y)
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Neural Network Accuracy: {accuracy:.2%}")
    print(f"Network architecture: {X.shape[1]} -> {nn.hidden_layers} -> 1")
