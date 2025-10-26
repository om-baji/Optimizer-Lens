import numpy as np

class SVM:
    """
    Support Vector Machine for binary classification
    Using SMO (Sequential Minimal Optimization) simplified version
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=0.1, degree=3, coef0=0.0, max_iter=1000, tol=1e-3):
        self.C = C  # Regularization parameter
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        
    def _kernel_function(self, x1, x2):
        """Compute kernel function"""
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # RBF (Gaussian) kernel
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
            dist = np.sum((x1[:, np.newaxis] - x2) ** 2, axis=2)
            return np.exp(-self.gamma * dist)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2.T) + self.coef0)
        else:
            return np.dot(x1, x2.T)
    
    def _decision_function(self, X):
        """Compute decision function"""
        if self.support_vectors is None:
            return np.zeros(X.shape[0])
        
        kernel_vals = self._kernel_function(X, self.support_vectors)
        decision = np.sum(self.support_vector_alphas * self.support_vector_labels * kernel_vals, axis=1) + self.b
        return decision
    
    def fit(self, X, y):
        """
        Train SVM using simplified SMO algorithm
        """
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and 1
        y = np.where(y <= 0, -1, 1)
        
        # Initialize alphas
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Kernel matrix
        K = self._kernel_function(X, X)
        
        # Simplified SMO
        for iteration in range(self.max_iter):
            alpha_changed = 0
            
            for i in range(n_samples):
                # Calculate error
                E_i = self._decision_function(X[i:i+1])[0] - y[i]
                
                # Check KKT conditions
                if ((y[i] * E_i < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * E_i > self.tol and self.alpha[i] > 0)):
                    
                    # Select random j != i
                    j = np.random.choice([k for k in range(n_samples) if k != i])
                    E_j = self._decision_function(X[j:j+1])[0] - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                break
        
        # Store support vectors
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = self.alpha[sv_indices]
    
    def predict(self, X):
        """Predict class labels"""
        decision = self._decision_function(X)
        return np.where(decision >= 0, 1, 0)
    
    def predict_proba(self, X):
        """Predict probability estimates (using Platt scaling approximation)"""
        decision = self._decision_function(X)
        # Simple sigmoid approximation
        proba = 1 / (1 + np.exp(-decision))
        return proba
    
    def get_params(self):
        """Return model parameters"""
        return {
            'support_vectors': self.support_vectors,
            'support_vector_labels': self.support_vector_labels,
            'support_vector_alphas': self.support_vector_alphas,
            'b': self.b
        }


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Test Linear SVM
    print("Testing Linear SVM...")
    svm = SVM(C=1.0, kernel='linear', max_iter=100)
    svm.fit(X, y)
    predictions = svm.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Linear SVM Accuracy: {accuracy:.2%}")
    
    # Test RBF SVM
    print("\nTesting RBF SVM...")
    svm_rbf = SVM(C=1.0, kernel='rbf', gamma=0.1, max_iter=100)
    svm_rbf.fit(X, y)
    predictions = svm_rbf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"RBF SVM Accuracy: {accuracy:.2%}")
