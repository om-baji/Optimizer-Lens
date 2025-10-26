import numpy as np

class LogisticRegression:
    
    def __init__(self,learning_rate = 0.01,iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_gradient(self,X : np.array, y : np.array, y_pred : np.array) -> tuple[float,float]:
        n_samples = X.shape[0]
        error = y_pred - y
        
        dw = (1/ n_samples) * np.dot(X.T, error)
        db = (1/ n_samples) * np.sum(error)
        
        return dw, db
    
    def _update_params(self,dw,db):
        
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def _compute_prediction(self,X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict_proba(self, X):
        return self._compute_prediction(X)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = self._compute_prediction(X)
            dw, db = self._compute_gradient(X, y, y_pred)
            self._update_params(dw, db)
    
    def get_params(self):
        return self.weights, self.bias

# Test the implementation
if __name__ == "__main__":
    # Simple binary classification test
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    lr = LogisticRegression(learning_rate=0.1, iterations=1000)
    lr.fit(X, y)
    
    predictions = lr.predict(X)
    print("Predictions:", predictions)
    print("Actual:", y)
    print("Accuracy:", np.mean(predictions == y))
    print("Weights:", lr.weights)
    print("Bias:", lr.bias)