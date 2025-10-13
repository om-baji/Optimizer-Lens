import numpy as np

class LogisticRegression:
    
    def __init__(self,learning_rate = 0.01,iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def _sigmoid(z):
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
        return self._sigmoid(np.dot(self.weights, X) + self.bias)
    
    def predict_proba(self, X):
        return self._compute_prediction(X)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)