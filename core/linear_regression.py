import numpy as np

class LinearRegression:
    
    def __init__(self,learning_rate = 0.01,iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def _compute_gradient(self, X : np.array, y : np.array, y_pred : np.array) -> tuple[float,float]:
        
        n_samples = X.shape[0]
        error = y_pred - y
        
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        return dw, db
    
    def _compute_stochastic_gradient(self, X : np.array, yi : float) -> tuple[float,float]:
        
        y_pred_i = self._compute_prediction(X)
        error = y_pred_i - yi
        
        dw = X * error
        db = error
        return dw, db
            
    def _compute_prediction(self,X : np.array) :
        return np.dot(X, self.weights) + self.bias
    
    def _update_params(self, dw: np.ndarray, db: float):
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            
            y_pred = self._compute_prediction(X)
            dw, db = self._compute_gradient(X, y, y_pred)
            self._update_params(dw,db)
            
    def fit_sgd(self,X,y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for i in indices:
                xi = X[i]
                yi = y[i]
                dw, db = self._compute_stochastic_gradient(xi,yi)
                self._update_params(dw,db)
    
    def predict(self,X):
        return self._compute_prediction(X)
    
    def get_params(self):
        return self.weights,self.bias
    
    @staticmethod
    def mean_sq_err(y_true,y):
        return np.mean((y_true - y) ** 2)
   
X = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9]) 

lr = LinearRegression(0.01)
lr.fit_sgd(X,y)

print(lr.predict(8))