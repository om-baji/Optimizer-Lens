import numpy as np
from decision_tree import DecisionTree

class GradientBoosting:
    """
    Gradient Boosting Classifier for binary classification
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None
        
    def _sigmoid(self, z):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _log_loss_gradient(self, y, y_pred):
        """Compute gradient of log loss"""
        return y - y_pred
    
    def fit(self, X, y):
        """Train the gradient boosting model"""
        # Initialize with log odds
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        if pos_count == 0 or neg_count == 0:
            self.initial_prediction = 0.0
        else:
            self.initial_prediction = np.log(pos_count / neg_count)
        
        # Current predictions (in log-odds space)
        F = np.full(len(y), self.initial_prediction)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Convert to probabilities
            p = self._sigmoid(F)
            
            # Calculate residuals (negative gradient)
            residuals = self._log_loss_gradient(y, p)
            
            # Fit a tree to residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            
            # Create regression targets (residuals)
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            
            self.trees.append(tree)
    
    def _predict_raw(self, X):
        """Predict raw scores (log-odds)"""
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add predictions from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict probability estimates"""
        raw_predictions = self._predict_raw(X)
        return self._sigmoid(raw_predictions)
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate test data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    print("Testing Gradient Boosting...")
    gb = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
    gb.fit(X, y)
    predictions = gb.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Gradient Boosting Accuracy: {accuracy:.2%}")
    print(f"Number of trees: {len(gb.trees)}")
