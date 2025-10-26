import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    """
    Random Forest Classifier
    """
    
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features):
        """Calculate max features for splitting"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features
    
    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Train decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        """Predict using majority voting"""
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote
        predictions = np.round(np.mean(tree_predictions, axis=0))
        return predictions
    
    def predict_proba(self, X):
        """Predict probability estimates"""
        # Average predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate test data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    print("Testing Random Forest...")
    rf = RandomForest(n_estimators=10, max_depth=10)
    rf.fit(X, y)
    predictions = rf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Random Forest Accuracy: {accuracy:.2%}")
    print(f"Number of trees: {len(rf.trees)}")
