import numpy as np

class Node:
    """Decision Tree Node"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Decision Tree Classifier using CART algorithm
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        m = len(y)
        if m == 0:
            return 0
        p = np.sum(y == 1) / m
        return 2 * p * (1 - p)
    
    def _entropy(self, y):
        """Calculate entropy"""
        m = len(y)
        if m == 0:
            return 0
        p = np.sum(y == 1) / m
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def _split(self, X, y, feature, threshold):
        """Split dataset based on feature and threshold"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask
    
    def _information_gain(self, y, left_mask, right_mask):
        """Calculate information gain"""
        parent_impurity = self._gini_impurity(y)
        
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        left_impurity = self._gini_impurity(y[left_mask])
        right_impurity = self._gini_impurity(y[right_mask])
        
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Find the best split for the data"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask, right_mask = self._split(X, y, feature, threshold)
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(y, left_mask, right_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = np.round(np.mean(y))
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.round(np.mean(y))
            return Node(value=leaf_value)
        
        # Split the data
        left_mask, right_mask = self._split(X, y, best_feature, best_threshold)
        
        # Recursively build left and right subtrees
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """Predict single sample"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict class labels"""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def predict_proba(self, X):
        """Predict probability (for compatibility, returns 0 or 1)"""
        return self.predict(X).astype(float)


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate test data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    print("Testing Decision Tree...")
    dt = DecisionTree(max_depth=10, min_samples_split=2)
    dt.fit(X, y)
    predictions = dt.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Decision Tree Accuracy: {accuracy:.2%}")
