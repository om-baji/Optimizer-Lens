"""
Test all ML algorithms
"""
import numpy as np
import sys

# Simple test data generator
def make_test_data(n_samples=100):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

print("=" * 60)
print("Testing All ML Algorithms")
print("=" * 60)

# Test Linear Regression
print("\n1. Linear Regression")
from linear_regression import LinearRegression
X, y = make_test_data()
model = LinearRegression(learning_rate=0.01, iterations=100)
model.fit(X, y)
predictions = model.predict(X)
print(f"   ✓ Trained successfully")
print(f"   Predictions shape: {predictions.shape}")

# Test Logistic Regression
print("\n2. Logistic Regression")
from logistic import LogisticRegression
X, y = make_test_data()
model = LogisticRegression(learning_rate=0.1, iterations=100)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

# Test SVM
print("\n3. Support Vector Machine (Linear)")
from svm import SVM
X, y = make_test_data()
model = SVM(C=1.0, kernel='linear', max_iter=100)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

# Test RBF SVM
print("\n4. Support Vector Machine (RBF)")
model = SVM(C=1.0, kernel='rbf', gamma=0.1, max_iter=100)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

# Test Neural Network
print("\n5. Neural Network")
from neural_network import NeuralNetwork
X, y = make_test_data(200)
model = NeuralNetwork(hidden_layers=[32, 16], activation='relu', 
                     learning_rate=0.01, epochs=100, batch_size=32)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

# Test Decision Tree
print("\n6. Decision Tree")
from decision_tree import DecisionTree
X, y = make_test_data()
model = DecisionTree(max_depth=5, min_samples_split=2)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

# Test Random Forest
print("\n7. Random Forest")
from random_forest import RandomForest
X, y = make_test_data()
model = RandomForest(n_estimators=10, max_depth=5)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

# Test Gradient Boosting
print("\n8. Gradient Boosting")
from gradient_boosting import GradientBoosting
X, y = make_test_data()
model = GradientBoosting(n_estimators=30, learning_rate=0.1, max_depth=3)
model.fit(X, y)
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"   ✓ Trained successfully")
print(f"   Accuracy: {accuracy:.2%}")

print("\n" + "=" * 60)
print("✅ All Algorithms Working!")
print("=" * 60)
